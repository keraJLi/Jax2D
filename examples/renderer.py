from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax2d import joint
from jax2d.engine import select_shape
from jax2d.sim_state import RigidBody
from examples.textures import (
    FJOINT_TEXTURE_16_RGBA,
    RJOINT_TEXTURE_16_RGBA,
    THRUSTER_TEXTURE_16_RGBA,
    RJOINT_TEXTURE_RGBA,
    THRUSTER_TEXTURE_RGBA,
    FJOINT_TEXTURE_RGBA,
)
from flax import struct


def get_fragments_and_positions(params, dimensions: tuple[int, int], downscale: int):
    downscaled_dim = (dimensions[0] // downscale, dimensions[1] // downscale)

    fragment_xs = jnp.arange(dimensions[0])[::downscale]  # / params.pixels_per_unit
    fragment_xs = jnp.repeat(fragment_xs[:, None], downscaled_dim[1], axis=1)

    fragment_ys = jnp.arange(dimensions[1])[::downscale]  # / params.pixels_per_unit
    fragment_ys = jnp.repeat(fragment_ys[None, :], downscaled_dim[0], axis=0)

    fragment_positions = jnp.stack([fragment_xs, fragment_ys], axis=0)
    fragment_positions = fragment_positions.transpose((1, 2, 0))
    fragment_positions_flat = fragment_positions.reshape((-1, 2))

    return fragment_positions_flat, downscaled_dim


def make_render_pixels(params, static_params, pixel_upscale=2):
    RENDER_VERTICES = False
    RENDER_POLYGON_CENTER = False
    OVERRIDE_COLOUR = jnp.array([128.0, 0.0, 128.0])
    screen_dim = static_params.screen_dim
    downscale = static_params.downscale

    if pixel_upscale == 2:
        RJOINT_TEX = RJOINT_TEXTURE_RGBA
        FJOINT_TEX = FJOINT_TEXTURE_RGBA
        THRUSTER_TEX = THRUSTER_TEXTURE_RGBA
    elif pixel_upscale > 2 and pixel_upscale % 2 == 0:
        RJOINT_TEX = jnp.repeat(
            jnp.repeat(RJOINT_TEXTURE_RGBA, repeats=pixel_upscale // 2, axis=0), repeats=pixel_upscale // 2, axis=1
        )
        FJOINT_TEX = jnp.repeat(
            jnp.repeat(FJOINT_TEXTURE_RGBA, repeats=pixel_upscale // 2, axis=0), repeats=pixel_upscale // 2, axis=1
        )
        THRUSTER_TEX = jnp.repeat(
            jnp.repeat(THRUSTER_TEXTURE_RGBA, repeats=pixel_upscale // 2, axis=0), repeats=pixel_upscale // 2, axis=1
        )
    elif pixel_upscale > 1:
        RJOINT_TEX = jnp.repeat(
            jnp.repeat(RJOINT_TEXTURE_16_RGBA, repeats=pixel_upscale, axis=0), repeats=pixel_upscale, axis=1
        )
        FJOINT_TEX = jnp.repeat(
            jnp.repeat(FJOINT_TEXTURE_16_RGBA, repeats=pixel_upscale, axis=0), repeats=pixel_upscale, axis=1
        )
        THRUSTER_TEX = jnp.repeat(
            jnp.repeat(THRUSTER_TEXTURE_16_RGBA, repeats=pixel_upscale, axis=0), repeats=pixel_upscale, axis=1
        )
    else:
        RJOINT_TEX = RJOINT_TEXTURE_16_RGBA
        FJOINT_TEX = FJOINT_TEXTURE_16_RGBA
        THRUSTER_TEX = THRUSTER_TEXTURE_16_RGBA

    CYAN = jnp.array([80, 80, 80])
    joint_colours = jnp.array(
        [
            [255, 255, 0],  # yellow
            [255, 0, 255],  # purple/magenta
            [0, 255, 255],  # cyan
            [255, 153, 51],  # white
            [120, 120, 120],
        ]
    )

    ROLE_COLOURS = jnp.array(
        [
            [160.0, 160.0, 160.0],  # None
            [0.0, 204.0, 0.0],  # Green:    The ball
            [0.0, 102.0, 204.0],  # Blue:   The goal
            [255.0, 102.0, 102.0],  # Red:      Death Objects
        ]
    )
    MAX_SIZE = static_params.max_shape_size * params.pixels_per_unit
    MAX_SIZE_ARR = jnp.array([MAX_SIZE, MAX_SIZE])
    EDGE_VALUE = jnp.maximum(downscale // 2, 2) / params.pixels_per_unit * pixel_upscale / 2

    # BACKGROUND_COLOUR = jnp.array([135.0, 206.0, 235.0])
    BACKGROUND_COLOUR = jnp.array([255.0, 255.0, 255.0])

    large_frags_dim, temp_large_dim = get_fragments_and_positions(params, screen_dim, downscale)

    frag_in_chunk_pos_flat, downscaled_chunk_dim = get_fragments_and_positions(params, (MAX_SIZE, MAX_SIZE), downscale)
    DOWNSCALE_CHUNK_DIM_ARR = jnp.array(downscaled_chunk_dim)

    # Precompute Distances
    all_distances = jnp.sqrt(
        (frag_in_chunk_pos_flat[:, 0] - MAX_SIZE // 2) ** 2 + (frag_in_chunk_pos_flat[:, 1] - MAX_SIZE // 2) ** 2
    )

    downscaled_dim = (
        (screen_dim[0] + MAX_SIZE * 2) // downscale,
        (screen_dim[1] + MAX_SIZE * 2) // downscale,
    )

    fragments_in_chunk_flat = jnp.zeros((downscaled_chunk_dim[0] * downscaled_chunk_dim[1], 3), dtype=jnp.float32) - 1
    fragments_flat_temp = (
        jnp.ones((downscaled_dim[0] * downscaled_dim[1], 3), dtype=jnp.float32) * BACKGROUND_COLOUR[None, :]
    )

    # Precompute this
    circ_angles = jnp.linspace(0, jnp.pi, 1, endpoint=False)
    circ_x = jnp.sin(circ_angles)
    circ_y = jnp.cos(circ_angles)

    def _get_colour(colour, shape_role, inverse_inertia):
        f = (inverse_inertia == 0) * 1
        is_not_normal = (shape_role != 0) * 1
        col = jnp.array(
            [
                colour,
                colour,
                CYAN,
                colour * 0.5,
            ]
        )[2 * f + is_not_normal]
        return col

    @jax.jit
    def render_pixels(state):
        fragments_flat = fragments_flat_temp

        # GENERAL
        def _make_shape_renderer_small(
            fragment_shader,
            shapes,
            shape_roles,
            shape_highlights,
            MAX_SIZE=MAX_SIZE_ARR,
            fragment_positions_to_use=frag_in_chunk_pos_flat,
            distances_to_use=all_distances,
        ):
            s = jnp.sin(shapes.rotation)
            c = jnp.cos(shapes.rotation)

            all_precompute = {
                "sin": s,
                "cos": c,
                "rmat": jax.vmap(_rmat_sc)(s, c),
                "inv_rmat": jax.vmap(_inv_rmat_sc)(s, c),
            }

            def _render_shape(fragments_flat, shape_index):
                # select a particular shape
                shape = jax.tree.map(lambda x: x[shape_index], shapes)
                shape_role = jax.tree.map(lambda x: x[shape_index], shape_roles)
                highlighted = shape_highlights[shape_index]
                precompute = jax.tree.map(lambda x: x[shape_index], all_precompute)

                # And for each pixel, see if we should (a) change its colour or (b) keep it the same
                fragments_flat = jax.vmap(fragment_shader, in_axes=(None, None, None, None, 0, 0, 0, None))(
                    shape,
                    shape_role,
                    highlighted,
                    precompute,
                    fragment_positions_to_use,
                    distances_to_use,
                    fragments_flat,
                    MAX_SIZE,
                )

                return fragments_flat, None

            return _render_shape

        @jax.jit
        def paint_single_chunk(carry, value):
            # This paints the small chunk into the large image.
            big_fragment = carry
            position, small_fragment = value

            # Location of chunk inside the big image
            where_to_put_rect = (position * params.pixels_per_unit - MAX_SIZE // 2).astype(
                jnp.int32
            ) // downscale + DOWNSCALE_CHUNK_DIM_ARR

            # Get the small chunk, possibly mask it out
            mask = jax.lax.select(
                where_to_put_rect.min() < 0,
                jnp.zeros_like(small_fragment, dtype=jnp.bool_),
                small_fragment >= 0,
            )

            # Get the corresponding slice from the large image
            slice_from_big = jax.lax.dynamic_slice(
                big_fragment,
                (where_to_put_rect[0], where_to_put_rect[1], 0),
                (downscaled_chunk_dim[0], downscaled_chunk_dim[1], 3),
            )

            # And update it
            updated_big = jax.lax.dynamic_update_slice(
                big_fragment,
                (small_fragment * mask + (1 - mask) * slice_from_big),
                (where_to_put_rect[0], where_to_put_rect[1], 0),
            )
            return updated_big, None

        # POLYGON
        def _signed_line_distance(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def _dist_from_line(a, b, c):
            return jnp.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def _rmat_sc(s, c):
            return jnp.array([[c, -s], [s, c]])

        def _inv_rmat_sc(s, c):
            return jnp.array([[c, s], [-s, c]])

        def _polygon_fragment_shader_small(
            polygon: RigidBody,
            shape_role,
            highlighted,
            precompute,
            fragment_position,
            fragment_distance,
            existing_colour,
            MAX_SIZE,
        ):
            M = precompute["inv_rmat"]
            pos = M @ (fragment_position - MAX_SIZE // 2) / params.pixels_per_unit

            next_vertices = jnp.concatenate([polygon.vertices[1:], polygon.vertices[:1]], axis=0)
            next_vertices = next_vertices.at[polygon.n_vertices - 1].set(polygon.vertices[0])

            edge_value = EDGE_VALUE  # * (highlighted * 3 + 2)
            a = _signed_line_distance(pos, polygon.vertices[0], next_vertices[0]) / jnp.linalg.norm(
                polygon.vertices[0] - next_vertices[0]
            )
            computed_colour = _signed_line_distance(pos, polygon.vertices[1], next_vertices[1]) / jnp.linalg.norm(
                polygon.vertices[1] - next_vertices[1]
            )
            c = _signed_line_distance(pos, polygon.vertices[2], next_vertices[2]) / jnp.linalg.norm(
                polygon.vertices[2] - next_vertices[2]
            )
            d = _signed_line_distance(pos, polygon.vertices[3], next_vertices[3]) / jnp.linalg.norm(
                polygon.vertices[3] - next_vertices[3]
            )
            is_inside_rect = (a <= 0) & (computed_colour <= 0) & (c <= 0) & ((d <= 0) | (polygon.n_vertices < 4))

            should_render_special_points = False
            if RENDER_VERTICES:
                for i in range(static_params.max_polygon_vertices):
                    should_render_special_points = should_render_special_points | (
                        (polygon.n_vertices >= i + 1) & (jnp.linalg.norm(pos - polygon.vertices[i]) < EDGE_VALUE * 1.5)
                    )

            on_edge = (
                ((-edge_value <= a) & (a <= 0))
                | ((-edge_value <= computed_colour) & (computed_colour <= 0))
                | ((-edge_value <= c) & (c <= 0))
                | ((-edge_value <= d) & (d <= 0) & (polygon.n_vertices == 4))
            )
            # on_edge &= False

            edge_colour = ((highlighted + (polygon.collision_mode == 0)) > 0) * jnp.array([255.0, 255.0, 255.0])
            col = _get_colour(ROLE_COLOURS[shape_role], shape_role, polygon.inverse_inertia)
            computed_colour = jax.lax.select(
                is_inside_rect & polygon.active,
                (col) * (1 - on_edge) + on_edge * edge_colour,
                existing_colour,
            )
            if RENDER_POLYGON_CENTER:
                should_render_special_points = should_render_special_points | (jnp.linalg.norm(pos) < EDGE_VALUE * 1.5)

            return jax.lax.select(
                should_render_special_points & polygon.active, jnp.array([128.0, 0.0, 128.0]), computed_colour
            )

        _render_polygon = _make_shape_renderer_small(
            _polygon_fragment_shader_small, state.polygon, state.polygon_shape_roles, state.polygon_highlighted
        )
        rect_fragments = jax.vmap(_render_polygon, in_axes=(None, 0))(
            fragments_in_chunk_flat, jnp.arange(static_params.num_polygons - 1) + 1  # ignore the first polygon
        )[0]

        # CIRCLE
        def _make_circle_fragment_shader(should_render_only_part_of_circle=False, override_colour=None):
            def _circle_fragment_shader_small(
                circle: RigidBody,
                shape_role,
                highlighted,
                precompute,
                fragment_position,
                fragment_distance,
                existing_colour,
                MAX_SIZE,
                min_angle,
                max_angle,
            ):
                distance = fragment_distance / params.pixels_per_unit
                inside = distance <= circle.radius
                M = precompute["rmat"]
                if should_render_only_part_of_circle:
                    diff = fragment_position - MAX_SIZE // 2
                    angle = -jnp.arctan2(diff[1], diff[0]) + circle.rotation
                    angle = (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                    inside = inside & (angle >= min_angle) & (angle <= max_angle)

                on_edge = circle.radius - distance <= EDGE_VALUE  # * (highlighted * 3 + 2)
                # on_edge &= False

                x = circ_x * circle.radius
                y = circ_y * circle.radius

                def single(M, x, y):
                    line_end = jnp.matmul(M, jnp.array([x, y])) + MAX_SIZE // 2
                    on_line = (
                        _dist_from_line(
                            MAX_SIZE // 2,
                            line_end,
                            fragment_position,
                        )
                        <= EDGE_VALUE * circle.radius / 2 * params.pixels_per_unit
                    )
                    # print("shapes", M.shape, line_end.shape, MAX_SIZE.shape, fragment_position.shape)
                    n = line_end - MAX_SIZE // 2
                    n /= jnp.linalg.norm(n)

                    point_on_line = jnp.dot(fragment_position - MAX_SIZE // 2, n)
                    # `abs` makes it go to both sides of the radius, without the abs it is only one side
                    # on_line &= jnp.abs(point_on_line) > circle.radius * 0.7 / (max(1, downscale // 2))
                    on_line &= point_on_line > circle.radius * 0.7 / (max(1, downscale // 2))

                    return on_line

                on_line = jax.vmap(single, (None, 0, 0))(M, x, y).any(axis=0)
                # on_line &= False

                edge_colour = ((highlighted + (circle.collision_mode == 0)) > 0) * jnp.array([255.0, 255.0, 255.0])

                col = _get_colour(ROLE_COLOURS[shape_role], shape_role, circle.inverse_inertia)
                col = jax.lax.select(
                    on_edge | on_line,
                    edge_colour,
                    col,
                )

                if override_colour is not None:
                    col = override_colour
                # col = jax.lax.select(override_colour, OVERRIDE_COLOUR, col)

                return jax.lax.select(
                    inside & circle.active,
                    col,
                    existing_colour,
                )

            if should_render_only_part_of_circle:
                return _circle_fragment_shader_small
            else:
                return partial(_circle_fragment_shader_small, min_angle=-jnp.pi / 2, max_angle=jnp.pi / 2)

        _render_circle = _make_shape_renderer_small(
            _make_circle_fragment_shader(False), state.circle, state.circle_shape_roles, state.circle_highlighted
        )
        circle_fragments = jax.vmap(_render_circle, in_axes=(None, 0))(
            fragments_in_chunk_flat, jnp.arange(static_params.num_circles)
        )[0]

        # Paint all of the small fragments onto the big image
        fragments = fragments_flat.reshape((downscaled_dim[0], downscaled_dim[1], 3))

        # First paint the first polygon, which may be bigger than the MAX_PIXEL_SIZE
        # This is somewhat of a hack
        _render_rect_large = _make_shape_renderer_small(
            _polygon_fragment_shader_small,
            state.polygon,
            state.polygon_shape_roles,
            state.polygon_highlighted,
            state.polygon.position[0] * 2 * params.pixels_per_unit,
            large_frags_dim,
            large_frags_dim[..., 0],
        )
        fragments2, _ = _render_rect_large(
            fragments[
                downscaled_chunk_dim[0] : -downscaled_chunk_dim[0],
                downscaled_chunk_dim[1] : -downscaled_chunk_dim[1],
            ].reshape(-1, 3),
            0,
        )
        fragments = jax.lax.dynamic_update_slice(
            fragments,
            fragments2.reshape(
                (
                    downscaled_dim[0] - 2 * downscaled_chunk_dim[0],
                    downscaled_dim[1] - 2 * downscaled_chunk_dim[1],
                    3,
                )
            ),
            (downscaled_chunk_dim[0], downscaled_chunk_dim[1], 0),
        )

        fragments, _ = jax.lax.scan(
            paint_single_chunk,
            fragments,
            (
                jnp.concatenate(
                    [state.polygon.position[1:], state.circle.position]
                ),  # since we ignore the first polygon
                jnp.concatenate(
                    [
                        rect_fragments.reshape((-1, downscaled_chunk_dim[0], downscaled_chunk_dim[1], 3)),
                        circle_fragments.reshape((-1, downscaled_chunk_dim[0], downscaled_chunk_dim[1], 3)),
                    ],
                    axis=0,
                ),
            ),
        )

        def make_joint_renderer(joints, motor_bindings):
            # joint_texture = joint_texture[::downscale, ::downscale]

            def _render_joint(fragments, joint_index):
                joint = jax.tree.map(lambda x: x[joint_index], joints)

                rjoint_tex = RJOINT_TEX[::downscale, ::downscale]
                fjoint_tex = FJOINT_TEX[::downscale, ::downscale]

                joint_texture = jax.lax.select(joint.is_fixed_joint, fjoint_tex, rjoint_tex)

                base_size = 16 * pixel_upscale
                size = base_size // downscale
                pos = jnp.round(
                    (joint.global_position * params.pixels_per_unit) // downscale
                    + jnp.array(
                        [
                            downscaled_chunk_dim[0] - base_size // 2 // downscale,
                            downscaled_chunk_dim[1] - base_size // 2 // downscale,
                        ]
                    )
                ).astype(int)

                # Get the corresponding slice from the large image
                slice_from_big = jax.lax.dynamic_slice(
                    fragments,
                    (pos[0], pos[1], 0),
                    (size, size, 3),
                )

                mask = jax.lax.select(
                    joint.is_fixed_joint | (jnp.logical_not(joint.motor_on)) | True,
                    joint_texture[:, :, 3] * joint.active,
                    jnp.ones_like(joint_texture[:, :, 3]) * joint.active,
                )
                # jt = jax.lax.select(
                #     jnp.logical_not(joint.is_fixed_joint) & joint.motor_on,
                #     joint_texture.at[:, :, :3].set(
                #         jnp.where(
                #             joint_texture[:, :, 3:], joint_texture[:, :, :3], joint_colours[motor_bindings[joint_index]]
                #         )
                #     ),
                #     joint_texture,
                # )
                jt = jax.lax.select(
                    jnp.logical_not(joint.is_fixed_joint) & joint.motor_on,
                    joint_texture.at[:, :, :3].set(
                        joint_texture[:, :, :3]
                        / 255.0
                        * joint_colours[motor_bindings[joint_index]][None, None, :]
                        # jnp.where(
                        #     joint_texture[:, :, 3:], joint_texture[:, :, :3], joint_colours[motor_bindings[joint_index]]
                        # )
                    ),
                    joint_texture,
                )
                # jt = joint_texture

                # And update it
                updated_big = jax.lax.dynamic_update_slice(
                    fragments,
                    (jt[:, :, :3] * mask[:, :, None] + (1 - mask[:, :, None]) * slice_from_big),
                    (pos[0], pos[1], 0),
                )

                return updated_big, None

            return _render_joint

        def make_thruster_renderer(thruster_texture, thrusters, thruster_bindings):
            THRUSTER_SIZE = 32 * pixel_upscale
            thruster_texture = thruster_texture[::downscale, ::downscale]

            def _render_thruster_into_32by32_rotated(fragment_position, thruster, thruster_colour):

                rotation = thruster.rotation + select_shape(state, thruster.object_index, static_params).rotation
                M = _inv_rmat_sc(jnp.sin(rotation), jnp.cos(rotation))

                pos = M @ (fragment_position - THRUSTER_SIZE // 2) + THRUSTER_SIZE // 4
                pos = jnp.round(pos).astype(jnp.int32)
                # col = jax.lax.select(
                #     (pos[0] < 0) | (pos[0] >= THRUSTER_SIZE // 2) | (pos[1] < 0) | (pos[1] >= THRUSTER_SIZE // 2),
                #     jnp.array([-1, -1, -1]),
                #     jnp.where(
                #         thruster_texture[pos[0], pos[1], 3] > 0,
                #         # (joint_colours[thruster_colour] * thruster_texture[pos[0], pos[1], :3] // 255).astype(jnp.int32),
                #         thruster_texture[pos[0], pos[1], :3],
                #         thruster_texture[pos[0], pos[1], :3],
                #     ),
                # )

                tt = thruster_texture.astype(jnp.float32)

                col = jax.lax.select(
                    (pos[0] < 0) | (pos[0] >= THRUSTER_SIZE // 2) | (pos[1] < 0) | (pos[1] >= THRUSTER_SIZE // 2),
                    jnp.zeros(4),
                    # thruster_texture[pos[0], pos[1], :]
                    jax.lax.select(
                        pos[0] < 9 * pixel_upscale,
                        tt[pos[0], pos[1], :].at[:3].mul(joint_colours[thruster_colour] / 255.0),
                        tt[pos[0], pos[1], :],
                    )
                    # joint_colours[thruster_colour] * thruster_texture[pos[0], pos[1], :3] // 255,
                )
                col = col.astype(jnp.int32)

                # col = jax.lax.select(jnp.linalg.norm(col) < 1, jnp.array([-1, -1, -1]), col)
                return col

            def _render_thruster(fragments, joint_index):
                fragment_positions_flat, downscaled_dim = get_fragments_and_positions(
                    params, (THRUSTER_SIZE, THRUSTER_SIZE), downscale
                )
                thruster = jax.tree.map(lambda x: x[joint_index], thrusters)
                thruster_colour = jax.tree.map(lambda x: x[joint_index], thruster_bindings)
                block = jax.vmap(_render_thruster_into_32by32_rotated, (0, None, None))(
                    fragment_positions_flat, thruster, thruster_colour
                )
                block = block.reshape((downscaled_dim[0], downscaled_dim[1], 4))
                # return make_joint_renderer(block, thruster, False)(fragments, joint_index)
                size = THRUSTER_SIZE // downscale

                pos = jnp.round(
                    (thruster.global_position * params.pixels_per_unit) // downscale
                    + jnp.array(
                        [
                            downscaled_chunk_dim[0] - THRUSTER_SIZE // 2 // downscale,
                            downscaled_chunk_dim[1] - THRUSTER_SIZE // 2 // downscale,
                        ]
                    )
                ).astype(int)

                # Get the corresponding slice from the large image
                slice_from_big = jax.lax.dynamic_slice(
                    fragments,
                    (pos[0], pos[1], 0),
                    (size, size, 3),
                )

                mask = (block[:, :, 0] >= 0) * thruster.active
                jt = block.at[:, :, 3].mul(thruster.active)

                # And update it
                # updated_big = jax.lax.dynamic_update_slice(
                #     fragments,
                #     (jt[:, :, :3] * mask[:, :, None] + (1 - mask[:, :, None]) * slice_from_big),
                #     (pos[0], pos[1], 0),
                # )
                updated_big = jax.lax.dynamic_update_slice(
                    fragments,
                    (jt[:, :, :3] * jt[:, :, 3][:, :, None] + (1 - jt[:, :, 3])[:, :, None] * slice_from_big),
                    (pos[0], pos[1], 0),
                )

                return updated_big, None
                # return fragments, None

            return _render_thruster

        joint_renderer = make_joint_renderer(state.joint, state.motor_bindings)
        fragments, _ = jax.lax.scan(joint_renderer, fragments, jnp.arange(static_params.num_joints))

        thruster_renderer = make_thruster_renderer(THRUSTER_TEX, state.thruster, state.thruster_bindings)
        fragments, _ = jax.lax.scan(thruster_renderer, fragments, jnp.arange(static_params.num_thrusters))

        return fragments[
            downscaled_chunk_dim[0] : -downscaled_chunk_dim[0], downscaled_chunk_dim[1] : -downscaled_chunk_dim[1]
        ]

    return render_pixels
