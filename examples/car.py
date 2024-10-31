import jax
import jax.numpy as jnp
from jaxgl.renderer import clear_screen, make_renderer

from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.sim_state import StaticSimParams, SimParams


def make_render_pixels(
    static_sim_params,
):
    screen_dim = (512, 512)

    screen_padding = 100
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )
    cleared_screen = clear_screen(full_screen_size, jnp.zeros(3))

    def fragment_shader_kinetix_circle(position, current_frag, unit_position, uniform):
        centre, radius, rotation, colour, mask = uniform

        dist = jnp.sqrt(jnp.square(position - centre).sum())
        inside = dist <= radius
        on_edge = dist > radius - 2

        # TODO - precompute?
        normal = jnp.array([jnp.sin(rotation), -jnp.cos(rotation)])

        dist = dist_from_line(position, centre, centre + normal)

        on_edge |= (dist < 1) & (jnp.dot(normal, position - centre) <= 0)

        fragment = jax.lax.select(on_edge, jnp.zeros(3), colour)

        return jax.lax.select(inside & mask, fragment, current_frag)

    patch_size = (128, 128)

    circle_renderer = make_renderer(full_screen_size, fragment_shader_kinetix_circle, patch_size, batched=True)
    quad_renderer = make_renderer(full_screen_size, fragment_shader_edged_quad, patch_size, batched=True)

    @jax.jit
    def render_pixels(state):
        pixels = cleared_screen

        # Rectangles
        rectangle_patch_positions = _world_space_to_pixel_space(
            state.polygon.position - (static_params.max_shape_size / 2.0)
        ).astype(jnp.int32)

        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))

        rectangle_uniforms = (
            rectangle_vertices_pixel_space,
            rectangle_colours,
            rectangle_edge_colours,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        circle_radii_pixel_space = state.circle.radius * ppud
        circle_patch_positions = _world_space_to_pixel_space(
            state.circle.position - (static_params.max_shape_size / 2.0)
        ).astype(jnp.int32)

        circle_colours = jax.vmap(_get_colour)(state.circle_shape_roles, state.circle.inverse_mass)

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            state.circle.rotation,
            circle_colours,
            state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Joints
        joint_patch_positions = jnp.round(
            _world_space_to_pixel_space(state.joint.global_position) - (joint_pixel_size // 2)
        ).astype(jnp.int32)
        joint_textures = jax.vmap(jax.lax.select, in_axes=(0, None, None))(
            state.joint.is_fixed_joint, FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA
        )
        joint_colours = JOINT_COLOURS[
            (state.motor_bindings + 1) * (state.joint.motor_on & (~state.joint.is_fixed_joint))
        ]

        joint_uniforms = (joint_textures, joint_colours, state.joint.active)

        pixels = joint_renderer(pixels, joint_patch_positions, joint_uniforms)

        # Thrusters
        thruster_positions = jnp.round(_world_space_to_pixel_space(state.thruster.global_position)).astype(jnp.int32)
        thruster_patch_positions = thruster_positions - (thruster_pixel_size_diagonal // 2)
        thruster_textures = coloured_thruster_textures[state.thruster_bindings + 1]
        thruster_rotations = (
            state.thruster.rotation
            + jax.vmap(select_shape, in_axes=(None, 0, None))(
                state, state.thruster.object_index, static_params
            ).rotation
        )
        thruster_uniforms = (thruster_positions, thruster_rotations, thruster_textures, state.thruster.active)

        pixels = thruster_renderer(pixels, thruster_patch_positions, thruster_uniforms)

        # Crop out the sides
        crop_amount = static_params.max_shape_size * ppud
        return pixels[crop_amount:-crop_amount, crop_amount:-crop_amount]

    return render_pixels


def main():
    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state = create_empty_sim(static_sim_params)

    # Step scene
    actions = jnp.zeros(static_sim_params.num_joints + static_sim_params.num_thrusters)
    engine.step(sim_state, sim_params, actions)


if __name__ == "__main__":
    main()
