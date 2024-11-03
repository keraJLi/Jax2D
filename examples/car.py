import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxgl.maths import signed_line_distance
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import fragment_shader_circle, fragment_shader_quad
from matplotlib import pyplot as plt

from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.maths import rmat
from jax2d.scene import (
    add_circle_to_scene,
    add_rectangle_to_scene,
    add_fixed_joint_to_scene,
    add_revolute_joint_to_scene,
    add_thruster_to_scene,
    add_polygon_to_scene,
)
from jax2d.sim_state import StaticSimParams, SimParams


def mask_shader(shader_fn):
    @jax.jit
    def masked_shader(position, current_frag, unit_position, uniform):
        inner_uniforms = uniform[:-1]
        mask = uniform[-1]

        inner_fragment = shader_fn(position, current_frag, unit_position, inner_uniforms)
        return jax.lax.select(mask, inner_fragment, current_frag)

    return masked_shader


def make_fragment_shader_convex_dynamic_ngon_with_edges(max_n, edge_thickness=2):
    def fragment_shader_convex_dynamic_ngon(position, current_frag, unit_position, uniform):
        vertices, colour, edge_colour, n = uniform
        assert vertices.shape == (max_n, 2)

        next_vertices_idx = (jnp.arange(max_n) + 1) % n
        next_vertices = vertices[next_vertices_idx]

        inside = True
        on_edge = False
        for i in range(max_n):
            side = signed_line_distance(position, vertices[i], next_vertices[i]) / jnp.linalg.norm(
                vertices[i] - next_vertices[i]
            )
            inside &= (side <= 0) | (i >= n)
            on_edge |= (side > -edge_thickness) & (side <= 0) & (i < n)

        on_edge &= inside

        return jax.lax.select(inside, jax.lax.select(on_edge, edge_colour, colour), current_frag)

    return fragment_shader_convex_dynamic_ngon


def make_render_pixels(static_sim_params, screen_dim):
    ppud = 100
    patch_size = 512
    screen_padding = patch_size
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    def _world_space_to_pixel_space(x):
        return x * ppud + screen_padding

    cleared_screen = clear_screen(full_screen_size, jnp.zeros(3))

    circle_renderer = make_renderer(
        full_screen_size, mask_shader(fragment_shader_circle), (patch_size, patch_size), batched=True
    )

    polygon_shader = mask_shader(make_fragment_shader_convex_dynamic_ngon_with_edges(4))
    quad_renderer = make_renderer(full_screen_size, polygon_shader, (patch_size, patch_size), batched=True)

    @jax.jit
    def render_pixels(state):
        pixels = cleared_screen

        # Rectangles
        rect_positions_pixel_space = _world_space_to_pixel_space(state.polygon.position)
        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(
            rectangle_rmats[:, None, :, :], repeats=static_sim_params.max_polygon_vertices, axis=1
        )
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rect_patch_positions = (rect_positions_pixel_space - (patch_size / 2)).astype(jnp.int32)
        rect_patch_positions = jnp.maximum(rect_patch_positions, 0)

        rect_colours = jnp.ones((static_sim_params.num_polygons, 3)) * 128.0
        rect_uniforms = (
            rectangle_vertices_pixel_space,
            rect_colours,
            rect_colours,
            state.polygon.n_vertices,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rect_patch_positions, rect_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        circle_radii_pixel_space = state.circle.radius * ppud
        circle_patch_positions = (circle_positions_pixel_space - (patch_size / 2)).astype(jnp.int32)
        circle_patch_positions = jnp.maximum(circle_patch_positions, 0)

        circle_colours = jnp.ones((static_sim_params.num_circles, 3)) * 255.0

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            circle_colours,
            state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Crop out the sides
        return pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]

    return render_pixels


def main():
    screen_dim = (500, 500)

    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state = create_empty_sim(static_sim_params, floor_offset=0.0)

    sim_state, (_, r1) = add_rectangle_to_scene(
        sim_state, jnp.array([2.0, 1.0]), jnp.array([1.0, 0.4]), static_sim_params
    )

    sim_state, (_, c1) = add_circle_to_scene(sim_state, jnp.array([1.5, 1.0]), 0.3, static_sim_params)
    sim_state, (_, c2) = add_circle_to_scene(sim_state, jnp.array([2.5, 1.0]), 0.3, static_sim_params)

    sim_state, _ = add_revolute_joint_to_scene(
        sim_state, r1, c1, jnp.array([-0.5, 0.0]), jnp.zeros(2), static_sim_params, motor_on=True
    )
    sim_state, _ = add_revolute_joint_to_scene(
        sim_state, r1, c2, jnp.array([0.5, 0.0]), jnp.zeros(2), static_sim_params, motor_on=True
    )

    triangle_vertices = jnp.array(
        [
            [
                -0.5,
                -0.5,
            ],
            [
                0.5,
                -0.2,
            ],
            [
                0.5,
                0.2,
            ],
        ][::-1]
    )
    sim_state, (_, t1) = add_polygon_to_scene(sim_state, jnp.array([3.0, 1.0]), triangle_vertices, 3, static_sim_params)

    # Renderer
    renderer = make_render_pixels(static_sim_params, screen_dim)

    # Step scene
    step_fn = jax.jit(engine.step)

    pygame.init()
    screen_surface = pygame.display.set_mode(screen_dim)

    while True:
        actions = -jnp.ones(static_sim_params.num_joints + static_sim_params.num_thrusters)
        sim_state, _ = step_fn(sim_state, sim_params, actions)

        # Render
        pixels = renderer(sim_state)

        # Update screen
        surface = pygame.surfarray.make_surface(np.array(pixels)[:, ::-1])
        screen_surface.blit(surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True


if __name__ == "__main__":
    debug = False

    if debug:
        print("JIT disabled")
        with jax.disable_jit():
            main()
    else:
        main()
