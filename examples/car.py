import jax
import jax.numpy as jnp
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import fragment_shader_circle
from matplotlib import pyplot as plt

from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.scene import add_circle_to_scene
from jax2d.sim_state import StaticSimParams, SimParams


def make_render_pixels(
    static_sim_params,
):
    screen_dim = (500, 500)
    screen_padding = 100
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    def _world_space_to_pixel_space(x):
        return x * 100 + screen_padding

    cleared_screen = clear_screen(full_screen_size, jnp.zeros(3))

    patch_size = 512

    circle_renderer = make_renderer(full_screen_size, fragment_shader_circle, (patch_size, patch_size), batched=True)
    # quad_renderer = make_renderer(full_screen_size, fragment_shader_edged_quad, patch_size, batched=True)

    @jax.jit
    def render_pixels(state):
        pixels = cleared_screen

        # Rectangles
        # rectangle_patch_positions = _world_space_to_pixel_space(
        #     state.polygon.position - (static_params.max_shape_size / 2.0)
        # ).astype(jnp.int32)
        #
        # rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        # rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        # rectangle_vertices_pixel_space = _world_space_to_pixel_space(
        #     state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        # )
        # rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        # rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))
        #
        # rectangle_uniforms = (
        #     rectangle_vertices_pixel_space,
        #     rectangle_colours,
        #     rectangle_edge_colours,
        #     state.polygon.active,
        # )
        #
        # pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        circle_radii_pixel_space = _world_space_to_pixel_space(state.circle.radius)
        circle_patch_positions = (circle_positions_pixel_space - patch_size / 2).astype(jnp.int32) * 0

        circle_colours = jnp.ones((static_sim_params.num_circles, 3)) * 255.0

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            circle_colours,
            # state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Crop out the sides
        return pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]

    return render_pixels


def main():
    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state = create_empty_sim(static_sim_params)
    sim_state = add_circle_to_scene(sim_state, jnp.ones(2), 0.5)

    # Renderer
    renderer = make_render_pixels(static_sim_params)

    # Step scene
    # actions = jnp.zeros(static_sim_params.num_joints + static_sim_params.num_thrusters)
    # sim_state, _ = engine.step(sim_state, sim_params, actions)

    # Render
    pixels = renderer(sim_state)
    plt.imshow(pixels.astype(jnp.uint8))
    plt.show()


if __name__ == "__main__":
    main()
