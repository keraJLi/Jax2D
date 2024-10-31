import jax
import jax.numpy as jnp

from jax2d.engine import calc_inverse_mass_circle, calc_inverse_inertia_circle
from jax2d.sim_state import SimState


def add_circle_to_scene(
    sim_state: SimState,
    position,
    radius,
    rotation=0.0,
    velocity=jnp.zeros(2),
    angular_velocity=0.0,
    density=1.0,
    friction=1.0,
    restitution=0.0,
):
    circle_index = jnp.argmin(sim_state.circle.active)
    can_add_circle = jnp.logical_not(sim_state.circle.active.all())

    inverse_mass = calc_inverse_mass_circle(radius, density)
    inverse_inertia = calc_inverse_inertia_circle(radius, density)

    new_sim_state = sim_state.replace(
        circle=sim_state.circle.replace(
            position=sim_state.circle.position.at[circle_index].set(position),
            radius=sim_state.circle.radius.at[circle_index].set(radius),
            rotation=sim_state.circle.rotation.at[circle_index].set(rotation),
            velocity=sim_state.circle.velocity.at[circle_index].set(velocity),
            angular_velocity=sim_state.circle.angular_velocity.at[circle_index].set(angular_velocity),
            friction=sim_state.circle.friction.at[circle_index].set(friction),
            restitution=sim_state.circle.restitution.at[circle_index].set(restitution),
            inverse_mass=sim_state.circle.inverse_mass.at[circle_index].set(inverse_mass),
            inverse_inertia=sim_state.circle.inverse_inertia.at[circle_index].set(inverse_inertia),
        )
    )

    return jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_circle, x, y), new_sim_state, sim_state)
