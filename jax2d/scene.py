from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax2d.engine import (
    calc_inverse_mass_circle,
    calc_inverse_inertia_circle,
    calc_inverse_mass_polygon,
    calc_inverse_inertia_polygon,
    calculate_collision_matrix,
)
from jax2d.sim_state import SimState


def add_thruster_to_scene(sim_state: SimState, object_index, relative_position, rotation, power=1.0):
    thruster_index = jnp.argmin(sim_state.thruster.active)
    can_add_thruster = jnp.logical_not(sim_state.thruster.active.all())

    new_sim_state = sim_state.replace(
        thruster=sim_state.thruster.replace(
            object_index=sim_state.thruster.object_index.at[thruster_index].set(object_index),
            relative_position=sim_state.thruster.relative_position.at[thruster_index].set(relative_position),
            rotation=sim_state.thruster.rotation.at[thruster_index].set(rotation),
            power=sim_state.thruster.power.at[thruster_index].set(power),
            active=sim_state.thruster.active.at[thruster_index].set(True),
        )
    )

    return (
        jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(can_add_thruster, x, y),
            new_sim_state,
            sim_state,
        ),
        thruster_index,
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_revolute_joint_to_scene(
    sim_state: SimState,
    static_sim_params,
    a_index,
    b_index,
    a_relative_pos,
    b_relative_pos,
    motor_on=False,
    motor_speed=1.0,
    motor_power=1.0,
    has_joint_limits=False,
    min_rotation=-np.pi,
    max_rotation=np.pi,
):
    joint_index = jnp.argmin(sim_state.joint.active)
    can_add_joint = jnp.logical_not(sim_state.joint.active.all())

    new_sim_state = sim_state.replace(
        joint=sim_state.joint.replace(
            a_index=sim_state.joint.a_index.at[joint_index].set(a_index),
            b_index=sim_state.joint.b_index.at[joint_index].set(b_index),
            a_relative_pos=sim_state.joint.a_relative_pos.at[joint_index].set(a_relative_pos),
            b_relative_pos=sim_state.joint.b_relative_pos.at[joint_index].set(b_relative_pos),
            active=sim_state.joint.active.at[joint_index].set(True),
            is_fixed_joint=sim_state.joint.is_fixed_joint.at[joint_index].set(False),
            motor_on=sim_state.joint.motor_on.at[joint_index].set(motor_on),
            motor_speed=sim_state.joint.motor_speed.at[joint_index].set(motor_speed),
            motor_power=sim_state.joint.motor_power.at[joint_index].set(motor_power),
            motor_has_joint_limits=sim_state.joint.motor_has_joint_limits.at[joint_index].set(has_joint_limits),
            min_rotation=sim_state.joint.min_rotation.at[joint_index].set(min_rotation),
            max_rotation=sim_state.joint.max_rotation.at[joint_index].set(max_rotation),
        )
    )

    new_sim_state = new_sim_state.replace(
        collision_matrix=calculate_collision_matrix(static_sim_params, new_sim_state.joint)
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_joint, x, y), new_sim_state, sim_state),
        joint_index,
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_fixed_joint_to_scene(
    sim_state: SimState,
    static_sim_params,
    a_index,
    b_index,
    a_relative_pos,
    b_relative_pos,
):
    joint_index = jnp.argmin(sim_state.joint.active)
    can_add_joint = jnp.logical_not(sim_state.joint.active.all())

    new_sim_state = sim_state.replace(
        joint=sim_state.joint.replace(
            a_index=sim_state.joint.a_index.at[joint_index].set(a_index),
            b_index=sim_state.joint.b_index.at[joint_index].set(b_index),
            a_relative_pos=sim_state.joint.a_relative_pos.at[joint_index].set(a_relative_pos),
            b_relative_pos=sim_state.joint.b_relative_pos.at[joint_index].set(b_relative_pos),
            active=sim_state.joint.active.at[joint_index].set(True),
            is_fixed_joint=sim_state.joint.is_fixed_joint.at[joint_index].set(True),
        )
    )

    new_sim_state = new_sim_state.replace(
        collision_matrix=calculate_collision_matrix(static_sim_params, new_sim_state.joint)
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_joint, x, y), new_sim_state, sim_state),
        joint_index,
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_circle_to_scene(
    sim_state: SimState,
    static_sim_params,
    position,
    radius,
    rotation=0.0,
    velocity=jnp.zeros(2),
    angular_velocity=0.0,
    density=1.0,
    friction=1.0,
    restitution=0.0,
    fixated=False,
):
    circle_index = jnp.argmin(sim_state.circle.active)
    can_add_circle = jnp.logical_not(sim_state.circle.active.all())

    inverse_mass = calc_inverse_mass_circle(radius, density)
    inverse_inertia = calc_inverse_inertia_circle(radius, density)

    inverse_mass *= jnp.logical_not(fixated)
    inverse_inertia *= jnp.logical_not(fixated)

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
            active=sim_state.circle.active.at[circle_index].set(True),
        )
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_circle, x, y), new_sim_state, sim_state),
        (
            circle_index,
            circle_index + static_sim_params.num_polygons,
        ),
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_rectangle_to_scene(
    sim_state: SimState,
    static_sim_params,
    position,
    dimensions,
    rotation=0.0,
    velocity=jnp.zeros(2),
    angular_velocity=0.0,
    density=1.0,
    friction=1.0,
    restitution=0.0,
    fixated=False,
):
    half_dims = dimensions / 2.0
    vertices = jnp.array(
        [
            [-half_dims[0], half_dims[1]],
            [half_dims[0], half_dims[1]],
            [half_dims[0], -half_dims[1]],
            [-half_dims[0], -half_dims[1]],
        ]
    )

    return add_polygon_to_scene(
        sim_state,
        static_sim_params,
        position,
        vertices,
        4,
        rotation,
        velocity,
        angular_velocity,
        density,
        friction,
        restitution,
        fixated,
    )


@partial(jax.jit, static_argnames=["static_sim_params", "n_vertices"])
def add_polygon_to_scene(
    sim_state: SimState,
    static_sim_params,
    position,
    vertices,
    n_vertices,
    rotation=0.0,
    velocity=jnp.zeros(2),
    angular_velocity=0.0,
    density=1.0,
    friction=1.0,
    restitution=0.0,
    fixated=False,
):
    # Fill vertices up to maxP
    vertices = jnp.zeros((static_sim_params.max_polygon_vertices, 2)).at[:n_vertices].set(vertices)

    polygon_index = jnp.argmin(sim_state.polygon.active)
    can_add_polygon = jnp.logical_not(sim_state.polygon.active.all())

    # Adjust position and vertices according to new CoM
    # This will only be non-zero if the current CoM is wrong
    inverse_mass, delta_centre_of_mass = calc_inverse_mass_polygon(vertices, n_vertices, static_sim_params, density)
    position += delta_centre_of_mass
    vertices -= delta_centre_of_mass[None, :]

    inverse_inertia = calc_inverse_inertia_polygon(vertices, n_vertices, static_sim_params, density)

    inverse_mass *= jnp.logical_not(fixated)
    inverse_inertia *= jnp.logical_not(fixated)

    new_sim_state = sim_state.replace(
        polygon=sim_state.polygon.replace(
            position=sim_state.polygon.position.at[polygon_index].set(position),
            vertices=sim_state.polygon.vertices.at[polygon_index].set(vertices),
            rotation=sim_state.polygon.rotation.at[polygon_index].set(rotation),
            velocity=sim_state.polygon.velocity.at[polygon_index].set(velocity),
            angular_velocity=sim_state.polygon.angular_velocity.at[polygon_index].set(angular_velocity),
            friction=sim_state.polygon.friction.at[polygon_index].set(friction),
            restitution=sim_state.polygon.restitution.at[polygon_index].set(restitution),
            inverse_mass=sim_state.polygon.inverse_mass.at[polygon_index].set(inverse_mass),
            inverse_inertia=sim_state.polygon.inverse_inertia.at[polygon_index].set(inverse_inertia),
            active=sim_state.polygon.active.at[polygon_index].set(True),
            n_vertices=sim_state.polygon.n_vertices.at[polygon_index].set(n_vertices),
        )
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_polygon, x, y), new_sim_state, sim_state),
        (polygon_index, polygon_index),
    )
