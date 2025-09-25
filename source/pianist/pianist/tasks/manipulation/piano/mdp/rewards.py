import torch
from typing import Tuple

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from pianist.tasks.manipulation.piano.mdp.commands import KeyPressCommand

# each reward term should return a tensor of shape (num_envs,)


def gaussian_sigmoid_func(x: torch.Tensor, value_at_margin: float) -> torch.Tensor:
    """Compute the Gaussian sigmoid function."""
    scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_margin)))
    return torch.exp(-0.5 * torch.pow(x * scale, 2))


def gaussian_tolerance(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """
    Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A torch tensor.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A torch tensor with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, gaussian_sigmoid_func(d, value_at_margin))

    return value


def key_on_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    key_close_enough_to_pressed: float = 0.05,
) -> torch.Tensor:
    """
    Reward for pressing the right keys at the right time.

    We separate the `_compute_key_press_reward` function from the original paper
    into two functions, `key_on_reward` and `key_off_reward`, to allow for more
    flexibility in the reward computation.
    """

    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    # get the keys that should be pressed from command
    # this is a boolean tensor of shape (num_envs, 88)
    on_keys = command_term.key_goal_states

    # compute the error between goal and actual for all keys
    # the boolean tensor will be mapped to 0.0 and 1.0
    errors = on_keys.float() - command_term.key_actual_states

    # apply gaussian tolerance to all key errors
    all_key_rewards = gaussian_tolerance(
        errors,
        bounds=(0.0, key_close_enough_to_pressed),
        margin=key_close_enough_to_pressed * 10,
    )

    # get the average reward across all keys that should be pressed
    # use the mask to only consider rewards for keys that should be pressed
    rewards = (all_key_rewards * on_keys).sum(dim=-1) / (on_keys.sum(dim=-1).float() + 1e-6)

    return rewards


def key_off_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    key_close_enough_to_pressed: float = 0.05,
) -> torch.Tensor:
    """Reward for not pressing the wrong keys."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    # get the keys that should not be pressed from command
    # this is a boolean tensor of shape (num_envs, 88)
    off_keys = ~command_term.key_goal_states

    # only consider keys that should not be pressed
    key_press_actual = command_term.key_actual_states * off_keys

    # if there are any false positives, do not grant any reward
    rewards = (1.0 - (key_press_actual > 0.5).any(dim=-1).float())

    return rewards


# def sustain_reward(self, physics) -> float:
#     """Reward for pressing the sustain pedal at the right time."""
#     del physics  # Unused.
#     return tolerance(
#         self._goal_current[-1] - self.piano.sustain_activation[0],
#         bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
#         margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
#         sigmoid="gaussian",
#     )


def key_on_perfect_reward(env: ManagerBasedRLEnv, command_name: str, std: float = 0.01) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    on_keys = command_term.key_goal_states

    errors = torch.square(command_term.key_goal_states.float() - command_term.key_actual_states)
    effective_errors = errors * on_keys
    return torch.exp(-effective_errors.mean(dim=-1) / std**2)


def key_off_perfect_reward(env: ManagerBasedRLEnv, command_name: str, std: float = 0.01) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    off_keys = ~command_term.key_goal_states

    errors = torch.square(command_term.key_goal_states.float() - command_term.key_actual_states)
    effective_errors = errors * off_keys
    return torch.exp(-effective_errors.mean(dim=-1) / std**2)


def key_position_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the error between the goal and actual key positions."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    errors = torch.square(command_term.key_goal_states.float() - command_term.key_actual_states)
    return errors.sum(dim=-1)


def energy_reward(env: ManagerBasedRLEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing energy."""
    robot: Articulation = env.scene[robot_asset_cfg.name]
    powers = torch.abs(robot.data.applied_torque * robot.data.joint_vel)
    rewards = torch.sum(powers, dim=-1)
    return rewards


def fingertip_to_key_distances(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the distances between each of the fingertips and the keys."""
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    # obtain the desired and current positions
    # key_locations and fingertip_positions are tensors of shape (num_envs, 5, 3)
    key_locations = command_term.key_goal_locations
    fingertip_positions = command_term.fingertip_positions

    # calculate L2 distance between the fingertip and the key
    distances = torch.norm(fingertip_positions - key_locations, dim=-1)

    return distances


def fingertip_to_key_distance_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the L2 distance between the fingertip and the key."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)
    all_distances = fingertip_to_key_distances(env, command_name, asset_cfg)
    distances = torch.sum(command_term.active_fingers * all_distances, dim=-1) / (command_term.active_fingers.sum(dim=-1).float() + 1e-6)
    return distances


def fingertip_to_key_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    finger_close_enough_to_key: float = 0.01,
) -> torch.Tensor:
    """Reward for minimizing the distance between the fingertip and the key."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    # get the distances between all of the fingertips and the keys
    all_distances = fingertip_to_key_distances(env, command_name, asset_cfg)
    all_distance_rewards = gaussian_tolerance(
        all_distances,
        bounds=(0, finger_close_enough_to_key),
        margin=(finger_close_enough_to_key * 10),
    )
    # mask off the rewards for the inactive fingers
    active_rewards = command_term.active_fingers * all_distance_rewards
    rewards = active_rewards.sum(dim=-1) / (command_term.active_fingers.sum(dim=-1).float() + 1e-6)

    return rewards
