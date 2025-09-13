import torch
from typing import Tuple

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from pianist.tasks.manipulation.piano.mdp.commands import RandomKeyPressCommand


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

    command_term: RandomKeyPressCommand = env.command_manager.get_term(command_name)
    # get the key indices of nonzero elements in the command
    on_keys = torch.nonzero(command_term.key_press_goals, as_tuple=True)

    # get the target and actual key press states
    # need to perform this reshape to convert the 1D indexed result back to (envs, num_on_keys)
    key_press_goal = command_term.key_press_goals[on_keys].reshape(env.num_envs, -1)
    key_press_actual = command_term.key_press_actual[on_keys].reshape(env.num_envs, -1)

    # if we have pressed the correct keys, reward according to the correct amount
    on_key_rewards = gaussian_tolerance(
        key_press_goal - key_press_actual,
        bounds=(0.0, key_close_enough_to_pressed),
        margin=key_close_enough_to_pressed * 10,
    )
    rewards = on_key_rewards.mean(dim=-1)
    return rewards


def key_off_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    key_close_enough_to_pressed: float = 0.05,
) -> torch.Tensor:
    """Reward for not pressing the wrong keys."""
    command_term: RandomKeyPressCommand = env.command_manager.get_term(command_name)
    # get the key indices of nonzero elements in the command
    off_keys = torch.nonzero(1 - command_term.key_press_goals, as_tuple=True)

    # get the actual key press states
    # need to perform this reshape to convert the 1D indexed result back to (envs, num_off_keys)
    key_press_actual = command_term.key_press_actual[off_keys].reshape(env.num_envs, -1)

    # if there are any false positives do not grant any reward
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


def energy_reward(env: ManagerBasedRLEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing energy."""
    robot: Articulation = env.scene[robot_asset_cfg.name]
    powers = torch.abs(robot.data.applied_torque * robot.data.joint_vel)
    rewards = torch.sum(powers, dim=-1)
    return rewards


def fingertip_to_key_distances(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the distances between the fingertip and the key."""
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: RandomKeyPressCommand = env.command_manager.get_term(command_name)

    # obtain the desired and current positions
    key_locations = command_term.target_key_locations[:, :, 0:3]
    fingertip_positions = asset.data.body_pos_w[:, command_term._finger_body_indices]

    distances = torch.norm(fingertip_positions - key_locations, dim=-1)
    return distances


def fingertip_to_key_distance_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the L2 distance between the fingertip and the key."""
    distances = fingertip_to_key_distances(env, command_name, asset_cfg)
    return torch.mean(distances, dim=-1)


def fingertip_to_key_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    finger_close_enough_to_key: float = 0.01,
) -> torch.Tensor:
    """Reward for minimizing the distance between the fingertip and the key."""
    distances = fingertip_to_key_distances(env, command_name, asset_cfg)

    distance_rewards = gaussian_tolerance(
        distances.flatten(start_dim=1),
        bounds=(0, finger_close_enough_to_key),
        margin=(finger_close_enough_to_key * 10),
    )
    rewards = distance_rewards.mean(dim=-1)

    return rewards
