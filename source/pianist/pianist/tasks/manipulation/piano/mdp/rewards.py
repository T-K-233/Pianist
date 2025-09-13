import torch
from typing import Tuple

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from pianist.tasks.manipulation.piano.mdp.commands import KeyPressCommand


def gaussian_sigmoid_func(
    x: torch.Tensor,
    value_at_margin: float
) -> torch.Tensor:
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


def key_press_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    key_close_enough_to_pressed: float = 0.05,
) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""

    # piano: Articulation = env.scene[piano_entity_cfg.name]
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)
    key_press_goal = command_term.key_press_goals
    key_press_actual = command_term.key_press_actual

    # on = np.flatnonzero(self._goal_current[:-1])
    # rew = 0.0
    # # It's possible we have no keys to press at this timestep, so we need to check
    # # that `on` is not empty.
    # if on.size > 0:
    #     actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
    #     rews = tolerance(
    #         self._goal_current[:-1][on] - actual[on],
    #         bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
    #         margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
    #         sigmoid="gaussian",
    #     )
    #     rew += 0.5 * rews.mean()
    # # If there are any false positives, the remaining 0.5 reward is lost.
    # off = np.flatnonzero(1 - self._goal_current[:-1])
    # rew += 0.5 * (1 - float(self.piano.activation[off].any()))
    # return rew

    # get the key indices of nonzero elements in the command
    on_keys = torch.nonzero(key_press_goal, as_tuple=True)
    off_keys = torch.nonzero(1 - key_press_goal, as_tuple=True)
    rewards = torch.zeros(env.num_envs, device=env.device)

    # if we have pressed the correct keys, reward according to the correct amount
    # rewards += 0.5 * torch.abs(keypress_command[on_keys] - piano_key_press_state[on_keys]).mean(dim=-1)
    on_key_rewards = gaussian_tolerance(
        key_press_goal[on_keys] - key_press_actual[on_keys],
        bounds=(0.0, key_close_enough_to_pressed),
        margin=key_close_enough_to_pressed * 10,
    )
    rewards[:] += 0.5 * on_key_rewards.mean(dim=-1)

    # if there are any false positives, the other half of the reward is lost.
    rewards[:] += 0.5 * (1.0 - key_press_actual[off_keys].any(dim=-1).float())

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
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    # obtain the desired and current positions
    key_locations = command_term.target_key_locations[:, :, 0:3]
    fingertip_positions = asset.data.body_pos_w[:, command_term._finger_body_indices]

    distances = torch.norm(fingertip_positions - key_locations, dim=-1)
    return distances


def fingertip_to_key_distance_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    distances = fingertip_to_key_distances(env, command_name, asset_cfg)
    return torch.mean(distances, dim=-1)


def fingertip_to_key_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    finger_close_enough_to_key: float = 0.01,
) -> torch.Tensor:
    """
    the fingering_reward
    """
    distances = fingertip_to_key_distances(env, command_name, asset_cfg)

    distance_rewards = gaussian_tolerance(
        distances.flatten(start_dim=1),
        bounds=(0, finger_close_enough_to_key),
        margin=(finger_close_enough_to_key * 10),
    )
    rewards = distance_rewards.mean(dim=-1)

    return rewards
