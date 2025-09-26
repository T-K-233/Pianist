import torch
from typing import Tuple

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from pianist.tasks.manipulation.piano.mdp.commands import KeyPressCommand
from pianist.tasks.manipulation.piano.mdp.math_functions import gaussian, windowed_gaussian


# each reward term should return a tensor of shape (num_envs,)

def key_on_perfect_reward(env: ManagerBasedRLEnv, command_name: str, std: float = 0.01) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    on_keys = command_term.key_goal_states

    # errors = torch.square(command_term.key_goal_states.float() - command_term.key_actual_states)
    # return torch.exp(-effective_errors.mean(dim=-1) / std**2)
    errors = gaussian(command_term.key_actual_states, command_term.key_goal_states.float(), std)
    effective_errors = errors * on_keys
    return effective_errors.mean(dim=-1)


def key_off_perfect_reward(env: ManagerBasedRLEnv, command_name: str, std: float = 0.01) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""
    command_term: KeyPressCommand = env.command_manager.get_term(command_name)

    off_keys = ~command_term.key_goal_states

    errors = gaussian(command_term.key_actual_states, command_term.key_goal_states.float(), std)
    effective_errors = errors * off_keys
    return effective_errors.mean(dim=-1)


# def sustain_reward(self, physics) -> float:
#     """Reward for pressing the sustain pedal at the right time."""
#     del physics  # Unused.
#     return tolerance(
#         self._goal_current[-1] - self.piano.sustain_activation[0],
#         bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
#         margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
#         sigmoid="gaussian",
#     )


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
    all_distance_rewards = windowed_gaussian(
        all_distances,
        lower=0,
        upper=finger_close_enough_to_key,
        std=0.05,
    )
    # mask off the rewards for the inactive fingers
    active_rewards = command_term.active_fingers * all_distance_rewards
    rewards = active_rewards.sum(dim=-1) / (command_term.active_fingers.sum(dim=-1).float() + 1e-6)

    return rewards
