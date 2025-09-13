import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from pianist.assets.piano_articulation import PianoArticulation


def forearm_pos(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The position of the forearm w.r.t. the world.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[robot_asset_cfg.name]
    body_index, _ = asset.find_bodies(["forearm"])
    return asset.data.body_pos_w[:, body_index].squeeze(1)


def active_fingers(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The active fingers of the robot."""
    command_term = env.command_manager.get_term(command_name)
    return command_term.active_fingers


def piano_key_pos(env: ManagerBasedEnv, piano_asset_cfg: SceneEntityCfg = SceneEntityCfg("piano")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`piano_asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: PianoArticulation = env.scene[piano_asset_cfg.name]
    return asset.key_press_states
