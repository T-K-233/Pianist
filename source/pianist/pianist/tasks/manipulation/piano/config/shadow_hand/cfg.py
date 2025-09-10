import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pianist.robots.shadow_hand import SHADOW_HAND_CFG
from pianist.tasks.manipulation.piano.piano_env import PianoEnvCfg


@configclass
class ShadowHandCfg(PianoEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/robot").replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.6),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={".*": 0.0},
            )
        )

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ffdistal"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ffdistal"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ffdistal"]

        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "ffdistal"
