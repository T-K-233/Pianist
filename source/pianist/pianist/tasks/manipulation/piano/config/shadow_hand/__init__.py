"""
Shadow Hand environment.
"""

import gymnasium as gym

from .cfg import ShadowHandCfg, ShadowHandPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="Piano-Shadow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandCfg,
        "rsl_rl_cfg_entry_point": ShadowHandPPORunnerCfg,
    },
)
