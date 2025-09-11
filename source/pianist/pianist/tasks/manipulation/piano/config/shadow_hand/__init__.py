"""
Shadow Hand environment.
"""

import gymnasium as gym

from .cfg import ShadowHandCfg
from .agents import rsl_rl_ppo_cfg
from pianist.tasks.manipulation.piano.piano_env import PianoEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Piano-Shadow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.ShadowHandPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
