"""
Shadow Hand environment.
"""

import gymnasium as gym

from .cfg import SelfPlayingPianoEnvCfg, SelfPlayingPPORunnerCfg


##
# Register Gym environments.
##

gym.register(
    id="Piano-Self-Playing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SelfPlayingPianoEnvCfg,
        "rsl_rl_cfg_entry_point": SelfPlayingPPORunnerCfg,
    },
)
