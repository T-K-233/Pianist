from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from pianist.tasks.manipulation.piano.piano_env import PianoEnvCfg
from pianist.robots.shadow_hand import SHADOW_HAND_CFG


@configclass
class ShadowHandCfg(PianoEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = SHADOW_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        # self.scene.robot.init_state.pos = (-0.35, 0.0, 0.6)
        self.scene.robot.init_state.pos = (-0.4, 0.0, 0.55)


@configclass
class ShadowHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 20000
    save_interval = 100
    experiment_name = "shadow_hand"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
