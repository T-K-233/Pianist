
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import pianist.tasks.manipulation.piano.mdp as mdp
from pianist.assets.piano_cfg import PIANO_CFG


KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05


@configclass
class SelfPlayingPianoSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a piano."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    piano = PIANO_CFG.replace(prim_path="{ENV_REGEX_NS}/piano")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    keypress = mdp.KeyPressCommandCfg(
        # song_name="simple",
        song_name="./source/pianist/data/music/pig_single_finger/nocturne_op9_no_2-1.proto",
        piano_name="piano",
        robot_name=None,  # no robot in self-playing mode
        robot_finger_body_names=None,
        key_trigger_threshold=0.70,
        lookahead_steps=10,
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        piano_key_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "keypress"})
        active_fingers = ObsTerm(func=mdp.active_fingers, params={"command_name": "keypress"})
        piano_key_positions = ObsTerm(func=mdp.piano_key_pos, params={"piano_asset_cfg": SceneEntityCfg("piano")})

        def __post_init__(self):
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="piano",
        joint_names=[".*"],
        scale=0.1,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    key_on_perfect = RewTerm(
        func=mdp.key_on_perfect_reward,
        params={
            "command_name": "keypress",
            "std": 0.01,
        },
        weight=1.0,
    )
    key_off_perfect = RewTerm(
        func=mdp.key_off_perfect_reward,
        params={
            "command_name": "keypress",
            "std": 0.01,
        },
        weight=1.0,
    )
    key_position_error = RewTerm(
        func=mdp.key_position_error,
        params={
            "command_name": "keypress",
        },
        weight=-0.1,
    )
    energy = RewTerm(
        func=mdp.energy_reward,
        params={"robot_asset_cfg": SceneEntityCfg("piano")},
        weight=-5e-3,
    )

    # we want to reproduce the same sparse key presses
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("piano", joint_names=[".*"])},
        weight=-0.5,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Configuration for events."""

    pass


##
# Environment configuration
##

@configclass
class SelfPlayingPianoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the piano environment."""
    # Scene settings
    scene: SelfPlayingPianoSceneCfg = SelfPlayingPianoSceneCfg(num_envs=4096, env_spacing=2.5)

    # Policy commands
    commands: CommandsCfg = CommandsCfg()

    # Policy observations
    observations: ObservationsCfg = ObservationsCfg()

    # Policy actions
    actions: ActionsCfg = ActionsCfg()

    # Policy rewards
    rewards: RewardsCfg = RewardsCfg()

    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg()

    # Randomization events
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (-0.5, 1.0, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 0.01


@configclass
class SelfPlayingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 20000
    save_interval = 200
    experiment_name = "self_playing"
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
