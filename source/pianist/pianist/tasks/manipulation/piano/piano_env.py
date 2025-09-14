from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import pianist.tasks.manipulation.piano.mdp as mdp
from pianist.assets.piano_cfg import PIANO_CFG


FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05


@configclass
class PianoSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a piano."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    piano = PIANO_CFG.replace(prim_path="{ENV_REGEX_NS}/piano")

    # a ball just for debugging
    # ball = AssetBaseCfg(
    #     prim_path="/World/ball",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.03,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.5, 1.0)),
    # )

    # robots
    robot: ArticulationCfg = MISSING

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

    keypress = mdp.RandomKeyPressCommandCfg(
        piano_name="piano",
        robot_name="robot",
        # robot_finger_body_names=["thdistal", "ffdistal", "mfdistal", "rfdistal", "lfdistal"],
        robot_finger_body_names=["rfdistal", "mfdistal", "ffdistal"],
        resampling_time_range=(1.0, 4.0),
        key_close_enough_to_pressed=KEY_CLOSE_ENOUGH_TO_PRESSED,
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        piano_key_goal_state = ObsTerm(func=mdp.generated_commands, params={"command_name": "keypress"})
        forearm_pos = ObsTerm(func=mdp.forearm_pos, params={"robot_asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        active_fingers = ObsTerm(func=mdp.active_fingers, params={"command_name": "keypress"})
        piano_key_positions = ObsTerm(func=mdp.piano_key_pos, params={"piano_asset_cfg": SceneEntityCfg("piano")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    key_on = RewTerm(
        func=mdp.key_on_reward,
        params={
            "command_name": "keypress",
            "key_close_enough_to_pressed": KEY_CLOSE_ENOUGH_TO_PRESSED,
        },
        weight=4.0,
    )
    key_off = RewTerm(
        func=mdp.key_off_reward,
        params={
            "command_name": "keypress",
            "key_close_enough_to_pressed": KEY_CLOSE_ENOUGH_TO_PRESSED,
        },
        weight=1.0,
    )
    energy = RewTerm(
        func=mdp.energy_reward,
        params={"robot_asset_cfg": SceneEntityCfg("robot")},
        weight=-5e-3,
    )
    minimize_fingertip_to_key_distance = RewTerm(
        func=mdp.fingertip_to_key_distance_reward,
        params={
            "command_name": "keypress",
            "asset_cfg": SceneEntityCfg("robot"),
            "finger_close_enough_to_key": FINGER_CLOSE_ENOUGH_TO_KEY,
        },
        weight=0.5,
    )
    # sustain_pedal = RewTerm(
    #     func=mdp.sustain_pedal_reward,
    #     weight=-1e-4,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Environment configuration
##

@configclass
class PianoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the piano environment."""
    # Scene settings
    scene: PianoSceneCfg = PianoSceneCfg(num_envs=1024, env_spacing=2.5)

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
    # events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (-0.5, 1.0, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
