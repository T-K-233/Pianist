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
from isaaclab.sensors import ContactSensorCfg
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

    # robots
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/.*",
        history_length=3,
        track_air_time=False,
    )

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
        robot_name="robot",
        robot_finger_body_names=["thtip", "fftip", "mftip", "rftip", "lftip"],
        key_close_enough_to_pressed=KEY_CLOSE_ENOUGH_TO_PRESSED,
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
        active_fingers = ObsTerm(func=mdp.active_fingers_lookahead, params={"command_name": "keypress"})
        forearm_pos = ObsTerm(func=mdp.forearm_pos, params={"robot_asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        piano_key_positions = ObsTerm(func=mdp.piano_key_pos, params={"piano_asset_cfg": SceneEntityCfg("piano")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        # observation terms (order preserved)
        distance_to_key = ObsTerm(func=mdp.distance_to_key, params={"command_name": "keypress"})

        def __post_init__(self):
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
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
        weight=2.0,
    )
    key_off = RewTerm(
        func=mdp.key_off_reward,
        params={
            "command_name": "keypress",
            "key_close_enough_to_pressed": KEY_CLOSE_ENOUGH_TO_PRESSED,
        },
        weight=0.2,
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
        weight=1.0,
    )
    # sustain_pedal = RewTerm(
    #     func=mdp.sustain_pedal_reward,
    #     weight=-1e-4,
    # )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )
    joint_deviation_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "WRJ2",
                ],
            )
        },
        weight=-0.5,
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "(LF|TH)J5",
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)",
                ],
            )
        },
        weight=-0.01,
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "palm",
                ".*proximal",
                ".*middle",
            ]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )


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
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 40.0
        self.viewer.eye = (-0.5, 1.0, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 0.01
