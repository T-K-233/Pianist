import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from pianist.assets.piano_articulation import PianoArticulationCfg


PIANO_CFG = PianoArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/pianist/data/assets/piano/usd/piano.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        ".*": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim={".*": 2.0},
            stiffness={".*": 2.0},
            damping={".*": 0.05},
            armature={".*": 0.001},
            # TODO: how to implement springref=-0.017453292519943295??
        ),
    },
)
