import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./source/pianist/data/robots/shadow_hand/usd/right_hand_translation.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        # fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "WRJ(X|Y)": 0.0,
            "WRJ2": 0.0,
            "WRJ1": -0.524,
            "(LF|TH)J5": 0.0,
            "(FF|MF|RF|LF)J4": 0.0,
            "(FF|MF|RF|LF)J(3|2|1)": 0.524,
            "THJ4": 0.524,
            "THJ(3|2|1)": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "WRJ(X|Y)",
                "WRJ(2|1)",
                "(LF|TH)J5",
                "(FF|MF|RF|LF|TH)J(4|3|2|1)",
            ],
            effort_limit_sim={
                "WRJ(X|Y)": 10,
                "WRJ2": 4.785,
                "WRJ1": 2.175,
                "LFJ5": 0.9,
                "(FF|MF|RF|LF)J(4|3)": 0.9,
                "(FF|MF|RF|LF)J(2|1)": 0.7245,
                "THJ5": 2.3722,
                "THJ4": 1.45,
                "THJ(3|2|1)": 0.99,
            },
            stiffness={
                "WRJ(X|Y)": 80.0,
                "WRJ(2|1)": 5.0,
                "(LF|TH)J5": 1.0,
                "(FF|MF|RF|LF|TH)J(4|3|2|1)": 1.0,
            },
            damping={
                "WRJ(X|Y)": 40.0,
                "WRJ(2|1)": 0.5,
                "(LF|TH)J5": 0.1,
                "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""
