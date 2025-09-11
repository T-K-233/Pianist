import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


PIANO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/pianist/data/assets/piano/usd/piano.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.5),
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
