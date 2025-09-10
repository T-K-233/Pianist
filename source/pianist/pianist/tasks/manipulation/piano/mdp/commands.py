
import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab.envs import ManagerBasedEnv


WHITE_KEY_INDICES = [
    0,
    2,
    3,
    5,
    7,
    8,
    10,
    12,
    14,
    15,
    17,
    19,
    20,
    22,
    24,
    26,
    27,
    29,
    31,
    32,
    34,
    36,
    38,
    39,
    41,
    43,
    44,
    46,
    48,
    50,
    51,
    53,
    55,
    56,
    58,
    60,
    62,
    63,
    65,
    67,
    68,
    70,
    72,
    74,
    75,
    77,
    79,
    80,
    82,
    84,
    86,
    87,
]

BLACK_TWIN_KEY_INDICES = [
    4,
    6,
    16,
    18,
    28,
    30,
    40,
    42,
    52,
    54,
    64,
    66,
    76,
    78,
]
BLACK_TRIPLET_KEY_INDICES = [
    1,
    9,
    11,
    13,
    21,
    23,
    25,
    33,
    35,
    37,
    45,
    47,
    49,
    57,
    59,
    61,
    69,
    71,
    73,
    81,
    83,
    85,
]

class KeyPressCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: "KeyPressCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "KeyPressCommandCfg", env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.piano = env.scene["piano"]
        self.body_idx = self.robot.find_bodies(["ffdistal"])[0][0]

        query_key_names = []
        for i in range(88):
            if i in WHITE_KEY_INDICES:
                query_key_names.append(f"white_key_{i}")
            else:
                query_key_names.append(f"black_key_{i}")
        self.key_body_index_mapping, _ = self.piano.find_bodies(query_key_names, preserve_order=True)
        self.key_body_index_mapping = torch.tensor(self.key_body_index_mapping, device=self.device)

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "KeyPressCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        random_note_index = torch.randint(20, 60, (len(env_ids),), device=self.device)
        body_indices = self.key_body_index_mapping[random_note_index]
        self.pose_command_w[env_ids, 0:3] = self.piano.data.body_state_w[env_ids, body_indices, 0:3]
        # r = torch.empty(len(env_ids), device=self.device)
        # self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        # self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        # self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # # -- orientation
        # euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        # euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        # euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        # euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        # quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # # make sure the quaternion has real part as positive
        # self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        # middle_c_key_index, _ = self.piano.find_bodies(["white_key_39"])
        # self.piano.data.body_state_w[:, middle_c_key_index]
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


@configclass
class KeyPressCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = KeyPressCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
