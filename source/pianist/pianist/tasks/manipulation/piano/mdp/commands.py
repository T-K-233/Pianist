
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from pianist.assets.piano_constants import WHITE_KEY_INDICES, NUM_KEYS, WHITE_KEY_LENGTH


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

    def __init__(self, cfg: "KeyPressCommandCfg", env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # TODO: move these to the cfg
        robot_name = "robot"
        piano_name = "piano"
        robot_finger_body_names = ["ffdistal"]

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[robot_name]
        self.piano: Articulation = env.scene[piano_name]
        self.finger_body_indices, _ = self.robot.find_bodies(robot_finger_body_names)
        self.finger_body_indices = torch.tensor(self.finger_body_indices, device=self.device)

        query_key_names = []
        for i in range(NUM_KEYS):
            if i in WHITE_KEY_INDICES:
                query_key_names.append(f"white_key_{i}")
            else:
                query_key_names.append(f"black_key_{i}")
        self.key_body_indices, _ = self.piano.find_bodies(query_key_names, preserve_order=True)
        self.key_joint_indices, _ = self.piano.find_joints([name + "_joint" for name in query_key_names], preserve_order=True)
        self.key_body_indices = torch.tensor(self.key_body_indices, device=self.device)

        # create buffers
        # discrete command to indicate if the key needs to be pressed
        self._keypress_command = torch.zeros(self.num_envs, 88, device=self.device)
        # target positions of the keys to be pressed, maximum 10 keys (one for each finger)
        self._keypress_target_positions = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # -- metrics
        self.metrics["finger_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correct_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correct_not_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["f1_error"] = torch.zeros(self.num_envs, device=self.device)

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
        return torch.cat(
            [
                self.keypress_command,
                self.keypress_target_positions.flatten(start_dim=1),
            ],
            dim=1,
        )

    @property
    def keypress_command(self) -> torch.Tensor:
        return self._keypress_command

    @property
    def piano_pressed_status(self) -> torch.Tensor:
        # print(self.piano.data.joint_pos[:, self.key_joint_indices].max())
        # max depression is 0.0887 for black and 0.0666 for white
        keypress_normalized = self.piano.data.joint_pos / self.piano.data.default_joint_pos_limits[:, :, 1]
        return keypress_normalized[:, self.key_joint_indices]

    @property
    def keypress_target_positions(self) -> torch.Tensor:
        return self._keypress_target_positions[:, :, 0:3]

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        fingertip_positions = self.piano.data.body_state_w[:, self.finger_body_indices, 0:3]
        pos_error = torch.norm(self._keypress_target_positions[:, :, 0:3] - fingertip_positions, dim=-1).mean(dim=-1)
        self.metrics["finger_position_error"] = pos_error

        correct_pressed_percentage = ((self.keypress_command > 0.5) * (torch.abs(self.keypress_command - self.piano_pressed_status) < 0.2)).sum(dim=-1).float() / self.keypress_command.sum(dim=-1)
        correct_not_pressed_percentage = ((self.keypress_command < 0.5) * (torch.abs(self.keypress_command - self.piano_pressed_status) < 0.2)).sum(dim=-1).float() / (NUM_KEYS - self.keypress_command.sum(dim=-1))
        f1_score = (torch.abs(self.keypress_command - self.piano_pressed_status) < 0.2).sum(dim=-1).float() / NUM_KEYS
        self.metrics["correct_pressed"] = correct_pressed_percentage
        self.metrics["correct_not_pressed"] = correct_not_pressed_percentage
        self.metrics["f1_error"] = f1_score

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        random_note_index = torch.randint(20, 60, (len(env_ids),), device=self.device)
        self._keypress_command[env_ids, :] = 0
        self._keypress_command[env_ids, random_note_index] = 1
        body_indices = self.key_body_indices[random_note_index]
        self._keypress_target_positions[env_ids, 0, 0:3] = self.piano.data.body_state_w[env_ids, body_indices, 0:3]

        # specify the contact position to be at 70% of the white key length
        self._keypress_target_positions[env_ids, 0, 0] -= 0.7 * WHITE_KEY_LENGTH

    def _update_command(self):
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
        piano_key_position = self.keypress_target_positions[:, 0, 0:3]
        self.goal_pose_visualizer.visualize(piano_key_position)
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.finger_body_indices]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, 0, 0:3], body_link_pose_w[:, 0, 3:7])


@configclass
class KeyPressCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = KeyPressCommand

    # robot_name: str = MISSING
    # """Name of the robot in the environment for which the commands are generated."""

    # piano_name: str = MISSING
    # """Name of the piano in the environment for which the commands are generated."""

    # robot_finger_body_names: list[str] = MISSING
    # """Names of the robot finger bodies for which the commands are generated."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
