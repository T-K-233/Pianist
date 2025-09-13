
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils import configclass

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from pianist.assets.piano_articulation import PianoArticulation
from pianist.assets.piano_constants import NUM_KEYS


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
        self.piano: PianoArticulation = env.scene[piano_name]
        self.piano.manual_init()

        finger_body_indices, _ = self.robot.find_bodies(robot_finger_body_names)
        self._finger_body_indices = torch.tensor(finger_body_indices, device=self.device)

        # create buffers
        # discrete command to indicate if the key needs to be pressed
        self._key_press_goals = torch.zeros(self.num_envs, 88, device=self.device)
        # target locations of the keys to be pressed, maximum 10 keys (one for each finger)
        self._target_key_locations = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # active fingers: one-hot vector with 5 elements (thumb, index, middle, ring, pinky)
        # for now, only index finger (element 1) is active
        self._active_fingers = torch.zeros(self.num_envs, 5, device=self.device)
        self._active_fingers[:, 1] = 1.0  # index finger is active

        # -- metrics
        self.metrics["fingertip_to_key_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correctly_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correctly_not_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["f1"] = torch.zeros(self.num_envs, device=self.device)

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
        return self.key_press_goals

    @property
    def key_press_goals(self) -> torch.Tensor:
        return self._key_press_goals

    @property
    def key_press_actual(self) -> torch.Tensor:
        return self.piano.key_press_states

    @property
    def target_key_locations(self) -> torch.Tensor:
        return self._target_key_locations

    @property
    def fingertip_positions(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._finger_body_indices]

    @property
    def active_fingers(self) -> torch.Tensor:
        return self._active_fingers

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error = torch.norm(self._target_key_locations - self.fingertip_positions, dim=-1).mean(dim=-1)
        key_on_threshold = self.cfg.key_close_enough_to_pressed

        correctly_pressed_percentage = (
            (self.key_press_goals > 0.5) * (torch.abs(self.key_press_goals - self.key_press_actual) < key_on_threshold)
        ).sum(dim=-1).float() / self.key_press_goals.sum(dim=-1)
        correctly_not_pressed_percentage = (
            (self.key_press_goals < 0.5) * (torch.abs(self.key_press_goals - self.key_press_actual) < key_on_threshold)
        ).sum(dim=-1).float() / (NUM_KEYS - self.key_press_goals.sum(dim=-1))

        f1_score = (torch.abs(self.key_press_goals - self.key_press_actual) < key_on_threshold).sum(dim=-1).float() / NUM_KEYS

        self.metrics["fingertip_to_key_distance"] = pos_error
        self.metrics["correctly_pressed"] = correctly_pressed_percentage
        self.metrics["correctly_not_pressed"] = correctly_not_pressed_percentage
        self.metrics["f1"] = f1_score

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        random_note_index = torch.randint(20, 60, (len(env_ids),), device=self.device)
        self._key_press_goals[env_ids, :] = 0
        self._key_press_goals[env_ids, random_note_index] = 1
        target_locations = self.piano.get_key_world_locations(env_ids, random_note_index)
        self._target_key_locations[env_ids, 0, 0:3] = target_locations

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_key_visualizer"):
                # -- goal pose
                self.goal_key_visualizer = VisualizationMarkers(self.cfg.goal_key_visualizer_cfg)
                # -- current body pose
                self.current_key_visualizer = VisualizationMarkers(self.cfg.current_key_visualizer_cfg)
            # set their visibility to true
            self.goal_key_visualizer.set_visibility(True)
            self.current_key_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_key_visualizer"):
                self.goal_key_visualizer.set_visibility(False)
                self.current_key_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        loc = self.target_key_locations[:, 0, 0:3]
        self.goal_key_visualizer.visualize(loc)
        # -- current body pose
        finger_quat = self.robot.data.body_quat_w[:, self._finger_body_indices][:, 0]
        pos = self.fingertip_positions[:, 0]
        self.current_key_visualizer.visualize(pos, finger_quat)


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

    key_close_enough_to_pressed: float = 0.05
    """The threshold for the key to be considered pressed."""

    goal_key_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_pos",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.05, 0.02, 0.025),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """The configuration for the goal pose visualization marker."""

    current_key_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/body_pos",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """The configuration for the current pose visualization marker."""
