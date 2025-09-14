from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils import configclass

from pianist.music.song_sequence import SongSequence
from pianist.assets.piano_articulation import PianoArticulation


FINGERTIP_COLORS = [
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
    (0.8, 0.8, 0.2),  # Yellow.
]

torch.set_printoptions(precision=2)


class KeyPressCommand(CommandTerm):
    """
    This command generates key press commands.
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

        if self.cfg.song_name.endswith(".proto"):
            self.song = SongSequence.from_midi(self.cfg.song_name, dt=env.step_dt, device=self.device)
        elif self.cfg.song_name == "simple":
            self.song = SongSequence.from_simple(num_frames=40, dt=env.step_dt, device=self.device)
        elif self.cfg.song_name == "random":
            # self.song = SongSequence.from_random(num_frames=40, dt=env.step_dt, device=self.device)
            pass
        else:
            raise ValueError(f"Invalid song name: {self.cfg.song_name}")

        # extract the robot and body index for which the command is generated
        self.piano: PianoArticulation = env.scene[self.cfg.piano_name]
        self.piano.manual_init()

        if self.cfg.robot_name:
            self.robot: Articulation = env.scene[self.cfg.robot_name]

            finger_body_indices, _ = self.robot.find_bodies(self.cfg.robot_finger_body_names, preserve_order=True)
            self._finger_body_indices = torch.tensor(finger_body_indices, device=self.device)

        # create buffers
        # discrete command to indicate if the key needs to be pressed
        self._key_goal_states = torch.zeros(self.num_envs, 88, dtype=torch.bool, device=self.device)
        # discrete vector with 5 elements (thumb, index, middle, ring, pinky)
        self._active_fingers = torch.zeros(self.num_envs, 5, dtype=torch.bool, device=self.device)
        # target locations of the keys to be pressed, maximum 10 keys (one for each finger)
        self._key_goal_locations = torch.zeros(self.num_envs, 5, 3, device=self.device)

        # step counter for the song
        self._song_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- metrics
        if self.cfg.robot_name:
            self.metrics["fingertip_to_key_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correctly_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correctly_not_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["f1"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "KeyPressCommand:\n"
        msg += f"\tSong name: {self.cfg.song_name}\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return self.key_goal_states.float()

    @property
    def key_goal_states(self) -> torch.Tensor:
        return self._key_goal_states

    @property
    def key_actual_states(self) -> torch.Tensor:
        return self.piano.key_press_states

    @property
    def key_goal_locations(self) -> torch.Tensor:
        return self._key_goal_locations

    @property
    def fingertip_positions(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._finger_body_indices]

    @property
    def active_fingers(self) -> torch.Tensor:
        return self._active_fingers

    def _resample_command(self, env_ids: torch.Tensor):
        self._song_steps[env_ids] = 0

        self._key_goal_states[env_ids] = 0
        self._active_fingers[env_ids] = 0
        self._key_goal_locations[env_ids] = 0.0

    def _update_command(self):
        self._song_steps[:] += 1
        env_ids = torch.where(self._song_steps >= self.song.num_frames)[0]
        self._resample_command(env_ids)

        # each of these tensors are (num_envs, dim)
        active_keys, active_fingers, fingerings = self.song.get_frames(self._song_steps)
        self._key_goal_states[:] = active_keys
        self._active_fingers[:] = active_fingers

        # self._key_goal_locations[:] = self.piano.get_key_world_locations(env_ids, keys_indices)
        # create a selection tensor for environment ids to get all the key locations
        env_id_sel = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(
            self.num_envs, active_fingers.shape[1]
        )

        key_locations = self.piano.data.body_pos_w[env_id_sel, self.piano._key_body_indices[fingerings]]

        # add the offset from the key rotate joints as the desired contact location
        key_locations += self.piano._key_contact_offsets[fingerings]

        # mask the key locations with the active fingers
        self._key_goal_locations[:] = key_locations * active_fingers.unsqueeze(-1)

    def _update_metrics(self):
        # compute the error
        key_on_threshold = self.cfg.key_close_enough_to_pressed

        num_active_fingers = self.active_fingers.sum(dim=-1).float()

        if self.cfg.robot_name:
            # compute the distance between goal and actual for all keys
            key_distances = torch.norm(self.key_goal_locations - self.fingertip_positions, dim=-1)
            # only consider the distances for the active fingers
            effective_distances = self.active_fingers * key_distances
            # get the average distance error across all active fingers
            distance_error = effective_distances.sum(dim=-1) / (num_active_fingers + 1e-6)

        # get the number of keys intended to be pressed and not pressed
        on_keys = self._key_goal_states
        off_keys = ~self._key_goal_states
        num_on_keys = on_keys.sum(dim=-1)
        num_off_keys = off_keys.sum(dim=-1)

        # the error between goal and actual for all keys
        key_joint_pos_errors = torch.abs(self.key_goal_states.float() - self.key_actual_states)

        # compute the number of keys that are correctly pressed and not pressed
        correctly_pressed_count = ((key_joint_pos_errors < key_on_threshold) * on_keys).sum(dim=-1)
        correctly_not_pressed_count = ((key_joint_pos_errors < key_on_threshold) * off_keys).sum(dim=-1)

        correctly_pressed_percentage = correctly_pressed_count.float() / (num_on_keys.float() + 1e-6)
        correctly_not_pressed_percentage = correctly_not_pressed_count.float() / (num_off_keys.float() + 1e-6)

        f1_score = (correctly_pressed_count + correctly_not_pressed_count).float() / ((num_on_keys + num_off_keys).float() + 1e-6)

        if self.cfg.robot_name:
            self.metrics["fingertip_to_key_distance"] = distance_error
        self.metrics["correctly_pressed"] = correctly_pressed_percentage
        self.metrics["correctly_not_pressed"] = correctly_not_pressed_percentage
        self.metrics["f1"] = f1_score

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_key_visualizer"):
                # -- goal pose
                self.goal_key_visualizers = []
                for i in range(5):
                    cfg = self.cfg.goal_key_visualizer_cfg.copy()
                    cfg.markers["cuboid"].visual_material.diffuse_color = FINGERTIP_COLORS[i]
                    self.goal_key_visualizers.append(VisualizationMarkers(cfg))
                # -- current body pose
                self.current_key_visualizers = []
                for i in range(5):
                    cfg = self.cfg.current_key_visualizer_cfg.copy()
                    cfg.markers["sphere"].visual_material.diffuse_color = FINGERTIP_COLORS[i]
                    self.current_key_visualizers.append(VisualizationMarkers(cfg))
            # set their visibility to true
            for goal_key_visualizer in self.goal_key_visualizers:
                goal_key_visualizer.set_visibility(True)
            for current_key_visualizer in self.current_key_visualizers:
                current_key_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_key_visualizers"):
                for goal_key_visualizer in self.goal_key_visualizers:
                    goal_key_visualizer.set_visibility(False)
                for current_key_visualizer in self.current_key_visualizers:
                    current_key_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if self.cfg.robot_name:
            if not self.robot.is_initialized:
                return
            # update the markers
            # -- goal pose
            for i in range(5):
                loc = self.key_goal_locations[:, i, 0:3]
                self.goal_key_visualizers[i].visualize(loc)
            # -- current body pose
            finger_quat = self.robot.data.body_quat_w[:, self._finger_body_indices][:, 0]
            for i in range(5):
                pos = self.fingertip_positions[:, i]
                self.current_key_visualizers[i].visualize(pos, finger_quat)
        else:
            for i in range(5):
                loc = self.key_goal_locations[:, i, 0:3]
                self.goal_key_visualizers[i].visualize(loc)
                self.current_key_visualizers[i].visualize(loc)


@configclass
class KeyPressCommandCfg(CommandTermCfg):
    """Configuration for key press command generator."""

    class_type: type = KeyPressCommand

    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    song_name: str = MISSING
    """Name of the song in the environment for which the commands are generated."""

    piano_name: str = MISSING
    """Name of the piano in the environment for which the commands are generated."""

    robot_name: str = MISSING
    """Name of the robot in the environment for which the commands are generated."""

    robot_finger_body_names: list[str] = MISSING
    """Names of the robot finger bodies for which the commands are generated."""

    key_close_enough_to_pressed: float = 0.05
    """The threshold for the key to be considered pressed."""

    goal_key_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_pos",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.05, 0.02, 0.025),
                visual_material=sim_utils.PreviewSurfaceCfg(),
            ),
        },
    )
    """The configuration for the goal pose visualization marker."""

    current_key_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/body_pos",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(),
            ),
        },
    )
    """The configuration for the current pose visualization marker."""
