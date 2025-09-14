from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils import configclass

from pianist.music.midi_file import MidiFile, NoteTrajectory
from pianist.assets.piano_articulation import PianoArticulation
from pianist.assets.piano_constants import NUM_KEYS


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

    cfg: CommandTermCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.piano: PianoArticulation = env.scene[self.cfg.piano_name]
        self.piano.manual_init()

        if self.cfg.robot_name:
            self.robot: Articulation = env.scene[self.cfg.robot_name]
            finger_body_indices, _ = self.robot.find_bodies(self.cfg.robot_finger_body_names, preserve_order=True)
            self._finger_body_indices = torch.tensor(finger_body_indices, device=self.device)

        # create buffers
        # discrete command to indicate if the key needs to be pressed
        self._key_press_goals = torch.zeros(self.num_envs, 88, device=self.device)
        # target locations of the keys to be pressed, maximum 10 keys (one for each finger)
        self._target_key_locations = torch.zeros(self.num_envs, 5, 3, device=self.device)

        # active fingers: one-hot vector with 5 elements (thumb, index, middle, ring, pinky)
        self._active_fingers = torch.zeros(self.num_envs, 5, device=self.device)

        # -- metrics
        if self.cfg.robot_name:
            self.metrics["fingertip_to_key_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correctly_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["correctly_not_pressed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["f1"] = torch.zeros(self.num_envs, device=self.device)

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
        key_on_threshold = self.cfg.key_close_enough_to_pressed

        if self.cfg.robot_name:
            pos_error = (self.active_fingers * torch.norm((self._target_key_locations - self.fingertip_positions), dim=-1)).sum(dim=-1)

        on_keys = self.key_press_goals > 0.5
        off_keys = self.key_press_goals < 0.5

        correctly_pressed_percentage = (
            (torch.abs(self.key_press_goals - self.key_press_actual) < key_on_threshold).float() * on_keys.float()
        ).sum(dim=-1).float() / (on_keys.sum(dim=-1).float() + 1e-6)
        correctly_not_pressed_percentage = (
            (torch.abs(self.key_press_goals - self.key_press_actual) < key_on_threshold).float() * off_keys.float()
        ).sum(dim=-1).float() / (off_keys.sum(dim=-1).float() + 1e-6)

        f1_score = (torch.abs(self.key_press_goals - self.key_press_actual) < key_on_threshold).sum(dim=-1).float() / NUM_KEYS

        # print(correctly_pressed_percentage, correctly_not_pressed_percentage)
        # print("metrics", self.key_press_goals[0])

        if self.cfg.robot_name:
            self.metrics["fingertip_to_key_distance"] = pos_error
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
                loc = self.target_key_locations[:, i, 0:3]
                self.goal_key_visualizers[i].visualize(loc)
            # -- current body pose
            finger_quat = self.robot.data.body_quat_w[:, self._finger_body_indices][:, 0]
            for i in range(5):
                pos = self.fingertip_positions[:, i]
                self.current_key_visualizers[i].visualize(pos, finger_quat)
        else:
            for i in range(5):
                loc = self.target_key_locations[:, i, 0:3]
                self.goal_key_visualizers[i].visualize(loc)
                self.current_key_visualizers[i].visualize(loc)


class SongKeyPressCommand(KeyPressCommand):
    """
    This command generates random key press commands.
    """

    cfg: "SongKeyPressCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "SongKeyPressCommandCfg", env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        midi = MidiFile.from_file(self.cfg.midi_file)
        self.trajectory = NoteTrajectory.from_midi(midi, dt=env.step_dt)
        self.trajectory = self.trajectory.trim_silence()

        self.song_num_frames = len(self.trajectory)

        # create tensor buffer for the notes and fingering
        self._key_goal_trajectory = torch.zeros(self.song_num_frames, 88, device=self.device)
        self._active_fingers_trajectory = torch.zeros(self.song_num_frames, 5, device=self.device)
        self._active_keys_trajectory = torch.zeros(self.song_num_frames, 5, dtype=torch.int32, device=self.device)

        for frame_index in range(self.song_num_frames):
            notes = self.trajectory.notes[frame_index]
            for note in notes:
                if note.fingering >= 5:
                    # this is left hand, pass for now
                    continue
                finger_idx = note.fingering
                self._key_goal_trajectory[frame_index, note.key] = 1.0
                self._active_fingers_trajectory[frame_index, finger_idx] = 1.0
                self._active_keys_trajectory[frame_index, finger_idx] = note.key

        self._env_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def __str__(self) -> str:
        msg = "SongKeyPressCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def _resample_command(self, env_ids: torch.Tensor):
        self._env_steps[env_ids] = 0

        self._key_press_goals[env_ids] = 0.0
        self._active_fingers[env_ids] = 0.0
        self._target_key_locations[:] = 0.0

    def _update_command(self):
        self._env_steps[:] += 1
        # motion time of these environments need to be resampled
        env_ids = torch.where(self._env_steps >= len(self.trajectory))[0]
        self._resample_command(env_ids)

        # reset the key press goals and target key locations
        self._key_press_goals[:] = self._key_goal_trajectory[self._env_steps]
        self._active_fingers[:] = self._active_fingers_trajectory[self._env_steps]

        # self.piano.get_key_world_locations()
        key_indices = self._active_keys_trajectory[self._env_steps]
        env_id_sel = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 5)
        key_locations = self.piano.data.body_pos_w[env_id_sel, self.piano._key_body_indices[key_indices]]
        key_locations += self.piano._key_contact_offsets[key_indices]

        self._target_key_locations[:] = key_locations * self._active_fingers.unsqueeze(-1)

        # # TODO: this is super slow, need to convert NoteTrajectory to tensor
        # for env_id in range(self.num_envs):
        #     frame_index = self._env_steps[env_id]
        #     keys = self._keys_trajectory[frame_index]
        #     target_locations = self.piano.get_key_world_locations(env_id, keys)
        #     self._target_key_locations[env_id, :target_locations.shape[0], 0:3] = target_locations

        # for env_id in range(self.num_envs):
        #     frame_index = self._env_steps[env_id]
        #     for note in notes:
        #         if note.fingering >= 5:
        #             # this is left hand, pass for now
        #             continue
        #         finger_idx = note.fingering
        #         self._key_press_goals[env_id, note.key] = 1.0
        #         target_locations = self.piano.get_key_world_locations(env_id, note.key)
        #         self._target_key_locations[env_id, finger_idx, 0:3] = target_locations
        #         if self.cfg.robot_name:
        #             self._active_fingers[env_id, finger_idx] = 1.0

        # print(self._env_steps[0], len(self.trajectory))
        # print("update", self.key_press_goals[0])

@configclass
class SongKeyPressCommandCfg(CommandTermCfg):
    """Configuration for song key press command generator."""

    class_type: type = SongKeyPressCommand

    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    piano_name: str = MISSING
    """Name of the piano in the environment for which the commands are generated."""

    robot_name: str = MISSING
    """Name of the robot in the environment for which the commands are generated."""

    robot_finger_body_names: list[str] = MISSING
    """Names of the robot finger bodies for which the commands are generated."""

    key_close_enough_to_pressed: float = 0.05
    """The threshold for the key to be considered pressed."""

    max_num_fingers: int = 5
    """The maximum number of fingerings possible, should match number of fingers of the robot."""

    midi_file: str = MISSING
    """The path to the MIDI file."""

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


class RandomKeyPressCommand(KeyPressCommand):
    """
    This command generates random key press commands.
    """

    cfg: "RandomKeyPressCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "RandomKeyPressCommandCfg", env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = "RandomKeyPressCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def _resample_command(self, env_ids: torch.Tensor):
        # sample new pose targets
        # -- position
        key_indices = torch.zeros(env_ids.shape[0], self.cfg.num_notes, dtype=torch.int32, device=self.device)
        base = torch.randint(20, 60, (env_ids.shape[0], 1), device=self.device)
        key_indices[:, :] = base
        for i in range(1, self.cfg.num_notes):
            offset = torch.randint(1, 5, (env_ids.shape[0], 1), device=self.device)
            key_indices[:, i:] += offset

        self._key_press_goals[env_ids, :] = 0.0
        self._target_key_locations[env_ids, :, :] = 0.0
        self._active_fingers[:, :] = 0.0

        # Convert env_ids to tensor and create proper indexing
        env_ids_sel = env_ids.unsqueeze(1).expand(-1, self.cfg.num_notes)
        self._key_press_goals[env_ids_sel, key_indices] = 1.0

        target_locations = self.piano.get_key_world_locations(env_ids_sel, key_indices)

        # for now, only index finger (element 1) is active
        # num notes must be less than 5, we are not using thumb
        self._target_key_locations[env_ids, 1:target_locations.shape[1] + 1, 0:3] = target_locations
        self._active_fingers[:, 1:target_locations.shape[1] + 1] = 1.0  # index finger is active

    def _update_command(self):
        pass
        # self.piano.write_joint_position_to_sim(self._key_press_goals, self.piano._key_joint_indices)


@configclass
class RandomKeyPressCommandCfg(CommandTermCfg):
    """Configuration for random key press command generator."""

    class_type: type = RandomKeyPressCommand

    piano_name: str = MISSING
    """Name of the piano in the environment for which the commands are generated."""

    robot_name: str = MISSING
    """Name of the robot in the environment for which the commands are generated."""

    robot_finger_body_names: list[str] = MISSING
    """Names of the robot finger bodies for which the commands are generated."""

    key_close_enough_to_pressed: float = 0.05
    """The threshold for the key to be considered pressed."""

    num_notes: int = 3
    """The number of notes to generate."""

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
