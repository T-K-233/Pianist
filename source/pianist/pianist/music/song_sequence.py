import torch

from pianist.music.midi_file import MidiFile, NoteTrajectory


class SongSequence:
    @classmethod
    def from_simple(cls, num_frames: int, dt: float, device: torch.device) -> "SongSequence":
        seq = cls(num_frames, dt, device)

        seg_length = num_frames // 4

        seq._active_fingers[0:seg_length, 1] = 1
        seq._active_keys[0:seg_length, 39] = 1
        seq._fingerings[0:seg_length, 1] = 39

        seq._active_fingers[seg_length:2 * seg_length, 2] = 1
        seq._active_keys[seg_length:2 * seg_length, 43] = 1
        seq._fingerings[seg_length:2 * seg_length, 2] = 43

        seq._active_fingers[2 * seg_length:3 * seg_length, 3] = 1
        seq._active_keys[2 * seg_length:3 * seg_length, 46] = 1
        seq._fingerings[2 * seg_length:3 * seg_length, 3] = 46

        seq._active_fingers[3 * seg_length:4 * seg_length, [1, 2, 3]] = 1
        seq._active_keys[3 * seg_length:4 * seg_length, [39, 43, 46]] = 1
        seq._fingerings[3 * seg_length:4 * seg_length, 1] = 39
        seq._fingerings[3 * seg_length:4 * seg_length, 2] = 43
        seq._fingerings[3 * seg_length:4 * seg_length, 3] = 46

        return seq

    @classmethod
    def from_random(cls, num_frames: int, dt: float, device: torch.device) -> "SongSequence":
        return cls(num_frames, dt, device)

    @classmethod
    def from_midi(cls, midi_file: str, dt: float, device: torch.device, stretch_factor: float = 1.0) -> "SongSequence":
        midi = MidiFile.from_file(midi_file)
        midi = midi.stretch(stretch_factor)
        trajectory = NoteTrajectory.from_midi(midi, dt=dt)  # HACK: slow down the tempo
        trajectory = trajectory.trim_silence()
        num_frames = len(trajectory)
        seq = cls(num_frames, dt, device)

        for frame_index in range(num_frames):
            notes = trajectory.notes[frame_index]
            for note in notes:
                if note.fingering >= 5:
                    # note is for left hand, pass for now
                    continue
                finger_idx = note.fingering
                seq._active_fingers[frame_index, finger_idx] = 1
                seq._active_keys[frame_index, note.key] = 1
                seq._fingerings[frame_index, finger_idx] = note.key
        return seq

    def __init__(self, num_frames: int, dt: float, device: torch.device):
        self.num_frames = num_frames
        self.dt = dt
        self.duration = num_frames * dt

        # stores boolean masks of fingers that are active at each frame
        self._active_fingers = torch.zeros(self.num_frames, 5, dtype=torch.bool, device=device)
        # stores boolean masks of keys that should be pressed at each frame
        self._active_keys = torch.zeros(self.num_frames, 88, dtype=torch.bool, device=device)
        # stores the fingerings (which finger should press which key)
        self._fingerings = torch.zeros(self.num_frames, 5, dtype=torch.int32, device=device)

    def get_frames(self, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        fingerings might not be full, they need to be masked with the active keys.

        Args:
            frame_indices: The indices of the frames to get.

        Returns:
            A tuple of the active keys, active fingers, and fingerings.
            active_keys contains boolean masks of keys among the total 88 keys that should be pressed
            active_fingers contains boolean masks of fingers among the total 5 fingers that are active (pressing a key)
            fingerings contains the fingerings (which finger should press which key)
        """
        frame_indices = frame_indices.clamp(0, self.num_frames - 1)
        return (
            self._active_keys[frame_indices],
            self._active_fingers[frame_indices],
            self._fingerings[frame_indices],
        )
