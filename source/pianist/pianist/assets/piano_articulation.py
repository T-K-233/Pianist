import torch
from isaaclab.utils import configclass
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

from pianist.assets.piano_constants import (
    WHITE_KEY_INDICES,
    BLACK_KEY_INDICES,
    NUM_KEYS,
    WHITE_KEY_LENGTH,
    BLACK_KEY_LENGTH,
    BLACK_KEY_HEIGHT,
    KEY_TRIGGER_THRESHOLD,
)


class PianoArticulation(Articulation):
    def __init__(self, cfg: "PianoArticulationCfg"):
        super().__init__(cfg)

        # add type hinting
        self.cfg: PianoArticulationCfg

    def post_scene_creation_init(self):
        # we need to do these initializations manually after the scene is created.
        self.key_names = []
        for i in range(NUM_KEYS):
            if i in WHITE_KEY_INDICES:
                self.key_names.append(f"white_key_{i}")
            else:
                self.key_names.append(f"black_key_{i}")

        joint_names = [name + "_joint" for name in self.key_names]
        key_body_indices, _ = self.find_bodies(self.key_names, preserve_order=True)
        key_joint_indices, _ = self.find_joints(joint_names, preserve_order=True)

        # ensure these are 1D tensors
        self._key_body_indices = torch.tensor(key_body_indices, device=self.device)
        self._key_joint_indices = torch.tensor(key_joint_indices, device=self.device)

        self._key_contact_offsets = torch.zeros(self.num_joints, 3, device=self.device)

        # specify the contact position to be at 80% of the key length
        self._key_contact_offsets[WHITE_KEY_INDICES, 0] += -0.8 * WHITE_KEY_LENGTH
        self._key_contact_offsets[BLACK_KEY_INDICES, 0] += -0.8 * BLACK_KEY_LENGTH
        self._key_contact_offsets[BLACK_KEY_INDICES, 2] += 0.2 * BLACK_KEY_HEIGHT

        # gives an upward force when the key is at rest
        spring_ref_position = torch.zeros(1, self.num_joints, device=self.device)
        spring_ref_position[:] = -0.2
        self.set_joint_position_target(spring_ref_position)

    @property
    def key_press_states(self) -> torch.Tensor:
        key_press_normalized = self.data.joint_pos / self.data.default_joint_pos_limits[:, :, 1]
        return key_press_normalized[:, self._key_joint_indices]

    def get_key_world_locations(self, env_ids: torch.Tensor, key_indices: torch.Tensor) -> torch.Tensor:
        key_locations = self.data.body_pos_w[env_ids, self._key_body_indices[key_indices]]

        # add the offset from the key rotate joints as the desired contact location
        key_locations += self._key_contact_offsets[key_indices]
        return key_locations


@configclass
class PianoArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    ##
    # Initialize configurations.
    ##

    class_type: type = PianoArticulation

    # TODO: currently we are using position as trigger, perhaps change to velocity?
    key_trigger_threshold: float = KEY_TRIGGER_THRESHOLD
