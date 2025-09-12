import torch
from isaaclab.utils import configclass
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

from pianist.assets.piano_constants import WHITE_KEY_INDICES, NUM_KEYS, WHITE_KEY_LENGTH


class PianoArticulation(Articulation):
    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)

    def manual_init(self):
        self.key_names = []
        for i in range(NUM_KEYS):
            if i in WHITE_KEY_INDICES:
                self.key_names.append(f"white_key_{i}")
            else:
                self.key_names.append(f"black_key_{i}")

        joint_names = [name + "_joint" for name in self.key_names]
        key_body_indices, _ = self.find_bodies(self.key_names, preserve_order=True)
        key_joint_indices, _ = self.find_joints(joint_names, preserve_order=True)
        self._key_body_indices = torch.tensor(key_body_indices, device=self.device)
        self._key_joint_indices = torch.tensor(key_joint_indices, device=self.device)

        # gives an upward force when the key is at rest
        spring_ref_position = torch.zeros(1, self.num_joints, device=self.device)
        spring_ref_position[:] = -0.017453292519943295
        self.set_joint_position_target(spring_ref_position)

    @property
    def key_press_states(self) -> torch.Tensor:
        key_press_normalized = self.data.joint_pos / self.data.default_joint_pos_limits[:, :, 1]
        return key_press_normalized[:, self._key_joint_indices]

    def get_key_world_locations(self, env_ids: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        key_locations = self.data.body_pos_w[env_ids, self._key_body_indices[key_index]]

        # specify the contact position to be at 70% of the white key length
        key_locations[:, 0] -= 0.7 * WHITE_KEY_LENGTH
        return key_locations


@configclass
class PianoArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    ##
    # Initialize configurations.
    ##

    class_type: type = PianoArticulation
