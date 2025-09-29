# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piano modeling constants.

Inspired by: https://kawaius.com/wp-content/uploads/2019/04/Kawai-Upright-Piano-Regulation-Manual.pdf
"""

from math import atan

NUM_KEYS = 88
NUM_WHITE_KEYS = 52
WHITE_KEY_WIDTH = 0.0225
WHITE_KEY_LENGTH = 0.15
WHITE_KEY_HEIGHT = WHITE_KEY_WIDTH
SPACING_BETWEEN_WHITE_KEYS = 0.001
N_SPACES_BETWEEN_WHITE_KEYS = NUM_WHITE_KEYS - 1
BLACK_KEY_WIDTH = 0.01
BLACK_KEY_LENGTH = 0.09
# Unlike the other dimensions, the height of the black key was roughly set such that
# when a white key is fully depressed, the bottom of the black key is barely visible.
BLACK_KEY_HEIGHT = 0.018
PIANO_LENGTH = (NUM_WHITE_KEYS * WHITE_KEY_WIDTH) + (
    N_SPACES_BETWEEN_WHITE_KEYS * SPACING_BETWEEN_WHITE_KEYS
)

WHITE_KEY_X_OFFSET = 0
WHITE_KEY_Z_OFFSET = WHITE_KEY_HEIGHT / 2
BLACK_KEY_X_OFFSET = -WHITE_KEY_LENGTH / 2 + BLACK_KEY_LENGTH / 2
# The top of the black key should be 12.5 mm above the top of the white key.
BLACK_OFFSET_FROM_WHITE = 0.0125
BLACK_KEY_Z_OFFSET = WHITE_KEY_HEIGHT + BLACK_OFFSET_FROM_WHITE - BLACK_KEY_HEIGHT / 2

BASE_HEIGHT = 0.04
BASE_LENGTH = 0.1
BASE_WIDTH = PIANO_LENGTH
BASE_SIZE = [BASE_LENGTH / 2, BASE_WIDTH / 2, BASE_HEIGHT / 2]
BASE_X_OFFSET = -WHITE_KEY_LENGTH / 2 - 0.5 * BASE_LENGTH - 0.002
BASE_POS = [BASE_X_OFFSET, 0, BASE_HEIGHT / 2]

# A key is designed to travel downward 3/8 of an inch (roughly 10mm).
# Assuming the joint is positioned at the back of the key, we can write:
# tan(θ) = d / l, where d is the distance the key travels and l is the length of the
# key. Solving for θ, we get: θ = arctan(d / l).
WHITE_KEY_TRAVEL_DISTANCE = 0.01
WHITE_KEY_JOINT_MAX_ANGLE = atan(WHITE_KEY_TRAVEL_DISTANCE / WHITE_KEY_LENGTH)
# TODO(kevin): Figure out black key travel distance.
BLACK_KEY_TRAVEL_DISTANCE = 0.008
BLACK_KEY_JOINT_MAX_ANGLE = atan(BLACK_KEY_TRAVEL_DISTANCE / BLACK_KEY_LENGTH)
# Mass in kg.
WHITE_KEY_MASS = 0.04
BLACK_KEY_MASS = 0.02
# Joint spring reference, in degrees.
# At equilibrium, the joint should be at 0 degrees.
WHITE_KEY_SPRINGREF = -1
BLACK_KEY_SPRINGREF = -1
# Joint spring stiffness, in Nm/rad.
# The spring should be stiff enough to support the weight of the key at equilibrium.
WHITE_KEY_STIFFNESS = 2
BLACK_KEY_STIFFNESS = 2
# Joint damping and armature for smoothing key motion.
WHITE_JOINT_DAMPING = 0.05
BLACK_JOINT_DAMPING = 0.05
WHITE_JOINT_ARMATURE = 0.001
BLACK_JOINT_ARMATURE = 0.001

# Actuator parameters (for self-actuated only).
ACTUATOR_DYNPRM = 1
ACTUATOR_GAINPRM = 1

# Colors.
WHITE_KEY_COLOR = [0.9, 0.9, 0.9, 1]
BLACK_KEY_COLOR = [0.1, 0.1, 0.1, 1]
BASE_COLOR = [0.15, 0.15, 0.15, 1]

WHITE_KEY_INDICES = [
    0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48, 50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87,
]
BLACK_TWIN_KEY_INDICES = [
    4, 6, 16, 18, 28, 30, 40, 42, 52, 54, 64, 66, 76, 78
]
BLACK_TRIPLET_KEY_INDICES = [
    1, 9, 11, 13, 21, 23, 25, 33, 35, 37, 45, 47, 49, 57, 59, 61, 69, 71, 73, 81, 83, 85
]
BLACK_KEY_INDICES = BLACK_TWIN_KEY_INDICES + BLACK_TRIPLET_KEY_INDICES

TWIN_INDICES = list(range(2, NUM_WHITE_KEYS - 1, 7))
TRIPLET_INDICES = list(range(5, NUM_WHITE_KEYS - 1, 7))

# TODO: currently we are using position as trigger, perhaps change to velocity?
KEY_TRIGGER_THRESHOLD = 0.70
"""The percentage of position travelled for the key to be considered pressed."""
