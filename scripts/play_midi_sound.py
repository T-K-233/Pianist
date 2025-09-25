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

"""Play a MIDI file using FluidSynth and PyAudio.

Example usage:
    python examples/play_midi_file.py --file robopianist/music/data/rousseau/nocturne-trimmed.mid
"""

import argparse

from pianist.music import midi_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="./source/pianist/data/music/twinkle-twinkle-trimmed.mid")
    parser.add_argument("--stretch", type=float, default=1.0)
    parser.add_argument("--shift", type=int, default=0)
    args = parser.parse_args()

    midi = midi_file.MidiFile.from_file(args.file)
    midi = midi.stretch(args.stretch)
    midi = midi.transpose(args.shift)
    midi = midi.trim_silence()
    midi.play()
