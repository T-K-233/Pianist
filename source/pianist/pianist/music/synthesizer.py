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

"""Library for synthesizing music from MIDI files."""

import fluidsynth

from pianist.music import constants as consts


_PROGRAM = 0  # Acoustic Grand Piano
_CHANNEL = 0
_BANK = 0


class Synthesizer:
    """FluidSynth-based synthesizer."""

    def __init__(
        self,
        soundfont_path: str = consts.SF2_PATH,
        sample_rate: int = consts.SAMPLING_RATE,
    ) -> None:
        self._soundfont_path = soundfont_path
        self._sample_rate = sample_rate
        self._muted: bool = False
        self._sustained: bool = False

        # Initialize FluidSynth.
        self._synth = fluidsynth.Synth(samplerate=sample_rate)
        soundfont_id = self._synth.sfload(soundfont_path)
        self._synth.program_select(_CHANNEL, soundfont_id, _BANK, _PROGRAM)

    def _validate_note(self, note: int) -> None:
        assert consts.MIN_MIDI_PITCH <= note <= consts.MAX_MIDI_PITCH

    def _validate_velocity(self, velocity: int) -> None:
        assert consts.MIN_VELOCITY <= velocity <= consts.MAX_VELOCITY

    def start(self) -> None:
        self._synth.start()

    def stop(self) -> None:
        self._synth.delete()

    def mute(self, value: bool) -> None:
        self._muted = value
        if value:
            self.all_sounds_off()

    def all_sounds_off(self) -> None:
        self._synth.all_sounds_off(_CHANNEL)

    def all_notes_off(self) -> None:
        self._synth.all_notes_off(_CHANNEL)

    def note_on(self, note: int, velocity: int) -> None:
        if not self._muted:
            self._validate_note(note)
            self._validate_velocity(velocity)
            self._synth.noteon(_CHANNEL, note, velocity)

    def note_off(self, note: int) -> None:
        if not self._muted:
            self._validate_note(note)
            self._synth.noteoff(_CHANNEL, note)

    def sustain_on(self) -> None:
        if not self._muted:
            self._synth.cc(
                _CHANNEL, consts.SUSTAIN_PEDAL_CC_NUMBER, consts.MAX_CC_VALUE
            )
            self._sustained = True

    def sustain_off(self) -> None:
        if not self._muted:
            self._synth.cc(
                _CHANNEL, consts.SUSTAIN_PEDAL_CC_NUMBER, consts.MIN_CC_VALUE
            )
            self._sustained = False

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def sustained(self) -> bool:
        return self._sustained
