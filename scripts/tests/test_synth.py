import time

from pianist.music.synthesizer import Synthesizer


if __name__ == "__main__":
    synthesizer = Synthesizer()
    synthesizer.start()
    synthesizer.note_on(60, 127)
    time.sleep(1)
    synthesizer.note_off(60)
    synthesizer.stop()
