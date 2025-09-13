from pianist.music.midi_file import MidiFile, NoteTrajectory


if __name__ == "__main__":
    # midi_file = "./source/pianist/data/music/twinkle-twinkle-trimmed.mid"
    midi_file = "/home/tk/Downloads/robopianist/robopianist/music/data/pig_single_finger/nocturne_op9_no_2-1.proto"
    dt = 0.2

    midi = MidiFile.from_file(midi_file)
    trajectory = NoteTrajectory.from_midi(midi, dt=dt)
    print(trajectory.notes)
    # print(trajectory.sustains)
    # print(trajectory.to_piano_roll())

    import time
    import mido

    outport = mido.open_output("Piano MIDI Device:Piano MIDI Device Digital Piano 24:0")

    t_start = time.time()
    for instant in trajectory.notes:
        print(time.time() - t_start, instant)
        for note in instant:
            outport.send(mido.Message("note_on", note=note.number, velocity=127, time=0))
        time.sleep(dt)
