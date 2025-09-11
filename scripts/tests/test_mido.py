import time
import mido


print(mido.get_input_names())

inport = mido.open_input("Piano MIDI Device:Piano MIDI Device Digital Piano 24:0")
outport = mido.open_output("Piano MIDI Device:Piano MIDI Device Digital Piano 24:0")


msg = mido.Message("note_on", note=60, velocity=127, time=0)
outport.send(msg)
time.sleep(0.1)

msg = mido.Message("note_on", note=64, velocity=127, time=0)
outport.send(msg)
time.sleep(0.1)

msg = mido.Message("note_on", note=67, velocity=127, time=0)
outport.send(msg)
time.sleep(0.5)

msg = mido.Message("note_on", note=60, velocity=127, time=0)
outport.send(msg)
msg = mido.Message("note_on", note=64, velocity=127, time=0)
outport.send(msg)
msg = mido.Message("note_on", note=67, velocity=127, time=0)
outport.send(msg)

while True:
    msg = inport.receive()
    print(f"Time: {msg.time}, Type: {msg.type}, Channel: {msg.channel}, Note: {msg.note}, Velocity: {msg.velocity}")
