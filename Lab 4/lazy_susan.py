# Lab 4: Lazy Susan driven by touch.
# Pad 8 on the capacitive sensor turns the tray 90 degrees, pad 3 turns it back.
# The speaker says "Rotating" two seconds before each move, using the Piper
# voice from Lab 3, loaded once at startup (falls back to espeak if missing).
#
#   python lazy_susan.py
import subprocess
import time
from pathlib import Path
import board
import busio
import adafruit_mpr121
import pi_servo_hat

SERVO_CH = 0
HOME = 105         # resting angle
TURNED = HOME - 90 # after a pad-8 touch: turns the other way, down to 15
PAD_TURN = 8
PAD_BACK = 3
STEP_DELAY = 0.01  # seconds per degree; bigger is slower and gentler
WARNING = 1.0      # seconds between the announcement and the move

VOICE = Path.home() / "Interactive-Lab-Hub" / "Lab 3" / "voices" / "en_US-lessac-medium.onnx"

# Load the Piper voice once at startup so each announcement is instant.
# Launching Piper fresh for every line costs about three seconds.
try:
    import numpy as np
    import sounddevice as sd
    from piper import PiperVoice
    _voice = PiperVoice.load(str(VOICE))
except Exception as e:  # noqa: BLE001
    print(f"Piper not available ({e}); using espeak")
    _voice = None


def say(text):
    """Speak through the USB speaker. Blocks until the speech is finished."""
    if _voice is None:
        subprocess.run(f'espeak -s 150 --stdout "{text}" | aplay -q', shell=True)
        return
    for chunk in _voice.synthesize(text):
        audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        sd.play(audio, samplerate=chunk.sample_rate)
        sd.wait()


i2c = busio.I2C(board.SCL, board.SDA)
touch = adafruit_mpr121.MPR121(i2c)
servo = pi_servo_hat.PiServoHat()
servo.restart()


def move_to(target):
    """Sweep one degree at a time so the tray doesn't jerk."""
    global angle
    # The speaker's current draw can dip the 5V and reset the servo chip into
    # sleep mode, so wake and re-initialize it before every move.
    servo.restart()
    servo.move_servo_position(SERVO_CH, angle)
    time.sleep(0.05)
    step = 1 if target > angle else -1
    for a in range(angle, target + step, step):
        servo.move_servo_position(SERVO_CH, a)
        time.sleep(STEP_DELAY)
    angle = target


angle = HOME
servo.move_servo_position(SERVO_CH, HOME)
print("Ready. Touch pad 8 to turn 90 degrees, pad 3 to turn back. Ctrl-C to quit.")

was_touched = {PAD_TURN: False, PAD_BACK: False}
try:
    while True:
        for pad in (PAD_TURN, PAD_BACK):
            now = touch[pad].value
            if now and not was_touched[pad]:        # only on a new touch
                if pad == PAD_TURN and angle != TURNED:
                    print("pad 8: turning")
                    say("Rotating")
                    time.sleep(WARNING)
                    move_to(TURNED)
                elif pad == PAD_BACK and angle != HOME:
                    print("pad 3: turning back")
                    say("Rotating")
                    time.sleep(WARNING)
                    move_to(HOME)
            was_touched[pad] = now
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopped.")
