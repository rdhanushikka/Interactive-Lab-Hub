#!/usr/bin/env python3
"""Door Greeter: the speech device for Lab 3 Part 2.

The Pi plays the device. It draws a door on the MiniPiTFT, notices a visitor
with the APDS9960 proximity sensor, listens through the USB mic (Silero VAD +
faster-whisper), and speaks through the USB speaker (Piper).

The scripted dialogue runs on its own: greet, match the name to a resident,
confirm it, wait for the door with progress lines every 10 s, offer to take a
message at 30 s. If it can't use what it heard twice in a row it says "try
again later" and resets. A web page on the wizard's phone shows the live
transcript and lets them override any step, or take over entirely by
switching auto dialogue off.

Run on the Pi, with the Lab 3 venv active and piscreen.service stopped:

    sudo systemctl stop piscreen.service
    cd ~/Interactive-Lab-Hub/"Lab 3"/greeter
    source ../.venv/bin/activate
    python greeter.py

Then open http://minimachine.local:5000 on a phone on the same network.

Options:
    --model base.en        whisper size (base.en catches names better than tiny)
    --prox 20              proximity reading (0-255) that counts as "someone is here"
    --no-auto-greet        don't greet on proximity; wizard presses Greet instead
"""

import argparse
import math
import sys
import threading
import time
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from flask import Flask, jsonify, request, Response
from piper import PiperVoice

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent.parent
VAD_MODEL = LAB_DIR / "models" / "silero_vad.onnx"
VOICE = LAB_DIR / "voices" / "en_US-lessac-medium.onnx"
MESSAGES_DIR = Path(__file__).resolve().parent / "messages"

RESIDENTS = ["Sam", "Pam", "Nicole"]

SAMPLE_RATE = 16000
TURN_SILENCE = 0.5      # Part C: ends a normal turn
MESSAGE_SILENCE = 1.5   # Part C: long threshold for composing a message
THINK_PAUSE = 0.3       # beat before the device replies
PROGRESS_EVERY = 10.0   # "Still waiting" cadence
WAIT_TIMEOUT = 30.0     # then offer to take a message
OFFER_TIMEOUT = 5.0     # how long to wait for an answer to the offer
DONE_HOLD = 3.0         # how long the "open door" screen stays before idle

# States
IDLE, LISTENING, THINKING, WAITING, OFFER, RECORDING, DONE = (
    "idle", "listening", "thinking", "waiting", "offer", "recording", "done")

LED_COLORS = {
    IDLE: (60, 60, 60),
    LISTENING: (255, 200, 0),
    THINKING: (255, 200, 0),
    WAITING: (60, 140, 255),
    OFFER: (255, 200, 0),
    RECORDING: (255, 50, 50),
    DONE: (60, 220, 90),
}
STATE_LABELS = {
    IDLE: "Door Greeter",
    LISTENING: "Listening...",
    THINKING: "Thinking",
    WAITING: "Waiting for",
    OFFER: "Message?",
    RECORDING: "Recording",
    DONE: "Door opening",
}


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = IDLE
        self.name = None            # resident being asked for
        self.pending_name = None    # name we asked "Did you say X?" about
        self.state_since = time.time()
        self.last_progress = 0.0
        self.heard = []             # [(ts, text)]
        self.said = []              # [(ts, text)]
        self.speaking = False
        self.last_transcript = ""
        self.prox = 0
        self.armed = True           # proximity trigger re-arms after the visitor leaves
        self.clear_since = None
        self.auto_greet = True
        self.auto_timers = True
        self.auto_dialogue = True   # device answers on its own; wizard can override
        self.fails = 0              # consecutive turns the device couldn't use
        self.messages = []          # saved message files

    def set(self, new_state, **kw):
        with self.lock:
            self.state = new_state
            self.state_since = time.time()
            self.last_progress = self.state_since
            for k, v in kw.items():
                setattr(self, k, v)

    def log_heard(self, text):
        with self.lock:
            self.heard.append((datetime.now().strftime("%H:%M:%S"), text))
            self.heard = self.heard[-20:]
            self.last_transcript = text

    def log_said(self, text):
        with self.lock:
            self.said.append((datetime.now().strftime("%H:%M:%S"), text))
            self.said = self.said[-20:]

    def snapshot(self):
        with self.lock:
            remaining = None
            if self.state == WAITING:
                remaining = max(0, int(WAIT_TIMEOUT - (time.time() - self.state_since)))
            return {
                "state": self.state,
                "name": self.name,
                "pending_name": self.pending_name,
                "heard": self.heard[-10:],
                "said": self.said[-10:],
                "speaking": self.speaking,
                "prox": self.prox,
                "armed": self.armed,
                "remaining": remaining,
                "auto_greet": self.auto_greet,
                "auto_timers": self.auto_timers,
                "auto_dialogue": self.auto_dialogue,
                "fails": self.fails,
                "messages": self.messages,
                "residents": RESIDENTS,
            }


S = State()


# ---------------------------------------------------------------------------
# Speaker (Piper -> default output), runs in its own thread off a queue
# ---------------------------------------------------------------------------
class Speaker(threading.Thread):
    def __init__(self, voice_path):
        super().__init__(daemon=True)
        self.voice = PiperVoice.load(str(voice_path))
        self.queue = []
        self.cv = threading.Condition()

    def say(self, text, then=None):
        """Queue text. `then` is a callable run after the line is spoken."""
        S.log_said(text)
        with self.cv:
            self.queue.append((text, then))
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                text, then = self.queue.pop(0)
            with S.lock:
                S.speaking = True
            try:
                for chunk in self.voice.synthesize(text):
                    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                    sd.play(audio, samplerate=chunk.sample_rate)
                    sd.wait()
            except Exception as e:  # keep the device alive if audio hiccups
                print("speak error:", e)
            time.sleep(0.25)  # let the mic settle before listening again
            with S.lock:
                S.speaking = False
            if then:
                then()


# ---------------------------------------------------------------------------
# Listener (mic -> VAD -> whisper), runs in its own thread
# ---------------------------------------------------------------------------
def build_vad(min_silence):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(VAD_MODEL)
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60), cfg.silero_vad.window_size


class Listener(threading.Thread):
    def __init__(self, model_name, on_utterance, on_message):
        super().__init__(daemon=True)
        self.recognizer = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.vad_turn, self.window = build_vad(TURN_SILENCE)
        self.vad_msg, _ = build_vad(MESSAGE_SILENCE)
        self.on_utterance = on_utterance
        self.on_message = on_message

    def transcribe(self, samples):
        segments, _ = self.recognizer.transcribe(samples, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    def run(self):
        buf = np.empty(0, dtype=np.float32)
        per_read = int(0.1 * SAMPLE_RATE)
        was_speaking = False
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as stream:
            while True:
                if S.speaking:
                    # Throw away everything the mic captures while we talk,
                    # including audio already queued in the stream buffer.
                    n = stream.read_available
                    if n:
                        stream.read(n)
                    was_speaking = True
                    time.sleep(0.05)
                    continue
                if was_speaking:
                    # Speech just ended: drop the tail, reset the detectors.
                    n = stream.read_available
                    if n:
                        stream.read(n)
                    self.vad_turn.reset()
                    self.vad_msg.reset()
                    buf = np.empty(0, dtype=np.float32)
                    was_speaking = False
                chunk, _ = stream.read(per_read)
                if S.speaking:
                    continue  # started talking mid-read; drop this chunk too
                buf = np.concatenate([buf, chunk.reshape(-1)])
                while len(buf) > self.window:
                    self.vad_turn.accept_waveform(buf[:self.window])
                    self.vad_msg.accept_waveform(buf[:self.window])
                    buf = buf[self.window:]

                recording = S.state == RECORDING
                while not self.vad_turn.empty():
                    utt = np.array(self.vad_turn.front.samples, dtype=np.float32)
                    self.vad_turn.pop()
                    if not recording:
                        text = self.transcribe(utt)
                        if text:
                            self.on_utterance(text)
                while not self.vad_msg.empty():
                    utt = np.array(self.vad_msg.front.samples, dtype=np.float32)
                    self.vad_msg.pop()
                    if recording:
                        self.on_message(utt, self.transcribe(utt))


# ---------------------------------------------------------------------------
# Display (MiniPiTFT): a door, a device box with an LED, and status text
# ---------------------------------------------------------------------------
class Display:
    def __init__(self):
        import board
        import digitalio
        from adafruit_rgb_display import st7789
        from PIL import Image, ImageDraw, ImageFont

        self.Image, self.ImageDraw = Image, ImageDraw
        cs = digitalio.DigitalInOut(board.D5)
        dc = digitalio.DigitalInOut(board.D25)
        self.disp = st7789.ST7789(board.SPI(), cs=cs, dc=dc, rst=None, baudrate=64000000,
                                  width=135, height=240, x_offset=53, y_offset=40)
        backlight = digitalio.DigitalInOut(board.D22)
        backlight.switch_to_output(value=True)
        self.w, self.h = self.disp.height, self.disp.width  # landscape 240x135
        self.image = Image.new("RGB", (self.w, self.h))
        self.draw = ImageDraw.Draw(self.image)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.big = ImageFont.truetype(font, 16)
        self.med = ImageFont.truetype(font, 13)
        self.small = ImageFont.truetype(font, 10)
        self.mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)

    def wrap(self, text, font, max_w):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self.draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines[:3]

    def render(self, snap, now):
        d = self.draw
        d.rectangle((0, 0, self.w, self.h), fill=(0, 0, 0))
        state = snap["state"]

        # --- door, right side ---
        fx0, fy0, fx1, fy1 = 158, 14, 228, 126
        opening = state == DONE
        d.rectangle((fx0 - 3, fy0 - 3, fx1 + 3, fy1), outline=(200, 200, 200), width=2)
        if opening:
            # door swung inward: draw as a narrow slab plus dark opening
            d.rectangle((fx0, fy0, fx1, fy1), fill=(15, 15, 25))
            d.polygon([(fx0, fy0), (fx0 + 22, fy0 + 10), (fx0 + 22, fy1 - 6), (fx0, fy1)],
                      fill=(120, 75, 40), outline=(230, 200, 160))
        else:
            d.rectangle((fx0, fy0, fx1, fy1), fill=(120, 75, 40), outline=(230, 200, 160))
            d.rectangle((fx0 + 10, fy0 + 10, fx1 - 10, fy0 + 48), outline=(90, 55, 30), width=2)
            d.rectangle((fx0 + 10, fy0 + 60, fx1 - 10, fy1 - 10), outline=(90, 55, 30), width=2)
            d.ellipse((fx1 - 18, 68, fx1 - 10, 76), fill=(240, 220, 120))
        if state == WAITING and snap["remaining"] is not None:
            txt = str(snap["remaining"])
            tw = d.textlength(txt, font=self.mono)
            d.text(((fx0 + fx1) / 2 - tw / 2, 50), txt, font=self.mono, fill=(255, 255, 255))

        # --- device box with LED, left of the door ---
        bx0, by0, bx1, by1 = 132, 40, 150, 78
        d.rectangle((bx0, by0, bx1, by1), fill=(40, 40, 40), outline=(160, 160, 160))
        color = LED_COLORS[state]
        if state == LISTENING:
            k = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(now * 5))
            color = tuple(int(c * k) for c in color)
        d.ellipse((bx0 + 4, by0 + 5, bx1 - 4, by0 + 15), fill=color)
        for y in (by0 + 22, by0 + 27, by0 + 32):  # speaker grille
            d.line((bx0 + 4, y, bx1 - 4, y), fill=(120, 120, 120))

        # --- text, left side ---
        label = STATE_LABELS[state]
        if state == WAITING and snap["name"]:
            label = f"Waiting for {snap['name']}"
        d.text((4, 2), label, font=self.big, fill=(255, 255, 255))
        if state == THINKING:
            dots = "." * (1 + int(now * 3) % 3)
            d.text((4, 22), dots, font=self.big, fill=(255, 200, 0))
        if snap["pending_name"] and state == LISTENING:
            d.text((4, 24), snap["pending_name"] + "?", font=self.big, fill=(255, 200, 0))
        y = 48
        if snap["said"]:
            for line in self.wrap("Pi: " + snap["said"][-1][1], self.small, 124):
                d.text((4, y), line, font=self.small, fill=(150, 200, 255)); y += 12
        y = max(y + 4, 90)
        if snap["heard"]:
            for line in self.wrap("You: " + snap["heard"][-1][1], self.small, 124):
                d.text((4, y), line, font=self.small, fill=(255, 230, 150)); y += 12
        if state == RECORDING:
            d.ellipse((4, 120, 12, 128), fill=(255, 50, 50))
            d.text((16, 118), "recording message", font=self.small, fill=(255, 120, 120))
        self.disp.image(self.image, 90)


# ---------------------------------------------------------------------------
# Sensors: proximity (required), Qwiic button (optional)
# ---------------------------------------------------------------------------
class Sensors:
    def __init__(self):
        import board
        import busio
        import adafruit_apds9960.apds9960
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.prox = adafruit_apds9960.apds9960.APDS9960(self.i2c)
        self.prox.enable_proximity = True
        self.button = None
        try:
            from i2c_button import I2C_Button
            self.button = I2C_Button(self.i2c)
            self.button.clear()
            self.button.led_bright = 0
            print("Qwiic button found; a press counts as 'yes'.")
        except Exception:
            print("No Qwiic button found (that's fine).")

    def proximity(self):
        return self.prox.proximity

    def button_clicked(self):
        if not self.button:
            return False
        try:
            clicked = bool(self.button.status & 0x2)
            if clicked:
                self.button.clear()
            return clicked
        except Exception:
            return False

    def button_led(self, on):
        if self.button:
            try:
                self.button.led_bright = 255 if on else 0
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dialogue actions (what the wizard can trigger)
# ---------------------------------------------------------------------------
speaker = None  # set in main


def best_resident(text):
    words = [w.strip(".,!?").capitalize() for w in text.split()]
    for w in words:
        if w in RESIDENTS:
            return w
    for w in words:
        m = get_close_matches(w, RESIDENTS, n=1, cutoff=0.6)
        if m:
            return m[0]
    return None


def think_then(fn):
    """Show 'thinking' for THINK_PAUSE, then run fn."""
    S.set(THINKING)
    threading.Timer(THINK_PAUSE, fn).start()


YES_WORDS = {"yes", "yeah", "yep", "yup", "correct", "right", "sure", "ya", "yes."}
NO_WORDS = {"no", "nope", "nah", "wrong", "not"}


def heard_yes_no(text):
    words = {w.strip(".,!?").lower() for w in text.split()}
    if words & YES_WORDS:
        return "yes"
    if words & NO_WORDS:
        return "no"
    return None


def act_greet():
    S.set(THINKING, name=None, pending_name=None, armed=False, fails=0)
    speaker.say("Who are you here for?", then=lambda: S.set(LISTENING))


def act_try_later():
    S.set(THINKING, pending_name=None)
    speaker.say("Sorry, I couldn't understand. Please try again later.",
                then=lambda: S.set(IDLE, name=None, pending_name=None))


def act_yes_or_no():
    name = S.pending_name
    def go():
        speaker.say(f"Please say yes or no. Did you say {name}?", then=lambda: S.set(LISTENING))
    think_then(go)


def act_confirm(name=None):
    name = name or best_resident(S.last_transcript) or S.last_transcript.strip(".?! ").title()
    if not name:
        return act_again()
    def go():
        S.set(THINKING, pending_name=name)
        speaker.say(f"Did you say {name}?", then=lambda: S.set(LISTENING))
    think_then(go)


def act_again():
    def go():
        speaker.say("Sorry, who are you here for?", then=lambda: S.set(LISTENING))
    think_then(go)


def act_list():
    names = ", ".join(RESIDENTS[:-1]) + f", and {RESIDENTS[-1]}"
    def go():
        speaker.say(f"Sorry. The people here are {names}. Which one?", then=lambda: S.set(LISTENING))
    think_then(go)


def act_wait():
    name = S.pending_name or S.name or "them"
    def go():
        S.set(THINKING, name=name, pending_name=None)
        speaker.say("One moment please.")
        speaker.say(f"Someone is here for {name}.", then=lambda: S.set(WAITING))
    think_then(go)


def act_still():
    name = S.name or "them"
    speaker.say(f"Still waiting for {name}.")
    with S.lock:
        S.last_progress = time.time()


def act_heard_you():
    name = S.name or "them"
    speaker.say(f"I heard you. Still waiting for {name}.")
    with S.lock:
        S.last_progress = time.time()


def act_offer():
    name = S.name or "They"
    S.set(THINKING)
    speaker.say(f"{name} is unavailable now. Want to leave a message?", then=lambda: S.set(OFFER))


def act_record():
    S.set(THINKING)
    speaker.say("Go ahead, I'm recording.", then=lambda: S.set(RECORDING))


def act_open():
    S.set(DONE)
    speaker.say("Here they come.")


def act_bye():
    S.set(THINKING)
    speaker.say("Okay, goodbye.", then=lambda: S.set(IDLE))


def act_say(text):
    if text.strip():
        speaker.say(text.strip())


def act_reset():
    S.set(IDLE, name=None, pending_name=None)


ACTIONS = {
    "greet": act_greet, "confirm": act_confirm, "again": act_again, "list": act_list,
    "wait": act_wait, "still": act_still, "heard_you": act_heard_you, "offer": act_offer,
    "record": act_record, "open": act_open, "bye": act_bye, "reset": act_reset,
    "try_later": act_try_later,
}


def fail_or(retry):
    """Count a turn the device couldn't use; retry once, then give up."""
    with S.lock:
        S.fails += 1
        fails = S.fails
    if fails >= 2:
        act_try_later()
    else:
        retry()


def auto_respond(text):
    """The predictable part of the dialogue, run by the device itself."""
    st = S.state
    if st == LISTENING and S.pending_name is None:
        # We asked "Who are you here for?"
        name = best_resident(text)
        if name:
            with S.lock:
                S.fails = 0
            act_confirm(name)
        else:
            fail_or(act_again)
    elif st == LISTENING and S.pending_name:
        # We asked "Did you say X?"
        answer = heard_yes_no(text)
        if answer == "yes":
            with S.lock:
                S.fails = 0
            act_wait()
        elif answer == "no":
            # maybe they said the right name in the same breath: "No, Pam."
            name = best_resident(text)
            if name and name != S.pending_name:
                act_confirm(name)
            else:
                S.pending_name = None
                fail_or(act_again)
        else:
            fail_or(act_yes_or_no)
    elif st == WAITING:
        if time.time() - S.last_progress > 2.0:
            act_heard_you()
    elif st == OFFER:
        if heard_yes_no(text) == "yes":
            act_record()
        else:
            act_bye()


def on_utterance(text):
    S.log_heard(text)
    print(f"heard: {text}")
    if S.auto_dialogue and not S.speaking:
        auto_respond(text)


def on_message(samples, text):
    MESSAGES_DIR.mkdir(exist_ok=True)
    path = MESSAGES_DIR / f"message_{datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(path, samples, SAMPLE_RATE)
    S.log_heard(f"[message] {text}")
    with S.lock:
        S.messages.append({"file": path.name, "text": text})
    S.set(THINKING)
    speaker.say("Got it. I'll pass that on.", then=lambda: S.set(IDLE))


# ---------------------------------------------------------------------------
# Wizard controller (Flask)
# ---------------------------------------------------------------------------
app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Greeter wizard</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;padding:12px;background:#111;color:#eee}
h1{font-size:18px;margin:0 0 8px}
.state{padding:8px 12px;border-radius:8px;background:#222;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center}
.led{width:14px;height:14px;border-radius:50%;display:inline-block;margin-right:8px;background:#555}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
button{font-size:15px;padding:12px 8px;border:0;border-radius:8px;background:#2a5;color:#fff}
button.alt{background:#36c}button.warn{background:#c53}button.dim{background:#444}
.log{background:#1a1a1a;border-radius:8px;padding:8px;font-size:14px;max-height:160px;overflow:auto;margin-bottom:10px}
.log div{padding:2px 0;border-bottom:1px solid #2a2a2a}.t{color:#888;font-size:12px;margin-right:6px}
.heard{color:#fd8}.said{color:#9cf}
input[type=text]{width:100%;box-sizing:border-box;font-size:15px;padding:10px;border-radius:8px;border:1px solid #444;background:#222;color:#eee}
select{font-size:15px;padding:8px;border-radius:8px;background:#222;color:#eee;border:1px solid #444}
label{font-size:13px;color:#aaa}
</style></head><body>
<h1>Door Greeter wizard</h1>
<div class="state"><span><span class="led" id="led"></span><b id="st">...</b> <span id="nm"></span></span><span id="rem"></span></div>
<div class="log" id="heard"></div>
<div class="grid">
<button onclick="act('greet')">Greet</button>
<button onclick="act('confirm', {name: document.getElementById('res').value})">Confirm name &#8594;</button>
<div style="grid-column:1/3"><label>Resident to confirm (blank = guess from transcript)</label><br>
<select id="res"><option value="">guess</option></select></div>
<button class="alt" onclick="act('again')">Ask again</button>
<button class="alt" onclick="act('list')">List residents</button>
<button onclick="act('wait')">One moment (start wait)</button>
<button class="alt" onclick="act('still')">Still waiting</button>
<button class="alt" onclick="act('heard_you')">I heard you</button>
<button class="warn" onclick="act('offer')">Offer message</button>
<button class="warn" onclick="act('record')">Record message</button>
<button onclick="act('open')">Door opened</button>
<button class="dim" onclick="act('bye')">Goodbye</button>
<button class="dim" onclick="act('try_later')">Try again later</button>
<button class="dim" style="grid-column:1/3" onclick="act('reset')">Reset</button>
</div>
<input type="text" id="free" placeholder="Say anything..." onkeydown="if(event.key==='Enter'){act('say',{text:this.value});this.value=''}">
<p><label><input type="checkbox" id="ad" onchange="act('auto_dialogue',{on:this.checked})"> <b>auto dialogue</b> (device answers by itself)</label><br>
<label><input type="checkbox" id="ag" onchange="act('auto_greet',{on:this.checked})"> auto-greet on proximity</label>
&nbsp; <label><input type="checkbox" id="at" onchange="act('auto_timers',{on:this.checked})"> auto progress / 30 s fallback</label>
&nbsp; <span id="prox" style="color:#888"></span></p>
<div class="log" id="said"></div>
<div id="msgs" style="font-size:13px;color:#aaa"></div>
<script>
async function act(a, extra){await fetch('/act',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({action:a},extra||{}))});poll()}
const colors={idle:'#555',listening:'#fc0',thinking:'#fc0',waiting:'#4af',offer:'#fc0',recording:'#f44',done:'#4d6'};
let filled=false;
async function poll(){
 const s=await (await fetch('/state')).json();
 document.getElementById('st').textContent=s.state.toUpperCase()+(s.speaking?' (speaking)':'');
 document.getElementById('led').style.background=colors[s.state];
 document.getElementById('nm').textContent=s.pending_name?('asked: '+s.pending_name+'?'):(s.name?('for '+s.name):'');
 document.getElementById('rem').textContent=s.remaining!==null?(s.remaining+' s'):'';
 document.getElementById('heard').innerHTML=s.heard.map(h=>'<div class="heard"><span class="t">'+h[0]+'</span>'+h[1]+'</div>').join('')||'<div style="color:#666">nothing heard yet</div>';
 document.getElementById('said').innerHTML=s.said.map(h=>'<div class="said"><span class="t">'+h[0]+'</span>'+h[1]+'</div>').join('');
 document.getElementById('prox').textContent='prox '+s.prox+(s.armed?' (armed)':'');
 document.getElementById('ag').checked=s.auto_greet; document.getElementById('at').checked=s.auto_timers; document.getElementById('ad').checked=s.auto_dialogue;
 document.getElementById('msgs').innerHTML=s.messages.map(m=>'&#9679; '+m.file+': '+m.text).join('<br>');
 if(!filled){const sel=document.getElementById('res');s.residents.forEach(r=>{const o=document.createElement('option');o.value=r;o.textContent=r;sel.appendChild(o)});filled=true}
 const h=document.getElementById('heard');h.scrollTop=h.scrollHeight;
}
setInterval(poll,1000);poll();
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/state")
def state():
    return jsonify(S.snapshot())


@app.route("/act", methods=["POST"])
def act():
    data = request.get_json(force=True) or {}
    a = data.get("action")
    if a == "say":
        act_say(data.get("text", ""))
    elif a == "confirm":
        act_confirm(data.get("name") or None)
    elif a == "auto_greet":
        with S.lock:
            S.auto_greet = bool(data.get("on"))
    elif a == "auto_timers":
        with S.lock:
            S.auto_timers = bool(data.get("on"))
    elif a == "auto_dialogue":
        with S.lock:
            S.auto_dialogue = bool(data.get("on"))
    elif a in ACTIONS:
        ACTIONS[a]()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Main loop: screen, proximity, timers
# ---------------------------------------------------------------------------
def main():
    global speaker
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="base.en")
    p.add_argument("--prox", type=int, default=20, help="proximity threshold 0-255")
    p.add_argument("--no-auto-greet", action="store_true")
    p.add_argument("--no-display", action="store_true", help="run without the MiniPiTFT")
    p.add_argument("--no-sensors", action="store_true", help="run without proximity/buttons")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    for path, what in [(VAD_MODEL, "VAD model"), (VOICE, "Piper voice")]:
        if not path.is_file():
            sys.exit(f"{what} not found at {path}. Run speech-scripts/setup.sh first.")
    S.auto_greet = not args.no_auto_greet

    print("Loading models...", flush=True)
    speaker = Speaker(VOICE)
    speaker.start()
    Listener(args.model, on_utterance, on_message).start()
    display = None if args.no_display else Display()
    sensors = None if args.no_sensors else Sensors()

    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=args.port, threaded=True,
                                            use_reloader=False, debug=False), daemon=True).start()
    print(f"Wizard page: http://0.0.0.0:{args.port}/  (use the Pi's hostname or IP)")
    speaker.say("Door greeter ready.")

    while True:
        now = time.time()
        snap = S.snapshot()
        st = snap["state"]

        if sensors:
            prox = sensors.proximity()
            with S.lock:
                S.prox = prox
            if st == IDLE:
                if prox < args.prox:
                    if S.clear_since is None:
                        S.clear_since = now
                    elif now - S.clear_since > 2.0 and not S.armed:
                        with S.lock:
                            S.armed = True
                elif S.armed and S.auto_greet and not snap["speaking"]:
                    S.clear_since = None
                    act_greet()
                if prox >= args.prox:
                    S.clear_since = None
            if sensors.button_clicked():
                on_utterance("yes (button)")
            sensors.button_led(st in (LISTENING, OFFER))

        if st == WAITING and snap["auto_timers"] and not snap["speaking"]:
            elapsed = now - S.state_since
            if elapsed >= WAIT_TIMEOUT:
                act_offer()
            elif now - S.last_progress >= PROGRESS_EVERY:
                act_still()
        elif st == OFFER and snap["auto_timers"] and now - S.state_since >= OFFER_TIMEOUT and not snap["speaking"]:
            act_bye()
        elif st == DONE and now - S.state_since >= DONE_HOLD:
            S.set(IDLE, name=None, pending_name=None)

        if display:
            display.render(snap, now)
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
