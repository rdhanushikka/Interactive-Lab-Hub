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
import queue
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
MESSAGE_SILENCE = 2.0   # long threshold for composing a message; people pause mid-thought
THINK_PAUSE = 0.3       # beat before the device replies
PROGRESS_EVERY = 10.0   # "Still waiting" cadence
HEARD_YOU_COOLDOWN = 5.0  # don't repeat "I heard you" more often than this
WAIT_TIMEOUT = 30.0     # then offer to take a message
OFFER_TIMEOUT = 8.0     # how long to wait for an answer to the offer
MAX_MESSAGE = 20.0      # cap on a recorded message, seconds
DONE_HOLD = 5.0         # door stays open this long, then "door closing" and re-arm
SPEECH_GUARD = 0.8      # keep the mic muted this long after playback "finishes";
                        # the USB speaker is still emitting the tail of the audio

# States
IDLE, LISTENING, THINKING, WAITING, OFFER, RECORDING, DONE, CLOSING = (
    "idle", "listening", "thinking", "waiting", "offer", "recording", "done", "closing")

STATE_LABELS = {
    IDLE: "Door Greeter",
    LISTENING: "Listening...",
    THINKING: "Thinking",
    WAITING: "Waiting for",
    OFFER: "Message?",
    RECORDING: "Recording",
    DONE: "Door opening",
    CLOSING: "Door closing",
}


# ---------------------------------------------------------------------------
# Event log: terminal + logs/session_<time>.log
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
_log_file = None
_log_lock = threading.Lock()
_t_start = time.time()


def log(kind, msg=""):
    """One line per event: wall clock, seconds since start, kind, message."""
    global _log_file
    line = f"{datetime.now():%H:%M:%S.%f}"[:-3] + f" +{time.time() - _t_start:7.2f}s  {kind:<9} {msg}"
    with _log_lock:
        print(line, flush=True)
        if _log_file is None:
            LOG_DIR.mkdir(exist_ok=True)
            _log_file = open(LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.log", "a")
        _log_file.write(line + "\n")
        _log_file.flush()


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
        self.last_progress = 0.0    # when the last timed "still waiting" went out
        self.last_ack = 0.0         # when we last acknowledged the visitor during a wait
        self.heard = []             # [(ts, text)]
        self.said = []              # [(ts, text)]
        self.speaking = False
        self.listening_busy = False  # speech detected or being transcribed right now
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
            old = self.state
            self.state = new_state
            self.state_since = time.time()
            self.last_progress = self.state_since
            self.last_ack = 0.0
            for k, v in kw.items():
                setattr(self, k, v)
        extra = ", ".join(f"{k}={v}" for k, v in kw.items() if k in ("name", "pending_name") and v)
        log("state", f"{old} -> {new_state}" + (f"  ({extra})" if extra else ""))

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
                "listening_busy": self.listening_busy,
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
    def __init__(self, voice_path, listener):
        super().__init__(daemon=True)
        self.voice = PiperVoice.load(str(voice_path))
        self.listener = listener  # told when to ignore the mic
        self.queue = []
        self.cv = threading.Condition()

    def say(self, text, then=None):
        """Queue text. `then` is a callable run after the line is spoken."""
        S.log_said(text)
        log("say", text)
        with self.cv:
            self.queue.append(("say", text, then))
            self.cv.notify()

    def play(self, samples, rate, then=None):
        """Queue raw audio (e.g. a recorded message) through the same gated path."""
        log("play", f"{len(samples) / rate:.1f}s of audio")
        with self.cv:
            self.queue.append(("play", (samples, rate), then))
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                kind, payload, then = self.queue.pop(0)
            with S.lock:
                S.speaking = True
            self.listener.mute(True)
            t0 = time.perf_counter()
            try:
                if kind == "say":
                    for chunk in self.voice.synthesize(payload):
                        audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                        sd.play(audio, samplerate=chunk.sample_rate)
                        sd.wait()
                else:
                    samples, rate = payload
                    sd.play(samples, samplerate=rate)
                    sd.wait()
            except Exception as e:  # keep the device alive if audio hiccups
                log("error", f"speak: {e}")
            log("spoke", f"{time.perf_counter() - t0:.2f}s")
            self.listener.mute(False)  # ignore the mic for SPEECH_GUARD more, in the audio clock
            time.sleep(SPEECH_GUARD)   # and don't act on anything until the tail has played
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
    """Two whisper models: the accurate one for name turns, the fast one for
    yes/no turns. Whisper pads every input to 30 s, so a one-word answer costs
    the same as a sentence, and the model size is the only lever on latency."""

    def __init__(self, model_name, fast_model_name, on_utterance, on_message):
        super().__init__(daemon=True)
        self.models = {"accurate": WhisperModel(model_name, device="cpu", compute_type="int8")}
        self.models["fast"] = (self.models["accurate"] if fast_model_name == model_name
                               else WhisperModel(fast_model_name, device="cpu", compute_type="int8"))
        self.names = {"accurate": model_name, "fast": fast_model_name}
        self.vad_turn, self.window = build_vad(TURN_SILENCE)
        self.vad_msg, _ = build_vad(MESSAGE_SILENCE)
        self.on_utterance = on_utterance
        self.on_message = on_message
        self.q = queue.Queue()
        self.stream = None
        self.mute_from = 0.0
        self.mute_until = 0.0
        self.flush_requested = False
        # First transcription after load is slow; do it now on a second of silence.
        for m in set(self.models.values()):
            m.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1)

    def pick(self):
        """Name turns need accuracy; everything else can use the fast model
        if one was given (by default both are base.en)."""
        if S.state == LISTENING and S.pending_name is None:
            return "accurate"
        return "fast"

    def transcribe(self, samples, which=None):
        which = which or self.pick()
        t0 = time.perf_counter()
        with S.lock:
            S.listening_busy = True
        try:
            segments, _ = self.models[which].transcribe(samples, beam_size=1)
            text = " ".join(s.text.strip() for s in segments).strip()
        finally:
            with S.lock:
                S.listening_busy = False
        log("asr", f"{self.names[which]}: {len(samples) / SAMPLE_RATE:.1f}s audio -> {time.perf_counter() - t0:.2f}s")
        return text

    # --- echo gating, in the audio clock -----------------------------------
    # The mic pipeline delivers audio a long and variable time after it was
    # captured, so "ignore the mic while speaking" fails if done in wall-clock
    # time. Instead every chunk carries its capture time from the audio
    # driver, and the speaker marks the window during which its sound could
    # be in the air, in that same clock. Chunks captured inside the window are
    # dropped no matter when they arrive.
    def stream_time(self):
        return self.stream.time if self.stream else 0.0

    def mute(self, on):
        now = self.stream_time()
        if on:
            self.mute_from = now
            self.mute_until = float("inf")
        else:
            self.mute_until = now + SPEECH_GUARD
        log("gate", f"{'closed' if on else 'reopens'} at audio-clock {now:.2f}"
                    + ("" if on else f", chunks stamped before {self.mute_until:.2f} are dropped"))

    def flush_message(self):
        self.flush_requested = True

    def _callback(self, indata, frames, time_info, status):
        t_adc = time_info.inputBufferAdcTime
        if not t_adc or t_adc != t_adc:  # 0 or NaN: driver didn't stamp it
            t_adc = time_info.currentTime - (self.stream.latency or 0.0)
        self.q.put((t_adc, t_adc + frames / SAMPLE_RATE, indata[:, 0].copy()))

    def run(self):
        buf = np.empty(0, dtype=np.float32)
        dropped = False
        self.stream = sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE,
                                     blocksize=int(0.1 * SAMPLE_RATE), callback=self._callback)
        with self.stream:
            try:
                dev = sd.query_devices(self.stream.device)["name"]
            except Exception:
                dev = "?"
            log("mic", f"{dev}, reported input latency {self.stream.latency:.3f}s, "
                       f"audio-clock now {self.stream.time:.2f}")
            drop_t0 = drop_t1 = None
            while True:
                t0, t1, chunk = self.q.get()
                if t1 > self.mute_from and t0 < self.mute_until:
                    # captured while we were (or may still be) talking
                    if not dropped:
                        drop_t0 = t0
                    drop_t1 = t1
                    dropped = True
                    continue
                if dropped:
                    # First clean chunk after a speech window: forget any partial
                    # utterance the detectors were building from our own voice.
                    log("gate", f"dropped audio stamped {drop_t0:.2f}-{drop_t1:.2f} "
                                f"({drop_t1 - drop_t0:.1f}s); listening again from {t0:.2f}")
                    self.vad_turn.reset()
                    self.vad_msg.reset()
                    buf = np.empty(0, dtype=np.float32)
                    dropped = False
                if self.flush_requested:
                    self.flush_requested = False
                    self.vad_msg.flush()
                buf = np.concatenate([buf, chunk])
                while len(buf) > self.window:
                    self.vad_turn.accept_waveform(buf[:self.window])
                    self.vad_msg.accept_waveform(buf[:self.window])
                    buf = buf[self.window:]
                # While someone is mid-sentence, hold off the timers.
                try:
                    in_speech = self.vad_turn.is_speech_detected()
                except AttributeError:
                    in_speech = False
                if in_speech != S.listening_busy:
                    with S.lock:
                        S.listening_busy = in_speech

                recording = S.state == RECORDING
                while not self.vad_turn.empty():
                    utt = np.array(self.vad_turn.front.samples, dtype=np.float32)
                    self.vad_turn.pop()
                    log("vad", f"utterance ended, {len(utt) / SAMPLE_RATE:.1f}s of speech")
                    if not recording:
                        text = self.transcribe(utt)
                        if text:
                            self.on_utterance(text)
                        else:
                            log("asr", "(empty transcript)")
                while not self.vad_msg.empty():
                    utt = np.array(self.vad_msg.front.samples, dtype=np.float32)
                    self.vad_msg.pop()
                    if recording:
                        self.on_message(utt)  # raw audio only; no transcription needed


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
        # The display's two buttons (top = A on GPIO 23, bottom = B on GPIO 24).
        # Either one means "the resident opened the door".
        self.buttons = {}
        for name, pin in (("A", board.D23), ("B", board.D24)):
            b = digitalio.DigitalInOut(pin)
            b.switch_to_input(pull=digitalio.Pull.UP)
            self.buttons[name] = b
        self._was_down = {"A": False, "B": False}
        self.w, self.h = self.disp.height, self.disp.width  # landscape 240x135
        self.image = Image.new("RGB", (self.w, self.h))
        self.draw = ImageDraw.Draw(self.image)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.big = ImageFont.truetype(font, 16)
        self.med = ImageFont.truetype(font, 13)
        self.small = ImageFont.truetype(font, 10)
        self.mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)

    def pressed(self):
        """Names of buttons that went down since the last call (they read LOW when pressed)."""
        hits = []
        for name, b in self.buttons.items():
            down = not b.value
            if down and not self._was_down[name]:
                hits.append(name)
            self._was_down[name] = down
        return hits

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

        # The whole screen is the door. Wall on either side, frame, door slab,
        # a caption above it. The pulsing light is a physical LED, not on screen.
        wall = (28, 30, 38)
        d.rectangle((0, 0, self.w, self.h), fill=wall)
        d.rectangle((0, self.h - 8, self.w, self.h), fill=(45, 40, 36))  # floor
        fx0, fy0, fx1, fy1 = 72, 22, 168, 128
        d.rectangle((fx0 - 5, fy0 - 5, fx1 + 5, fy1), fill=(215, 215, 220))  # frame
        if state == DONE:
            # door swung inward: narrow slab plus a lit room behind it
            d.rectangle((fx0, fy0, fx1, fy1), fill=(255, 240, 200))
            d.polygon([(fx0, fy0), (fx0 + 30, fy0 + 12), (fx0 + 30, fy1 - 8), (fx0, fy1)],
                      fill=(120, 75, 40), outline=(230, 200, 160))
        else:
            d.rectangle((fx0, fy0, fx1, fy1), fill=(120, 75, 40), outline=(230, 200, 160))
            d.rectangle((fx0 + 12, fy0 + 12, fx1 - 12, fy0 + 52), outline=(90, 55, 30), width=2)
            d.rectangle((fx0 + 12, fy0 + 62, fx1 - 12, fy1 - 10), outline=(90, 55, 30), width=2)
            d.ellipse((fx1 - 22, 72, fx1 - 12, 82), fill=(240, 220, 120))
        if state == WAITING and snap["remaining"] is not None:
            txt = str(snap["remaining"])
            tw = d.textlength(txt, font=self.mono)
            d.rectangle((fx0 + 14, 56, fx1 - 14, 92), fill=(60, 40, 25))
            d.text(((fx0 + fx1) / 2 - tw / 2, 58), txt, font=self.mono, fill=(255, 255, 255))
        if state == RECORDING:
            d.ellipse((fx0 + 16, fy0 + 18, fx0 + 26, fy0 + 28), fill=(255, 50, 50))

        # caption above the door
        label = STATE_LABELS[state]
        if state == WAITING and snap["name"]:
            label = f"Waiting for {snap['name']}"
        elif state == LISTENING and snap["pending_name"]:
            label = f"{snap['pending_name']}?"
        elif state == THINKING:
            label = "Thinking" + "." * (1 + int(now * 3) % 3)
        tw = d.textlength(label, font=self.big)
        d.text((self.w / 2 - tw / 2, 2), label, font=self.big, fill=(255, 255, 255))

        # small text either side of the door: what we said, what we heard
        if snap["said"]:
            for i, line in enumerate(self.wrap(snap["said"][-1][1], self.small, 62)[:4]):
                d.text((4, 30 + i * 12), line, font=self.small, fill=(150, 200, 255))
        if snap["heard"]:
            for i, line in enumerate(self.wrap(snap["heard"][-1][1], self.small, 62)[:4]):
                d.text((fx1 + 10, 30 + i * 12), line, font=self.small, fill=(255, 230, 150))
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
        log("sensor", "APDS9960 proximity sensor ready")
        self.button = None
        try:
            from i2c_button import I2C_Button
            self.button = I2C_Button(self.i2c)
            self.button.clear()
            self.button.led_bright = 0
            log("sensor", "Qwiic button found; a press counts as 'yes'")
        except Exception:
            log("sensor", "no Qwiic button (optional)")

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
# Indicator light: a physical LED that shows listening / thinking / waiting.
# Uses the Qwiic button's LED if one is connected (it pulses in hardware);
# otherwise the Pi's own green ACT LED, which can only blink.
# ---------------------------------------------------------------------------
OFF, PULSE, SOLID, BLINK_SLOW, BLINK_FAST = "off", "pulse", "solid", "blink_slow", "blink_fast"
LIGHT_FOR_STATE = {
    IDLE: OFF, LISTENING: PULSE, THINKING: SOLID, WAITING: BLINK_SLOW,
    OFFER: PULSE, RECORDING: BLINK_FAST, DONE: SOLID, CLOSING: OFF,
}


class Indicator:
    def __init__(self, button=None):
        self.button = button
        self.mode = None
        self.act = None
        self.act_on = None
        if not button:
            try:
                with open("/sys/class/leds/ACT/trigger", "w") as f:
                    f.write("none")
                self.act = open("/sys/class/leds/ACT/brightness", "w")
                log("light", "using the Pi's green ACT LED (blink only)")
            except Exception as e:
                log("light", f"no LED available ({e.__class__.__name__}); "
                             "for the ACT LED run: sudo chmod a+w /sys/class/leds/ACT/{trigger,brightness}")
        else:
            log("light", "using the Qwiic button LED")

    def set(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        if self.button:
            try:
                b = self.button
                if mode == OFF:
                    b.led_bright = 0
                elif mode == SOLID:
                    b.led_cycle_ms = 0; b.led_off_ms = 0; b.led_bright = 255
                elif mode == PULSE:
                    b.led_cycle_ms = 1200; b.led_off_ms = 100; b.led_bright = 255
                elif mode == BLINK_SLOW:
                    b.led_cycle_ms = 100; b.led_off_ms = 1500; b.led_bright = 255
                elif mode == BLINK_FAST:
                    b.led_cycle_ms = 100; b.led_off_ms = 250; b.led_bright = 255
            except Exception as e:
                log("error", f"button LED: {e}")

    def tick(self, now):
        """Software blink for the ACT LED; called from the main loop."""
        if not self.act:
            return
        if self.mode == OFF:
            on = False
        elif self.mode == SOLID:
            on = True
        elif self.mode == PULSE:
            on = (now % 0.8) < 0.4
        elif self.mode == BLINK_SLOW:
            on = (now % 2.0) < 0.15
        else:  # BLINK_FAST
            on = (now % 0.4) < 0.2
        if on != self.act_on:
            self.act_on = on
            try:
                self.act.write("1" if on else "0")
                self.act.flush()
            except Exception:
                pass

    def restore(self):
        self.set(OFF)
        if self.act:
            try:
                with open("/sys/class/leds/ACT/trigger", "w") as f:
                    f.write("mmc0")  # back to disk activity
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dialogue actions (what the wizard can trigger)
# ---------------------------------------------------------------------------
speaker = None   # set in main
listener = None  # set in main
light = None     # set in main


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


FILLER = {"i", "im", "i'm", "am", "here", "for", "to", "see", "the", "a", "an", "um", "uh",
          "looking", "visit", "visiting", "meet", "is", "it", "its", "it's", "my", "friend",
          "hi", "hello", "hey", "please", "and", "with", "want", "need", "can", "in", "at"}


def guess_name(text):
    """The word the visitor most likely meant as a name, so we can say
    'there's no Samantha here' instead of pretending we didn't understand."""
    words = [w.strip(".,!?'\"") for w in text.split()]
    for w in words:
        if w and w.lower() not in FILLER and w.isalpha():
            return w.capitalize()
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


def act_again(no_such=None):
    line = (f"There's no {no_such} here. Who are you here for?" if no_such
            else "Sorry, who are you here for?")
    def go():
        speaker.say(line, then=lambda: S.set(LISTENING))
    think_then(go)


def act_no_such(name):
    S.set(THINKING, pending_name=None)
    speaker.say(f"Sorry, there's no {name} here either. Please try again later.",
                then=lambda: S.set(IDLE, name=None, pending_name=None))


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
        S.last_ack = S.last_progress


def act_heard_you():
    name = S.name or "them"
    speaker.say(f"I heard you. Still waiting for {name}.")
    with S.lock:
        S.last_progress = time.time()
        S.last_ack = S.last_progress


def act_offer():
    name = S.name or "They"
    S.set(THINKING)
    speaker.say(f"{name} is unavailable now. Want to leave a message?", then=lambda: S.set(OFFER))


def act_record():
    S.set(THINKING)
    def start():
        S.set(RECORDING)
        log("record", f"recording; ends after {MESSAGE_SILENCE:.1f}s of silence or {MAX_MESSAGE:.0f}s")
    speaker.say("Go ahead, I'm recording.", then=start)


def act_open():
    """The resident opened the door: end the conversation, show it open."""
    S.set(DONE, pending_name=None)
    speaker.say("Door opening.")


def act_close():
    S.set(CLOSING)
    speaker.say("Door closing.",
                then=lambda: S.set(IDLE, name=None, pending_name=None, armed=True))


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


def fail_or(retry, give_up=None):
    """Count a turn the device couldn't use; retry once, then give up."""
    with S.lock:
        S.fails += 1
        fails = S.fails
    if fails >= 2:
        log("decide", f"couldn't use the turn, fail #{fails}: give up")
        (give_up or act_try_later)()
    else:
        log("decide", f"couldn't use the turn, fail #{fails}: retry")
        retry()


def auto_respond(text):
    """The predictable part of the dialogue, run by the device itself."""
    st = S.state
    if st == LISTENING and S.pending_name is None:
        # We asked "Who are you here for?"
        name = best_resident(text)
        if name:
            log("decide", f"matched resident '{name}' in '{text}'")
            with S.lock:
                S.fails = 0
            act_confirm(name)
        else:
            heard_name = guess_name(text)
            log("decide", f"no resident matched in '{text}'"
                          + (f" (they seem to want '{heard_name}')" if heard_name else ""))
            if heard_name:
                fail_or(lambda: act_again(no_such=heard_name), lambda: act_no_such(heard_name))
            else:
                fail_or(act_again)
    elif st == LISTENING and S.pending_name:
        # We asked "Did you say X?"
        answer = heard_yes_no(text)
        if answer == "yes":
            log("decide", f"confirmed {S.pending_name}")
            with S.lock:
                S.fails = 0
            act_wait()
        elif answer == "no":
            # maybe they said the right name in the same breath: "No, Pam."
            name = best_resident(text)
            if name and name != S.pending_name:
                log("decide", f"rejected {S.pending_name}, heard '{name}' instead")
                act_confirm(name)
            else:
                log("decide", f"rejected {S.pending_name}, no new name")
                S.pending_name = None
                fail_or(act_again)
        else:
            log("decide", f"not a yes/no: '{text}'")
            fail_or(act_yes_or_no)
    elif st == WAITING:
        if time.time() - S.last_ack > HEARD_YOU_COOLDOWN:
            log("decide", "visitor spoke during the wait")
            act_heard_you()
        else:
            log("decide", "visitor spoke during the wait (acknowledged recently, ignoring)")
    elif st == OFFER:
        if heard_yes_no(text) == "yes":
            log("decide", "accepted the message offer")
            act_record()
        else:
            log("decide", f"did not accept the offer: '{text}'")
            act_bye()
    else:
        log("decide", f"ignored (state {st})")


def on_utterance(text):
    S.log_heard(text)
    log("heard", text)
    if S.auto_dialogue and not S.speaking:
        auto_respond(text)


def on_message(samples):
    MESSAGES_DIR.mkdir(exist_ok=True)
    path = MESSAGES_DIR / f"message_{datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(path, samples, SAMPLE_RATE)
    secs = len(samples) / SAMPLE_RATE
    S.log_heard(f"[message recorded, {secs:.1f}s]")
    log("message", f"saved {path.name} ({secs:.1f}s)")
    with S.lock:
        S.messages.append({"file": path.name, "text": f"{secs:.1f}s"})
    S.set(THINKING)
    speaker.say("Got it. Here's your message.")
    speaker.play(samples, SAMPLE_RATE)
    speaker.say("I'll pass that on.", then=lambda: S.set(IDLE, name=None, pending_name=None))


# ---------------------------------------------------------------------------
# Wizard controller (Flask)
# ---------------------------------------------------------------------------
app = Flask(__name__)
import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # the page polls every second; keep it out of the log

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
<button onclick="act('open')">Door opening</button>
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
const colors={idle:'#555',listening:'#fc0',thinking:'#fc0',waiting:'#4af',offer:'#fc0',recording:'#f44',done:'#4d6',closing:'#777'};
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
    log("wizard", " ".join(f"{k}={v}" for k, v in data.items()))
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
    global speaker, listener, light
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="base.en", help="whisper model for name turns")
    p.add_argument("--fast-model", default="base.en",
                   help="whisper model for yes/no turns (tiny.en is faster but mishears)")
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

    log("start", f"model={args.model} fast_model={args.fast_model} prox_threshold={args.prox} auto_greet={S.auto_greet}")
    log("start", "loading models...")
    sd.default.latency = "low"  # shrink the gap between "playback done" and sound actually ending
    listener = Listener(args.model, args.fast_model, on_utterance, on_message)
    speaker = Speaker(VOICE, listener)
    listener.start()
    speaker.start()
    display = None if args.no_display else Display()
    sensors = None if args.no_sensors else Sensors()
    light = Indicator(sensors.button if sensors else None)

    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=args.port, threaded=True,
                                            use_reloader=False, debug=False), daemon=True).start()
    log("start", f"wizard page on port {args.port}; log file in {LOG_DIR}")
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
                        log("prox", f"clear for 2 s (reading {prox}), trigger re-armed")
                elif S.armed and S.auto_greet and not snap["speaking"]:
                    S.clear_since = None
                    log("prox", f"visitor detected (reading {prox} >= {args.prox})")
                    act_greet()
                if prox >= args.prox:
                    S.clear_since = None
            if sensors.button_clicked():
                log("button", "pressed")
                on_utterance("yes (button)")

        light.set(LIGHT_FOR_STATE[st])
        light.tick(now)

        # Timers never fire while the device is talking or the visitor is
        # mid-sentence / being transcribed; otherwise a late "yes" gets cut off.
        quiet = not snap["speaking"] and not snap["listening_busy"]
        if st == WAITING and snap["auto_timers"] and quiet:
            elapsed = now - S.state_since
            if elapsed >= WAIT_TIMEOUT:
                log("timer", f"{WAIT_TIMEOUT:.0f} s wait is up, offering a message")
                act_offer()
            elif now - S.last_progress >= PROGRESS_EVERY:
                log("timer", f"{elapsed:.0f} s into the wait, progress line")
                act_still()
        elif st == OFFER and snap["auto_timers"] and quiet and now - S.state_since >= OFFER_TIMEOUT:
            log("timer", f"no answer to the offer in {OFFER_TIMEOUT:.0f} s")
            act_bye()
        elif st == RECORDING and now - S.state_since >= MAX_MESSAGE:
            log("timer", f"message hit the {MAX_MESSAGE:.0f} s cap, saving what we have")
            listener.flush_message()
            S.set(THINKING)  # so this fires once; on_message moves us on from here
        elif st == DONE and now - S.state_since >= DONE_HOLD and not snap["speaking"]:
            log("timer", f"door open for {DONE_HOLD:.0f} s, closing; sensor armed again after")
            act_close()

        if display and st not in (IDLE, DONE, CLOSING):
            hit = display.pressed()
            if hit:
                log("button", f"display button {hit[0]} pressed: door opened")
                act_open()
        elif display:
            display.pressed()  # keep edge tracking current, ignore presses while idle

        if display:
            display.render(snap, now)
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            light.restore()
        except Exception:
            pass
