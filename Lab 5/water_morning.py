#!/usr/bin/env python3
"""Water by the bed: the morning report.

Reads night_log.jsonl from water_night.py, works out when the bottle was
reached for, and says so in one line. The counting is done here, in plain
code, so the number is always right. Qwen is only asked to phrase it.

    python water_morning.py                  # print the report
    python water_morning.py --speak          # also read it aloud with the Lab 3 Piper voice
    python water_morning.py --screen         # also draw the night as a timeline on the MiniPiTFT
    python water_morning.py --qwen           # let Qwen phrase it (facts are passed in; it adds no new ones)
    python water_morning.py --log other.jsonl

A "drink" is a reach: the bottle moved, or it left the frame and came back. Events
closer together than --gap seconds are one reach. Minutes the camera couldn't
see (dark) are reported, not guessed about.
"""
import argparse
import datetime
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log", default="night_log.jsonl")
parser.add_argument("--gap", type=int, default=90, help="seconds within which events are one reach")
parser.add_argument("--speak", action="store_true")
parser.add_argument("--screen", action="store_true")
parser.add_argument("--qwen", action="store_true")
parser.add_argument("--model", default="qwen3.5:0.8b")
args = parser.parse_args()

entries = [json.loads(l) for l in open(args.log) if l.strip()]
if not entries:
    raise SystemExit("empty log")


def t(e):
    return datetime.datetime.fromisoformat(e["t"])


# Count from the moment the lights went off, if they did.
starts = [e for e in entries if e.get("night_start")]
night_start = t(starts[0]) if starts else t(entries[0])
night_end = t(entries[-1])
night = [e for e in entries if t(e) >= night_start]

# Reaches: moved, or gone-then-back. Merge events within --gap seconds.
reaches = []
for e in night:
    if e.get("reading") in ("moved", "gone"):
        when = t(e)
        if reaches and (when - reaches[-1]).total_seconds() < args.gap:
            continue
        reaches.append(when)

# How much of the night was too dark to see?
dark_minutes = 0.0
prev_t, prev_state = None, None
for e in night:
    if "reading" in e:
        if prev_state == "dark" and prev_t:
            dark_minutes += (t(e) - prev_t).total_seconds() / 60
        prev_t, prev_state = t(e), e["reading"]
if prev_state == "dark" and prev_t:
    dark_minutes += (night_end - prev_t).total_seconds() / 60

hours = (night_end - night_start).total_seconds() / 3600
times = [r.strftime("%-I:%M") for r in reaches]

# The honest version, written by code.
if not reaches:
    line = f"The bottle never moved between {night_start:%-I:%M} and {night_end:%-I:%M}."
elif len(reaches) == 1:
    line = f"You reached for the water once, at {times[0]}."
else:
    line = f"You reached for the water {len(reaches)} times, at " + ", ".join(times[:-1]) + f" and {times[-1]}."
if dark_minutes >= 5:
    line += f" The camera couldn't see for {dark_minutes:.0f} minutes of that."
if not starts:
    line += " (The lights never went off, so this counts the whole log.)"

print(f"night {night_start:%H:%M} to {night_end:%H:%M} ({hours:.1f} h), {len(reaches)} reach(es), "
      f"{dark_minutes:.0f} dark min, {sum(1 for e in night if e.get('alive'))} heartbeats")
print(line)

report = line
if args.qwen:
    import requests
    prompt = f"""You are a bedside device. Rewrite the following as one warm sentence a half-awake
person can take in. Do not add any facts. Keep every number and time.

{line}"""
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": args.model, "prompt": prompt, "think": False, "stream": False,
        "options": {"temperature": 0.3, "num_predict": 80}}, timeout=600)
    r.raise_for_status()
    report = r.json()["response"].strip()
    print("\nQwen:", report)

if args.speak:
    try:
        import numpy as np
        import sounddevice as sd
        from piper import PiperVoice
        voice = PiperVoice.load(str(Path.home() / "Interactive-Lab-Hub/Lab 3/voices/en_US-lessac-medium.onnx"))
        for chunk in voice.synthesize(report):
            sd.play(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16), samplerate=chunk.sample_rate)
            sd.wait()
    except Exception as e:
        print("could not speak:", e)

if args.screen:
    # The night as a bar across the MiniPiTFT, a tick for each reach, dark stretches greyed.
    try:
        import board
        import digitalio
        from adafruit_rgb_display import st7789
        from PIL import Image, ImageDraw, ImageFont
        disp = st7789.ST7789(board.SPI(), cs=digitalio.DigitalInOut(board.D5), dc=digitalio.DigitalInOut(board.D25),
                             rst=None, baudrate=64000000, width=135, height=240, x_offset=53, y_offset=40)
        bl = digitalio.DigitalInOut(board.D22); bl.switch_to_output(value=True)
        W, H = 240, 135
        img = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        span = max((night_end - night_start).total_seconds(), 1)
        x0, x1, y = 12, W - 12, 70
        d.text((12, 4), "Last night", font=font, fill=(180, 180, 180))
        d.text((12, 20), f"{len(reaches)} reach" + ("" if len(reaches) == 1 else "es") + " for water",
               font=big, fill=(255, 255, 255))
        d.rectangle((x0, y - 6, x1, y + 6), fill=(50, 90, 140))
        for r in reaches:
            x = x0 + (x1 - x0) * (r - night_start).total_seconds() / span
            d.rectangle((x - 2, y - 12, x + 2, y + 12), fill=(120, 200, 255))
        d.text((x0, y + 14), night_start.strftime("%-I:%M"), font=font, fill=(150, 150, 150))
        d.text((x1 - 36, y + 14), night_end.strftime("%-I:%M"), font=font, fill=(150, 150, 150))
        if dark_minutes >= 5:
            d.text((12, 108), f"{dark_minutes:.0f} min too dark to see", font=font, fill=(200, 160, 80))
        disp.image(img, 90)
    except Exception as e:
        print("could not draw on the screen:", e)
