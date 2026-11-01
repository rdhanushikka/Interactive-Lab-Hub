#!/usr/bin/env python3
"""Water by the bed: the night shift.

The camera sits on the nightstand looking at the water bottle. Every few seconds
YOLO looks for a bottle or cup. The interpreter turns the boxes into one of a
few readings, and only changes are logged, so a quiet night is a short file:

    present     the bottle is where it was
    moved       its box jumped, so someone reached for it
    gone        it left the frame (picked up)
    back        it returned
    dark        too dark to see anything; absence is NOT counted as a drink

The hand-off is silent: when the room goes dark for half a minute, the night
has started. The device acknowledges by turning its green LED off rather than
on. Everything before that is logged too, but the morning report counts from
night_start.

    python water_night.py                    # one look every 5 s until Ctrl-C
    python water_night.py --interval 10
    python water_night.py --dark 30          # brightness (0-255) below which the room counts as dark
    python water_night.py --move 50          # pixels the box has to jump to count as moved

Writes night_log.jsonl (one JSON object per line) and a snapshot in snaps/ on
every event. Run it in tmux so it survives closing the laptop:

    tmux new -s night
    python water_night.py
"""
import argparse
import datetime
import json
import math
import pathlib
import time

import cv2
from ultralytics import YOLO

BOTTLE, WINE_GLASS, CUP = 39, 40, 41        # COCO class ids
DRINKWARE = {BOTTLE: "bottle", WINE_GLASS: "glass", CUP: "cup"}

parser = argparse.ArgumentParser()
parser.add_argument("--interval", type=float, default=5, help="seconds between looks")
parser.add_argument("--dark", type=int, default=35, help="mean brightness below which it's dark")
parser.add_argument("--move", type=int, default=50, help="pixels of box movement that count as a reach")
parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
parser.add_argument("--night-after", type=int, default=30, help="seconds of dark before the night starts")
parser.add_argument("--log", default="night_log.jsonl")
args = parser.parse_args()

model = YOLO("yolo26n_ncnn_model")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("no webcam found: is it plugged in?")
pathlib.Path("snaps").mkdir(exist_ok=True)
log = open(args.log, "a")


def write(entry):
    log.write(json.dumps(entry) + "\n")
    log.flush()


# The acknowledgement: the Pi's green ACT LED goes OFF when the night starts.
# Needs: sudo chmod a+w /sys/class/leds/ACT/trigger /sys/class/leds/ACT/brightness
def led(on):
    try:
        with open("/sys/class/leds/ACT/trigger", "w") as f:
            f.write("none")
        with open("/sys/class/leds/ACT/brightness", "w") as f:
            f.write("1" if on else "0")
    except Exception:
        pass


def look(frame):
    """One frame -> (reading, name, center, brightness)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    r = model.predict(frame, imgsz=320, conf=args.conf, classes=list(DRINKWARE), verbose=False)[0]
    best = None
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[0]:
            best = (area, DRINKWARE[int(box.cls)], ((x1 + x2) / 2, (y1 + y2) / 2), float(box.conf))
    if best:
        _, name, center, conf = best
        return "present", name, center, brightness, conf
    if brightness < args.dark:
        return "dark", None, None, brightness, 0.0
    return "gone", None, None, brightness, 0.0


state, last_center, last_beat = None, None, 0.0
dark_since, night_started = None, False
led(True)
print("water watch started, Ctrl-C to stop", flush=True)

while True:
    try:
        for _ in range(4):                  # skip the webcam's buffered stale frames
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("no frame")
        reading, name, center, brightness, conf = look(frame)
        now = datetime.datetime.now()
        ts = now.isoformat(timespec="seconds")

        # The silent hand-off: dark for long enough means the night has begun.
        if brightness < args.dark:
            dark_since = dark_since or time.time()
            if not night_started and time.time() - dark_since >= args.night_after:
                night_started = True
                led(False)
                write({"t": ts, "night_start": True, "light": round(brightness, 1)})
                print(f"{ts} night started (lights off)", flush=True)
        else:
            dark_since = None

        # The interpreter's state machine.
        event = None
        if reading == "present":
            if state == "gone":
                event = "back"
            elif state == "present" and last_center and \
                    math.dist(center, last_center) > args.move:
                event = "moved"
            elif state in (None, "dark"):
                event = "present"
            last_center = center
            state = "present"
        elif reading == "gone":
            if state == "present":
                event = "gone"
            state = "gone"
        else:  # dark: we can't see; don't change what we believe about the bottle
            if state != "dark":
                event = "dark"
            state = "dark"

        if event:
            entry = {"t": ts, "reading": event, "light": round(brightness, 1)}
            if name:
                entry.update({"what": name, "x": round(center[0]), "y": round(center[1]),
                              "conf": round(conf, 2)})
            write(entry)
            print(f"{ts} {event}" + (f" ({name} at {round(center[0])},{round(center[1])})" if name else ""),
                  flush=True)
            if event in ("moved", "gone", "back"):
                cv2.imwrite(f"snaps/{now:%Y%m%d-%H%M%S}-{event}.jpg", frame)

        if time.time() - last_beat > 600:
            write({"t": ts, "alive": True, "light": round(brightness, 1), "state": state})
            last_beat = time.time()
        time.sleep(args.interval)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print("hiccup:", e, flush=True)
        time.sleep(args.interval)

cap.release()
led(True)
write({"t": datetime.datetime.now().isoformat(timespec="seconds"), "stopped": True})
print("water watch stopped", flush=True)
