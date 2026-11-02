#!/usr/bin/env python3
"""Water by the bed: the night shift.

The camera sits on the nightstand looking at the water bottle. Every few seconds
YOLO looks for drinkware. The tracker locks onto ONE bottle by position and
ignores everything else in the frame (a cup next to it, a lamp base). A change
has to hold for several looks in a row before it is logged, so detection
flicker at the confidence line doesn't turn into events. Readings:

    present     the bottle is where it was
    moved       its position changed and stayed changed
    gone        it left the frame and stayed gone (picked up)
    back        it returned
    dark        too dark to see anything; absence is NOT counted as a drink

The hand-off is silent: when the room goes dark for half a minute, the night
has started. The device acknowledges by turning its green LED off rather than
on. Everything before that is logged too, but the morning report counts from
night_start.

    python water_night.py                    # one look every 5 s until Ctrl-C
    python water_night.py --interval 10
    python water_night.py --dark 30          # brightness (0-255) below which the room counts as dark
    python water_night.py --move 50          # pixels the bottle has to shift to count as moved
    python water_night.py --confirm 3        # looks in a row a change must hold before it's logged (gone: twice that)
    python water_night.py --radius 120       # how far from its last spot we still believe it's the same bottle

Writes night_log.jsonl (one JSON object per line) and a snapshot in snaps/ on
every event. On a Mac, run it from a Terminal window (camera permission) with
caffeinate -i so the laptop doesn't sleep; the lid has to stay open.
"""
import argparse
import datetime
import json
import math
import pathlib
import time

BOTTLE, WINE_GLASS, CUP = 39, 40, 41        # COCO class ids
DRINKWARE = {BOTTLE: "bottle", WINE_GLASS: "glass", CUP: "cup"}


class Tracker:
    """Turns per-look detections into settled events.

    Feed it a list of (name, (cx, cy), conf) and the frame brightness. It
    returns a list of events for this look, usually empty."""

    def __init__(self, move=50, radius=120, confirm=3, confirm_gone=None, dark=35):
        self.move, self.radius, self.confirm, self.dark = move, radius, confirm, dark
        self.confirm_gone = confirm_gone or 2 * confirm   # absence needs longer to believe
        self.state = None        # settled: present / gone / dark
        self.pos = None          # settled position of the bottle
        self.name = None
        self.cand = None         # (raw reading, position) seen in a row
        self.cand_n = 0

    def _observe(self, dets, brightness):
        """Raw reading for this look, before debouncing."""
        if self.pos is None:
            # Not locked on yet: take the best bottle, else the best other drinkware.
            bottles = [d for d in dets if d[0] == "bottle"]
            pick = max(bottles or dets, key=lambda d: d[2], default=None)
            if pick:
                self.name = pick[0]
                return "present", pick
            return ("dark" if brightness < self.dark else "gone"), None
        # Locked on: the nearest detection of the SAME class within radius. A cup
        # sitting next to the bottle is not the bottle, however close it is.
        near = [d for d in dets if d[0] == self.name and math.dist(d[1], self.pos) <= self.radius]
        if near:
            return "present", min(near, key=lambda d: math.dist(d[1], self.pos))
        # Nothing near it. Something elsewhere isn't our bottle, unless it's a
        # bottle and ours is missing: then it probably got put down somewhere new.
        far_bottles = [d for d in dets if d[0] == "bottle"]
        if far_bottles:
            return "present", max(far_bottles, key=lambda d: d[2])
        if brightness < self.dark:
            return "dark", None
        return "gone", None

    def step(self, dets, brightness):
        raw, det = self._observe(dets, brightness)
        pos = det[1] if det else None
        # Does this look agree with the settled state? Then nothing is pending.
        if raw == self.state and (raw != "present" or math.dist(pos, self.pos) <= self.move):
            self.cand, self.cand_n = None, 0
            if raw == "present":
                self.pos = pos           # follow small drift
            return []
        # Otherwise it's a candidate change; count how many looks in a row agree.
        same = (self.cand is not None and self.cand[0] == raw and
                (raw != "present" or math.dist(pos, self.cand[1]) <= self.move))
        if same:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = (raw, pos), 1
        needed = self.confirm_gone if raw in ("gone", "dark") else self.confirm
        if self.cand_n < needed:
            return []
        # Confirmed. Emit the event and settle.
        events = []
        if raw == "present":
            if self.state is None:
                events.append("present")
            elif self.state == "present":
                events.append("moved")
            else:                        # was gone or dark
                events.append("back")
            self.pos, self.name = pos, det[0]
        else:
            events.append(raw)           # gone or dark
        self.state = raw
        self.cand, self.cand_n = None, 0
        return events


def main():
    import cv2
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2, help="seconds between looks")
    parser.add_argument("--dark", type=int, default=35, help="mean brightness below which it's dark")
    parser.add_argument("--move", type=int, default=50, help="pixels of movement that count as a reach")
    parser.add_argument("--radius", type=int, default=120, help="pixels within which it's still our bottle")
    parser.add_argument("--confirm", type=int, default=3, help="looks in a row before a change is logged")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="YOLO confidence threshold; low is fine, position filtering does the rest")
    parser.add_argument("--night-after", type=int, default=30, help="seconds of dark before the night starts")
    parser.add_argument("--log", default="night_log.jsonl")
    args = parser.parse_args()

    model = YOLO("yolo26n_ncnn_model")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("no webcam found: is it plugged in, and does this terminal have camera permission?")
    pathlib.Path("snaps").mkdir(exist_ok=True)
    log = open(args.log, "a")

    def write(entry):
        log.write(json.dumps(entry) + "\n")
        log.flush()

    # The acknowledgement: the Pi's green ACT LED goes OFF when the night starts.
    # (No-op on a Mac.) Needs: sudo chmod a+w /sys/class/leds/ACT/{trigger,brightness}
    def led(on):
        try:
            with open("/sys/class/leds/ACT/trigger", "w") as f:
                f.write("none")
            with open("/sys/class/leds/ACT/brightness", "w") as f:
                f.write("1" if on else "0")
        except Exception:
            pass

    tracker = Tracker(move=args.move, radius=args.radius, confirm=args.confirm, dark=args.dark)
    print(f"looking every {args.interval:g} s; a change must hold {args.confirm} looks, an absence {2 * args.confirm}", flush=True)
    last_beat, dark_since, night_started = 0.0, None, False
    led(True)
    write({"t": datetime.datetime.now().isoformat(timespec="seconds"), "started": True,
           "interval": args.interval, "dark": args.dark, "move": args.move, "confirm": args.confirm})
    print("water watch started, Ctrl-C to stop", flush=True)

    while True:
        try:
            for _ in range(4):                  # skip the webcam's buffered stale frames
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("no frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            r = model.predict(frame, imgsz=320, conf=args.conf, classes=list(DRINKWARE), verbose=False)[0]
            dets = []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append((DRINKWARE[int(box.cls)], ((x1 + x2) / 2, (y1 + y2) / 2), float(box.conf)))
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

            for event in tracker.step(dets, brightness):
                entry = {"t": ts, "reading": event, "light": round(brightness, 1),
                         "seen": len(dets)}
                if tracker.pos and event in ("present", "moved", "back"):
                    entry.update({"what": tracker.name, "x": round(tracker.pos[0]), "y": round(tracker.pos[1])})
                write(entry)
                where = f" ({tracker.name} at {round(tracker.pos[0])},{round(tracker.pos[1])})" \
                    if tracker.pos and event in ("present", "moved", "back") else ""
                print(f"{ts} {event}{where}  light {brightness:.0f}", flush=True)
                if event in ("moved", "gone", "back"):
                    cv2.imwrite(f"snaps/{now:%Y%m%d-%H%M%S}-{event}.jpg", frame)

            if time.time() - last_beat > 600:
                write({"t": ts, "alive": True, "light": round(brightness, 1), "state": tracker.state,
                       "seen": len(dets)})
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


if __name__ == "__main__":
    main()
