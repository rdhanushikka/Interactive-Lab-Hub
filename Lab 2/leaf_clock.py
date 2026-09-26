# Lab 2 Part 2 - The Fall Tree.
# A tree that measures the season. It starts full on the first day of fall and
# loses one leaf a day, so the leaves on the ground are the days already spent
# and the leaves still on the tree are the days of fall that remain. On the
# last day of fall the tree is bare.
#
# Button A (top):    reset the tree to day 1 of fall
# Button B (bottom): advance one day
#
# Left alone, the tree follows the real calendar and drops a leaf at midnight.
#
# Usage:
#   python leaf_clock.py
import time
import math
import random
import datetime
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7789 as st7789

# The season. Astronomical fall 2026: equinox to solstice, 91 days inclusive.
FALL_START = datetime.date(2026, 9, 22)
FALL_END = datetime.date(2026, 12, 21)
TOTAL_DAYS = (FALL_END - FALL_START).days + 1   # 91
N_LEAVES = TOTAL_DAYS - 1                       # 90: one leaf falls per day, bare on the last day

cs_pin = digitalio.DigitalInOut(board.D5)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = None
BAUDRATE = 64000000
spi = board.SPI()

disp = st7789.ST7789(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=135,
    height=240,
    x_offset=53,
    y_offset=40,
)

height = disp.width   # swap to rotate into landscape
width = disp.height
image = Image.new("RGB", (width, height))
rotation = 90
draw = ImageDraw.Draw(image)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

backlight = digitalio.DigitalInOut(board.D22)
backlight.switch_to_output()
backlight.value = True

buttonA = digitalio.DigitalInOut(board.D23)   # top button: reset to day 1
buttonB = digitalio.DigitalInOut(board.D24)   # bottom button: advance a day
buttonA.switch_to_input(pull=digitalio.Pull.UP)
buttonB.switch_to_input(pull=digitalio.Pull.UP)

# Scene geometry (landscape: 240 wide, 135 tall)
SKY = (18, 24, 44)
GROUND_Y = 118
GROUND = (72, 52, 30)
TRUNK = (96, 62, 34)
TEXT = (235, 235, 235)
HINT = (140, 160, 190)
PALETTE = [(226, 88, 34), (240, 140, 30), (250, 200, 50), (196, 48, 40), (255, 110, 70)]

TREE_X = 138          # trunk centre
CANOPY_CX, CANOPY_CY = TREE_X, 60
CANOPY_RX, CANOPY_RY = 62, 40

# The tree is laid out once from a fixed seed, so every reboot shows the same tree
# and the same leaves fall in the same order.
rng = random.Random(2026)
leaves = []
while len(leaves) < N_LEAVES:
    x = rng.uniform(CANOPY_CX - CANOPY_RX, CANOPY_CX + CANOPY_RX)
    y = rng.uniform(CANOPY_CY - CANOPY_RY, CANOPY_CY + CANOPY_RY)
    if ((x - CANOPY_CX) / CANOPY_RX) ** 2 + ((y - CANOPY_CY) / CANOPY_RY) ** 2 <= 1.0:
        leaves.append({
            "x": x, "y": y,
            "color": rng.choice(PALETTE),
            "phase": rng.uniform(0, 2 * math.pi),
            # where this leaf ends up on the ground, piling toward the trunk
            "gx": TREE_X + rng.gauss(0, 34),
            "gy": GROUND_Y - rng.uniform(0, 5),
        })
fall_order = list(range(N_LEAVES))
rng.shuffle(fall_order)
for rank, idx in enumerate(fall_order):
    leaves[idx]["rank"] = rank        # rank r falls on day r + 2


def draw_leaf(x, y, color):
    draw.ellipse((x - 3, y - 2, x + 3, y + 2), fill=color)


def real_day():
    """Day of fall according to the calendar, clamped to the season."""
    d = (datetime.date.today() - FALL_START).days + 1
    return max(1, min(TOTAL_DAYS, d))


# The day on show is the real day plus whatever the buttons have added. Button A
# sets the offset so that day 1 shows; button B adds one. Midnight still moves
# the real day, so the tree keeps time on its own.
offset = 0
shown_day = real_day()
falling = []          # leaves mid-air: [leaf, start_time]
strays = []           # decorative loose leaves that do not count
last_stray = time.time()
prevA = prevB = True  # buttons read HIGH when released

while True:
    now = time.time()
    a, b = buttonA.value, buttonB.value
    if prevA and not a:                     # A pressed: back to day 1
        offset = 1 - real_day()
        falling = []
    if prevB and not b:                     # B pressed: one day on
        offset += 1
    prevA, prevB = a, b

    new_day = max(1, min(TOTAL_DAYS, real_day() + offset))
    if new_day > shown_day:
        # every leaf that falls between the old day and the new one drifts down
        for leaf in leaves:
            if shown_day - 2 < leaf["rank"] <= new_day - 2:
                falling.append([leaf, now + 0.15 * len(falling)])
    shown_day = new_day
    fallen = shown_day - 1                  # leaves on the ground
    shown_date = FALL_START + datetime.timedelta(days=shown_day - 1)

    # an occasional stray leaf keeps the tree alive between days
    if now - last_stray > 7 and shown_day < TOTAL_DAYS:
        src = random.choice(leaves)
        strays.append([src["x"], src["y"], now, random.choice(PALETTE)])
        last_stray = now

    draw.rectangle((0, 0, width, height), fill=SKY)
    draw.rectangle((0, GROUND_Y, width, height), fill=GROUND)

    # trunk and branches
    draw.polygon([(TREE_X - 6, GROUND_Y), (TREE_X + 6, GROUND_Y),
                  (TREE_X + 3, CANOPY_CY + 10), (TREE_X - 3, CANOPY_CY + 10)], fill=TRUNK)
    for dx, dy in ((-30, -16), (30, -20), (-16, -34), (18, -32)):
        draw.line([(TREE_X, CANOPY_CY + 8), (TREE_X + dx, CANOPY_CY + dy)], fill=TRUNK, width=2)

    # leaves on the ground
    for leaf in leaves:
        if leaf["rank"] < fallen and not any(f[0] is leaf for f in falling):
            draw_leaf(leaf["gx"], leaf["gy"], leaf["color"])

    # leaves still on the tree, swaying a little in the wind
    for leaf in leaves:
        if leaf["rank"] >= fallen:
            sway = math.sin(now * 1.6 + leaf["phase"])
            draw_leaf(leaf["x"] + sway, leaf["y"] + 0.4 * sway, leaf["color"])

    # leaves on their way down, about two seconds each
    still = []
    for leaf, t0 in falling:
        p = (now - t0) / 2.0
        if p < 0:
            sway = math.sin(now * 1.6 + leaf["phase"])
            draw_leaf(leaf["x"] + sway, leaf["y"], leaf["color"])
            still.append([leaf, t0])
        elif p < 1:
            x = leaf["x"] + (leaf["gx"] - leaf["x"]) * p + 6 * math.sin(p * 9 + leaf["phase"])
            y = leaf["y"] + (leaf["gy"] - leaf["y"]) * (p * p)
            draw_leaf(x, y, leaf["color"])
            still.append([leaf, t0])
        else:
            draw_leaf(leaf["gx"], leaf["gy"], leaf["color"])
    falling = still

    kept = []
    for sx, sy, t0, color in strays:
        p = (now - t0) / 3.0
        if p < 1:
            draw_leaf(sx + 20 * math.sin(p * 5) - 10 * p, sy + (GROUND_Y - sy) * p, color)
            kept.append([sx, sy, t0, color])
    strays = kept

    # header: date on the left, leaves left on the right
    if shown_day >= TOTAL_DAYS:
        label = "last day of fall"
    else:
        label = "%d leaves left" % (N_LEAVES - fallen)
    draw.text((4, 1), "%s  day %d/%d" % (shown_date.strftime("%b %-d"), shown_day, TOTAL_DAYS),
              font=font, fill=TEXT)
    draw.text((width - 4 - draw.textlength(label, font=font), 1), label, font=font, fill=TEXT)

    # button hints next to the physical buttons on the left edge
    draw.text((2, 20), "< A day 1", font=small, fill=HINT)
    draw.text((2, GROUND_Y + 3), "< B +1 day", font=small, fill=HINT)

    disp.image(image, rotation)
    time.sleep(0.05)
