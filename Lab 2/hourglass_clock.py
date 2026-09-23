# Lab 2 Part 2 - The Day Hourglass.
# Sand level shows how much of today is left. Button A pauses the sand,
# button B resumes it, so you can hold onto a moment instead of watching it drain.
#
# Usage:
#   python hourglass_clock.py        # real time: one full drain per day
#   python hourglass_clock.py 60     # demo: one full drain every 60 seconds
import sys
import time
import math
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7789 as st7789

# A full cycle is one day, unless a shorter period is passed in for demoing.
PERIOD = float(sys.argv[1]) if len(sys.argv) > 1 else 86400.0

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

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

backlight = digitalio.DigitalInOut(board.D22)
backlight.switch_to_output()
backlight.value = True

buttonA = digitalio.DigitalInOut(board.D23)   # top button: pause
buttonB = digitalio.DigitalInOut(board.D24)   # bottom button: resume
buttonA.switch_to_input(pull=digitalio.Pull.UP)
buttonB.switch_to_input(pull=digitalio.Pull.UP)

# Hourglass geometry (landscape: 240 wide, 135 tall)
CX = width // 2
TOPY, NECK, BOTY = 17, 69, 121
CHAMBER = NECK - TOPY
HW = 48
SAND = (232, 176, 75)
SAND_PAUSED = (120, 130, 145)
GLASS = (120, 170, 200)

paused = False
frozen_frac = 1.0


def fraction_left():
    """How much of the current cycle is still ahead of us, 1.0 -> 0.0."""
    if PERIOD >= 86400.0:
        now = time.localtime()
        elapsed = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
        return 1.0 - (elapsed % 86400) / 86400.0
    return 1.0 - (time.time() % PERIOD) / PERIOD


while True:
    # Buttons read LOW when pressed.
    if not buttonA.value:
        paused = True
    if not buttonB.value:
        paused = False

    if not paused:
        frozen_frac = fraction_left()
    frac_left = frozen_frac
    color = SAND_PAUSED if paused else SAND

    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))

    # sand still in the top chamber, sized by area so it drains evenly
    h = CHAMBER * math.sqrt(frac_left)
    if h > 0.5:
        w = HW * h / CHAMBER
        draw.polygon(
            [(CX - w, NECK - h), (CX + w, NECK - h), (CX, NECK)], fill=color
        )

    # sand piled in the bottom chamber
    y = NECK + CHAMBER * math.sqrt(frac_left)
    if y < BOTY - 0.5:
        w = HW * (y - NECK) / CHAMBER
        draw.polygon(
            [(CX - w, y), (CX + w, y), (CX + HW, BOTY), (CX - HW, BOTY)], fill=color
        )

    # falling grain, only while the sand is actually running
    if not paused and 0.001 < frac_left < 0.999:
        draw.line([(CX, NECK), (CX, min(BOTY, y + 6))], fill=color, width=2)

    # glass outline on top of the sand
    draw.polygon([(CX - HW, TOPY), (CX + HW, TOPY), (CX, NECK)], outline=GLASS)
    draw.polygon([(CX - HW, BOTY), (CX + HW, BOTY), (CX, NECK)], outline=GLASS)

    hrs_left = int(frac_left * 24)
    mins_left = int(frac_left * 1440) % 60
    if PERIOD >= 86400.0:
        clock = time.strftime("%H:%M:%S")
    else:
        # In demo mode the sand runs on a compressed day, so show the simulated
        # time of day instead of the wall clock - otherwise the two disagree.
        sim = int((1.0 - frac_left) * 86400)
        clock = "%02d:%02d:%02d" % (sim // 3600, (sim % 3600) // 60, sim % 60)
    draw.text((4, 1), clock, font=font, fill="#FFFFFF")
    label = "%dh %dm left" % (hrs_left, mins_left)
    draw.text((width - 4 - draw.textlength(label, font=font), 1),
              label, font=font, fill=color)

    status = "PAUSED - B resumes" if paused else "A to pause"
    if PERIOD < 86400.0:
        # e.g. a 60s period means one real minute stands in for a whole day
        status += "  [%gmin = 1 day]" % (PERIOD / 60.0)
    draw.text((4, BOTY + 2), status, font=small, fill=GLASS)

    disp.image(image, rotation)
    time.sleep(0.2)
