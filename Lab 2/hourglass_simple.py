# Lab 2 Part 2 - first pass: the barebones clock redrawn as an hourglass.
# One element of the bigger idea: sand level = how much of today is left.
import time
import math
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7789 as st7789

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

backlight = digitalio.DigitalInOut(board.D22)
backlight.switch_to_output()
backlight.value = True

# Hourglass geometry (landscape: 240 wide, 135 tall)
CX = width // 2
TOPY, NECK, BOTY = 17, 69, 121
CHAMBER = NECK - TOPY
HW = 48
SAND = (232, 176, 75)
GLASS = (120, 170, 200)

while True:
    now = time.localtime()
    secs_today = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    frac_left = 1.0 - (secs_today / 86400.0)

    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))

    # sand still in the top chamber, by area
    h = CHAMBER * math.sqrt(frac_left)
    if h > 0.5:
        w = HW * h / CHAMBER
        draw.polygon(
            [(CX - w, NECK - h), (CX + w, NECK - h), (CX, NECK)], fill=SAND
        )

    # sand piled in the bottom chamber
    y = NECK + CHAMBER * math.sqrt(frac_left)
    if y < BOTY - 0.5:
        w = HW * (y - NECK) / CHAMBER
        draw.polygon(
            [(CX - w, y), (CX + w, y), (CX + HW, BOTY), (CX - HW, BOTY)], fill=SAND
        )

    # glass outline on top of the sand
    draw.polygon([(CX - HW, TOPY), (CX + HW, TOPY), (CX, NECK)], outline=GLASS)
    draw.polygon([(CX - HW, BOTY), (CX + HW, BOTY), (CX, NECK)], outline=GLASS)

    hrs_left = int(frac_left * 24)
    mins_left = int(frac_left * 1440) % 60
    draw.text((4, 1), time.strftime("%H:%M:%S"), font=font, fill="#FFFFFF")
    label = "%dh %dm left" % (hrs_left, mins_left)
    draw.text((width - 4 - draw.textlength(label, font=font), 1),
              label, font=font, fill=SAND)

    disp.image(image, rotation)
    time.sleep(1)
