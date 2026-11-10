# MQTT Walkie-Talkie Instructions

These instructions set up the Lab 6 push-to-talk system on three or more Raspberry Pis. Each station uses a Qwiic button, USB microphone, and USB speaker. All Pis communicate through the course MQTT broker, so they can be on different networks.

## 1. Pull the code

On every Pi:

```bash
cd ~/Interactive-Lab-Hub
git pull origin Fall2026
cd "Lab 6"
```

If a team member does not have the repository yet:

```bash
git clone --branch Fall2026 https://github.com/Elliot-verified/Interactive-Lab-Hub.git
cd Interactive-Lab-Hub/"Lab 6"
```

## 2. Install dependencies

On every Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
sudo apt-get update
sudo apt-get install -y alsa-utils i2c-tools
pip install -r requirements-pi.txt
```

Activate the environment again whenever you open a new terminal:

```bash
cd ~/Interactive-Lab-Hub/"Lab 6"
source .venv/bin/activate
```

## 3. Configure each Pi

Edit `walkie_config.json`.

- Every team member must use the same short, unique `team_id`.
- Every Pi must have a different `device_id`, such as `elliot-pi`, `maya-pi`, and `sam-pi`.
- Do not commit each Pi's personal `device_id` back to the shared branch.

Example:

```json
{
  "team_id": "night-owls",
  "device_id": "elliot-pi"
}
```

Keep the other existing configuration fields. Leaving `device_id` empty uses the Pi's hostname, but explicitly setting it is safer because many Pis are named `raspberrypi`.

## 4. Check the hardware

Connect the Qwiic button while the Pi is powered off. After booting, check the button and USB audio devices:

```bash
i2cdetect -y 1
arecord -l
aplay -l
```

The Qwiic button normally appears at address `0x6f`.

Test recording and playback locally:

```bash
arecord -D default -d 3 -f S16_LE -r 16000 -c 1 /tmp/walkie-test.wav
aplay -D default /tmp/walkie-test.wav
```

If `default` does not select the correct USB device, update `capture_device` or `playback_device` in `walkie_config.json` with an ALSA device such as `plughw:2,0`.

## 5. Start the coordinator

Choose one Pi as the coordinator. Only one team member should run this process.

In the coordinator Pi's first terminal:

```bash
cd ~/Interactive-Lab-Hub/"Lab 6"
source .venv/bin/activate
python walkie_coordinator.py
```

Leave this process running throughout the test.

## 6. Start every walkie-talkie

On every Pi, including the coordinator Pi, open another terminal and run:

```bash
cd ~/Interactive-Lab-Hub/"Lab 6"
source .venv/bin/activate
python walkie_talkie.py
```

For a test without the physical button:

```bash
python walkie_talkie.py --keyboard
```

In keyboard mode, press Enter once to request the channel and press Enter again to release it.

## 7. Use the system

1. Hold the Qwiic button.
2. Keep holding for at least a quarter second and wait until its LED becomes bright, confirming that the channel was granted.
3. Speak into the USB microphone.
4. Release the button when finished.

LED states:

- Off: connected and idle
- Blinking: requesting the channel
- Bright: speaking
- Dim: listening or the channel is busy
- Slow blinking: disconnected

Only one person can speak at a time. A transmission ends when the button is released, the sender disconnects, or the 30-second limit is reached.

## 8. Toggle voice obfuscation

The Pi running `walkie_coordinator.py` controls the team-wide voice-obfuscation mode. Quickly double-click that Pi's Qwiic button while it is idle. The coordinator terminal and every client terminal will print whether voice obfuscation is `ON` or `OFF`.

When enabled, every sending Pi applies a robotic ring-modulation effect before publishing its microphone audio. The retained MQTT setting also reaches clients that connect later. Double-click the coordinator button again to turn the effect off.

A quick click is reserved for this gesture, so push-to-talk begins only after the button has been held for 250 ms. Other Pis ignore double-clicks.

This effect makes casual recognition harder, but it does not guarantee anonymity against voice analysis. The MQTT connection is still unencrypted, so continue to avoid private or identifying speech.

## 9. Run tests

```bash
cd ~/Interactive-Lab-Hub/"Lab 6"
source .venv/bin/activate
python -m unittest -v test_walkie_talkie.py
```

Then test with the team in stages:

1. Confirm local recording and playback on every Pi.
2. Start the coordinator and two clients.
3. Confirm audio in both directions.
4. Add the third or remaining clients.
5. Press two buttons simultaneously and verify that only one is granted.
6. Disconnect the active speaker and verify that the channel recovers.

## Privacy

The course MQTT broker is shared and unencrypted. Team-specific topics prevent accidental interference but do not make audio private. Only test with consenting participants, do not transmit sensitive speech, and do not record or retain audio.
