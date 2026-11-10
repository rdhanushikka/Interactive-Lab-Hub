#!/usr/bin/env python3
"""Push-to-talk MQTT walkie-talkie for a Qwiic button and USB audio."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid

import paho.mqtt.client as mqtt

from walkie_protocol import (
    AudioPacketFilter,
    PushToTalkGesture,
    Topics,
    VoiceDistorter,
    json_bytes,
    load_config,
    pack_audio,
    parse_json,
    status_brightness,
    unpack_audio,
)


DEFAULT_CONFIG = Path(__file__).with_name("walkie_config.json")


class QwiicButton:
    def __init__(self, address: int):
        try:
            import qwiic_button
        except ImportError as exc:
            raise RuntimeError("install sparkfun-qwiic-button") from exc
        self.button = qwiic_button.QwiicButton(address)
        if not self.button.begin():
            raise RuntimeError(f"Qwiic button not found at 0x{address:02x}")
        self._brightness = -1

    def is_pressed(self) -> bool:
        return bool(self.button.is_button_pressed())

    def set_led(self, brightness: int):
        brightness = max(0, min(255, int(brightness)))
        if brightness == self._brightness:
            return
        if brightness:
            self.button.LED_on(brightness)
        else:
            self.button.LED_off()
        self._brightness = brightness


class KeyboardToggleButton:
    """Development fallback: each Enter press toggles push-to-talk."""

    def __init__(self):
        self._pressed = False
        self._stop = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop.is_set():
            try:
                input()
            except EOFError:
                return
            self._pressed = not self._pressed
            print("[BUTTON] pressed" if self._pressed else "[BUTTON] released", flush=True)

    def is_pressed(self) -> bool:
        return self._pressed

    def set_led(self, brightness: int):
        return

    def close(self):
        self._stop.set()


@dataclass
class FloorState:
    speaker: str | None = None
    stream_id: str | None = None
    request_id: str | None = None


class AudioPlayer:
    def __init__(self, config: dict, active_stream):
        self.config = config
        self.active_stream = active_stream
        self.items: queue.Queue[tuple[str, bytes]] = queue.Queue(
            maxsize=int(config.get("jitter_chunks", 8))
        )
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="audio-player", daemon=True)
        self.process: subprocess.Popen | None = None
        self.process_stream: str | None = None
        self.thread.start()

    def put(self, stream_id: str, pcm: bytes):
        try:
            self.items.put_nowait((stream_id, pcm))
        except queue.Full:
            try:
                self.items.get_nowait()
            except queue.Empty:
                pass
            try:
                self.items.put_nowait((stream_id, pcm))
            except queue.Full:
                pass

    def _start(self, stream_id: str):
        self._stop_process()
        command = [
            "aplay",
            "-q",
            "-D",
            self.config.get("playback_device", "default"),
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(self.config.get("sample_rate", 16000)),
            "-c",
            str(self.config.get("channels", 1)),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
        self.process_stream = stream_id

    def _stop_process(self):
        if self.process:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
            self.process = None
            self.process_stream = None

    def _run(self):
        while not self.stop_event.is_set():
            try:
                stream_id, pcm = self.items.get(timeout=0.1)
            except queue.Empty:
                if self.process_stream and self.active_stream() != self.process_stream:
                    self._stop_process()
                continue
            if stream_id != self.active_stream():
                continue
            if self.process_stream != stream_id:
                self._start(stream_id)
            try:
                assert self.process and self.process.stdin
                self.process.stdin.write(pcm)
            except (BrokenPipeError, OSError):
                self._stop_process()
        self._stop_process()

    def clear(self):
        while True:
            try:
                self.items.get_nowait()
            except queue.Empty:
                return

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


class WalkieTalkie:
    def __init__(self, config: dict, keyboard: bool = False):
        if not shutil.which("arecord") or not shutil.which("aplay"):
            raise RuntimeError("arecord/aplay are required; install the alsa-utils package")
        self.config = config
        self.device_id = config["device_id"]
        self.topics = Topics(config["team_id"])
        self.button = KeyboardToggleButton() if keyboard else QwiicButton(int(config.get("qwiic_address", 0x6F)))
        self.floor = FloorState()
        self.floor_lock = threading.Lock()
        self.connected = False
        self.coordinator_seen = 0.0
        self.coordinator_id: str | None = None
        self.anonymize_enabled = False
        self.button_pressed = False
        self.gesture = PushToTalkGesture(
            hold_seconds=float(config.get("hold_to_talk_ms", 250)) / 1000,
            double_click_seconds=float(config.get("double_click_ms", 450)) / 1000,
        )
        self.request_id: str | None = None
        self.status = "disconnected"
        self.stop_event = threading.Event()
        self.record_stop = threading.Event()
        self.record_thread: threading.Thread | None = None
        self.packet_filter = AudioPacketFilter()
        self.player = AudioPlayer(config, self.active_remote_stream)

        self.client = mqtt.Client(client_id=f"{self.device_id}-walkie")
        self.client.username_pw_set(config["username"], config["password"])
        self.client.will_set(
            self.topics.presence(self.device_id),
            json_bytes({"device_id": self.device_id, "online": False}),
            qos=1,
            retain=True,
        )
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

    def active_remote_stream(self) -> str | None:
        with self.floor_lock:
            if self.floor.speaker and self.floor.speaker != self.device_id:
                return self.floor.stream_id
        return None

    def on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"[ERROR] MQTT connection failed: {rc}", flush=True)
            return
        self.connected = True
        self.status = "idle"
        client.subscribe(
            [
                (self.topics.grant, 1),
                (self.topics.deny(self.device_id), 1),
                (self.topics.audio_wildcard, 0),
                (self.topics.heartbeat, 0),
                (self.topics.anonymize_state, 1),
            ]
        )
        client.publish(
            self.topics.presence(self.device_id),
            json_bytes({"device_id": self.device_id, "online": True}),
            qos=1,
            retain=True,
        )
        print(f"[OK] Connected as {self.device_id}. Hold the button to talk.", flush=True)

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.status = "disconnected"
        self.stop_recording()
        if rc and not self.stop_event.is_set():
            print(f"[WARNING] MQTT disconnected ({rc}); reconnecting...", flush=True)

    def on_message(self, client, userdata, message):
        try:
            if message.topic == self.topics.grant:
                self.handle_floor(parse_json(message.payload))
            elif message.topic == self.topics.deny(self.device_id):
                self.handle_denied(parse_json(message.payload))
            elif message.topic == self.topics.heartbeat:
                data = parse_json(message.payload)
                if data.get("online"):
                    self.coordinator_seen = time.monotonic()
                    self.coordinator_id = data.get("coordinator")
            elif message.topic == self.topics.anonymize_state:
                self.handle_anonymize_state(parse_json(message.payload))
            elif message.topic.startswith(f"{self.topics.base}/audio/"):
                self.handle_audio(message.topic, message.payload)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[WARNING] Invalid message on {message.topic}: {exc}", flush=True)

    def handle_floor(self, data: dict):
        new_floor = FloorState(
            speaker=data.get("speaker"),
            stream_id=data.get("stream_id"),
            request_id=data.get("request_id"),
        )
        with self.floor_lock:
            old_stream = self.floor.stream_id
            self.floor = new_floor
        if old_stream != new_floor.stream_id:
            self.player.clear()
            self.packet_filter.reset()

        if new_floor.speaker == self.device_id:
            if self.button_pressed:
                self.status = "speaking"
                self.start_recording(new_floor.stream_id)
            else:
                self.publish_release(new_floor.stream_id)
        else:
            self.stop_recording()
            if new_floor.speaker:
                self.status = "listening"
                print(f"[LISTENING] {new_floor.speaker}", flush=True)
            elif self.connected:
                self.status = "idle"

    def handle_denied(self, data: dict):
        if data.get("request_id") == self.request_id and self.button_pressed:
            self.status = "busy"
            print(f"[BUSY] {data.get('speaker', 'another device')} is speaking", flush=True)

    def handle_anonymize_state(self, data: dict):
        self.anonymize_enabled = bool(data.get("enabled", False))
        state = "ON" if self.anonymize_enabled else "OFF"
        print(f"[VOICE OBFUSCATION] {state}", flush=True)

    def handle_audio(self, topic: str, payload: bytes):
        sender = topic.rsplit("/", 1)[-1]
        if sender == self.device_id:
            return
        stream_id, sequence, sent_at_ms, pcm = unpack_audio(payload)
        with self.floor_lock:
            expected = self.floor.stream_id if self.floor.speaker == sender else None
        if stream_id != expected:
            return
        if not self.packet_filter.accept(
            stream_id,
            sequence,
            sent_at_ms,
            now_ms=int(time.time() * 1000),
            stale_ms=int(self.config.get("stale_audio_ms", 1500)),
        ):
            return
        self.player.put(stream_id, pcm)

    def request_floor(self):
        self.request_id = str(uuid.uuid4())
        self.status = "waiting"
        self.client.publish(
            self.topics.request,
            json_bytes(
                {
                    "device_id": self.device_id,
                    "request_id": self.request_id,
                    "at_ms": int(time.time() * 1000),
                }
            ),
            qos=1,
        )
        print("[WAITING] Requesting the channel...", flush=True)

    def publish_release(self, stream_id: str | None):
        self.client.publish(
            self.topics.release,
            json_bytes({"device_id": self.device_id, "stream_id": stream_id}),
            qos=1,
        )

    def start_recording(self, stream_id: str | None):
        if not stream_id or (self.record_thread and self.record_thread.is_alive()):
            return
        self.record_stop.clear()
        self.record_thread = threading.Thread(
            target=self._record, args=(stream_id,), name="audio-recorder", daemon=True
        )
        self.record_thread.start()
        print("[SPEAKING] Channel granted", flush=True)

    def _record(self, stream_id: str):
        rate = int(self.config.get("sample_rate", 16000))
        channels = int(self.config.get("channels", 1))
        chunk_ms = int(self.config.get("chunk_ms", 40))
        chunk_bytes = max(1, rate * channels * 2 * chunk_ms // 1000)
        command = [
            "arecord",
            "-q",
            "-D",
            self.config.get("capture_device", "default"),
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(rate),
            "-c",
            str(channels),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0)
        sequence = 0
        distorter = VoiceDistorter(
            rate,
            frequency_hz=float(self.config.get("distortion_frequency_hz", 70)),
        )
        try:
            assert process.stdout
            while not self.record_stop.is_set() and self.button_pressed:
                pcm = process.stdout.read(chunk_bytes)
                if not pcm:
                    break
                if self.anonymize_enabled:
                    pcm = distorter.process(pcm)
                packet = pack_audio(stream_id, sequence, int(time.time() * 1000), pcm)
                self.client.publish(self.topics.audio(self.device_id), packet, qos=0)
                sequence += 1
        finally:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    def stop_recording(self):
        self.record_stop.set()
        thread = self.record_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=1)
        self.record_thread = None

    def update_button(self):
        pressed = self.button.is_pressed()
        self.button_pressed = pressed
        actions = self.gesture.update(
            pressed,
            time.monotonic(),
            allow_double_click=self.device_id == self.coordinator_id,
        )
        for action in actions:
            if action == "request":
                if self.connected:
                    self.request_floor()
                else:
                    self.status = "disconnected"
            elif action == "release":
                with self.floor_lock:
                    own_stream = self.floor.stream_id if self.floor.speaker == self.device_id else None
                self.stop_recording()
                if own_stream:
                    self.publish_release(own_stream)
                self.request_id = None
                self.status = "idle" if self.connected else "disconnected"
                print("[IDLE] Button released", flush=True)
            elif action == "toggle_anonymize":
                self.client.publish(
                    self.topics.anonymize_toggle,
                    json_bytes({"device_id": self.device_id, "at_ms": int(time.time() * 1000)}),
                    qos=1,
                )
                print("[VOICE OBFUSCATION] Toggle requested", flush=True)

    def update_led(self):
        phase_rate = 2 if self.status == "disconnected" else 4
        brightness = status_brightness(self.status, int(time.monotonic() * phase_rate))
        self.button.set_led(brightness)

    def run(self):
        self.client.connect(
            self.config["broker"],
            port=int(self.config.get("port", 1883)),
            keepalive=30,
        )
        self.client.loop_start()
        try:
            while not self.stop_event.wait(0.04):
                self.update_button()
                self.update_led()
        finally:
            self.cleanup()

    def cleanup(self):
        self.stop_event.set()
        self.stop_recording()
        self.player.stop()
        self.button.set_led(0)
        if hasattr(self.button, "close"):
            self.button.close()
        if self.connected:
            self.client.publish(
                self.topics.presence(self.device_id),
                json_bytes({"device_id": self.device_id, "online": False}),
                qos=1,
                retain=True,
            ).wait_for_publish(timeout=1)
        self.client.loop_stop()
        self.client.disconnect()

    def stop(self, *_):
        self.stop_event.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="use Enter to toggle push-to-talk instead of a Qwiic button",
    )
    args = parser.parse_args()
    try:
        walkie = WalkieTalkie(load_config(args.config), keyboard=args.keyboard)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Startup error: {exc}") from exc
    signal.signal(signal.SIGINT, walkie.stop)
    signal.signal(signal.SIGTERM, walkie.stop)
    walkie.run()


if __name__ == "__main__":
    main()
