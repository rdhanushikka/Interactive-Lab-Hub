#!/usr/bin/env python3
"""Arbitrate the single speaking channel for the MQTT walkie-talkie."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from walkie_protocol import (
    FloorArbiter,
    Topics,
    idle_floor_message,
    json_bytes,
    load_config,
    parse_json,
)


DEFAULT_CONFIG = Path(__file__).with_name("walkie_config.json")


class Coordinator:
    def __init__(self, config: dict):
        self.config = config
        self.device_id = config["device_id"]
        self.topics = Topics(config["team_id"])
        self.arbiter = FloorArbiter(float(config.get("max_talk_seconds", 30)))
        self.anonymize_enabled = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.client = mqtt.Client(client_id=f"{self.device_id}-walkie-coordinator")
        self.client.username_pw_set(config["username"], config["password"])
        self.client.will_set(
            self.topics.heartbeat,
            json_bytes({"online": False, "coordinator": self.device_id}),
            qos=1,
            retain=True,
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"[ERROR] MQTT connection failed: {rc}", flush=True)
            return
        print(f"[OK] Coordinator connected as {self.device_id}", flush=True)
        client.subscribe(
            [
                (self.topics.request, 1),
                (self.topics.release, 1),
                (self.topics.presence_wildcard, 1),
                (self.topics.anonymize_toggle, 1),
            ]
        )
        client.publish(
            self.topics.heartbeat,
            json_bytes({"online": True, "coordinator": self.device_id, "at_ms": int(time.time() * 1000)}),
            qos=1,
            retain=True,
        )
        with self.lock:
            state = self.arbiter.current.as_message() if self.arbiter.current else idle_floor_message()
        client.publish(self.topics.grant, json_bytes(state), qos=1, retain=True)
        self.publish_anonymize_state()

    def on_disconnect(self, client, userdata, rc):
        if rc and not self.stop_event.is_set():
            print(f"[WARNING] MQTT disconnected ({rc}); reconnecting...", flush=True)

    def on_message(self, client, userdata, message):
        try:
            data = parse_json(message.payload)
            if message.topic == self.topics.request:
                self.handle_request(data)
            elif message.topic == self.topics.release:
                self.handle_release(data, "released")
            elif message.topic.startswith(f"{self.topics.base}/presence/"):
                self.handle_presence(data)
            elif message.topic == self.topics.anonymize_toggle:
                self.handle_anonymize_toggle(data)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[WARNING] Ignoring invalid message on {message.topic}: {exc}", flush=True)

    def handle_request(self, data: dict):
        speaker = str(data["device_id"])
        request_id = str(data["request_id"])
        with self.lock:
            granted, floor = self.arbiter.request(speaker, request_id)
        if granted:
            self.client.publish(self.topics.grant, json_bytes(floor.as_message()), qos=1, retain=True)
            print(f"[GRANT] {floor.speaker} for {floor.lease_seconds:.0f}s", flush=True)
        else:
            self.client.publish(
                self.topics.deny(speaker),
                json_bytes(
                    {
                        "device_id": speaker,
                        "request_id": request_id,
                        "reason": "busy",
                        "speaker": floor.speaker,
                    }
                ),
                qos=1,
            )
            print(f"[BUSY] Denied {speaker}; {floor.speaker} is speaking", flush=True)

    def handle_release(self, data: dict, reason: str):
        speaker = str(data["device_id"])
        stream_id = data.get("stream_id")
        with self.lock:
            released = self.arbiter.release(speaker, stream_id)
        if released:
            self.publish_idle(reason)
            print(f"[IDLE] Released {speaker}: {reason}", flush=True)

    def handle_presence(self, data: dict):
        if data.get("online", True):
            return
        device_id = str(data.get("device_id", ""))
        if device_id:
            self.handle_release({"device_id": device_id}, "speaker disconnected")

    def publish_idle(self, reason: str):
        self.client.publish(
            self.topics.grant,
            json_bytes(idle_floor_message(reason)),
            qos=1,
            retain=True,
        )

    def handle_anonymize_toggle(self, data: dict):
        if data.get("device_id") != self.device_id:
            print("[WARNING] Ignoring anonymize toggle from a non-coordinator device", flush=True)
            return
        self.anonymize_enabled = not self.anonymize_enabled
        self.publish_anonymize_state()
        state = "ON" if self.anonymize_enabled else "OFF"
        print(f"[VOICE OBFUSCATION] {state}", flush=True)

    def publish_anonymize_state(self):
        self.client.publish(
            self.topics.anonymize_state,
            json_bytes(
                {
                    "enabled": self.anonymize_enabled,
                    "coordinator": self.device_id,
                    "at_ms": int(time.time() * 1000),
                }
            ),
            qos=1,
            retain=True,
        )

    def run(self):
        self.client.connect(
            self.config["broker"],
            port=int(self.config.get("port", 1883)),
            keepalive=30,
        )
        self.client.loop_start()
        print("Coordinator running. Press Ctrl-C to stop.", flush=True)
        try:
            while not self.stop_event.wait(0.25):
                with self.lock:
                    expired = self.arbiter.expire()
                if expired:
                    self.publish_idle("talk-time limit reached")
                    print(f"[TIMEOUT] Released {expired.speaker}", flush=True)
                self.client.publish(
                    self.topics.heartbeat,
                    json_bytes(
                        {
                            "online": True,
                            "coordinator": self.device_id,
                            "at_ms": int(time.time() * 1000),
                        }
                    ),
                    qos=0,
                    retain=True,
                )
                self.stop_event.wait(1.75)
        finally:
            self.publish_idle("coordinator stopped")
            self.client.publish(
                self.topics.heartbeat,
                json_bytes({"online": False, "coordinator": self.device_id}),
                qos=1,
                retain=True,
            )
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self, *_):
        self.stop_event.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    try:
        coordinator = Coordinator(load_config(args.config))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    signal.signal(signal.SIGINT, coordinator.stop)
    signal.signal(signal.SIGTERM, coordinator.stop)
    coordinator.run()


if __name__ == "__main__":
    main()
