"""Shared MQTT protocol and floor-control state for the Lab 6 walkie-talkie."""

from __future__ import annotations

from dataclasses import dataclass
from array import array
import json
import math
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any
import uuid


AUDIO_MAGIC = b"WTK1"
AUDIO_HEADER = struct.Struct("!4s16sIQ")


def clean_name(value: str) -> str:
    """Return a value that is safe to use as one MQTT topic segment."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.strip())
    if not cleaned:
        raise ValueError("MQTT topic names cannot be empty")
    return cleaned


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config["team_id"] = clean_name(config["team_id"])
    if config["team_id"] == "replace-with-your-team-name":
        raise ValueError("Set team_id in walkie_config.json before running")
    config["device_id"] = clean_name(config.get("device_id") or socket.gethostname())
    return config


@dataclass(frozen=True)
class Topics:
    team_id: str

    @property
    def base(self) -> str:
        return f"IDD/{self.team_id}/walkie"

    @property
    def request(self) -> str:
        return f"{self.base}/control/request"

    @property
    def release(self) -> str:
        return f"{self.base}/control/release"

    @property
    def grant(self) -> str:
        return f"{self.base}/control/grant"

    @property
    def heartbeat(self) -> str:
        return f"{self.base}/control/coordinator"

    @property
    def anonymize_toggle(self) -> str:
        return f"{self.base}/control/anonymize/toggle"

    @property
    def anonymize_state(self) -> str:
        return f"{self.base}/control/anonymize/state"

    @property
    def presence_wildcard(self) -> str:
        return f"{self.base}/presence/+"

    def presence(self, device_id: str) -> str:
        return f"{self.base}/presence/{clean_name(device_id)}"

    def deny(self, device_id: str) -> str:
        return f"{self.base}/control/deny/{clean_name(device_id)}"

    @property
    def deny_wildcard(self) -> str:
        return f"{self.base}/control/deny/+"

    def audio(self, device_id: str) -> str:
        return f"{self.base}/audio/{clean_name(device_id)}"

    @property
    def audio_wildcard(self) -> str:
        return f"{self.base}/audio/+"


def json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_json(payload: bytes) -> dict[str, Any]:
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected an MQTT JSON object")
    return data


def pack_audio(stream_id: str, sequence: int, sent_at_ms: int, pcm: bytes) -> bytes:
    return AUDIO_HEADER.pack(
        AUDIO_MAGIC,
        uuid.UUID(stream_id).bytes,
        sequence,
        sent_at_ms,
    ) + pcm


def unpack_audio(payload: bytes) -> tuple[str, int, int, bytes]:
    if len(payload) <= AUDIO_HEADER.size:
        raise ValueError("Audio packet is too short")
    magic, stream_bytes, sequence, sent_at_ms = AUDIO_HEADER.unpack_from(payload)
    if magic != AUDIO_MAGIC:
        raise ValueError("Unknown audio packet format")
    return str(uuid.UUID(bytes=stream_bytes)), sequence, sent_at_ms, payload[AUDIO_HEADER.size :]


class AudioPacketFilter:
    """Reject stale, duplicate, and out-of-order packets for each stream."""

    def __init__(self):
        self.last_sequence: dict[str, int] = {}

    def reset(self):
        self.last_sequence.clear()

    def accept(
        self,
        stream_id: str,
        sequence: int,
        sent_at_ms: int,
        *,
        now_ms: int,
        stale_ms: int,
    ) -> bool:
        if now_ms - sent_at_ms > stale_ms:
            return False
        previous = self.last_sequence.get(stream_id, -1)
        if sequence <= previous:
            return False
        self.last_sequence[stream_id] = sequence
        return True


def button_edge(previous: bool, current: bool) -> str | None:
    if previous == current:
        return None
    return "pressed" if current else "released"


class PushToTalkGesture:
    """Turn a hold into push-to-talk and a coordinator double-click into a toggle."""

    def __init__(self, hold_seconds: float = 0.25, double_click_seconds: float = 0.45):
        self.hold_seconds = hold_seconds
        self.double_click_seconds = double_click_seconds
        self.pressed = False
        self.talking = False
        self.press_started = 0.0
        self.last_quick_release: float | None = None

    def update(self, pressed: bool, now: float, allow_double_click: bool) -> list[str]:
        actions: list[str] = []
        if pressed and not self.pressed:
            self.press_started = now
        if pressed and not self.talking and now - self.press_started >= self.hold_seconds:
            self.talking = True
            self.last_quick_release = None
            actions.append("request")
        if not pressed and self.pressed:
            if self.talking:
                self.talking = False
                actions.append("release")
            elif allow_double_click:
                if (
                    self.last_quick_release is not None
                    and now - self.last_quick_release <= self.double_click_seconds
                ):
                    self.last_quick_release = None
                    actions.append("toggle_anonymize")
                else:
                    self.last_quick_release = now
        self.pressed = pressed
        return actions


class VoiceDistorter:
    """Apply continuous ring modulation to little-endian signed 16-bit PCM."""

    def __init__(self, sample_rate: int, frequency_hz: float = 70.0):
        self.phase = 0.0
        self.phase_step = 2.0 * math.pi * frequency_hz / sample_rate

    def process(self, pcm: bytes) -> bytes:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        for index, sample in enumerate(samples):
            carrier = math.sin(self.phase)
            value = int(sample * carrier * 1.35)
            samples[index] = max(-32768, min(32767, value))
            self.phase = (self.phase + self.phase_step) % (2.0 * math.pi)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples.tobytes()


def status_brightness(status: str, phase: int = 0) -> int:
    if status == "speaking":
        return 255
    if status == "waiting":
        return 100 if phase % 2 else 0
    if status in {"busy", "listening"}:
        return 25
    if status == "disconnected":
        return 180 if phase % 2 else 0
    return 0


@dataclass
class FloorGrant:
    speaker: str
    request_id: str
    stream_id: str
    granted_at_ms: int
    lease_seconds: float

    def as_message(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "request_id": self.request_id,
            "stream_id": self.stream_id,
            "granted_at_ms": self.granted_at_ms,
            "lease_seconds": self.lease_seconds,
        }


class FloorArbiter:
    """First-request-wins, single-speaker floor state with a fixed lease."""

    def __init__(self, lease_seconds: float, clock=time.monotonic):
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.current: FloorGrant | None = None
        self._expires_at = 0.0

    def request(self, speaker: str, request_id: str) -> tuple[bool, FloorGrant]:
        self.expire()
        if self.current is not None:
            return self.current.speaker == speaker, self.current
        self.current = FloorGrant(
            speaker=clean_name(speaker),
            request_id=request_id,
            stream_id=str(uuid.uuid4()),
            granted_at_ms=int(time.time() * 1000),
            lease_seconds=self.lease_seconds,
        )
        self._expires_at = self.clock() + self.lease_seconds
        return True, self.current

    def release(self, speaker: str, stream_id: str | None = None) -> FloorGrant | None:
        if self.current is None or self.current.speaker != speaker:
            return None
        if stream_id and stream_id != self.current.stream_id:
            return None
        released = self.current
        self.current = None
        self._expires_at = 0.0
        return released

    def expire(self) -> FloorGrant | None:
        if self.current is not None and self.clock() >= self._expires_at:
            expired = self.current
            self.current = None
            self._expires_at = 0.0
            return expired
        return None


def idle_floor_message(reason: str = "idle") -> dict[str, Any]:
    return {"speaker": None, "stream_id": None, "reason": reason, "at_ms": int(time.time() * 1000)}
