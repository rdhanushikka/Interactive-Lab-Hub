import unittest
import uuid

from walkie_protocol import (
    AudioPacketFilter,
    FloorArbiter,
    PushToTalkGesture,
    Topics,
    VoiceDistorter,
    button_edge,
    pack_audio,
    status_brightness,
    unpack_audio,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ProtocolTests(unittest.TestCase):
    def test_team_topics_are_isolated(self):
        topics = Topics("team-owl")
        self.assertEqual(topics.request, "IDD/team-owl/walkie/control/request")
        self.assertEqual(topics.audio("pi-one"), "IDD/team-owl/walkie/audio/pi-one")

    def test_audio_packet_round_trip(self):
        stream_id = str(uuid.uuid4())
        packed = pack_audio(stream_id, 42, 123456, b"\x00\x01pcm")
        self.assertEqual(unpack_audio(packed), (stream_id, 42, 123456, b"\x00\x01pcm"))

    def test_packet_filter_rejects_stale_duplicate_and_out_of_order_audio(self):
        packets = AudioPacketFilter()
        stream = str(uuid.uuid4())
        self.assertTrue(packets.accept(stream, 0, 1000, now_ms=1100, stale_ms=500))
        self.assertFalse(packets.accept(stream, 0, 1050, now_ms=1100, stale_ms=500))
        self.assertTrue(packets.accept(stream, 2, 1100, now_ms=1200, stale_ms=500))
        self.assertFalse(packets.accept(stream, 1, 1150, now_ms=1200, stale_ms=500))
        self.assertFalse(packets.accept(stream, 3, 1000, now_ms=2000, stale_ms=500))

    def test_button_edges_and_led_feedback(self):
        self.assertIsNone(button_edge(False, False))
        self.assertEqual(button_edge(False, True), "pressed")
        self.assertEqual(button_edge(True, False), "released")
        self.assertEqual(status_brightness("speaking"), 255)
        self.assertEqual(status_brightness("waiting", 0), 0)
        self.assertEqual(status_brightness("waiting", 1), 100)
        self.assertEqual(status_brightness("listening"), 25)

    def test_hold_talks_but_coordinator_double_click_toggles(self):
        gesture = PushToTalkGesture(hold_seconds=0.25, double_click_seconds=0.45)
        self.assertEqual(gesture.update(True, 0.0, True), [])
        self.assertEqual(gesture.update(True, 0.3, True), ["request"])
        self.assertEqual(gesture.update(False, 0.4, True), ["release"])

        gesture = PushToTalkGesture(hold_seconds=0.25, double_click_seconds=0.45)
        gesture.update(True, 1.0, True)
        self.assertEqual(gesture.update(False, 1.1, True), [])
        gesture.update(True, 1.2, True)
        self.assertEqual(gesture.update(False, 1.3, True), ["toggle_anonymize"])

    def test_voice_distortion_changes_pcm_without_changing_packet_size(self):
        pcm = (b"\x10\x27" * 160)
        distorted = VoiceDistorter(16000, frequency_hz=70).process(pcm)
        self.assertEqual(len(distorted), len(pcm))
        self.assertNotEqual(distorted, pcm)


class FloorArbiterTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.floor = FloorArbiter(lease_seconds=5, clock=self.clock)

    def test_first_request_wins(self):
        granted, first = self.floor.request("pi-one", "request-1")
        self.assertTrue(granted)
        granted, current = self.floor.request("pi-two", "request-2")
        self.assertFalse(granted)
        self.assertEqual(current.stream_id, first.stream_id)
        self.assertEqual(current.speaker, "pi-one")

    def test_only_current_speaker_can_release(self):
        _, grant = self.floor.request("pi-one", "request-1")
        self.assertIsNone(self.floor.release("pi-two"))
        self.assertIsNone(self.floor.release("pi-one", "wrong-stream"))
        self.assertEqual(self.floor.release("pi-one", grant.stream_id), grant)
        self.assertIsNone(self.floor.current)

    def test_lease_expires_and_next_device_can_speak(self):
        self.floor.request("pi-one", "request-1")
        self.clock.now = 5.1
        expired = self.floor.expire()
        self.assertEqual(expired.speaker, "pi-one")
        granted, second = self.floor.request("pi-two", "request-2")
        self.assertTrue(granted)
        self.assertEqual(second.speaker, "pi-two")


if __name__ == "__main__":
    unittest.main()
