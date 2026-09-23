#!/usr/bin/env python3
"""Ask a numerical question out loud, record the answer, and transcribe it.

The device speaks a question with Piper, records the respondent for a fixed
number of seconds, transcribes the recording with faster-whisper, and then
reads back the digits it thinks it heard. The raw transcript and the extracted
digits are both printed, because the gap between them is what this exercise is
about: transcription systems make characteristic mistakes on digit strings
("fifteen" vs "fifty", "oh" vs "zero", "two" vs "to", dropped or doubled
digits) and you want to see them before designing around them.

    python ask_number.py
    python ask_number.py --question "What is your zip code?" --seconds 4
    python ask_number.py --model base.en

The recording is saved next to this script as answer_<timestamp>.wav so you
can re-run transcribe.py on it with other model sizes.
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from piper import PiperVoice

SAMPLE_RATE = 16000
LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICE = LAB_DIR / "voices" / "en_US-lessac-medium.onnx"

# Whisper sometimes returns digit strings as words. Map the common ones.
WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "for": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "ate": "8",
    "nine": "9",
}


def extract_digits(transcript: str) -> str:
    """Pull a digit string out of a transcript, converting number words too."""
    digits = []
    for token in re.findall(r"[a-zA-Z]+|\d", transcript.lower()):
        if token.isdigit():
            digits.append(token)
        elif token in WORD_TO_DIGIT:
            digits.append(WORD_TO_DIGIT[token])
        elif token == "double" or token == "triple":
            # "double five" -> handled when the next digit arrives
            digits.append(token)
    # Expand "double"/"triple" markers.
    out = []
    i = 0
    while i < len(digits):
        if digits[i] in ("double", "triple") and i + 1 < len(digits):
            out.append(digits[i + 1] * (2 if digits[i] == "double" else 3))
            i += 2
        elif digits[i] in ("double", "triple"):
            i += 1
        else:
            out.append(digits[i])
            i += 1
    return "".join(out)


def spaced(digits: str) -> str:
    """'6071234' -> '6 0 7 1 2 3 4', so Piper reads digits one at a time."""
    return " ".join(digits)


class Speaker:
    """Synthesizes with Piper and plays through the default output device."""

    def __init__(self, voice_path: Path) -> None:
        self.voice = PiperVoice.load(str(voice_path))

    def say(self, text: str) -> None:
        for chunk in self.voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            sd.play(audio, samplerate=chunk.sample_rate)
            sd.wait()


def record(seconds: float) -> np.ndarray:
    """Record from the default input device for a fixed duration."""
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Hi. What is your phone number?",
                        help="what the device asks")
    parser.add_argument("--seconds", type=float, default=6.0,
                        help="how long to record the answer (default: 6)")
    parser.add_argument("--model", default="tiny.en",
                        help="whisper model size (default: tiny.en)")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--no-readback", action="store_true",
                        help="don't speak the digits back")
    args = parser.parse_args()

    if not args.voice.is_file():
        sys.exit(f"Piper voice not found at {args.voice}. Run ./setup.sh first.")

    print("Loading models...", flush=True)
    speaker = Speaker(args.voice)
    recognizer = WhisperModel(args.model, device="cpu", compute_type="int8")

    speaker.say(args.question)
    print(f"Recording for {args.seconds:.0f} seconds. Answer now.", flush=True)

    audio = record(args.seconds)
    recorded_at = time.perf_counter()

    out_path = Path(__file__).resolve().parent / (
        f"answer_{datetime.now():%Y%m%d_%H%M%S}.wav")
    sf.write(out_path, audio, SAMPLE_RATE)

    segments, info = recognizer.transcribe(audio, beam_size=1)
    transcript = " ".join(seg.text.strip() for seg in segments)
    asr_seconds = time.perf_counter() - recorded_at

    digits = extract_digits(transcript)

    print()
    print(f"transcript       {transcript!r}")
    print(f"digits heard     {digits or '(none)'}")
    print(f"digit count      {len(digits)}")
    print(f"model            {args.model}")
    print(f"transcription    {asr_seconds:.2f}s for {info.duration:.2f}s of audio "
          f"(real-time factor {asr_seconds / info.duration:.2f}x)")
    print(f"saved            {out_path}")

    if args.no_readback:
        return
    if digits:
        speaker.say(f"I heard {spaced(digits)}. Is that right?")
    else:
        speaker.say("Sorry, I didn't catch any numbers.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
