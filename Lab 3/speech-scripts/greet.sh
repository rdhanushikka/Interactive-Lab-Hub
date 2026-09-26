#!/usr/bin/env bash
# Lab 3 Part A: have the Pi greet me by name.
#
# Uses Piper (neural TTS) streamed straight to the speaker, so speech starts
# before the whole sentence is synthesized. Pass a different name to greet
# someone else:
#   ./greet.sh
#   ./greet.sh Nicole
#
# To hear the same greeting from the classic engines for comparison:
#   ./greet.sh --espeak
#   ./greet.sh --festival

set -euo pipefail
VOICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/voices"
VOICE="en_US-lessac-medium"

ENGINE="piper"
NAME="Dhanushikka"
for arg in "$@"; do
  case "$arg" in
    --espeak)   ENGINE="espeak" ;;
    --festival) ENGINE="festival" ;;
    *)          NAME="$arg" ;;
  esac
done

HOUR=$(date +%H)
if   [ "$HOUR" -lt 12 ]; then TOD="morning"
elif [ "$HOUR" -lt 18 ]; then TOD="afternoon"
else                          TOD="evening"; fi

GREETING="Good $TOD, $NAME. Welcome back. It's $(date +%-I:%M %p) and your Pi is ready."

case "$ENGINE" in
  piper)
    python3 -m piper --model "$VOICE" --data-dir "$VOICES_DIR" --output-raw -- "$GREETING" \
      | aplay -q -r 22050 -f S16_LE -t raw -
    ;;
  espeak)
    espeak-ng -s 150 "$GREETING"
    ;;
  festival)
    echo "$GREETING" | festival --tts
    ;;
esac
