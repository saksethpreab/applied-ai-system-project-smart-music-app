"""
scripts/generate_genre_map.py — One-time script to generate data/spotify_genre_map.json.

Uses Claude to map common Spotify artist genre tags to the project's KNOWN_GENRES vocabulary.
Run once from the project root:

    python scripts/generate_genre_map.py

Requires ANTHROPIC_API_KEY in .env or environment.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "spotify_genre_map.json"
MODEL = "claude-haiku-4-5-20251001"

KNOWN_GENRES = [
    "pop", "indie pop", "disco", "latin", "reggae", "ballad", "k-pop",
    "lofi", "ambient", "electronic", "synthwave", "techno", "house", "trance", "chillwave",
    "indie", "folk", "dream pop", "shoegaze",
    "rock", "metal", "punk", "grunge", "emo", "christian rock",
    "hip-hop", "r&b", "soul", "funk", "trap",
    "jazz", "blues", "classical", "country", "gospel", "bossa nova", "hymn", "spiritual", "worship",
    "afrobeats", "flamenco", "celtic",
]

SPOTIFY_GENRES = [
    "synth-pop", "electropop", "art pop", "bedroom pop", "canadian pop", "australian pop",
    "dance pop", "pop rock", "pop soul", "indie pop rock", "alt pop", "hyperpop",
    "alternative rock", "indie rock", "post-rock", "math rock", "prog rock", "classic rock",
    "hard rock", "soft rock", "southern rock", "garage rock", "noise rock", "post-punk",
    "new wave", "power pop", "britpop", "shoegaze revival", "dream pop revival",
    "death metal", "black metal", "heavy metal", "thrash metal", "doom metal", "nu-metal",
    "metalcore", "deathcore", "post-metal",
    "melodic rap", "conscious hip-hop", "trap rap", "cloud rap", "lo-fi hip-hop", "boom bap",
    "drill", "phonk", "hypnotic trap", "southern hip-hop", "east coast hip-hop",
    "contemporary r&b", "neo soul", "quiet storm", "new jack swing",
    "deep house", "progressive house", "tech house", "tropical house", "future house",
    "minimal techno", "detroit techno", "uk techno", "hard techno",
    "drum and bass", "jungle", "breakbeat", "dubstep", "future bass", "bass music",
    "ambient pop", "chillhop", "downtempo", "trip-hop", "lo-fi beats", "vapor soul",
    "vaporwave", "chillwave revival", "new age", "meditation",
    "indie folk", "folk rock", "freak folk", "contemporary folk", "americana",
    "bluegrass", "country pop", "country rock", "outlaw country",
    "bossa nova jazz", "smooth jazz", "contemporary jazz", "jazz fusion",
    "latin pop", "reggaeton", "latin trap", "bachata", "salsa", "cumbia",
    "afropop", "afrobeats fusion", "highlife",
    "flamenco pop", "celtic folk", "world music",
    "worship music", "ccm", "gospel rap", "christian metal",
    "j-pop", "j-rock", "city pop", "anime",
    "uk garage", "grime", "jersey club",
]

SYSTEM_PROMPT = (
    "You are a music genre taxonomy expert. Given a list of Spotify artist genre tags "
    "and a fixed vocabulary of allowed genre values, map each Spotify tag to the single "
    "closest allowed value. Respond with ONLY a valid JSON object — no markdown fences, "
    "no prose. Every key must be a Spotify tag from the input list. Every value must be "
    "exactly one string from the allowed vocabulary list."
)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "sk-ant-YOUR_KEY_HERE":
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Spotify genre tags to map:\n{json.dumps(SPOTIFY_GENRES, indent=2)}\n\n"
        f"Allowed vocabulary (map to exactly one of these):\n{json.dumps(KNOWN_GENRES, indent=2)}"
    )

    print(f"Calling Claude ({MODEL}) to generate genre map...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        print("ERROR: Could not parse Claude's response as JSON.", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(raw, file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"Written {len(mapping)} mappings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
