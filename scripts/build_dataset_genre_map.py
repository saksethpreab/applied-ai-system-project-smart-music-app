"""
scripts/build_dataset_genre_map.py — Generate data/dataset_genre_map.json.

Maps each unique track_genre value in data/dataset.csv to a value in
KNOWN_GENRES (from src/prompts.py) using Claude. Run once.

    python scripts/build_dataset_genre_map.py

Requires ANTHROPIC_API_KEY in .env or environment.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from prompts import KNOWN_GENRES  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "dataset_genre_map.json"
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a music genre taxonomy expert. Given a list of dataset genre tags "
    "and a fixed vocabulary of allowed genre values, map each dataset tag to the "
    "single closest allowed value. Respond with ONLY a valid JSON object — no "
    "markdown fences, no prose. Every key must be a dataset tag from the input "
    "list. Every value must be exactly one string from the allowed vocabulary list."
)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "sk-ant-YOUR_KEY_HERE":
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, usecols=["track_genre"])
    dataset_genres = sorted(df["track_genre"].dropna().unique().tolist())
    print(f"Found {len(dataset_genres)} unique dataset genres.")

    client = anthropic.Anthropic(api_key=api_key)
    user_message = (
        f"Dataset genre tags to map:\n{json.dumps(dataset_genres, indent=2)}\n\n"
        f"Allowed vocabulary (map to exactly one of these):\n"
        f"{json.dumps(KNOWN_GENRES, indent=2)}"
    )

    print(f"Calling Claude ({MODEL}) to generate genre map...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
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

    missing = sorted(set(dataset_genres) - set(mapping.keys()))
    invalid = {k: v for k, v in mapping.items() if v not in KNOWN_GENRES}
    if missing:
        print(
            f"WARNING: {len(missing)} dataset genres missing from response. "
            f"First 10: {missing[:10]}",
            file=sys.stderr,
        )
    if invalid:
        print(
            f"ERROR: {len(invalid)} mappings have values not in KNOWN_GENRES.",
            file=sys.stderr,
        )
        for k, v in list(invalid.items())[:10]:
            print(f"  {k!r} -> {v!r}", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"Written {len(mapping)} mappings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
