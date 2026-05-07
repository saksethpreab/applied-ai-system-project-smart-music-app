"""
scripts/prepare_dataset.py — Transform data/dataset.csv into data/songs_full.csv.

Steps:
  1. Drop rows with popularity == 0.
  2. Dedupe by track_id (keep highest-popularity occurrence).
  3. Apply data/dataset_genre_map.json to remap track_genre -> KNOWN_GENRES.
  4. Derive mood from (valence, energy) via 2x3 grid:
        E >= 0.7   ->  energetic (V>=0.5)  / intense (V<0.5)
        E in [0.3, 0.7) ->  happy (V>=0.5)  / moody   (V<0.5)
        E < 0.3    ->  relaxed   (V>=0.5)  / sad     (V<0.5)
  5. Write CSV with the existing schema plus track_id and popularity.

Run after scripts/build_dataset_genre_map.py:
    python scripts/prepare_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).parent.parent
DATASET_PATH   = PROJECT_ROOT / "data" / "dataset.csv"
GENRE_MAP_PATH = PROJECT_ROOT / "data" / "dataset_genre_map.json"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "songs_full.csv"


def derive_moods(valence: pd.Series, energy: pd.Series) -> pd.Series:
    """2x3 grid mapping (valence, energy) -> canonical mood from KNOWN_MOODS.

    Thresholds: V at 0.5; E at 0.3 and 0.7.
    """
    high_v = valence >= 0.5
    conditions = [
        (energy >= 0.7) &  high_v,
        (energy >= 0.7) & ~high_v,
        (energy >= 0.3) &  high_v,
        (energy >= 0.3) & ~high_v,
        high_v,                       # low E, high V
    ]
    choices = ["energetic", "intense", "happy", "moody", "relaxed"]
    return pd.Series(np.select(conditions, choices, default="sad"), index=valence.index)


def main() -> None:
    if not GENRE_MAP_PATH.exists():
        print(
            f"ERROR: {GENRE_MAP_PATH} not found. "
            f"Run scripts/build_dataset_genre_map.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(GENRE_MAP_PATH, encoding="utf-8") as f:
        genre_map = json.load(f)

    print(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    initial = len(df)
    print(f"Initial rows: {initial:,}")

    df = df[df["popularity"] > 0]
    print(f"After popularity > 0 filter: {len(df):,} ({initial - len(df):,} dropped)")

    df = (
        df.sort_values("popularity", ascending=False)
          .drop_duplicates(subset=["track_id"], keep="first")
    )
    print(f"After dedupe by track_id: {len(df):,}")

    df["genre"] = df["track_genre"].map(genre_map)
    unmapped = int(df["genre"].isna().sum())
    if unmapped:
        unmapped_tags = sorted(df.loc[df["genre"].isna(), "track_genre"].unique().tolist())
        print(
            f"WARNING: {unmapped:,} rows had unmapped genres ({len(unmapped_tags)} tags). "
            f"Dropping. Tags: {unmapped_tags[:10]}",
            file=sys.stderr,
        )
        df = df.dropna(subset=["genre"])

    df["mood"] = derive_moods(df["valence"], df["energy"])

    df = df.reset_index(drop=True)
    df["id"] = df.index

    out = pd.DataFrame({
        "id":           df["id"],
        "title":        df["track_name"],
        "artist":       df["artists"].str.replace(";", ", ", regex=False),
        "genre":        df["genre"],
        "mood":         df["mood"],
        "energy":       df["energy"],
        "tempo_bpm":    df["tempo"],
        "valence":      df["valence"],
        "danceability": df["danceability"],
        "acousticness": df["acousticness"],
        "track_id":     df["track_id"],
        "popularity":   df["popularity"],
    })

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWritten {len(out):,} rows to {OUTPUT_PATH}")
    print("\nMood distribution:")
    print(out["mood"].value_counts().to_string())
    print("\nGenre distribution (top 20):")
    print(out["genre"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
