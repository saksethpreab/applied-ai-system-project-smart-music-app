"""
scripts/prepare_dataset.py — Transform data/dataset.csv into data/songs_full.csv.

Steps:
  1. Drop rows with popularity == 0.
  2. Dedupe by track_id (keep highest-popularity occurrence).
  3. Apply data/dataset_genre_map.json to remap track_genre -> KNOWN_GENRES.
  4. Derive mood from (valence, energy) via 2x3 grid:
        E >= 0.7   ->  energetic (V>=0.5)  / intense (V<0.5)
        E in [0.4, 0.7) ->  happy (V>=0.5)  / moody   (V<0.5)
        E < 0.4    ->  relaxed   (V>=0.5)  / sad     (V<0.5)
  5. Detect language from "track_name + artists" via lingua-language-detector
     (pure Python, 75 languages). Low-confidence detections -> "unknown".
  6. Write CSV with the existing schema plus track_id, popularity, language.

Run after scripts/build_dataset_genre_map.py:
    python scripts/prepare_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lingua import LanguageDetectorBuilder, IsoCode639_1

PROJECT_ROOT   = Path(__file__).parent.parent
DATASET_PATH   = PROJECT_ROOT / "data" / "dataset.csv"
GENRE_MAP_PATH = PROJECT_ROOT / "data" / "dataset_genre_map.json"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "songs_full.csv"

LANG_MIN_CHARS = 4  # texts shorter than this -> "unknown" without prediction

# Candidate languages for detection. Restricting the candidate set physically
# prevents lingua from false-positiving short text into rare languages
# (la, eo, cy, yo, sw, sn, ts, ...) that have effectively zero presence in a
# modern Spotify catalog. Anything not in this list is detected as "unknown".
_CANDIDATE_LANGUAGES = [
    IsoCode639_1.EN, IsoCode639_1.ES, IsoCode639_1.PT, IsoCode639_1.FR,
    IsoCode639_1.DE, IsoCode639_1.IT, IsoCode639_1.NL, IsoCode639_1.SV,
    IsoCode639_1.DA, IsoCode639_1.NB, IsoCode639_1.FI, IsoCode639_1.PL,
    IsoCode639_1.CS, IsoCode639_1.HU, IsoCode639_1.RO, IsoCode639_1.RU,
    IsoCode639_1.UK, IsoCode639_1.TR, IsoCode639_1.AR, IsoCode639_1.HE,
    IsoCode639_1.HI, IsoCode639_1.BN, IsoCode639_1.ID, IsoCode639_1.VI,
    IsoCode639_1.TH, IsoCode639_1.JA, IsoCode639_1.KO, IsoCode639_1.ZH,
    IsoCode639_1.EL,
]


def _build_lingua_detector():
    # High-accuracy mode (default), restricted to _CANDIDATE_LANGUAGES so lingua
    # cannot false-positive into rare languages we don't expect. With a curated
    # candidate set, the relative-distance filter is unnecessary and would
    # over-reject — high-accuracy mode's internal logic already handles
    # genuinely unparseable input by returning None.
    return (
        LanguageDetectorBuilder
        .from_iso_codes_639_1(*_CANDIDATE_LANGUAGES)
        .build()
    )


def detect_languages(detector, texts: list[str]) -> list[str]:
    """Predict ISO-639-1 codes for each text. Short or unconfident -> 'unknown'.

    Uses lingua's parallel batch detector for speed on large catalogs.
    """
    cleaned = [(t or "").replace("\n", " ").strip() for t in texts]

    # Skip detection for short texts; they go straight to "unknown".
    eligible_idx = [i for i, t in enumerate(cleaned) if len(t) >= LANG_MIN_CHARS]
    eligible_texts = [cleaned[i] for i in eligible_idx]

    detections = (
        detector.detect_languages_in_parallel_of(eligible_texts)
        if eligible_texts else []
    )

    result = ["unknown"] * len(cleaned)
    for idx, lang in zip(eligible_idx, detections):
        if lang is not None:
            result[idx] = lang.iso_code_639_1.name.lower()
    return result


def derive_moods(valence: pd.Series, energy: pd.Series) -> pd.Series:
    """2x3 grid mapping (valence, energy) -> canonical mood from KNOWN_MOODS.

    Thresholds: V at 0.5; E at 0.4 and 0.7.
    """
    high_v = valence >= 0.5
    conditions = [
        (energy >= 0.7) &  high_v,
        (energy >= 0.7) & ~high_v,
        (energy >= 0.4) &  high_v,
        (energy >= 0.4) & ~high_v,
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

    print("Detecting languages with lingua...")
    detector = _build_lingua_detector()
    artists_clean = df["artists"].fillna("").str.replace(";", ", ", regex=False)
    detection_text = (df["track_name"].fillna("") + " " + artists_clean).tolist()
    df["language"] = detect_languages(detector, detection_text)

    df = df.reset_index(drop=True)
    df["id"] = df.index

    out = pd.DataFrame({
        "id":           df["id"],
        "title":        df["track_name"],
        "artist":       artists_clean.values,
        "genre":        df["genre"],
        "mood":         df["mood"],
        "energy":       df["energy"],
        "tempo_bpm":    df["tempo"],
        "valence":      df["valence"],
        "danceability": df["danceability"],
        "acousticness": df["acousticness"],
        "track_id":     df["track_id"],
        "popularity":   df["popularity"],
        "language":     df["language"],
    })

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWritten {len(out):,} rows to {OUTPUT_PATH}")
    print("\nMood distribution:")
    print(out["mood"].value_counts().to_string())
    print("\nGenre distribution (top 20):")
    print(out["genre"].value_counts().head(20).to_string())
    print("\nLanguage distribution (top 20):")
    print(out["language"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
