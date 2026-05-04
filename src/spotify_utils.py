"""
src/spotify_utils.py — Translate Spotify artist genre tags to KNOWN_GENRES values.

Usage:
    from spotify_utils import translate_spotify_genres
    genre = translate_spotify_genres(artist["genres"])  # artist["genres"] is list[str]

Lookup order:
  1. data/spotify_genre_map.json  (generated once by scripts/generate_genre_map.py)
  2. Substring matching against KNOWN_GENRES
  3. Default: "pop"
"""

import json
from pathlib import Path

_MAP_PATH = Path(__file__).parent.parent / "data" / "spotify_genre_map.json"
_MAP: dict | None = None


def _load_map() -> dict:
    try:
        with open(_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _get_map() -> dict:
    global _MAP
    if _MAP is None:
        _MAP = _load_map()
    return _MAP


def translate_spotify_genres(spotify_genres: list[str]) -> str:
    """Return the single best-matching KNOWN_GENRES value for a list of Spotify genre tags."""
    from prompts import KNOWN_GENRES

    genre_map = _get_map()

    for tag in spotify_genres:
        normalized = tag.strip().lower()
        if normalized in genre_map:
            return genre_map[normalized]

    for tag in spotify_genres:
        normalized = tag.strip().lower()
        for known in KNOWN_GENRES:
            if known in normalized or normalized in known:
                return known

    return "pop"
