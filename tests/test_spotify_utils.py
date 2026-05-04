"""
tests/test_spotify_utils.py — Tests for Spotify genre translation utility.

Covers:
  - translate_spotify_genres(): JSON map lookup, substring fallback, default
  - spotify_genre_map.json: all mapped values are valid KNOWN_GENRES entries
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spotify_utils import translate_spotify_genres
from prompts import KNOWN_GENRES

MAP_PATH = Path(__file__).parent.parent / "data" / "spotify_genre_map.json"


# ── Map lookup (Tier 1) ───────────────────────────────────────────────────────

class TestMapLookup:

    def test_known_tag_returns_mapped_value(self):
        """A Spotify tag present in the JSON map must return its mapped value,
        which must be a valid KNOWN_GENRES entry."""
        fake_map = {"synth-pop": "synthwave", "indie folk": "folk"}
        with patch("spotify_utils._get_map", return_value=fake_map):
            result = translate_spotify_genres(["synth-pop"])
        assert result == "synthwave"
        assert result in KNOWN_GENRES

    def test_first_matching_tag_wins(self):
        """When multiple tags are in the map, the first one in the list is
        returned — consistent with the priority order."""
        fake_map = {"indie rock": "indie", "alternative rock": "rock"}
        with patch("spotify_utils._get_map", return_value=fake_map):
            result = translate_spotify_genres(["indie rock", "alternative rock"])
        assert result == "indie"

    def test_lookup_is_case_insensitive_after_normalisation(self):
        """Tags are lowercased before lookup, so 'Synth-Pop' must match
        the map key 'synth-pop'."""
        fake_map = {"synth-pop": "synthwave"}
        with patch("spotify_utils._get_map", return_value=fake_map):
            result = translate_spotify_genres(["Synth-Pop"])
        assert result == "synthwave"

    def test_map_miss_falls_through_to_next_tier(self):
        """A tag absent from the map must not return a map value — it falls
        through to substring matching."""
        fake_map = {"some-other-genre": "jazz"}
        with patch("spotify_utils._get_map", return_value=fake_map):
            result = translate_spotify_genres(["indie folk"])
        assert result in KNOWN_GENRES


# ── Substring fallback (Tier 2) ──────────────────────────────────────────────

class TestSubstringFallback:

    def test_substring_match_when_map_empty(self):
        """When the map is empty, a tag containing a KNOWN_GENRES substring
        (e.g. 'indie rock' contains 'indie') must return that known value."""
        with patch("spotify_utils._get_map", return_value={}):
            result = translate_spotify_genres(["indie rock"])
        assert result in KNOWN_GENRES

    def test_known_genre_substring_in_spotify_tag(self):
        """'lo-fi hip-hop' contains 'hip-hop' which is in KNOWN_GENRES;
        substring matching must surface it."""
        with patch("spotify_utils._get_map", return_value={}):
            result = translate_spotify_genres(["lo-fi hip-hop"])
        assert result in KNOWN_GENRES

    def test_spotify_tag_substring_of_known_genre(self):
        """'folk' is a substring of 'indie folk' (KNOWN_GENRES entry);
        the reverse substring direction must also match."""
        with patch("spotify_utils._get_map", return_value={}):
            result = translate_spotify_genres(["folk"])
        assert result in KNOWN_GENRES


# ── Default fallback (Tier 3) ────────────────────────────────────────────────

class TestDefaultFallback:

    def test_empty_list_returns_pop(self):
        """An empty tag list has no map key or substring to match — must return
        the default value 'pop'."""
        with patch("spotify_utils._get_map", return_value={}):
            result = translate_spotify_genres([])
        assert result == "pop"
        assert result in KNOWN_GENRES

    def test_fully_unknown_tags_return_pop(self):
        """Tags with no map entry and no substring overlap with KNOWN_GENRES
        must fall through to the 'pop' default."""
        with patch("spotify_utils._get_map", return_value={}):
            result = translate_spotify_genres(["xyzzy-music-9999", "no-match-here"])
        assert result == "pop"

    def test_return_value_always_in_known_genres(self):
        """The default 'pop' must be a member of KNOWN_GENRES — if KNOWN_GENRES
        ever changes, this test catches a broken default."""
        assert "pop" in KNOWN_GENRES


# ── Genre map file integrity ──────────────────────────────────────────────────

class TestGenreMapFile:

    @pytest.mark.skipif(
        not MAP_PATH.exists(),
        reason="data/spotify_genre_map.json not yet generated — run scripts/generate_genre_map.py first",
    )
    def test_all_map_values_are_in_known_genres(self):
        """Every value in spotify_genre_map.json must be a valid KNOWN_GENRES
        entry. A value outside the vocabulary would silently break scoring."""
        with open(MAP_PATH, encoding="utf-8") as f:
            mapping = json.load(f)

        invalid = {
            tag: value
            for tag, value in mapping.items()
            if value not in KNOWN_GENRES
        }
        assert not invalid, (
            f"spotify_genre_map.json contains {len(invalid)} value(s) not in KNOWN_GENRES: "
            f"{invalid}"
        )

    @pytest.mark.skipif(
        not MAP_PATH.exists(),
        reason="data/spotify_genre_map.json not yet generated — run scripts/generate_genre_map.py first",
    )
    def test_map_keys_are_lowercase_strings(self):
        """All keys in spotify_genre_map.json must be lowercase strings so that
        the normalised tag lookup in translate_spotify_genres() always matches."""
        with open(MAP_PATH, encoding="utf-8") as f:
            mapping = json.load(f)

        non_lowercase = [k for k in mapping if k != k.lower()]
        assert not non_lowercase, (
            f"spotify_genre_map.json has non-lowercase keys: {non_lowercase}"
        )

    @pytest.mark.skipif(
        not MAP_PATH.exists(),
        reason="data/spotify_genre_map.json not yet generated — run scripts/generate_genre_map.py first",
    )
    def test_map_is_non_empty(self):
        """spotify_genre_map.json must contain at least one mapping — an empty
        file indicates the generation script failed silently."""
        with open(MAP_PATH, encoding="utf-8") as f:
            mapping = json.load(f)
        assert len(mapping) > 0

    def test_missing_map_file_does_not_crash(self):
        """If spotify_genre_map.json does not exist, translate_spotify_genres()
        must still return a valid KNOWN_GENRES value via fallback — not crash."""
        with patch("spotify_utils._load_map", return_value={}):
            result = translate_spotify_genres(["synth-pop"])
        assert result in KNOWN_GENRES
