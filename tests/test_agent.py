"""
tests/test_agent.py — Mocked tests for the agentic pipeline in src/agent.py.

All tests patch the Anthropic client so no real API key is required.
The existing test_recommender.py is unaffected.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import (
    analyze_prompt,
    draft_playlist,
    self_correct,
    _compute_metrics,
    run_agent,
    _ANALYZE_FALLBACK_PREFS,
)
from recommender import load_songs

# ── Fixtures ──────────────────────────────────────────────────────────────────

SONGS_PATH = Path(__file__).parent.parent / "data" / "songs.csv"


@pytest.fixture
def songs():
    return load_songs(str(SONGS_PATH))


@pytest.fixture
def mock_client():
    return MagicMock()


def _make_analyze_response(
    genre="lofi", mood="chill", current_genre="lofi", current_mood="chill",
    energy=0.25, valence=0.50, dance=0.35, acoustic=0.80, tempo=85.0,
    reasoning="Chill studying vibes.", confidence=0.88,
    languages=None, artists=None,
):
    payload = {
        "user_prefs": {
            "genre":               genre,
            "mood":                mood,
            "current_genre":       current_genre,
            "current_mood":        current_mood,
            "target_energy":       energy,
            "target_valence":      valence,
            "target_danceability": dance,
            "target_acousticness": acoustic,
            "target_tempo":        tempo,
        },
        "reasoning":  reasoning,
        "confidence": confidence,
        "languages":  languages,
        "artists":    artists,
    }
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    msg.usage   = MagicMock(output_tokens=200)
    return msg


def _make_correct_response(evaluations, approved=True, reasoning="Looks good."):
    payload = {
        "evaluations":       evaluations,
        "approved":          approved,
        "overall_reasoning": reasoning,
    }
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    msg.usage   = MagicMock(output_tokens=300)
    return msg


# ── analyze_prompt() tests ────────────────────────────────────────────────────

class TestAnalyzePrompt:

    def test_valid_prompt_returns_all_keys(self, mock_client):
        """LLM returns valid JSON — result must contain all 9 required user_prefs keys,
        a non-empty reasoning string, and a confidence value in [0.0, 1.0]."""
        mock_client.messages.create.return_value = _make_analyze_response()
        prefs, reasoning, confidence, _, _, _ = analyze_prompt("chill studying music", mock_client)

        assert set(prefs.keys()) == {
            "genre", "mood", "current_genre", "current_mood",
            "target_energy", "target_valence", "target_danceability",
            "target_acousticness", "target_tempo",
        }
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reasoning, str) and len(reasoning) > 0

    def test_floats_are_in_range(self, mock_client):
        """All numeric preference fields returned by the LLM must stay within
        their valid ranges: energy/valence/danceability/acousticness in [0, 1]
        and tempo in [50, 200] BPM."""
        mock_client.messages.create.return_value = _make_analyze_response(
            energy=0.25, valence=0.50, dance=0.35, acoustic=0.80, tempo=85.0
        )
        prefs, _, _, _, _, _ = analyze_prompt("any prompt", mock_client)

        for key in ("target_energy", "target_valence", "target_danceability", "target_acousticness"):
            assert 0.0 <= prefs[key] <= 1.0, f"{key} out of range"
        assert 50.0 <= prefs["target_tempo"] <= 200.0

    def test_out_of_range_floats_are_clamped(self, mock_client):
        """If the LLM returns floats outside the valid range (e.g. energy=1.9,
        tempo=300), they must be silently clamped to the boundary values
        (1.0 and 200.0 respectively)."""
        mock_client.messages.create.return_value = _make_analyze_response(
            energy=1.9, tempo=300.0
        )
        prefs, _, _, _, _, _ = analyze_prompt("any prompt", mock_client)

        assert prefs["target_energy"] == 1.0
        assert prefs["target_tempo"] == 200.0

    def test_unknown_genre_is_substituted(self, mock_client):
        """If the LLM returns a genre not in KNOWN_GENRES (e.g. 'chillwave'),
        the function must substitute it with a valid known genre rather than
        passing an unrecognised value to the rule engine."""
        mock_client.messages.create.return_value = _make_analyze_response(
            genre="chillwave", current_genre="chillwave"
        )
        prefs, _, _, _, _, _ = analyze_prompt("any prompt", mock_client)

        from prompts import KNOWN_GENRES
        assert prefs["genre"] in KNOWN_GENRES
        assert prefs["current_genre"] in KNOWN_GENRES

    def test_json_retry_succeeds(self, mock_client):
        """If the first LLM response is not valid JSON, the function retries
        once with a stricter instruction. The second valid response must be
        parsed successfully and returned as if no failure occurred."""
        bad_msg  = MagicMock()
        bad_msg.content = [MagicMock(text="not valid json!!!")]
        bad_msg.usage   = MagicMock(output_tokens=5)
        good_msg = _make_analyze_response()

        mock_client.messages.create.side_effect = [bad_msg, good_msg]
        prefs, reasoning, confidence, _, _, _ = analyze_prompt("any prompt", mock_client)

        assert confidence == 0.88

    def test_double_json_failure_returns_fallback(self, mock_client):
        """If both the initial call and the retry return unparseable JSON,
        the function must return the safe lofi/chill fallback profile with
        confidence=0.0 instead of raising an exception."""
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="not json")]
        bad_msg.usage   = MagicMock(output_tokens=5)

        mock_client.messages.create.side_effect = [bad_msg, bad_msg]
        prefs, _, confidence, _, _, _ = analyze_prompt("???", mock_client)

        assert confidence == 0.0
        assert prefs["genre"] == _ANALYZE_FALLBACK_PREFS["genre"]


# ── draft_playlist() tests ────────────────────────────────────────────────────

class TestDraftPlaylist:

    def test_returns_five_songs(self, songs):
        """The rule engine must return exactly 5 songs when the full catalog
        is available and k defaults to 5."""
        prefs = {
            "genre": "lofi", "mood": "chill",
            "current_genre": "lofi", "current_mood": "chill",
            "target_energy": 0.25, "target_valence": 0.50,
            "target_danceability": 0.35, "target_acousticness": 0.80,
            "target_tempo": 85.0,
        }
        results = draft_playlist(prefs, songs)
        assert len(results) == 5

    def test_scores_in_range(self, songs):
        """Every score produced by the rule engine must lie within [0.0, 1.0];
        this validates the combined categorical + numeric scoring formula."""
        prefs = {
            "genre": "pop", "mood": "happy",
            "current_genre": "pop", "current_mood": "happy",
            "target_energy": 0.85, "target_valence": 0.75,
            "target_danceability": 0.78, "target_acousticness": 0.15,
            "target_tempo": 128.0,
        }
        for _, score, _ in draft_playlist(prefs, songs):
            assert 0.0 <= score <= 1.0

    def test_explanations_non_empty(self, songs):
        """Each recommended song must carry a non-empty explanation string
        so the pipeline can surface the rule engine's reasoning to the user."""
        prefs = {
            "genre": "rock", "mood": "intense",
            "current_genre": "rock", "current_mood": "angry",
            "target_energy": 0.88, "target_valence": 0.25,
            "target_danceability": 0.45, "target_acousticness": 0.10,
            "target_tempo": 125.0,
        }
        for _, _, explanation in draft_playlist(prefs, songs):
            assert isinstance(explanation, str) and len(explanation) > 0

    def test_empty_song_list_returns_empty(self):
        """Passing an empty catalog must return an empty list without raising
        an exception — the rule engine already handles this edge case."""
        prefs = {
            "genre": "pop", "mood": "happy",
            "current_genre": "pop", "current_mood": "happy",
            "target_energy": 0.8, "target_valence": 0.7,
            "target_danceability": 0.7, "target_acousticness": 0.2,
            "target_tempo": 120.0,
        }
        assert draft_playlist(prefs, []) == []


# ── self_correct() tests ──────────────────────────────────────────────────────

class TestSelfCorrect:

    def _prefs(self):
        return {
            "genre": "lofi", "mood": "chill",
            "current_genre": "lofi", "current_mood": "chill",
            "target_energy": 0.25, "target_valence": 0.50,
            "target_danceability": 0.35, "target_acousticness": 0.80,
            "target_tempo": 85.0,
        }

    def _draft(self, songs):
        return draft_playlist(self._prefs(), songs)

    def test_all_approved_returns_draft_unchanged(self, mock_client, songs):
        """When the LLM marks every song as keep=True, the final playlist must
        be identical to the draft — same songs, same order, corrections_made=0."""
        draft = self._draft(songs)
        evaluations = [
            {"song_id": s["id"], "title": s["title"], "fit_score": 0.9,
             "fit_reason": "Good fit.", "keep": True}
            for s, _, _ in draft
        ]
        mock_client.messages.create.return_value = _make_correct_response(
            evaluations, approved=True
        )
        final, corrections = self_correct("chill music", self._prefs(), draft, songs, mock_client)

        assert corrections == 0
        assert len(final) == len(draft)
        orig_ids = {s["id"] for s, _, _ in draft}
        final_ids = {s["id"] for s, _, _ in final}
        assert orig_ids == final_ids

    def test_one_rejection_replaces_song(self, mock_client, songs):
        """When the LLM marks one song as keep=False, that song must be absent
        from the final playlist and corrections_made must equal 1."""
        draft = self._draft(songs)
        first_song = draft[0][0]
        evaluations = [
            {"song_id": first_song["id"], "title": first_song["title"],
             "fit_score": 0.2, "fit_reason": "Wrong vibe.", "keep": False},
        ] + [
            {"song_id": s["id"], "title": s["title"], "fit_score": 0.85,
             "fit_reason": "Good.", "keep": True}
            for s, _, _ in draft[1:]
        ]
        mock_client.messages.create.return_value = _make_correct_response(
            evaluations, approved=False
        )
        final, corrections = self_correct("chill music", self._prefs(), draft, songs, mock_client)

        assert corrections == 1
        final_ids = {s["id"] for s, _, _ in final}
        assert first_song["id"] not in final_ids

    def test_json_failure_fallback_returns_draft(self, mock_client, songs):
        """If both LLM calls return unparseable JSON, self_correct must fall
        back to returning the original draft unchanged with corrections_made=0,
        trusting the rule engine's output rather than crashing."""
        draft = self._draft(songs)
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="bad json")]
        bad_msg.usage   = MagicMock(output_tokens=5)
        mock_client.messages.create.side_effect = [bad_msg, bad_msg]

        final, corrections = self_correct("chill music", self._prefs(), draft, songs, mock_client)

        assert corrections == 0
        assert len(final) == len(draft)

    def test_replacement_ids_not_in_original_draft(self, mock_client, songs):
        """A replacement song must be drawn from outside the original draft pool —
        it cannot be a song already considered or rejected in this run."""
        draft = self._draft(songs)
        first_song = draft[0][0]
        evaluations = [
            {"song_id": first_song["id"], "title": first_song["title"],
             "fit_score": 0.1, "fit_reason": "Bad.", "keep": False},
        ] + [
            {"song_id": s["id"], "title": s["title"], "fit_score": 0.8,
             "fit_reason": "OK.", "keep": True}
            for s, _, _ in draft[1:]
        ]
        mock_client.messages.create.return_value = _make_correct_response(
            evaluations, approved=False
        )
        final, corrections = self_correct("chill music", self._prefs(), draft, songs, mock_client)

        original_ids = {s["id"] for s, _, _ in draft}
        kept_ids     = {s["id"] for s, _, _ in draft[1:]}
        final_ids    = {s["id"] for s, _, _ in final}

        assert kept_ids.issubset(final_ids)
        assert first_song["id"] not in final_ids
        new_ids = final_ids - kept_ids
        assert new_ids.isdisjoint(original_ids)


# ── _compute_metrics() tests ──────────────────────────────────────────────────

class TestComputeMetrics:

    def _dummy_playlist(self, scores):
        song = {"id": 1, "title": "T", "artist": "A", "genre": "pop", "mood": "happy",
                "energy": 0.8, "tempo_bpm": 120, "valence": 0.7,
                "danceability": 0.7, "acousticness": 0.2}
        return [(song, s, "expl") for s in scores]

    def test_no_corrections(self):
        """When no songs were replaced, draft_score and final_avg_score must be
        equal, correction_count must be 0, and analysis_confidence must match
        the value passed in."""
        draft = self._dummy_playlist([0.8, 0.7, 0.6, 0.5, 0.4])
        metrics = _compute_metrics(0.9, draft, draft, 0)

        assert metrics["correction_count"] == 0
        assert metrics["draft_score"] == metrics["final_avg_score"]
        assert metrics["analysis_confidence"] == 0.9

    def test_with_corrections(self):
        """When songs were replaced with lower-scoring alternatives,
        correction_count must reflect the number of replacements and
        final_avg_score must differ from draft_score."""
        draft = self._dummy_playlist([0.8, 0.7, 0.6, 0.5, 0.4])
        final = self._dummy_playlist([0.7, 0.6, 0.5, 0.4, 0.3])
        metrics = _compute_metrics(0.75, draft, final, 2)

        assert metrics["correction_count"] == 2
        assert metrics["draft_score"] != metrics["final_avg_score"]

    def test_all_keys_present(self):
        """The returned dict must contain exactly the four expected metric keys:
        analysis_confidence, draft_score, correction_count, final_avg_score."""
        draft = self._dummy_playlist([0.5])
        metrics = _compute_metrics(0.5, draft, draft, 0)
        assert set(metrics.keys()) == {
            "analysis_confidence", "draft_score",
            "correction_count", "final_avg_score",
        }


# ── run_agent() integration tests (fully mocked) ──────────────────────────────

class TestRunAgent:

    def _setup_mocks(self, mock_build, mock_client, songs):
        mock_build.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_analyze_response(),
            _make_correct_response([
                {"song_id": s["id"], "title": s["title"], "fit_score": 0.8,
                 "fit_reason": "Good.", "keep": True}
                for s, _, _ in draft_playlist(
                    {"genre": "lofi", "mood": "chill", "current_genre": "lofi",
                     "current_mood": "chill", "target_energy": 0.25,
                     "target_valence": 0.50, "target_danceability": 0.35,
                     "target_acousticness": 0.80, "target_tempo": 85.0},
                    load_songs(str(SONGS_PATH)),
                )
            ], approved=True),
        ]

    def test_returns_full_result_dict(self):
        """The top-level orchestrator must return a dict containing all expected
        top-level keys: user_prompt, analysis, draft_playlist, final_playlist,
        metrics, and recommended_ids."""
        with patch("agent._build_client") as mock_build:
            mock_client = MagicMock()
            self._setup_mocks(mock_build, mock_client, None)
            result = run_agent("chill studying music")

        assert set(result.keys()) == {
            "user_prompt", "analysis", "draft_playlist",
            "final_playlist", "metrics", "recommended_ids",
        }

    def test_analysis_structure(self):
        """The 'analysis' section of the result must contain the structured profile
        (user_prefs), LLM explanation (reasoning), confidence, plus the emotion
        intent and language metadata used by the UI."""
        with patch("agent._build_client") as mock_build:
            mock_client = MagicMock()
            self._setup_mocks(mock_build, mock_client, None)
            result = run_agent("chill studying music")

        assert set(result["analysis"].keys()) == {
            "user_prefs", "reasoning", "confidence",
            "emotion_intent", "detected_languages", "applied_languages",
            "detected_artists", "applied_artists",
        }

    def test_metrics_structure_and_types(self):
        """All four metric values must be present and have the correct Python types:
        floats for scores and confidence, int for correction_count."""
        with patch("agent._build_client") as mock_build:
            mock_client = MagicMock()
            self._setup_mocks(mock_build, mock_client, None)
            result = run_agent("chill studying music")

        m = result["metrics"]
        assert isinstance(m["analysis_confidence"], float)
        assert isinstance(m["draft_score"], float)
        assert isinstance(m["correction_count"], int)
        assert isinstance(m["final_avg_score"], float)

    def test_final_playlist_has_five_songs(self):
        """The final playlist returned by the full pipeline must contain
        exactly 5 songs when all draft songs are approved and the catalog
        has sufficient entries."""
        with patch("agent._build_client") as mock_build:
            mock_client = MagicMock()
            self._setup_mocks(mock_build, mock_client, None)
            result = run_agent("chill studying music")

        assert len(result["final_playlist"]) == 5

    def test_missing_api_key_raises_environment_error(self):
        """If ANTHROPIC_API_KEY is absent or empty, the agent must raise an
        EnvironmentError immediately before making any API calls."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            with pytest.raises(EnvironmentError):
                run_agent("anything")


# ── Artist detection + filtering tests ────────────────────────────────────────

class TestArtistFilter:
    """Tests for the artist detection + hard-filter capability added in the
    ANALYZE step. The filter is a case-insensitive substring match with OR
    logic across multiple artist names; playlist length adapts to the
    filtered pool (no padding from non-matching artists)."""

    # The test fixture data/songs.csv has no `language` column, so language
    # filtering would empty the catalog. All these tests use languages=None.
    # We patch DATA_PATH so the pipeline reads the small test fixture rather
    # than data/songs_full.csv.

    def _approve_all_correct_response(self):
        """A CORRECT response with no per-song evaluations approves the entire
        draft by default (eval_by_id.get(song['id'], {}).get('keep', True))."""
        return _make_correct_response([], approved=True)

    def test_analyze_extracts_artists(self, mock_client):
        """The LLM's `artists` field must surface as detected_artists, preserving
        the LLM's canonicalization (e.g. 'John Mayer' rather than 'john mayer')."""
        mock_client.messages.create.return_value = _make_analyze_response(
            artists=["John Mayer"]
        )
        _, _, _, _, _, detected_artists = analyze_prompt("john mayer songs", mock_client)
        assert detected_artists == ["John Mayer"]

    def test_analyze_returns_none_when_no_artist(self, mock_client):
        """When the LLM returns null for artists, detected_artists must be None
        so downstream code skips the filter rather than erroring."""
        mock_client.messages.create.return_value = _make_analyze_response(artists=None)
        _, _, _, _, _, detected_artists = analyze_prompt("workout vibes", mock_client)
        assert detected_artists is None

    def test_analyze_returns_none_for_empty_artist_list(self, mock_client):
        """An empty artist list from the LLM must coalesce to None."""
        mock_client.messages.create.return_value = _make_analyze_response(artists=[])
        _, _, _, _, _, detected_artists = analyze_prompt("workout vibes", mock_client)
        assert detected_artists is None

    def test_analyze_caps_artist_count(self, mock_client):
        """The parser must cap the number of artists at 5 to bound filter cost,
        even if the LLM returns more."""
        mock_client.messages.create.return_value = _make_analyze_response(
            artists=["A", "B", "C", "D", "E", "F", "G"]
        )
        _, _, _, _, _, detected_artists = analyze_prompt("any prompt", mock_client)
        assert len(detected_artists) == 5

    def test_artist_filter_keeps_only_matching_songs(self):
        """When run_agent detects an artist, every song in the final playlist
        must contain that artist name (case-insensitive) in its artist field."""
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    genre="pop", mood="melancholic",
                    current_genre="pop", current_mood="sad",
                    energy=0.40, valence=0.40, dance=0.50, acoustic=0.55, tempo=100.0,
                    artists=["John Mayer"],
                ),
                self._approve_all_correct_response(),
            ]
            result = run_agent("sob rock john mayer songs")

        assert len(result["final_playlist"]) > 0
        for song in result["final_playlist"]:
            assert "john mayer" in song["artist"].lower()

    def test_artist_filter_substring_match_includes_collabs(self):
        """Substring match must include collaborations where the artist name
        appears as one entry in a comma-separated list (e.g. 'Taylor Swift ft.
        Bon Iver' should match a Taylor Swift filter)."""
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    genre="pop", mood="sad",
                    current_genre="pop", current_mood="sad",
                    energy=0.40, valence=0.30, dance=0.50, acoustic=0.50, tempo=100.0,
                    artists=["Taylor Swift"],
                ),
                self._approve_all_correct_response(),
            ]
            result = run_agent("Taylor Swift breakup tracks")

        for song in result["final_playlist"]:
            assert "taylor swift" in song["artist"].lower()
        # Confirm the collab row is reachable: at least one final entry should
        # come from the post-filter pool that includes 'Taylor Swift ft. Bon Iver'
        # (id 199 in songs.csv).
        all_artist_strings = [s["artist"].lower() for s in result["final_playlist"]]
        assert any("taylor swift" in a for a in all_artist_strings)

    def test_artist_filter_empty_pool_raises(self):
        """When the LLM returns an artist not present in the catalog, the
        pipeline must raise RuntimeError (no silent fallback to other artists)."""
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    artists=["Some Random Indie Band That Does Not Exist"]
                ),
                self._approve_all_correct_response(),
            ]
            with pytest.raises(RuntimeError):
                run_agent("songs by Some Random Indie Band That Does Not Exist")

    def test_no_artist_skips_filter(self):
        """When the LLM returns artists=null, no filter is applied — the full
        catalog is available to the rule engine and a 5-song playlist results."""
        prefs = {
            "genre": "lofi", "mood": "chill",
            "current_genre": "lofi", "current_mood": "chill",
            "target_energy": 0.25, "target_valence": 0.50,
            "target_danceability": 0.35, "target_acousticness": 0.80,
            "target_tempo": 85.0,
        }
        full_draft = draft_playlist(prefs, load_songs(str(SONGS_PATH)))

        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(artists=None),
                self._approve_all_correct_response(),
            ]
            result = run_agent("chill studying music")

        assert len(result["final_playlist"]) == 5
        # No artist constraint => same set of recommended IDs as the unfiltered draft
        assert {s["id"] for s in result["final_playlist"]} == {s["id"] for s, _, _ in full_draft}

    def test_multi_artist_or_logic(self):
        """A list of artists must apply OR logic — every song in the final
        playlist must match at least one of the named artists."""
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    genre="pop", mood="sad",
                    current_genre="pop", current_mood="sad",
                    energy=0.40, valence=0.30, dance=0.50, acoustic=0.50, tempo=100.0,
                    artists=["John Mayer", "Taylor Swift"],
                ),
                self._approve_all_correct_response(),
            ]
            result = run_agent("John Mayer and Taylor Swift sad songs")

        for song in result["final_playlist"]:
            artist_lower = song["artist"].lower()
            assert "john mayer" in artist_lower or "taylor swift" in artist_lower

    def test_artist_filter_case_insensitive(self):
        """Lowercase artist names from the LLM must still match capitalized
        catalog entries — the filter applies .lower() to both sides."""
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    genre="pop", mood="melancholic",
                    current_genre="pop", current_mood="sad",
                    energy=0.40, valence=0.40, dance=0.50, acoustic=0.55, tempo=100.0,
                    artists=["john mayer"],
                ),
                self._approve_all_correct_response(),
            ]
            result = run_agent("john mayer songs")

        assert len(result["final_playlist"]) > 0
        for song in result["final_playlist"]:
            assert "john mayer" in song["artist"].lower()

    def test_analysis_dict_includes_artist_metadata(self):
        """The result dict's analysis section must surface both detected_artists
        and applied_artists, parallel to the language metadata."""
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    genre="pop", mood="melancholic",
                    current_genre="pop", current_mood="sad",
                    energy=0.40, valence=0.40, dance=0.50, acoustic=0.55, tempo=100.0,
                    artists=["John Mayer"],
                ),
                self._approve_all_correct_response(),
            ]
            result = run_agent("John Mayer sad songs")

        assert result["analysis"]["detected_artists"] == ["John Mayer"]
        assert result["analysis"]["applied_artists"] == ["John Mayer"]

    def test_short_playlist_when_artist_pool_small(self):
        """If the artist-filtered pool has fewer than 5 songs, the final playlist
        shrinks to match — no padding from non-matching artists, no error."""
        # 'Phoebe Bridgers' has only 1 song in the test fixture (id 62)
        with patch("agent._build_client") as mock_build, \
             patch("agent.DATA_PATH", SONGS_PATH):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_client.messages.create.side_effect = [
                _make_analyze_response(
                    genre="indie", mood="sad",
                    current_genre="indie", current_mood="sad",
                    energy=0.35, valence=0.30, dance=0.40, acoustic=0.50, tempo=110.0,
                    artists=["Phoebe Bridgers"],
                ),
                self._approve_all_correct_response(),
            ]
            result = run_agent("sad Phoebe Bridgers tracks")

        assert 0 < len(result["final_playlist"]) <= 5
        # All entries must still be Phoebe Bridgers (no padding from other artists)
        for song in result["final_playlist"]:
            assert "phoebe bridgers" in song["artist"].lower()
