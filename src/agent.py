"""
src/agent.py — Agentic Plan-Act-Check orchestrator for the Smart Music Recommender.

Pipeline:
  1. ANALYZE  (LLM Call #1) — natural language → structured user_prefs dict
  2. DRAFT    (rule engine) — user_prefs → 5 scored candidate songs
  3. CORRECT  (LLM Call #2) — review draft; replace mismatches via rule engine
  4. OUTPUT                 — full result dict + evaluation metrics

Usage (CLI):
    python src/agent.py "give me something chill for late-night studying"

Usage (module):
    from agent import run_agent
    result = run_agent("energetic morning workout vibes")

Environment:
    Export ANTHROPIC_API_KEY in your shell before running:
        export ANTHROPIC_API_KEY=sk-ant-...   (Mac/Linux)
        set ANTHROPIC_API_KEY=sk-ant-...      (Windows cmd)
        $env:ANTHROPIC_API_KEY="sk-ant-..."   (PowerShell)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src/ to path so this file can be run directly or imported
sys.path.insert(0, str(Path(__file__).parent))

from recommender import load_songs, recommend_songs
from prompts import (
    KNOWN_GENRES,
    KNOWN_MOODS,
    KNOWN_LANGUAGES,
    ANALYZE_SYSTEM_PROMPT,
    ANALYZE_USER_TEMPLATE,
    CORRECT_SYSTEM_PROMPT,
    CORRECT_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
DATA_PATH = Path(__file__).parent.parent / "data" / "songs_full.csv"

# ── Safe defaults used when ANALYZE JSON parsing fails completely ─────────────

_ANALYZE_FALLBACK_PREFS = {
    "genre":               "pop",
    "mood":                "relaxed",
    "current_genre":       "pop",
    "current_mood":        "relaxed",
    "target_energy":       0.50,
    "target_valence":      0.50,
    "target_danceability": 0.50,
    "target_acousticness": 0.50,
    "target_tempo":        110.0,
}
_ANALYZE_FALLBACK_INTENT = "match"
VALID_INTENTS = {"match", "uplift", "energize", "calm", "contrast"}
MAX_PROMPT_LENGTH = 500


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "sk-ant-YOUR_KEY_HERE":
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return anthropic.Anthropic(api_key=key)


def _call_llm(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    step_label: str,
    max_tokens: int = 1024,
) -> str:
    """Call Claude; retry once on API error. Return raw text content."""
    print(f"[LLM]     Sending to {MODEL}...")
    for attempt in range(2):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = msg.content[0].text
            print(f"[LLM]     Response received ({msg.usage.output_tokens} tokens)")
            return raw
        except (anthropic.APIError, anthropic.RateLimitError) as exc:
            if attempt == 0:
                print(f"[LLM]     API error ({exc}); retrying in 2s...")
                time.sleep(2)
            else:
                raise RuntimeError(f"[{step_label}] LLM call failed: {exc}") from exc
    return ""  # unreachable


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences the model sometimes wraps around JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _parse_json_with_retry(
    raw: str,
    client: anthropic.Anthropic,
    system: str,
    user: str,
    step_label: str,
) -> dict:
    """Parse raw as JSON; retry the LLM once on parse failure; return fallback on second failure."""
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        print(f"[{step_label}] JSON parse error — retrying with stricter instruction...")
        retry_user = user + "\n\nIMPORTANT: Your previous response was not valid JSON. Respond with ONLY the JSON object, no other text."
        try:
            raw2 = _call_llm(client, system, retry_user, step_label)
            return json.loads(_strip_fences(raw2))
        except json.JSONDecodeError:
            print(f"[{step_label}] JSON parse failed twice — using fallback.")
            if step_label == "ANALYZE":
                return {
                    "user_prefs":     _ANALYZE_FALLBACK_PREFS,
                    "emotion_intent": _ANALYZE_FALLBACK_INTENT,
                    "reasoning":      "Failed to parse LLM response; using safe defaults.",
                    "confidence":     0.0,
                }
            # CORRECT fallback: approve everything
            return {
                "evaluations":       [],
                "approved":          True,
                "overall_reasoning": "Failed to parse LLM response; keeping draft unchanged.",
            }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _nearest_vocab(value: str, vocab: list[str], default: str) -> str:
    """Return value if it's in vocab; try substring match; else return default."""
    if value in vocab:
        return value
    for v in vocab:
        if value in v or v in value:
            logger.warning("[WARN] '%s' not in vocab — substituting '%s'", value, v)
            return v
    logger.warning("[WARN] '%s' not in vocab — substituting '%s'", value, default)
    return default


def _apply_intent_nudge(user_prefs: dict, emotion_intent: str) -> dict:
    prefs = dict(user_prefs)
    if emotion_intent == "uplift":
        prefs["target_valence"] = _clamp(prefs["target_valence"] + 0.15, 0.0, 1.0)
        prefs["target_energy"]  = _clamp(prefs["target_energy"]  + 0.10, 0.0, 1.0)
    elif emotion_intent == "energize":
        prefs["target_energy"]  = _clamp(prefs["target_energy"]  + 0.15, 0.0, 1.0)
        if prefs["target_tempo"] < 120.0:
            prefs["target_tempo"] = _clamp(prefs["target_tempo"] + 20.0, 50.0, 200.0)
    elif emotion_intent == "calm":
        prefs["target_energy"]  = _clamp(prefs["target_energy"]  - 0.15, 0.0, 1.0)
        prefs["target_valence"] = _clamp(prefs["target_valence"] + 0.05, 0.0, 1.0)
        if prefs["target_tempo"] > 100.0:
            prefs["target_tempo"] = _clamp(prefs["target_tempo"] - 15.0, 50.0, 200.0)
    elif emotion_intent == "contrast":
        prefs["target_valence"] = _clamp(1.0 - prefs["target_valence"], 0.0, 1.0)
        prefs["target_energy"]  = _clamp(1.0 - prefs["target_energy"],  0.0, 1.0)
    return prefs


def _validate_user_prompt(user_prompt: str) -> str:
    user_prompt = user_prompt.strip()
    if len(user_prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt too long ({len(user_prompt)} chars). "
            f"Please keep your request under {MAX_PROMPT_LENGTH} characters."
        )
    return user_prompt


def _validate_analyze_output(user_prefs: dict, emotion_intent: str) -> None:
    for field in ("genre", "current_genre"):
        if user_prefs.get(field) not in KNOWN_GENRES:
            logger.warning("[SECURITY] %s='%s' not in KNOWN_GENRES after sanitisation", field, user_prefs.get(field))
    for field in ("mood", "current_mood"):
        if user_prefs.get(field) not in KNOWN_MOODS:
            logger.warning("[SECURITY] %s='%s' not in KNOWN_MOODS after sanitisation", field, user_prefs.get(field))
    for field in ("target_energy", "target_valence", "target_danceability", "target_acousticness"):
        v = user_prefs.get(field, 0.0)
        if not (0.0 <= v <= 1.0):
            logger.warning("[SECURITY] %s=%s out of [0.0, 1.0] after clamping", field, v)
    tempo = user_prefs.get("target_tempo", 110.0)
    if not (50.0 <= tempo <= 200.0):
        logger.warning("[SECURITY] target_tempo=%s out of [50.0, 200.0] after clamping", tempo)
    if emotion_intent not in VALID_INTENTS:
        logger.warning("[SECURITY] emotion_intent='%s' not in VALID_INTENTS after sanitisation", emotion_intent)


# ── Public pipeline functions ─────────────────────────────────────────────────

def analyze_prompt(
    user_prompt: str,
    client: anthropic.Anthropic,
) -> tuple[dict, str, float, str, list[str] | None]:
    """
    STEP 1 — ANALYZE (LLM Call #1).

    Translate a natural-language prompt into a user_prefs dict that
    recommend_songs() can consume directly.

    Returns: (user_prefs, reasoning, confidence, emotion_intent, detected_languages)
    detected_languages is a list of ISO 639-1 codes the LLM extracted from the
    prompt, or None if no language was implied.
    """
    system = ANALYZE_SYSTEM_PROMPT.format(
        genres=", ".join(sorted(KNOWN_GENRES)),
        moods=", ".join(sorted(KNOWN_MOODS)),
        languages=", ".join(sorted(KNOWN_LANGUAGES)),
    )
    user_msg = ANALYZE_USER_TEMPLATE.format(user_prompt=user_prompt)

    raw = _call_llm(client, system, user_msg, "ANALYZE")
    parsed = _parse_json_with_retry(raw, client, system, user_msg, "ANALYZE")

    prefs_raw = parsed.get("user_prefs", _ANALYZE_FALLBACK_PREFS)
    reasoning  = parsed.get("reasoning", "")
    confidence = _clamp(float(parsed.get("confidence", 0.0)), 0.0, 1.0)

    raw_intent = parsed.get("emotion_intent", "match")
    emotion_intent = raw_intent if raw_intent in VALID_INTENTS else "match"
    if raw_intent not in VALID_INTENTS:
        logger.warning("[WARN] emotion_intent '%s' not valid — defaulting to 'match'", raw_intent)

    # Parse and validate detected languages. Accept null/missing/empty -> None.
    raw_langs = parsed.get("languages")
    detected_languages: list[str] | None
    if isinstance(raw_langs, list) and raw_langs:
        detected_languages = [code for code in raw_langs if isinstance(code, str) and code in KNOWN_LANGUAGES]
        invalid = [c for c in raw_langs if c not in KNOWN_LANGUAGES]
        if invalid:
            logger.warning("[WARN] languages contained invalid codes %s — dropped", invalid)
        if not detected_languages:
            detected_languages = None
    else:
        detected_languages = None

    # Validate and sanitise every field
    user_prefs = {
        "genre":               _nearest_vocab(prefs_raw.get("genre",         "lofi"),  KNOWN_GENRES, "electronic"),
        "mood":                _nearest_vocab(prefs_raw.get("mood",          "chill"), KNOWN_MOODS,  "chill"),
        "current_genre":       _nearest_vocab(prefs_raw.get("current_genre", "lofi"),  KNOWN_GENRES, "electronic"),
        "current_mood":        _nearest_vocab(prefs_raw.get("current_mood",  "chill"), KNOWN_MOODS,  "chill"),
        "target_energy":       _clamp(float(prefs_raw.get("target_energy",       0.30)), 0.0, 1.0),
        "target_valence":      _clamp(float(prefs_raw.get("target_valence",      0.50)), 0.0, 1.0),
        "target_danceability": _clamp(float(prefs_raw.get("target_danceability", 0.40)), 0.0, 1.0),
        "target_acousticness": _clamp(float(prefs_raw.get("target_acousticness", 0.70)), 0.0, 1.0),
        "target_tempo":        _clamp(float(prefs_raw.get("target_tempo",        85.0)), 50.0, 200.0),
    }

    # Apply intent-driven nudges to numeric targets
    user_prefs = _apply_intent_nudge(user_prefs, emotion_intent)
    _validate_analyze_output(user_prefs, emotion_intent)

    print(f"[ANALYZE] genre={user_prefs['genre']}  mood={user_prefs['mood']}  "
          f"current_genre={user_prefs['current_genre']}  current_mood={user_prefs['current_mood']}")
    print(f"[ANALYZE] energy={user_prefs['target_energy']:.2f}  "
          f"valence={user_prefs['target_valence']:.2f}  "
          f"dance={user_prefs['target_danceability']:.2f}  "
          f"acoustic={user_prefs['target_acousticness']:.2f}  "
          f"tempo={user_prefs['target_tempo']:.0f} bpm")
    print(f"[ANALYZE] emotion_intent={emotion_intent}  confidence={confidence:.2f}")
    print(f"[ANALYZE] detected_languages={detected_languages}")
    print(f"[ANALYZE] reasoning: \"{reasoning}\"")

    return user_prefs, reasoning, confidence, emotion_intent, detected_languages


def draft_playlist(
    user_prefs: dict,
    songs: list[dict],
    seen_ids: set = None,
) -> list[tuple[dict, float, str]]:
    """
    STEP 2 — DRAFT (rule engine only, no LLM).

    Call recommend_songs() with the structured prefs to get 5 candidates.
    Returns list of (song_dict, score, explanation) tuples.
    """
    results = recommend_songs(user_prefs, songs, k=5, seen_ids=seen_ids)
    print(f"[DRAFT]   {len(results)} songs selected:")
    for i, (song, score, _) in enumerate(results, 1):
        print(f"[DRAFT]   {i}. {song['title']} — {song['artist']}  score={score:.3f}")
    return results


def self_correct(
    user_prompt: str,
    user_prefs: dict,
    draft: list[tuple[dict, float, str]],
    songs: list[dict],
    client: anthropic.Anthropic,
    seen_ids: set = None,
) -> tuple[list[tuple[dict, float, str]], int]:
    """
    STEP 3 — SELF-CORRECT (LLM Call #2 + conditional rule engine call).

    Ask the LLM to evaluate each draft song's fit with the original prompt.
    Rejected songs are replaced by calling recommend_songs() on the remaining
    catalog (excluding already-present and rejected IDs).

    Returns: (final_playlist, corrections_made)
    """
    # Serialise draft for the prompt
    draft_records = [
        {
            "song_id":      song["id"],
            "title":        song["title"],
            "artist":       song["artist"],
            "genre":        song["genre"],
            "mood":         song["mood"],
            "energy":       song["energy"],
            "valence":      song["valence"],
            "danceability": song["danceability"],
            "acousticness": song["acousticness"],
            "tempo_bpm":    song["tempo_bpm"],
            "rule_score":   round(score, 3),
            "rule_explanation": explanation,
        }
        for song, score, explanation in draft
    ]
    draft_json = json.dumps(draft_records, indent=2)

    user_msg = CORRECT_USER_TEMPLATE.format(
        user_prompt=user_prompt,
        draft_json=draft_json,
    )

    raw = _call_llm(client, CORRECT_SYSTEM_PROMPT, user_msg, "CORRECT")
    parsed = _parse_json_with_retry(raw, client, CORRECT_SYSTEM_PROMPT, user_msg, "CORRECT")

    evaluations  = parsed.get("evaluations", [])
    approved     = parsed.get("approved", True)
    overall_note = parsed.get("overall_reasoning", "")

    # Print per-song evaluations
    eval_by_id = {e["song_id"]: e for e in evaluations}
    for song, score, _ in draft:
        ev = eval_by_id.get(song["id"], {})
        keep      = ev.get("keep", True)
        fit_score = ev.get("fit_score", score)
        reason    = ev.get("fit_reason", "")
        flag = "✓" if keep else "✗"
        print(f"[CORRECT] {flag} {song['title']:<30} fit={fit_score:.2f}  {reason}")

    rejected_ids = {e["song_id"] for e in evaluations if not e.get("keep", True)}
    corrections_made = len(rejected_ids)

    print(f"[CORRECT] approved={approved}  corrections_made={corrections_made}")
    if overall_note:
        print(f"[CORRECT] {overall_note}")

    if corrections_made == 0:
        return list(draft), 0

    # Build replacement pool: exclude songs already in draft AND rejected ones
    present_ids = {song["id"] for song, _, _ in draft}
    pool = [s for s in songs if s["id"] not in present_ids and s["id"] not in rejected_ids]

    replacements = recommend_songs(user_prefs, pool, k=corrections_made, seen_ids=seen_ids)
    print(f"[CORRECT] Replacements found: {len(replacements)}")
    for song, score, _ in replacements:
        print(f"[CORRECT]   + {song['title']} — {song['artist']}  score={score:.3f}")

    # Splice: keep approved songs in order, append replacements
    final_playlist = [
        (song, score, expl)
        for song, score, expl in draft
        if song["id"] not in rejected_ids
    ] + replacements

    return final_playlist, corrections_made


def _compute_metrics(
    confidence: float,
    draft: list[tuple[dict, float, str]],
    final_playlist: list[tuple[dict, float, str]],
    corrections_made: int,
) -> dict[str, Any]:
    draft_scores = [score for _, score, _ in draft]
    final_scores = [score for _, score, _ in final_playlist]
    return {
        "analysis_confidence": round(confidence, 3),
        "draft_score":         round(sum(draft_scores) / len(draft_scores), 3) if draft_scores else 0.0,
        "correction_count":    corrections_made,
        "final_avg_score":     round(sum(final_scores) / len(final_scores), 3) if final_scores else 0.0,
    }


def analyze_only(user_prompt: str) -> dict[str, Any]:
    """
    Phase 1 of the two-phase Streamlit flow: run only the ANALYZE step.

    Returns the analysis dict (including detected_languages) so the caller
    can decide whether to prompt the user for a language selection before
    proceeding to DRAFT.

    Returns a dict with keys:
        user_prefs, reasoning, confidence, emotion_intent, detected_languages
    """
    user_prompt = _validate_user_prompt(user_prompt)
    print(f'\n[AGENT] Analyzing: "{user_prompt}"\n')

    client = _build_client()

    print("--- STEP 1: ANALYZE (LLM Call #1) ---")
    user_prefs, reasoning, confidence, emotion_intent, detected_languages = (
        analyze_prompt(user_prompt, client)
    )

    return {
        "user_prefs":         user_prefs,
        "reasoning":          reasoning,
        "confidence":         confidence,
        "emotion_intent":     emotion_intent,
        "detected_languages": detected_languages,
    }


def run_with_analysis(
    user_prompt: str,
    analysis: dict[str, Any],
    languages: list[str] | None,
    seen_ids: set = None,
) -> dict[str, Any]:
    """
    Phase 2 of the two-phase Streamlit flow: given a pre-computed analysis
    and a (possibly user-confirmed) language filter, run DRAFT + CORRECT.

    `languages` is the final language filter to apply: either auto-detected
    from the prompt (analysis["detected_languages"]) or chosen by the user
    via the multiselect menu. None or empty means no filtering.
    """
    user_prompt    = _validate_user_prompt(user_prompt)
    user_prefs     = analysis["user_prefs"]
    reasoning      = analysis["reasoning"]
    confidence     = analysis["confidence"]
    emotion_intent = analysis["emotion_intent"]

    client = _build_client()
    songs  = load_songs(str(DATA_PATH))

    if languages:
        lang_set = set(languages)
        before = len(songs)
        songs = [s for s in songs if s.get("language") in lang_set]
        print(f"[FILTER] Language filter {sorted(lang_set)}: {len(songs):,} of {before:,} songs")

    # ── Step 2: DRAFT ──────────────────────────────────────────────────────────
    print("\n--- STEP 2: DRAFT (Rule Engine) ---")
    draft = draft_playlist(user_prefs, songs, seen_ids=seen_ids)
    if not draft:
        raise RuntimeError(
            "No new songs available — try a different prompt, change your "
            "language filter, or clear your session history."
        )

    # ── Step 3: SELF-CORRECT ───────────────────────────────────────────────────
    print("\n--- STEP 3: SELF-CORRECT (LLM Call #2) ---")
    final_playlist, corrections_made = self_correct(
        user_prompt, user_prefs, draft, songs, client, seen_ids=seen_ids
    )

    # ── Step 4: Metrics ────────────────────────────────────────────────────────
    metrics = _compute_metrics(confidence, draft, final_playlist, corrections_made)

    print("\n--- STEP 4: METRICS ---")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    def _serialise(playlist):
        return [
            {
                **{k: v for k, v in song.items() if k != "_numeric_explanation"},
                "score":               round(score, 3),
                "explanation":         expl,
                "explanation_numeric": song.get("_numeric_explanation", ""),
            }
            for song, score, expl in playlist
        ]

    recommended_ids = {song["id"] for song, _, _ in final_playlist}

    return {
        "user_prompt": user_prompt,
        "analysis": {
            "user_prefs":         user_prefs,
            "reasoning":          reasoning,
            "confidence":         confidence,
            "emotion_intent":     emotion_intent,
            "detected_languages": analysis.get("detected_languages"),
            "applied_languages":  list(languages) if languages else None,
        },
        "draft_playlist":   _serialise(draft),
        "final_playlist":   _serialise(final_playlist),
        "metrics":          metrics,
        "recommended_ids":  recommended_ids,
    }


def run_agent(
    user_prompt: str,
    seen_ids: set = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """
    Top-level orchestrator. Run the full Plan-Act-Check pipeline in one call.

    `languages`: explicit language filter. If None, falls back to whatever
    the LLM detected from the prompt. Pass an explicit list to override
    detection (e.g., from the Streamlit multiselect).

    Returns a dict with keys:
        user_prompt, analysis, draft_playlist, final_playlist, metrics
    """
    print(f'\n[AGENT] Starting pipeline for: "{user_prompt}"\n')
    analysis = analyze_only(user_prompt)
    effective_languages = languages if languages is not None else analysis["detected_languages"]
    return run_with_analysis(user_prompt, analysis, effective_languages, seen_ids=seen_ids)


# ── Refresh / session helpers ─────────────────────────────────────────────────

# A session is just a plain dict: { prompt_string -> set of seen song IDs }
# Pass the same dict on every call and it accumulates history automatically.

def refresh_playlist(
    user_prompt: str,
    session: dict,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the pipeline for user_prompt, skipping every song already recommended
    for that prompt in this session. Optionally apply a language filter.

    Usage:
        session = {}
        r1 = refresh_playlist("chill late-night vibes", session)  # songs 1-5
        r2 = refresh_playlist("chill late-night vibes", session)  # songs 6-10
    """
    seen = session.get(user_prompt, set())
    result = run_agent(user_prompt, seen_ids=seen, languages=languages)
    session[user_prompt] = seen | result["recommended_ids"]
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/agent.py \"your music request here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    result = run_agent(prompt)

    print("\n=== FINAL PLAYLIST ===")
    for entry in result["final_playlist"]:
        print(f"  {entry['title']:<32} — {entry['artist']:<20}  score={entry['score']:.3f}")

    print("\n=== METRICS ===")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
