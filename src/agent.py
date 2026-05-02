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
    ANALYZE_SYSTEM_PROMPT,
    ANALYZE_USER_TEMPLATE,
    CORRECT_SYSTEM_PROMPT,
    CORRECT_USER_TEMPLATE,
)

MODEL = "claude-haiku-4-5-20251001"
DATA_PATH = Path(__file__).parent.parent / "data" / "songs.csv"

# ── Safe defaults used when ANALYZE JSON parsing fails completely ─────────────

_ANALYZE_FALLBACK_PREFS = {
    "genre":               "lofi",
    "mood":                "chill",
    "current_genre":       "lofi",
    "current_mood":        "chill",
    "target_energy":       0.30,
    "target_valence":      0.50,
    "target_danceability": 0.40,
    "target_acousticness": 0.70,
    "target_tempo":        85.0,
}


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


def _parse_json_with_retry(
    raw: str,
    client: anthropic.Anthropic,
    system: str,
    user: str,
    step_label: str,
) -> dict:
    """Parse raw as JSON; retry the LLM once on parse failure; return fallback on second failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[{step_label}] JSON parse error — retrying with stricter instruction...")
        retry_user = user + "\n\nIMPORTANT: Your previous response was not valid JSON. Respond with ONLY the JSON object, no other text."
        try:
            raw2 = _call_llm(client, system, retry_user, step_label)
            return json.loads(raw2)
        except json.JSONDecodeError:
            print(f"[{step_label}] JSON parse failed twice — using fallback.")
            if step_label == "ANALYZE":
                return {
                    "user_prefs": _ANALYZE_FALLBACK_PREFS,
                    "reasoning":  "Failed to parse LLM response; using safe defaults.",
                    "confidence": 0.0,
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
            print(f"[WARN]    '{value}' not in vocab — substituting '{v}'")
            return v
    print(f"[WARN]    '{value}' not in vocab — substituting '{default}'")
    return default


# ── Public pipeline functions ─────────────────────────────────────────────────

def analyze_prompt(
    user_prompt: str,
    client: anthropic.Anthropic,
) -> tuple[dict, str, float]:
    """
    STEP 1 — ANALYZE (LLM Call #1).

    Translate a natural-language prompt into a user_prefs dict that
    recommend_songs() can consume directly.

    Returns: (user_prefs, reasoning, confidence)
    """
    system = ANALYZE_SYSTEM_PROMPT.format(
        genres=", ".join(KNOWN_GENRES),
        moods=", ".join(KNOWN_MOODS),
    )
    user_msg = ANALYZE_USER_TEMPLATE.format(user_prompt=user_prompt)

    raw = _call_llm(client, system, user_msg, "ANALYZE")
    parsed = _parse_json_with_retry(raw, client, system, user_msg, "ANALYZE")

    prefs_raw = parsed.get("user_prefs", _ANALYZE_FALLBACK_PREFS)
    reasoning  = parsed.get("reasoning", "")
    confidence = _clamp(float(parsed.get("confidence", 0.0)), 0.0, 1.0)

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

    print(f"[ANALYZE] genre={user_prefs['genre']}  mood={user_prefs['mood']}  "
          f"current_genre={user_prefs['current_genre']}  current_mood={user_prefs['current_mood']}")
    print(f"[ANALYZE] energy={user_prefs['target_energy']:.2f}  "
          f"valence={user_prefs['target_valence']:.2f}  "
          f"dance={user_prefs['target_danceability']:.2f}  "
          f"acoustic={user_prefs['target_acousticness']:.2f}  "
          f"tempo={user_prefs['target_tempo']:.0f} bpm")
    print(f"[ANALYZE] reasoning: \"{reasoning}\"")
    print(f"[ANALYZE] confidence: {confidence:.2f}")

    return user_prefs, reasoning, confidence


def draft_playlist(
    user_prefs: dict,
    songs: list[dict],
) -> list[tuple[dict, float, str]]:
    """
    STEP 2 — DRAFT (rule engine only, no LLM).

    Call recommend_songs() with the structured prefs to get 5 candidates.
    Returns list of (song_dict, score, explanation) tuples.
    """
    results = recommend_songs(user_prefs, songs, k=5)
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

    replacements = recommend_songs(user_prefs, pool, k=corrections_made)
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


def run_agent(user_prompt: str) -> dict[str, Any]:
    """
    Top-level orchestrator. Run the full Plan-Act-Check pipeline.

    Returns a dict with keys:
        user_prompt, analysis, draft_playlist, final_playlist, metrics
    """
    print(f'\n[AGENT] Starting pipeline for: "{user_prompt}"\n')

    client = _build_client()
    songs  = load_songs(str(DATA_PATH))

    # ── Step 1: ANALYZE ────────────────────────────────────────────────────────
    print("--- STEP 1: ANALYZE (LLM Call #1) ---")
    user_prefs, reasoning, confidence = analyze_prompt(user_prompt, client)

    # ── Step 2: DRAFT ──────────────────────────────────────────────────────────
    print("\n--- STEP 2: DRAFT (Rule Engine) ---")
    draft = draft_playlist(user_prefs, songs)

    # ── Step 3: SELF-CORRECT ───────────────────────────────────────────────────
    print("\n--- STEP 3: SELF-CORRECT (LLM Call #2) ---")
    final_playlist, corrections_made = self_correct(
        user_prompt, user_prefs, draft, songs, client
    )

    # ── Step 4: Metrics ────────────────────────────────────────────────────────
    metrics = _compute_metrics(confidence, draft, final_playlist, corrections_made)

    print("\n--- STEP 4: METRICS ---")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    # Serialise playlists as plain dicts for JSON safety
    def _serialise(playlist):
        return [
            {**song, "score": round(score, 3), "explanation": expl}
            for song, score, expl in playlist
        ]

    return {
        "user_prompt": user_prompt,
        "analysis": {
            "user_prefs": user_prefs,
            "reasoning":  reasoning,
            "confidence": confidence,
        },
        "draft_playlist": _serialise(draft),
        "final_playlist": _serialise(final_playlist),
        "metrics": metrics,
    }


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
