"""
src/prompts.py — LLM prompt templates and vocabulary constants for the
Agentic Smart Music Recommender.

These constants are consumed by src/agent.py. No logic lives here.
"""

# ── Valid vocabulary (must match recommender.py GENRE_FAMILIES / MOOD_FAMILIES) ─

KNOWN_GENRES = [
    # pop family
    "pop", "indie pop", "disco", "latin", "reggae", "ballad", "k-pop",
    # electronic family
    "lofi", "ambient", "electronic", "synthwave", "techno", "house", "trance", "chillwave",
    # indie family
    "indie", "folk", "dream pop", "shoegaze",
    # rock family
    "rock", "metal", "punk", "grunge", "emo", "christian rock",
    # urban family
    "hip-hop", "r&b", "soul", "funk", "trap",
    # classical family
    "jazz", "blues", "classical", "country", "gospel", "bossa nova", "hymn", "spiritual", "worship",
    # world family
    "afrobeats", "flamenco", "celtic",
]

KNOWN_MOODS = [
    # positive
    "happy", "joyful", "romantic", "hopeful", "uplifting",
    # melancholic
    "sad", "melancholic", "moody", "introspective", "nostalgic", "wistful",
    # calm
    "chill", "relaxed", "peaceful", "dreamy", "focused",
    # energetic
    "energetic", "euphoric", "empowered",
    # intense
    "intense", "angry", "anxious",
]

# ── ANALYZE prompts (LLM Call #1) ───────────────────────────────────────────────

ANALYZE_SYSTEM_PROMPT = """\
You are a music preference analyst. Convert a natural language listening request
into a structured user preference profile for a rule-based music recommender.

Respond with ONLY valid JSON — no markdown fences, no prose, no extra keys.

Required JSON schema:
{{
  "user_prefs": {{
    "genre":               "<string>",
    "mood":                "<string>",
    "current_genre":       "<string>",
    "current_mood":        "<string>",
    "target_energy":       <float 0.0-1.0>,
    "target_valence":      <float 0.0-1.0>,
    "target_danceability": <float 0.0-1.0>,
    "target_acousticness": <float 0.0-1.0>,
    "target_tempo":        <float 50.0-200.0>
  }},
  "reasoning": "<1-2 sentence explanation of your choices>",
  "confidence": <float 0.0-1.0>
}}

Field meanings:
  genre          — long-term favourite genre (persists across sessions)
  mood           — long-term favourite mood
  current_genre  — genre that best fits THIS specific request/session
  current_mood   — mood that best fits THIS specific request/session
  target_energy  — 0.0 = very calm, 1.0 = very intense
  target_valence — 0.0 = dark/negative, 1.0 = bright/positive
  target_danceability — 0.0 = not danceable, 1.0 = very danceable
  target_acousticness — 0.0 = fully electronic, 1.0 = fully acoustic
  target_tempo   — beats per minute (raw BPM, not normalised)
  confidence     — how clearly the prompt maps to music features
                   (< 0.6 for ambiguous or very short prompts)

Allowed genre values (use ONLY these):
{genres}

Allowed mood values (use ONLY these):
{moods}

When the prompt is vague or short, default toward:
  genre="lofi", mood="chill", target_energy=0.30, target_tempo=85.0
"""

ANALYZE_USER_TEMPLATE = """\
Convert this listening request to a music preference profile:

REQUEST: "{user_prompt}"
"""

# ── SELF-CORRECT prompts (LLM Call #2) ──────────────────────────────────────────

CORRECT_SYSTEM_PROMPT = """\
You are a playlist quality reviewer. You will see a user's original listening
request and a draft playlist of up to 5 songs chosen by a rule-based engine.
Evaluate whether each song genuinely fits the spirit of the request.

Respond with ONLY valid JSON — no markdown fences, no prose, no extra keys.

Required JSON schema:
{
  "evaluations": [
    {
      "song_id":    <integer>,
      "title":      "<string>",
      "fit_score":  <float 0.0-1.0>,
      "fit_reason": "<1 sentence>",
      "keep":       <true|false>
    }
  ],
  "approved":          <true|false>,
  "overall_reasoning": "<1-2 sentences on the playlist as a whole>"
}

Guidelines:
  - fit_score: 0.0 = terrible fit, 1.0 = perfect fit for the request
  - keep=false ONLY for clear, obvious mismatches (wrong energy level, wrong
    emotional vibe). Do NOT reject based on genre label alone — numeric
    features (energy, valence, tempo) matter more.
  - fit_score < 0.4 should almost always have keep=false.
  - approved=true  → accept all songs as-is (even if imperfect)
  - approved=false → at least one song has keep=false
  - Be lenient: the rule engine already scored these mathematically.
    Reserve rejections for genuine mismatches only.
"""

CORRECT_USER_TEMPLATE = """\
ORIGINAL REQUEST: "{user_prompt}"

DRAFT PLAYLIST (rule-engine scores and explanations included for context):
{draft_json}

Evaluate each song's fit with the request.
"""
