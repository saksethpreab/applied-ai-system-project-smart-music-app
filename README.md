# LOOM DEMO
[Link](https://www.loom.com/share/9dd881e55c404108854b5d4459f6dbc1) 

# MoodSync — Agentic Smart Music Recommender

A music recommendation system that combines a **rule-based scoring engine** with a **Claude-powered agentic pipeline** to turn natural language prompts into ranked playlists. Every recommendation is fully explainable: each score is a deterministic, auditable formula with fixed weights.

---

## How It Works

The system runs a three-step Plan-Act-Check pipeline for every request:

```
User prompt ("something dark and atmospheric for a rainy drive")
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1 — PLAN (Claude LLM)                             │
│  Deconstruct the prompt → structured user preference    │
│  profile: genre, mood, energy, valence, tempo targets   │
│  + emotion_intent (match / uplift / energize / calm /   │
│    contrast)                                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 — ACT (Rule Engine)                             │
│  Score every song in the catalog against the profile.   │
│  40% categorical (genre + mood) + 60% numeric           │
│  (energy, valence, danceability, acousticness, tempo).  │
│  Genre gate caps foreign-genre numeric contribution.    │
│  Greedy variety re-ranker penalizes consecutive         │
│  same-genre/mood picks. Return top 5.                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3 — CHECK (Claude LLM)                            │
│  Review each of the 5 draft songs against the original  │
│  prompt. Reject clear mismatches; replace them via the  │
│  rule engine on the remaining catalog. Return final     │
│  playlist with fit scores and reasoning.                │
└─────────────────────────────────────────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- (Optional) A [Spotify Developer account](https://developer.spotify.com/dashboard) for live catalog fetching

### Install

```bash
pip install -r requirements.txt
```

### Configure `.env`

```
ANTHROPIC_API_KEY=sk-ant-...
SPOTIFY_CLIENT_ID=your_id_here        # optional — enables live Spotify catalog
SPOTIFY_CLIENT_SECRET=your_secret_here
```

### Run

**Web UI (Streamlit):**
```bash
streamlit run src/app.py
```

**CLI demo:**
```bash
python src/agent.py "energetic morning workout vibes"
```

**Generate Spotify genre map (one-time, required if using Spotify):**
```bash
python scripts/generate_genre_map.py
```

---

## Scoring Formula

### Step 1 — Categorical Score

Genre and mood are each matched against two signals — the user's long-term preference and the current session — blended 50/50:

```
family_match(a, b) → 1.0 exact · 0.5 same family · 0.0 otherwise

genre_match = 0.50 × family_match(song.genre, favorite_genre)
            + 0.50 × family_match(song.genre, current_genre)

mood_match  = 0.50 × family_match(song.mood,  favorite_mood)
            + 0.50 × family_match(song.mood,  current_mood)

categorical_score = 0.33 × genre_match + 0.67 × mood_match
```

**Genre families:** indie · electronic · rock · urban · classical · pop · world

**Mood families:** melancholic · calm · energetic (singleton) · intense · positive

### Step 2 — Numeric Similarity

```
energy_sim   = 1 − |song.energy       − target_energy|
valence_sim  = 1 − |song.valence      − target_valence|
dance_sim    = 1 − |song.danceability − target_danceability|
acoustic_sim = 1 − |song.acousticness − target_acousticness|
tempo_sim    = 1 − |(song.tempo_bpm − 60) / 140 − (target_tempo − 60) / 140|
```

### Step 3 — Genre Gate + Final Score

```
genre_gate = max(
    family_match(song.genre, favorite_genre),
    family_match(song.genre, current_genre),
    0.25   ← floor preserves cross-genre discovery
)

numeric_score = (0.35 × energy_sim
              + 0.25 × valence_sim
              + 0.25 × dance_sim
              + 0.10 × acoustic_sim
              + 0.05 × tempo_sim) × genre_gate

final_score = 0.40 × categorical_score + 0.60 × numeric_score
```

| Term | Max contribution |
|---|---|
| Genre (categorical) | 0.132 |
| Mood (categorical) | 0.268 |
| Energy | ≤ 0.210 |
| Valence | ≤ 0.150 |
| Danceability | ≤ 0.150 |
| Acousticness | ≤ 0.060 |
| Tempo | ≤ 0.030 |

### Step 4 — Variety Re-ranking

A greedy re-ranker penalizes consecutive picks that repeat genre (−0.15) or mood (−0.15), max −0.30 combined. Effective scores are clamped at 0.0.

---

## Security

User-supplied prompts are isolated from LLM system instructions using `<user_input>` trust-boundary tags in both Claude calls. A 500-character prompt length limit is enforced before any API call. Post-sanitisation validation logs anomalies in extracted preference fields.

---

## Data Sources

### Default — CSV Catalog

301 hand-curated songs with manually assigned audio features. Sufficient for demonstration and offline use.

### Live — Spotify Web API (optional)

When `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` are set, the system fetches up to 100 candidate tracks from Spotify using the user's analyzed genre and mood. Audio features (`energy`, `valence`, `danceability`, `acousticness`, `tempo`) are pulled directly from Spotify's API. Artist genre tags are translated to the project's vocabulary via `data/spotify_genre_map.json`. Mood is inferred from `valence + energy` quadrants. The CSV catalog is used as a fallback if Spotify is unavailable or credentials are not set.

---

## Bias Fixes Applied

Five scoring biases were identified and corrected. All fixes are weight constants or small formula additions — no new fields or function signatures.

| Fix | Problem | Change |
|---|---|---|
| Energy dominance | W_ENERGY = 0.60 made energy 36% of final score; low-energy users under-served | Reduced to 0.35; valence and danceability absorb slack at 0.25 each |
| Session filter bubble | W_SESSION = 0.70 let one off-genre session override all-time history | Rebalanced to 50/50 long-term/session |
| Genre bleed | Genre-foreign songs outscored exact-genre matches on energy alone | Added `genre_gate` multiplier (1.0 / 0.5 / 0.25 floor) to numeric score |
| Mood conflation | "energetic", "intense", "angry" grouped as equals | Split "energetic" into singleton family; angry/intense no longer get proximity credit toward energetic users |
| Weak variety penalty | −0.10/−0.10 max −0.20 couldn't break real catalog score gaps (0.25+) | Raised to −0.15/−0.15 max −0.30 |

---

## Key Results

**Acoustic Intensity (most dramatic):** Storm Runner had exact rock/intense match but ranked last (score 0.025) before fixes — one acoustic feature mismatch dragged it down and the variety re-ranker compounded the penalty. After fixes: rank 1 at 0.829. +0.804 score increase, +4 rank gain.

**Deep Intense Rock:** Urban Beats (hip-hop) scored 0.709 for a rock user before the genre gate. After: 0.175. Genre-foreign songs no longer competitive.

**Relaxed Workout:** Angry metal disappeared from results after the energetic/intense mood split.

---

## Limitations

- **Small catalog.** 301 songs covers common genres well but cannot represent full musical diversity. Spotify integration expands the candidate pool significantly.
- **Static numeric targets.** Energy, valence, and other targets are set per-prompt, not learned from history.
- **Contradictory profiles.** High energy + sad mood is an inherently conflicting signal the formula averages rather than resolves.
- **Genre floor allows bleed.** GENRE_FLOOR = 0.25 means foreign-genre songs are never fully excluded — intentional for niche users, but visible at positions 4–5 for well-represented genres.
- **Linear similarity.** `1 − |song − target|` is perceptually linear; human perception of musical features is not.
- **Mood inference from Spotify.** Mood is inferred from `valence + energy` quadrants, not from actual song content.

---

## Documentation

- [Model Card](model_card.md) — intended use, strengths, limitations, evaluation, reflection
- [System Design](design/design.md) — architecture, scoring formula, data model, security, Spotify integration
- [Visualization Specs](design/VISUALIZATION.md) — instructions for generating score breakdowns and sensitivity plots
