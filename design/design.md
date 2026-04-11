# Music Recommender — System Design

## System Overview

A rule-based music recommender that scores songs against a user profile and returns a ranked playlist. Fully explainable: every score is traceable to a hard-coded formula with fixed weights. No learning from data — all behavior is deterministic and auditable.

**Interface:** Command-line (CLI). Run via:
```bash
python -m src.main
```

---

## Data Model

### Song

| Field | Type | Used in Scoring |
|---|---|---|
| `id` | int | No |
| `title` | str | No |
| `artist` | str | No |
| `genre` | str | Yes — categorical |
| `mood` | str | Yes — categorical |
| `energy` | float | Yes — numeric |
| `valence` | float | Yes — numeric |
| `danceability` | float | Yes — numeric |
| `acousticness` | float | Yes — numeric |
| `tempo_bpm` | float | No (unused) |

### UserProfile (Proposed)

Splits taste signals into **long-term** (all-time history) and **short-term** (current session):

| Field | Type | Signal Type |
|---|---|---|
| `favorite_genre` | str | Long-term — most played genre across all sessions |
| `favorite_mood` | str | Long-term — most played mood across all sessions |
| `current_genre` | str | Short-term — most common genre in last N songs played |
| `current_mood` | str | Short-term — most common mood in last N songs played |
| `target_energy` | float | Numeric target (0–1) |
| `target_valence` | float | Numeric target (0–1) |
| `target_danceability` | float | Numeric target (0–1) |
| `target_acousticness` | float | Numeric target (0–1) |

**Previous design issue:** `UserProfile` had `likes_acoustic: bool` and was missing `target_valence`, `target_danceability`, and `target_acousticness` — fields required by the scoring formula. Also lacked any session-aware signal.

---

## Scoring Formula

### Step 1: Categorical Score

Blends long-term preference with current session signal:

```
genre_match = 0.30 × (1.0 if song.genre == favorite_genre else 0.0)
            + 0.70 × (1.0 if song.genre == current_genre  else 0.0)

mood_match  = 0.30 × (1.0 if song.mood == favorite_mood  else 0.0)
            + 0.70 × (1.0 if song.mood == current_mood   else 0.0)

categorical_score = (genre_match + mood_match) / 2
```

**Why the 0.30 / 0.70 split:** Recommendations should respond to what the user is listening to right now (0.70), while still respecting long-term taste as a fallback (0.30). This reduces the filter bubble effect where a user would otherwise be locked into their all-time favorite genre/mood forever.

**Previous design issue:** The original formula used only `favorite_genre` / `favorite_mood` — a static, session-blind match that always returned 1.0 or 0.0 with no sensitivity to current listening context.

### Step 2: Numeric Similarity

```
energy_sim   = 1 - |song.energy       - user.target_energy|
valence_sim  = 1 - |song.valence      - user.target_valence|
dance_sim    = 1 - |song.danceability - user.target_danceability|
acoustic_sim = 1 - |song.acousticness - user.target_acousticness|
```

Each similarity is in range [0, 1]. Peak (1.0) when song value exactly matches user target.

### Step 3: Numeric Score

```
numeric_score = 0.30 × energy_sim
              + 0.25 × valence_sim
              + 0.25 × dance_sim
              + 0.20 × acoustic_sim
```

### Step 4: Final Score

```
SCORE = 0.40 × categorical_score + 0.60 × numeric_score
```

### Max Contribution per Term

| Term | Max contribution to SCORE |
|---|---|
| Categorical (genre + mood) | 0.40 |
| Energy | 0.60 × 0.30 = **0.18** |
| Valence | 0.60 × 0.25 = **0.15** |
| Danceability | 0.60 × 0.25 = **0.15** |
| Acousticness | 0.60 × 0.20 = **0.12** |
| **Total** | **1.00** |

---

## Ranking Rule

After scoring, songs are re-ordered using a greedy variety re-ranker to prevent playlist monotony:

```
For each pick from the remaining pool:
    effective_score = final_score
                    - 0.10 if same genre as previous pick
                    - 0.10 if same mood as previous pick
    Pick the song with highest effective_score
```

Maximum variety penalty per song: −0.20 (both genre and mood repeat).

This is applied **after** scoring, so it only shuffles the top pool — it does not surface low-scoring songs.

---

## Profile Update Rules (Design Intent)

These rules define how `UserProfile` fields should update over time as the user listens:

- `favorite_genre` / `favorite_mood` → most frequent genre/mood across **all-time** listen history
- `current_genre` / `current_mood` → most frequent genre/mood in the **last N songs** played (session window, recommended N = 5–10)

Not yet implemented in `src/recommender.py`.

---

## Known Limitations

- **Small catalog** — 10 songs. Real recommenders operate on millions of tracks.
- **Static numeric targets** — `target_energy`, `target_valence`, etc. do not adapt over time. Only categorical signals are session-aware.
- **Partial filter bubble mitigation** — the 0.70 current session weight reduces but does not eliminate echo chamber risk. If the user always listens to the same genre, `current_genre` and `favorite_genre` converge.
- **No content understanding** — lyrics, instrumentation, and context (time of day, activity) are ignored.
- **Variety ranking is post-hoc** — applies only to the already-scored top pool. Low-scoring songs from different genres cannot surface regardless of variety.

---

## Design Decisions

**Explainability over accuracy.** All weights are hand-set and fixed. Any score can be calculated by hand and fully audited. This is the opposite of production ML recommenders (Spotify, YouTube) where weights are learned from billions of interactions and individual predictions cannot be explained.

**Short-term signal weighted higher than long-term.** A 0.70 / 0.30 split favors current session context. The assumption is that listening behavior in a session is a stronger signal of immediate intent than historical averages.

**Categorical weight (0.40) is high by design.** Genre and mood are the most interpretable features — matching them produces results that feel "obviously right" to users. Numeric features (energy, valence, etc.) provide fine-grained differentiation within a genre/mood bucket.
