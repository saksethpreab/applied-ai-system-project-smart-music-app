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

### UserProfile

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
| `target_tempo` | float | Raw BPM — normalized at score time using catalog range |

**Previous design issue:** `UserProfile` had `likes_acoustic: bool` and was missing `target_valence`, `target_danceability`, `target_acousticness`, and `target_tempo` — fields required by the scoring formula. Also lacked session-aware signals.

**Fix:** Removed `likes_acoustic: bool`. Added `target_valence`, `target_danceability`, `target_acousticness`, `target_tempo` as explicit numeric targets. Added `current_genre` and `current_mood` for session-aware categorical scoring.

---

## Data Flow

### Two Interfaces

`src/recommender.py` exposes two parallel interfaces:

**Functional** — used by `src/main.py` (CLI):
```
load_songs(csv_path: str)                              → List[Dict]
recommend_songs(user_prefs: Dict, songs: List[Dict], k) → List[Tuple[Dict, float, str]]
```

**OOP** — used by tests:
```
Recommender(songs: List[Song])
Recommender.recommend(user: UserProfile, k)            → List[Song]
```

Both interfaces must stay aligned with the same field structure defined in the Data Model section above.

### CLI Flow (Functional Path)

```
data/songs.csv
    ↓  load_songs()
List[Dict]          ← one dict per song: id, title, artist, genre, mood,
                       energy, valence, danceability, acousticness, tempo_bpm
    +
user_prefs Dict     ← hardcoded in main.py today; future: prompted from CLI
    { genre, mood, current_genre, current_mood,
      target_energy, target_valence,
      target_danceability, target_acousticness, target_tempo }
    ↓  recommend_songs(user_prefs, songs, k=5)
List[Tuple]         ← (song_dict, score: float, explanation: str)
    ↓  main() iterates and prints
CLI output:
    "Sunrise City - Score: 0.99"
    "Because: ..."
```

---

## Scoring Formula

### Step 1: Categorical Score

Blends long-term preference with current session signal using **proximity matching** instead of binary equality:

```
family_match(a, b) →
    1.0   if a == b                          (exact match)
    0.5   if a and b share a genre/mood family (same-family partial credit)
    0.0   otherwise

genre_match = 0.30 × family_match(song.genre, favorite_genre)
            + 0.70 × family_match(song.genre, current_genre)

mood_match  = 0.30 × family_match(song.mood, favorite_mood)
            + 0.70 × family_match(song.mood, current_mood)

categorical_score = (genre_match + mood_match) / 2
```

**Genre families** (songs in the same family receive 0.5 partial credit):
- indie: indie, indie pop, folk, ballad
- electronic: electronic, synthwave, techno, lofi, ambient
- rock: rock, metal, punk
- urban: hip-hop, r&b, soul, funk
- classical: classical, country, blues, gospel
- pop: pop, disco, latin, reggae

**Mood families:**
- melancholic: sad, melancholic, moody, introspective
- calm: chill, relaxed, peaceful, dreamy
- energetic: energetic, intense, angry
- positive: happy, joyful, romantic
- focused: focused

**Why the 0.30 / 0.70 split:** Recommendations should respond to what the user is listening to right now (0.70), while still respecting long-term taste as a fallback (0.30). This reduces the filter bubble effect where a user would otherwise be locked into their all-time favorite genre/mood forever.

**Previous design issue:** The original formula used only `favorite_genre` / `favorite_mood` with binary matching (1.0 or 0.0) — session-blind with no sensitivity to current listening context and no partial credit for related genres/moods.

**Fix:** Added `current_genre` / `current_mood` for session awareness (0.70 weight). Replaced binary equality with `family_match()` to give partial credit (0.5) to related genres and moods, so a user listening to "indie" still surfaces "folk" and "indie pop" results.

### Step 2: Numeric Similarity

Tempo is normalized dynamically using the catalog's actual BPM range — no hard-coded values:

```
bpm_min   = min(song.tempo_bpm for all songs in catalog)
bpm_max   = max(song.tempo_bpm for all songs in catalog)
bpm_range = bpm_max - bpm_min

tempo_norm        = (song.tempo_bpm    - bpm_min) / bpm_range
target_tempo_norm = (user.target_tempo - bpm_min) / bpm_range

energy_sim   = 1 - |song.energy       - user.target_energy|
valence_sim  = 1 - |song.valence      - user.target_valence|
dance_sim    = 1 - |song.danceability - user.target_danceability|
acoustic_sim = 1 - |song.acousticness - user.target_acousticness|
tempo_sim    = 1 - |tempo_norm        - target_tempo_norm|
```

Each similarity is in range [0, 1]. Peak (1.0) when song value exactly matches user target.

**Previous design issue:** `tempo_bpm` was a Song field but unused in scoring.

**Fix:** Added `tempo_sim` as a scored numeric feature. Normalized dynamically from catalog min/max so the formula auto-adapts if the catalog changes — no hard-coded BPM constants.

### Step 3: Numeric Score

```
numeric_score = 0.30 × energy_sim
              + 0.25 × valence_sim
              + 0.25 × dance_sim
              + 0.10 × acoustic_sim
              + 0.10 × tempo_sim
```

**Previous design issue:** Acousticness weight was 0.20, no tempo term (weights summed to 1.0 without tempo).

**Fix:** Reduced acousticness from 0.20 → 0.10 to make room for `tempo_sim` at 0.10. Weights still sum to 1.00.

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
| Acousticness | 0.60 × 0.10 = **0.06** |
| Tempo | 0.60 × 0.10 = **0.06** |
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
