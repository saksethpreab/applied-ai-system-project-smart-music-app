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

genre_match = 0.50 × family_match(song.genre, favorite_genre)
            + 0.50 × family_match(song.genre, current_genre)

mood_match  = 0.50 × family_match(song.mood, favorite_mood)
            + 0.50 × family_match(song.mood, current_mood)

categorical_score = 0.33 × genre_match + 0.67 × mood_match
```

**Genre families** (songs in the same family receive 0.5 partial credit):
- indie: indie, indie pop, folk
- electronic: electronic, synthwave, techno, lofi, ambient
- rock: rock, metal, punk
- urban: hip-hop, r&b, soul, funk
- classical: classical, country, blues, gospel, jazz
- pop: pop, disco, latin, reggae, ballad

**Mood families:**
- melancholic: sad, melancholic, moody, introspective
- calm: chill, relaxed, peaceful, dreamy, focused
- energetic: energetic
- intense: intense, angry
- positive: happy, joyful, romantic

**Why the 0.50 / 0.50 split:** Equal weighting balances current listening context against long-term taste. An earlier 0.30/0.70 split caused a filter bubble: one off-genre session could override all-time history, locking recommendations into the session's genre. At 50/50, neither signal dominates — a user's established taste remains equally weighted against what they're listening to right now.

**Previous design issue:** The original formula used only `favorite_genre` / `favorite_mood` with binary matching (1.0 or 0.0) — session-blind with no sensitivity to current listening context and no partial credit for related genres/moods.

**Fix:** Added `current_genre` / `current_mood` for session awareness (0.70 weight). Replaced binary equality with `family_match()` to give partial credit (0.5) to related genres and moods, so a user listening to "indie" still surfaces "folk" and "indie pop" results.

### Step 2: Numeric Similarity

Tempo is normalized using fixed BPM constants (`BPM_MIN = 60`, `BPM_MAX = 200`, `BPM_RANGE = 140`):

```
BPM_MIN   = 60.0
BPM_MAX   = 200.0
BPM_RANGE = 140.0

tempo_norm        = (song.tempo_bpm    - BPM_MIN) / BPM_RANGE
target_tempo_norm = (user.target_tempo - BPM_MIN) / BPM_RANGE

energy_sim   = 1 - |song.energy       - user.target_energy|
valence_sim  = 1 - |song.valence      - user.target_valence|
dance_sim    = 1 - |song.danceability - user.target_danceability|
acoustic_sim = 1 - |song.acousticness - user.target_acousticness|
tempo_sim    = 1 - |tempo_norm        - target_tempo_norm|
```

Each similarity is in range [0, 1]. Peak (1.0) when song value exactly matches user target.

**Previous design issue:** `tempo_bpm` was a Song field but unused in scoring.

**Fix:** Added `tempo_sim` as a scored numeric feature. Normalized using fixed constants (`BPM_MIN = 60.0`, `BPM_MAX = 200.0`, `BPM_RANGE = 140.0`) for stable scores regardless of catalog composition.

### Step 3: Numeric Score

```
genre_gate = max(
    family_match(song.genre, favorite_genre),
    family_match(song.genre, current_genre),
    0.25,                                      ← floor prevents complete exclusion
)

numeric_score = (0.35 × energy_sim
              + 0.25 × valence_sim
              + 0.25 × dance_sim
              + 0.10 × acoustic_sim
              + 0.05 × tempo_sim) × genre_gate
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
| Genre (categorical) | 0.40 × 0.33 = **0.132** |
| Mood (categorical) | 0.40 × 0.67 = **0.268** |
| Energy | 0.60 × 0.35 × genre_gate ≤ **0.210** |
| Valence | 0.60 × 0.25 × genre_gate ≤ **0.150** |
| Danceability | 0.60 × 0.25 × genre_gate ≤ **0.150** |
| Acousticness | 0.60 × 0.10 × genre_gate ≤ **0.060** |
| Tempo | 0.60 × 0.05 × genre_gate ≤ **0.030** |
| **Total** | **1.00** |

`genre_gate` scales all numeric terms simultaneously: exact-genre match = 1.00, same-family = 0.50, foreign genre = 0.25 (floor).

---

## Ranking Rule

After scoring, songs are re-ordered using a greedy variety re-ranker to prevent playlist monotony:

```
For each pick from the remaining pool:
    effective_score = final_score
                    - 0.15 if same genre as previous pick
                    - 0.15 if same mood as previous pick
    effective_score = max(effective_score, 0.0)   ← clamped to keep scores in [0, 1]
    Pick the song with highest effective_score
```

Maximum variety penalty per song: −0.30 (both genre and mood repeat).

This is applied **after** scoring, so it only shuffles the top pool — it does not surface low-scoring songs.

---

## Profile Update Rules (Design Intent)

These rules define how `UserProfile` fields should update over time as the user listens:

- `favorite_genre` / `favorite_mood` → most frequent genre/mood across **all-time** listen history
- `current_genre` / `current_mood` → most frequent genre/mood in the **last N songs** played (session window, recommended N = 5–10)

Not yet implemented in `src/recommender.py`.

---

## Known Limitations

- **Small catalog** — 35 songs. Real recommenders operate on millions of tracks.
- **Static numeric targets** — `target_energy`, `target_valence`, etc. do not adapt over time. Only categorical signals are session-aware.
- **Contradictory profiles** — profiles that combine conflicting signals (e.g. high energy + sad mood, high energy + relaxed mood) cannot be fully reconciled. The formula averages conflicting features rather than resolving them.
- **Genre floor still allows bleed** — the `GENRE_FLOOR = 0.25` minimum means genre-foreign songs are never fully excluded. Niche users with under-represented genres benefit from this, but a rock user can still see hip-hop at position 4–5.
- **Partial filter bubble mitigation** — the 50/50 session/long-term split reduces but does not eliminate echo chamber risk. If a user always listens to the same genre, `current_genre` and `favorite_genre` converge and the split has no effect.
- **No content understanding** — lyrics, instrumentation, and context (time of day, activity) are ignored.
- **Variety ranking is post-hoc** — applies only to the already-scored top pool. Low-scoring songs from different genres cannot surface regardless of variety.

---

## Design Decisions

**Explainability over accuracy.** All weights are hand-set and fixed. Any score can be calculated by hand and fully audited. This is the opposite of production ML recommenders (Spotify, YouTube) where weights are learned from billions of interactions and individual predictions cannot be explained.

**Session and long-term signals weighted equally.** A 0.50 / 0.50 split treats current session and all-time history as equally valid taste signals. An earlier 0.70 session weight caused a filter bubble — one off-genre session overrode established taste. Equal weighting prevents any single session from dominating.

**Genre floor prevents full exclusion.** `GENRE_FLOOR = 0.25` means genre-foreign songs always contribute at least 25% of their numeric score. This is intentional: niche genres (ambient, gospel, folk) are under-represented in the catalog, so a hard genre gate would leave those users with near-zero scores for most songs. The floor preserves cross-genre discovery while still heavily penalizing genre mismatch.

**Mood families split energetic from intense/angry.** These had been grouped as equals, but "energetic" (upbeat, positive activation) and "angry/intense" (negative valence, aggression) describe fundamentally different emotional states. Splitting them prevents workout-profile users from receiving aggressive metal recommendations at family-match proximity.

**Categorical weight (0.40) is high by design.** Genre and mood are the most interpretable features — matching them produces results that feel "obviously right" to users. Numeric features (energy, valence, etc.) provide fine-grained differentiation within a genre/mood bucket.
