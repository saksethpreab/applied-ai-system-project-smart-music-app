# 🎵 Music Recommender Simulation

![User Preference](user_preference.png)

![Recommendations](recommendations.png)

## Project Summary

This system recommends up to 5 songs from a hand-crafted 35-song catalog by scoring each song against a user profile. Every score is the weighted sum of a categorical match (genre and mood, 40%) and a numeric feature similarity (energy, valence, danceability, acousticness, tempo, 60%). A genre gate caps how much numeric similarity a genre-foreign song can contribute, preventing a hip-hop track from outscoring a rock track for a rock user purely on energy. A greedy variety re-ranker then applies small penalties for consecutive same-genre or same-mood picks so the final list is diverse. The entire pipeline is hand-coded with fixed weights — nothing is learned from data — making every recommendation fully auditable by hand.

---

## How The System Works

### Song features

Each `Song` carries two categorical labels and five numeric features:

| Field | Type | What it captures |
|---|---|---|
| `genre` | string | Broad musical category (e.g. pop, lofi, rock, hip-hop) |
| `mood` | string | Emotional tone (e.g. happy, chill, intense, sad) |
| `energy` | 0–1 float | Perceived intensity and activity level |
| `valence` | 0–1 float | Musical positiveness / brightness |
| `danceability` | 0–1 float | How suitable the track is for dancing |
| `acousticness` | 0–1 float | Acoustic (1.0) vs. electronic (0.0) character |
| `tempo_bpm` | float | Beats per minute (catalog range: 50–168 BPM) |

### UserProfile fields

A `UserProfile` holds two layers of preference:

- **Long-term taste** — `favorite_genre`, `favorite_mood` (stable, cross-session identity)
- **Current session** — `current_genre`, `current_mood` (the genre/mood the user has been playing in this session)
- **Numeric targets** — `target_energy`, `target_valence`, `target_danceability`, `target_acousticness`, `target_tempo`

Long-term and session signals are blended 50/50 in the final categorical score so a single off-genre session does not override established taste.

### Scoring formula

```
final_score = 0.40 × categorical_score + 0.60 × numeric_score
```

**Categorical score** (genre + mood proximity):

- Genre match and mood match are each computed against both long-term and session preference (50/50 blend).
- Match values: exact match = 1.0, same genre/mood family = 0.5, no relation = 0.0.
- Genre families group similar genres (e.g. rock/metal/punk share a family; lofi/ambient/synthwave/electronic share another). Mood families group similar moods (e.g. chill/relaxed/peaceful/dreamy/focused share a family).
- `categorical_score = 0.33 × genre_match + 0.67 × mood_match`

**Numeric score** (feature similarity):

Each feature similarity = `1 − |song_value − user_target|`. Tempo is first normalized to [0, 1] across the catalog range before comparison.

| Feature | Weight |
|---|---|
| Energy | 0.35 |
| Valence | 0.25 |
| Danceability | 0.25 |
| Acousticness | 0.10 |
| Tempo | 0.05 |

A **genre gate** is applied to the numeric score before combining: songs with an exact genre match pass through at full credit (gate = 1.0), same-family songs are capped at 50% (gate = 0.5), and genre-foreign songs are capped at 25% (gate = 0.25). The 0.25 floor preserves some cross-genre discovery rather than excluding foreign-genre songs entirely.

### Recommendation selection

1. Every song in the 35-song catalog is scored against the user profile.
2. A greedy **variety re-ranker** selects songs one at a time. Each time a song would repeat the previous pick's genre, it receives a −0.15 penalty; repeating the mood adds another −0.15. Effective scores are clamped to 0.0.
3. The top 5 songs from the re-ranked list are returned with a plain-language explanation of why each was chosen.

---

## Experiments You Tried

### What Worked Well

- **High-Energy Pop** and **Chill Lofi** produced clean, intuitive results. Top picks had exact genre and mood matches with strong numeric similarity across energy, valence, and danceability.

### Edge Cases and Surprising Results

**Acoustic Intensity — exact match penalized to last place**
Storm Runner has exact rock/intense genre and mood match but acousticness of 0.18 against a target of 0.92 (sim = 0.26). That mismatch pushed it to the bottom of the raw ranking. The variety re-ranker then applied an additional −0.20 penalty for repeated genre/mood, resulting in an effective score of 0.025 — the lowest of any recommendation shown. A perfect categorical match ended up ranked last because one numeric feature was far off.

**High-Energy Sadness — contradictory profile exposes formula limits**
High energy + sad mood is an inherently conflicting signal. The system cannot reconcile them — it averages. Result: Punk Energy (rock/angry) surfaces for an electronic/sad user purely because energy similarity (0.98 vs 0.90 target) overrides both genre and mood mismatch. The formula has no concept of "this combination doesn't make sense."

**Genre bleed in Deep Intense Rock and High-Energy Sadness**
Urban Beats (hip-hop) and Digital Rush (electronic) appear in rock-profile results. Genre is marked "different" in both cases, but mood and energy alignment score high enough to include them. The system has no hard genre filter — a sufficiently close numeric match can always override a genre miss.

### Bias Fixes Applied

After identifying six scoring biases, five targeted fixes were made to `src/recommender.py`. All changes are weight constants or small formula additions — no new fields or function signatures.

**Fix 1 — Rebalance numeric weights (energy dominance)**
`W_ENERGY` was 0.60, making energy alone 36% of the final score. With 51% of the catalog high-energy, low-energy users were systematically under-served. Reduced to 0.35; valence and danceability absorb the slack at 0.25 each. Energy remains the largest single numeric factor but no longer dominates.

| Feature | Before | After |
|---|---|---|
| Energy | 0.60 | 0.35 |
| Valence | 0.14 | 0.25 |
| Danceability | 0.14 | 0.25 |
| Acousticness | 0.06 | 0.10 |
| Tempo | 0.06 | 0.05 |

**Fix 2 — Rebalance session vs long-term (filter bubble)**
`W_SESSION = 0.70` meant one off-genre session overrode long-term taste. Changed to 50/50. A single accidental genre no longer locks the session; long-term and current signals now contribute equally.

**Fix 3 — Genre floor gate on numeric score**
Genre-foreign songs could outscore exact-genre matches on energy alone. Added a `genre_gate` multiplier to the numeric score based on the song's best genre proximity to either the long-term or session preference. Foreign-genre songs are capped at 25% of full numeric contribution (floor = 0.25 to preserve cross-genre discovery for niche users).

| Genre relationship | Gate | Max numeric contribution |
|---|---|---|
| Exact match | 1.00 | 60% |
| Same family | 0.50 | 30% |
| Foreign genre | 0.25 | 15% |

**Fix 4 — Split energetic/intense mood family**
`"energetic"`, `"intense"`, and `"angry"` were grouped as equals. A workout user wanting "energetic" got angry metal at the same 0.5 proximity as upbeat pop. Split into two families: `energetic: {energetic}` (singleton) and `intense: {intense, angry}`. Angry and intense songs no longer get proximity credit toward energetic users.

**Fix 5 — Increase variety re-ranker penalty**
The −0.10/−0.10 max penalty of −0.20 couldn't break score gaps above 0.20 (observed gaps reach 0.25+). Raised to −0.15/−0.15 (max −0.30). Scores clamped at 0.0 to keep all outputs in [0.0, 1.0].

**Result after all fixes:**

- Deep Intense Rock: Urban Beats (hip-hop) dropped from score 0.709 → 0.175 effective — genre-foreign songs no longer competitive
- Relaxed Workout: no angry/metal songs in top 5 (mood split fix)
- High-Energy Sadness: rock songs gone from results entirely (genre gate + mood split combined)
- Acoustic Intensity: Storm Runner still #1 at 0.829 — exact match rewarded correctly
- All scores remain in [0.0, 1.0]

### Sample Outputs: Before vs. After Bias Fixes

Legend: 🟢 exact genre · 🟡 same genre family · 🔴 different genre

Status tracks where each **After** entry ranked **Before**:
➖ No Change &nbsp;·&nbsp; 🔺 Up [N] &nbsp;·&nbsp; 🔻 Down [N] &nbsp;·&nbsp; 🆕 New Entry (not in top 5 before)

*Before scores are approximate from screenshots. E/V/D/A = similarity scores [0–1] against user targets. Before-only E/V/D/A values marked ~ are estimated from song features derived from other profiles.*

---

#### High-Energy Pop — genre: pop · mood: happy

| # | Before | After (Current) | Status |
|---|---|---|---|
| 1 | 🟢 **Sunrise City** ~0.974<br>genre: exact (pop/pop) · mood: exact (happy/happy)<br>E: 0.97 · V: 0.91 · D: 0.98 · A: 0.97 | 🟢 **Sunrise City** 0.975<br>genre: exact (pop/pop) · mood: exact (happy/happy)<br>E: 0.97 · V: 0.91 · D: 0.99 · A: 0.97 | ➖ No Change |
| 2 | 🟡 **Disco Magic** ~0.768<br>genre: close (disco/pop) · mood: exact (happy/happy)<br>E: 0.97 · V: 0.91 · D: 0.92 · A: 1.00 | 🟢 **Festival Lights** 0.648<br>genre: exact (pop/pop) · mood: exact (happy/happy)<br>E: 0.92 · V: 0.89 · D: 0.91 · A: 0.96 | 🔺 Up 1 |
| 3 | 🟢 **Festival Lights** ~0.847<br>genre: exact (pop/pop) · mood: exact (happy/happy)<br>E: 0.92 · V: 0.89 · D: 0.91 · A: 0.96 | 🟢 **Gym Hero** 0.540<br>genre: exact (pop/pop) · mood: different (intense/happy)<br>E: 0.92 · V: 0.98 · D: 0.90 · A: 0.90 | 🔺 Up 2 |
| 4 | 🟡 **Latin Fiesta** ~0.841<br>genre: close (latin/pop) · mood: exact (happy/happy)<br>E: 0.94 · V: 0.94 · D: 0.96 · A: 0.53 | 🟡 **Disco Magic** 0.618<br>genre: close (disco/pop) · mood: exact (happy/happy)<br>E: 0.97 · V: 0.91 · D: 0.92 · A: 1.00 | 🔻 Down 2 |
| 5 | 🟢 **Gym Hero** ~0.760<br>genre: exact (pop/pop) · mood: different (intense/happy)<br>E: 0.92 · V: 0.98 · D: 0.90 · A: 0.90 | 🟡 **Latin Fiesta** 0.455<br>genre: close (latin/pop) · mood: exact (happy/happy)<br>E: 0.94 · V: 0.94 · D: 0.96 · A: 0.53 | 🔻 Down 1 |

Exact-genre picks (🟢) rose relative to same-family picks (🟡 Disco Magic, Latin Fiesta). Energy weight reduction shrunk absolute scores for all, but shifted ranking toward songs with balanced feature similarity rather than just high energy.

---

#### Chill Lofi — genre: lofi · mood: relaxed / current: chill

| # | Before | After (Current) | Status |
|---|---|---|---|
| 1 | 🟢 **Library Rain** ~0.881<br>genre: exact (lofi/lofi) · mood: exact (chill/chill)<br>E: 0.90 · V: 0.85 · D: 0.77 · A: 0.89 | 🟢 **Library Rain** 0.845<br>genre: exact (lofi/lofi) · mood: exact (chill/chill)<br>E: 0.90 · V: 0.85 · D: 0.77 · A: 0.89 | ➖ No Change |
| 2 | 🟡 **Ambient Dream** ~0.727<br>genre: close (ambient/lofi) · mood: close (dreamy/chill)<br>E: 0.95 · V: 0.78 · D: 0.93 · A: 0.86 | 🟢 **Focus Flow** 0.622<br>genre: exact (lofi/lofi) · mood: close (focused/chill)<br>E: 0.85 · V: 0.86 · D: 0.75 · A: 0.97 | 🔺 Up 3 |
| 3 | 🟢 **Midnight Coding** ~0.875<br>genre: exact (lofi/lofi) · mood: exact (chill/chill)<br>E: 0.83 · V: 0.89 · D: 0.73 · A: 0.96 | 🟢 **Midnight Coding** 0.685<br>genre: exact (lofi/lofi) · mood: exact (chill/chill)<br>E: 0.83 · V: 0.89 · D: 0.73 · A: 0.96 | ➖ No Change |
| 4 | 🟡 **Deep Space** ~0.815<br>genre: close (ambient/lofi) · mood: exact (chill/chill)<br>E: 1.00 · V: 0.81 · D: 0.96 · A: 0.88 | 🟡 **Ambient Dream** 0.466<br>genre: close (ambient/lofi) · mood: close (dreamy/chill)<br>E: 0.95 · V: 0.78 · D: 0.93 · A: 0.86 | 🔻 Down 2 |
| 5 | 🟢 **Focus Flow** ~0.808<br>genre: exact (lofi/lofi) · mood: close (focused/chill)<br>E: 0.85 · V: 0.86 · D: 0.75 · A: 0.97 | 🟡 **Deep Space** 0.393<br>genre: close (ambient/lofi) · mood: exact (chill/chill)<br>E: 1.00 · V: 0.81 · D: 0.96 · A: 0.88 | 🔻 Down 1 |

Ambient songs (Deep Space E:1.00, Ambient Dream E:0.95) had strong energy and danceability sims — before the genre gate they competed with exact-lofi songs on numeric score alone. After, their same-family numeric contribution is capped at 50%, and exact-lofi Focus Flow (A:0.97) overtakes both.

---

#### Deep Intense Rock — genre: rock · mood: intense / angry

| # | Before | After (Current) | Status |
|---|---|---|---|
| 1 | 🟢 **Storm Runner** ~0.867<br>genre: exact (rock/rock) · mood: close (intense/angry)<br>E: 0.97 · V: 0.77 · D: 0.79 · A: 1.00 | 🟢 **Storm Runner** 0.855<br>genre: exact (rock/rock) · mood: close (intense/angry)<br>E: 0.97 · V: 0.77 · D: 0.79 · A: 1.00 | ➖ No Change |
| 2 | 🟡 **Punk Energy** ~0.801<br>genre: close (punk/rock) · mood: exact (angry/angry)<br>E: 0.98 · V: 0.90 · D: 0.73 · A: 0.98 | 🟡 **Punk Energy** 0.534<br>genre: close (punk/rock) · mood: exact (angry/angry)<br>E: 0.98 · V: 0.90 · D: 0.73 · A: 0.98 | ➖ No Change |
| 3 | 🟡 **Metal Storm** ~0.754<br>genre: close (metal/rock) · mood: close (intense/angry)<br>E: 0.91 · V: 0.93 · D: 0.81 · A: 0.96 | 🟡 **Metal Storm** 0.532<br>genre: close (metal/rock) · mood: close (intense/angry)<br>E: 0.91 · V: 0.93 · D: 0.81 · A: 0.96 | ➖ No Change |
| 4 | 🔴 **Urban Beats** ~0.620<br>genre: diff (hip-hop/rock) · mood: close (intense/angry)<br>E: 0.99 · V: 0.64 · D: 0.73 · A: 0.92 | 🔴 **Urban Beats** 0.175<br>genre: diff (hip-hop/rock) · mood: close (intense/angry)<br>E: 0.99 · V: 0.64 · D: 0.73 · A: 0.92 | ➖ No Change\* |
| 5 | 🔴 **Digital Rush** ~0.573<br>genre: diff (electronic/rock) · mood: **close→diff** (energetic/angry)<br>E: 0.94 · V: 0.62 · D: 0.53 · A: 0.98 | 🔴 **Gym Hero** 0.162<br>genre: diff (pop/rock) · mood: close (intense/angry)<br>E: 0.95 · V: 0.48 · D: 0.57 · A: 0.95 | 🆕 New Entry |

\*Urban Beats held rank 4 but score collapsed ~0.620 → 0.175 — genre gate capped its foreign-genre numeric contribution to 25%. Digital Rush fell off entirely: the genre gate cut its score, and the mood split changed its label from **close** (energetic was in the same old family as angry) to **different** (energetic is now a singleton), removing the 0.5 proximity credit it previously received.

---

#### High-Energy Sadness — genre: electronic · mood: sad

| # | Before | After (Current) | Status |
|---|---|---|---|
| 1 | 🟢 **Digital Rush** ~0.711<br>genre: exact (electronic/electronic) · mood: diff (energetic/sad)<br>E: 0.96 · V: 0.57 · D: 0.93 · A: 0.93 | 🟢 **Digital Rush** 0.643<br>genre: exact (electronic/electronic) · mood: diff (energetic/sad)<br>E: 0.96 · V: 0.57 · D: 0.93 · A: 0.93 | ➖ No Change |
| 2 | 🟡 **Night Drive Loop** ~0.694<br>genre: close (synthwave/electronic) · mood: close (moody/sad)<br>E: 0.85 · V: 0.71 · D: 0.88 · A: 0.93 | 🟢 **Neon Nights** 0.483<br>genre: exact (electronic/electronic) · mood: diff (happy/sad)<br>E: 0.97 · V: 0.41 · D: 1.00 · A: 0.97 | 🆕 New Entry |
| 3 | 🔴 **Punk Energy** ~0.641<br>genre: diff (punk/electronic) · mood: diff (angry/sad)<br>E: ~1.00 · V: ~0.85 · D: ~0.87 · A: ~0.97 | 🟡 **Night Drive Loop** 0.448<br>genre: close (synthwave/electronic) · mood: close (moody/sad)<br>E: 0.85 · V: 0.71 · D: 0.88 · A: 0.93 | 🔻 Down 1 |
| 4 | 🔴 **Metal Storm** ~0.503<br>genre: diff (metal/electronic) · mood: diff (intense/sad)<br>E: ~0.89 · V: ~0.98 · D: ~0.79 · A: ~0.91 | 🔴 **Autumn Thoughts** 0.357<br>genre: diff (indie/electronic) · mood: exact (sad/sad)<br>E: 0.58 · V: 0.68 · D: 0.59 · A: 0.37 | 🆕 New Entry |
| 5 | 🔴 **Urban Beats** ~0.480<br>genre: diff (hip-hop/electronic) · mood: diff (intense/sad)<br>E: ~0.99 · V: ~0.59 · D: ~0.87 · A: ~0.97 | 🟡 **Techno Pulse** 0.329<br>genre: close (techno/electronic) · mood: diff (energetic/sad)<br>E: 0.98 · V: 0.62 · D: 0.96 · A: 0.90 | 🆕 New Entry |

Punk Energy, Metal Storm, and Urban Beats all had near-perfect E/D/A sims against this user's targets — before the genre gate those numeric scores were uncapped, letting them outrank correct-genre songs. After, genre-foreign numeric contribution is capped at 25%. All three drop off; three electronic-family songs replace them. Before-only E/V/D/A marked ~ are estimated from song features derived from other profile runs.

---

#### Acoustic Intensity — genre: rock · mood: intense ⚡ Most dramatic change

| # | Before | After (Current) | Status |
|---|---|---|---|
| 1 | 🟡 **Metal Storm** ~0.732<br>genre: close (metal/rock) · mood: exact (intense/intense)<br>E: 0.91 · V: 0.93 · D: 0.51 · A: 0.14 | 🟢 **Storm Runner** 0.829<br>genre: exact (rock/rock) · mood: exact (intense/intense)<br>E: 0.97 · V: 0.77 · D: 0.49 · **A: 0.18** | 🔺 **Up 4** |
| 2 | 🟡 **Punk Energy** ~0.641<br>genre: close (punk/rock) · mood: close (angry/intense)<br>E: 0.98 · V: 0.90 · D: 0.43 · A: 0.20 | 🟡 **Punk Energy** 0.422<br>genre: close (punk/rock) · mood: close (angry/intense)<br>E: 0.98 · V: 0.90 · D: 0.43 · A: 0.20 | ➖ No Change |
| 3 | 🔴 **Urban Beats** ~0.583<br>genre: diff (hip-hop/rock) · mood: exact (intense/intense)<br>E: 0.99 · V: 0.64 · D: 0.43 · A: 0.26 | 🟡 **Metal Storm** 0.553<br>genre: close (metal/rock) · mood: exact (intense/intense)<br>E: 0.91 · V: 0.93 · D: 0.51 · A: 0.14 | 🔻 Down 2 |
| 4 | 🔴 **Digital Rush** ~0.481<br>genre: diff (electronic/rock) · mood: **close→diff** (energetic/intense)<br>E: 0.94 · V: 0.62 · D: 0.23 · A: 0.16 | 🔴 **Urban Beats** 0.220<br>genre: diff (hip-hop/rock) · mood: exact (intense/intense)<br>E: 0.99 · V: 0.64 · D: 0.43 · A: 0.26 | 🔻 Down 1 |
| 5 | 🟢 **Storm Runner** **~0.025**<br>genre: exact (rock/rock) · mood: exact (intense/intense)<br>E: 0.97 · V: 0.77 · D: 0.49 · **A: 0.18** ← culprit | 🔴 **Gym Hero** 0.205<br>genre: diff (pop/rock) · mood: exact (intense/intense)<br>E: 0.95 · V: 0.48 · D: 0.27 · A: 0.13 | 🆕 New Entry |

Storm Runner had perfect genre+mood categorical score (exact on both) yet ranked dead last at **0.025** effective. Its **A:0.18** acousticness sim (song acousticness ≈ 0.10 vs user target 0.92) dragged its raw score down, then the old variety re-ranker applied a −0.20 penalty for repeated rock/intense — pushing effective to near zero. After fixes: energy weight reduction (0.60→0.35) gives acousticness more relative influence; the genre gate removes genre-foreign competitors Urban Beats and Digital Rush from competing on equal numeric footing with an exact-genre song. Storm Runner jumps to rank **1** at **0.829** — a **+0.804** score increase and **+4** rank gain.

---
**Summary**
Each fix addressed a root cause rather than a symptom. Energy dominance was caused by a single feature weight being too large relative to the others — redistributing weight to valence and danceability restored balance without discarding energy as a signal. The session filter bubble stemmed from over-weighting current listening context, which let a single off-genre session override established taste — equal 50/50 long-term/session weighting keeps both signals honest. Genre bleed existed because the numeric score had no mechanism to penalize genre mismatch — the genre gate provides that cap without fully excluding cross-genre discovery. Mood conflation was a taxonomy error: grouping "energetic" with "angry/intense" treated fundamentally different emotional states as equivalent — separating "energetic" into its own singleton family corrects the grouping at the source. The variety re-ranker was too weak because the −0.10 penalty could not break score gaps that commonly exceeded 0.10 — raising it to −0.15 makes the penalty meaningful against real catalog variance. Taken together, the fixes push the system toward a consistent principle: **categorical signals (genre, mood) should constrain the result pool; numeric signals (energy, valence, danceability) should differentiate within that pool.** Before the fixes, strong numeric similarity could override weak categorical fit entirely. After, it cannot.

---

## Limitations and Risks

- **Tiny, static catalog.** 35 songs cannot represent musical diversity. Adding even one new song requires manually entering all numeric features — there is no ingestion pipeline.
- **Catalog imbalance.** ~51% of the catalog is high-energy, so low-energy users have fewer viable matches. Rock and classical are also underrepresented (6 and 7 songs respectively), while the electronic and pop families have 9 and 8 songs each.
- **No listening history or feedback loop.** The system cannot learn from plays, skips, or ratings. User profiles are defined up front and never updated. Real recommenders improve with every interaction; this one cannot.
- **No lyrics, language, or audio content.** The system knows nothing about what a song is actually saying or how it sounds beyond five hand-estimated numeric features. Songs in a language the user doesn't speak score identically to those in their native language.
- **Hand-coded genre and mood families.** Grouping lofi with ambient or angry with intense is a subjective editorial choice. A different taxonomy would produce different results, and the current one cannot capture nuance (e.g. "focus lofi" vs. "aesthetic lofi").
- **Contradictory profiles cannot be reconciled.** A user with high energy and sad mood presents an inherent conflict. The formula averages the signals rather than recognizing the tension, which can surface emotionally mismatched results.
- **Greedy variety re-ranker is not globally optimal.** Penalties are applied one pick at a time; a different ordering of the same songs might produce higher overall diversity with fewer penalties.

---

## Reflection

The [**Model Card**](model_card.md) documents this system's intended use, design decisions, strengths, limitations, and evaluation in detail.

Real-world recommendation systems like Spotify or YouTube learn their behavior from billions of user interactions — plays, skips, replays, and ratings. The model adjusts millions of internal parameters during training until it gets good at predicting what a user will engage with next. Because those parameters are learned rather than hand-written, no one can point to a single formula and explain exactly why a specific song was recommended. The behavior emerges from the data, not from explicit rules.

Our system is explainable precisely because it works the opposite way. Every score comes from a hard-coded formula with fixed, human-readable weights: 40% from categorical match (genre and mood) and 60% from numeric feature similarity (energy, valence, danceability, acousticness). Every contribution to a final score can be traced and calculated by hand. A variety re-ranking step then applies explicit penalties for consecutive genre or mood repeats. Nothing is learned or hidden. This makes our system fully transparent, at the cost of personalization — it cannot improve with use, but any result can be fully audited and explained.