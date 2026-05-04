# Model Card: MoodSync — Agentic Smart Music Recommender

## 1. Model Name

**MoodSync 1.0**

---

## 2. Intended Use

MoodSync 1.0 accepts a natural language listening request (e.g., "something chill for late-night studying" or "pump me up for a morning run") and returns a ranked 5-song playlist with per-song explanations.

The system is built for classroom exploration of applied AI engineering. It demonstrates how a rule-based scoring engine and an LLM-powered agentic pipeline can be combined to produce explainable, reliable recommendations. It is not connected to real user listening history and does not learn from behavior.

---

## 3. How the Model Works

The system runs a three-step Plan-Act-Check pipeline.

**Step 1 — Plan (Claude LLM, `claude-haiku-4-5`)**

The user's natural language prompt is analyzed by Claude to produce a structured preference profile: `genre`, `mood`, `current_genre`, `current_mood`, numeric targets (`target_energy`, `target_valence`, `target_danceability`, `target_acousticness`, `target_tempo`), and an `emotion_intent` signal (match / uplift / energize / calm / contrast). The intent signal nudges numeric targets before scoring — for example, "uplift" increases `target_valence` and `target_energy`.

**Step 2 — Act (rule engine)**

Every song in the catalog is scored 0–1 using a deterministic formula:

- **Categorical score (40%):** Genre and mood are matched against both the user's long-term preference and current session signal (50/50 blend). Matching uses a family-proximity function: exact match = 1.0, same family = 0.5, unrelated = 0.0.
- **Numeric score (60%):** Five audio features are compared to user targets using `1 − |song_value − target|`. A `genre_gate` multiplier (1.0 / 0.5 / 0.25) caps how much a genre-foreign song can contribute to the numeric component.
- **Variety re-ranking:** A greedy re-ranker selects the final 5 one at a time, applying −0.15 penalties for consecutive same-genre or same-mood picks (max −0.30).

**Step 3 — Check (Claude LLM)**

Claude reviews the 5 draft songs against the original prompt. Songs with clear mismatches (`keep=false`) are replaced by the rule engine from the remaining catalog. The final playlist is returned with per-song `fit_score` and `fit_reason`.

**Security guardrails:** User prompts are wrapped in `<user_input>` trust-boundary tags before insertion into LLM templates, preventing prompt injection. A 500-character length limit is enforced before any API call. Extracted preference fields are validated and clamped post-LLM-response.

---

## 4. Data

### Default Catalog

301 hand-curated songs with manually assigned audio features. Genres include pop, rock, metal, punk, hip-hop, electronic, synthwave, techno, lofi, ambient, indie, folk, classical, r&b, soul, disco, latin, and jazz. Moods include happy, sad, energetic, intense, angry, relaxed, chill, dreamy, focused, moody, melancholic, romantic, and peaceful.

About 51% of songs are high-energy (energy 0.7–1.0), so low-energy users have fewer viable candidates. Only 3 songs are in the rock genre. Classical, gospel, and folk are underrepresented.

### Spotify Catalog (optional)

When Spotify credentials are configured, the system fetches up to 100 candidate tracks per request using the analyzed genre and mood as search parameters. Audio features are pulled from Spotify's API. Artist genre tags are translated to the project's vocabulary via a pre-generated mapping (`data/spotify_genre_map.json`). Mood is inferred from `valence + energy` quadrants. The CSV catalog serves as a fallback.

---

## 5. Strengths

**Fully explainable.** Every score is traceable to a hand-coded formula. No learned parameters.

**Natural language input.** The LLM Plan step removes the need for users to specify numeric targets. A vague prompt like "something for a sad Sunday morning" produces a complete preference profile.

**Self-correcting.** The Check step catches cases where the rule engine surfaces songs that technically score well but don't match the spirit of the request.

**Session-aware.** The 50/50 blend of long-term and session signals means a single off-genre listening session doesn't override established taste.

**Prompt injection resistant.** Trust-boundary tags and length limits protect the LLM system instructions from user-controlled input.

**Scalable catalog.** Spotify integration allows the same scoring formula to operate over millions of tracks rather than 35.

---

## 6. Limitations and Bias

### Bias 1 — Energy Dominance

**Observed:** With W_ENERGY = 0.60, energy alone accounted for 36% of the final score. Because 51% of the catalog is high-energy, low-energy users were systematically under-served.

**Fix:** Reduced W_ENERGY from 0.60 to 0.35. Redistributed weight to valence (0.25) and danceability (0.25).

### Bias 2 — Session Filter Bubble

**Observed:** W_SESSION = 0.70 meant one off-genre session could lock recommendations into that session's genre.

**Fix:** Rebalanced to W_LONG_TERM = 0.50 / W_SESSION = 0.50.

### Bias 3 — Genre Bleed

**Observed:** Genre-foreign songs could outscore exact-genre matches purely on energy similarity.

**Fix:** Added `genre_gate` multiplier. Exact match = 1.00, same family = 0.50, foreign = 0.25 floor.

| Genre relationship | Gate | Max numeric contribution |
|---|---|---|
| Exact match | 1.00 | 60% |
| Same family | 0.50 | 30% |
| Foreign genre | 0.25 | 15% |

### Bias 4 — Mood Conflation

**Observed:** "energetic", "intense", and "angry" were grouped as equals. Workout users received angry metal at the same 0.5 proximity as upbeat pop.

**Fix:** Split "energetic" into a singleton family. Angry/intense songs receive 0.0 proximity toward energetic users.

### Bias 5 — Weak Variety Penalty

**Observed:** −0.10/−0.10 max penalty of −0.20 couldn't break catalog score gaps that routinely exceeded 0.20.

**Fix:** Raised each penalty to −0.15 (max −0.30). Scores clamped at 0.0.

### Residual Limitations

- **Linear similarity.** `1 − |song − target|` doesn't model perceptual non-linearity in musical features.
- **Mood inference from audio features.** Mood from Spotify is inferred via valence/energy quadrants — imprecise for edge cases.
- **Contradictory profiles.** High energy + sad mood is an unresolvable conflict; the formula averages the signals.
- **Genre floor allows residual bleed.** GENRE_FLOOR = 0.25 means genre-foreign songs appear at positions 4–5 for users with well-represented genres.
- **No content understanding.** Lyrics, language, instrumentation, and context are fully ignored.
- **Non-deterministic Check step.** Claude's self-correction is probabilistic; the same draft may produce different rejections across runs.

---

## 7. Evaluation

Six user profiles were tested by inspecting top-5 output for each. For each profile, the expectation was that top results would share the user's genre and mood, and that numeric features would be close to targets.

| Profile | Genre | Mood | Outcome |
|---|---|---|---|
| High-Energy Pop | pop | happy | Clean results — exact genre/mood matches dominated top 5 |
| Chill Lofi | lofi | relaxed | Clean results — exact-lofi songs ranked above ambient after genre gate |
| Deep Intense Rock | rock | intense | Urban Beats (hip-hop) dropped from 0.709 → 0.175 after genre gate |
| High-Energy Sadness | electronic | sad | Punk/rock songs removed after genre gate + mood split |
| Acoustic Intensity | rock | intense | Storm Runner: rank 5 (0.025) → rank 1 (0.829) after all fixes |
| Relaxed Workout | pop | relaxed | Angry metal removed after mood family split |

**Most dramatic result:** Storm Runner (exact rock/intense match) ranked last at 0.025 effective score before fixes. A single acoustic feature mismatch dragged its raw score below genre-foreign competitors, and the variety re-ranker added a further −0.20 penalty. After fixes, Storm Runner ranks first at 0.829 — a +0.804 score increase and +4 rank gain.

All 8 unit tests in `tests/test_recommender.py` pass after all fixes.

---

## 8. Future Work

- **Adaptive weights.** Allow numeric weights to shift based on user feedback (thumbs up/skip), replacing fixed weights with per-user profiles.
- **Richer mood inference.** Move beyond the 5-region valence/energy heuristic — use genre context or additional audio features to assign more nuanced moods from Spotify data.
- **Contradiction detection.** Flag inherently conflicting preference profiles (high energy + sad mood) before scoring rather than silently averaging.
- **Artist diversity.** Add a variety penalty for consecutive songs by the same artist.
- **Perceptual similarity function.** Replace linear `1 − |song − target|` with a non-linear curve that better matches human perception.
- **Per-feature score breakdowns in UI.** Show users exactly which features helped or hurt each recommendation.
- **Tests for security features.** Add unit tests for `_validate_user_prompt()` (length limit, stripping) and `_validate_analyze_output()` (out-of-range fields). Add integration tests for the Spotify genre mapping utility.

---

## 9. Personal Reflection

I learned that small weight choices have large downstream effects. Changing W_ENERGY from 0.35 to 0.60 was enough to let a hip-hop song outscore a rock song for a rock user purely because their energy values were close. The formula looks precise, but it requires careful calibration to behave as intended.

The most surprising result was Storm Runner ranking last despite being the only exact genre and mood match in the Acoustic Intensity profile. A single mismatched numeric feature dragged its raw score down, and the variety re-ranker then compounded the penalty. Fixing the weights and genre gate moved it from rank 5 to rank 1.

Adding the LLM pipeline changed how I think about the system. The rule engine is fully auditable but requires a perfectly structured input. The LLM Plan step bridges the gap between vague human language and a structured numeric profile — but introduces non-determinism. The Check step adds a qualitative filter the formula alone cannot provide. The combination is more capable than either component alone, but harder to reason about end-to-end.

This changed how I think about production recommenders like Spotify. Their system likely faces the same tradeoffs, but with millions of parameters learned from listening data instead of six hand-coded weights. The biases observable and fixable by hand in 301 songs are invisible in production systems — buried inside learned parameters that no one can directly read.

The 35-song hand-coded catalog turned out to be one of the most valuable parts of the project for learning purposes. Because every song was manually curated with known feature values, I could predict exactly which song should rank where and verify the formula was behaving correctly. When Storm Runner ranked last despite being the perfect categorical match, I could trace the exact cause — acousticness similarity 0.26 — without ambiguity. A larger catalog would have made that kind of debugging nearly impossible. The small catalog forced precision: every bias had a clear, traceable example, and every fix had a measurable, verifiable result.

That said, the catalog's limitations are just as clear. With only 301 songs and significant imbalances — 51% high-energy, just 3 rock songs, almost no classical or gospel — the system cannot serve many user profiles well regardless of how good the formula is. There simply aren't enough candidates in some genres and moods. The scoring logic is sound, but a good recommender ultimately depends on having enough songs to recommend. A real system would need a catalog orders of magnitude larger, with consistent and accurate audio feature data across all entries.

This is why I hope to integrate the Spotify Web API as the next step. Spotify already provides the exact audio features this formula uses — energy, valence, danceability, acousticness, tempo — for millions of tracks. Rather than replacing the scoring engine, the Spotify integration would feed it a much richer candidate pool: up to 100 relevant tracks per request, fetched live based on the user's analyzed genre and mood. The formula, the genre gate, the variety re-ranker, and the Plan-Act-Check pipeline all remain unchanged. The only difference is the catalog stops being a bottleneck. The groundwork is already in place — the genre mapping utility, the translation layer, and the integration architecture are designed and ready. Completing the integration is the most impactful next step this project could take.

### Limitations and Biases

The deepest limitation I encountered was not a bug but a structural one: the formula is only as good as what it can measure. Energy, valence, danceability, acousticness, and tempo are all surface-level audio features. They say nothing about lyrics, language, cultural context, or instrumentation. A protest song and a party anthem can share nearly identical audio features and rank identically for an energetic user — the formula has no way to tell them apart. The same blindspot applies to mood inference from Spotify data: mapping a track's valence and energy to a single mood word collapses a lot of nuance. A slow, melancholic jazz ballad and a quiet, intimate folk song can share the same quadrant even though a listener would experience them very differently.

The catalog imbalance bias was also harder to fully fix than I expected. Raising the genre gate and splitting mood families helped, but they only work when there are enough songs in the right genre to promote in the first place. With only 3 rock songs in the default catalog, a rock user will always see genre-foreign results at positions 4–5 — not because the formula is wrong, but because the candidates simply don't exist. The scoring logic and the catalog quality are two separate problems, and fixing the formula cannot substitute for a richer dataset.

### Potential Misuse

A music recommender seems low-stakes, but there are real misuse risks once the LLM Plan step is involved. The most direct one is prompt injection: a user could craft an input designed to override the system instructions embedded in the LLM template, causing the model to return a malformed preference profile, emit arbitrary text, or leak the system prompt itself. I addressed this by wrapping all user input in `<user_input>` trust-boundary tags and enforcing a 500-character length limit before any API call. These controls make injection harder without making the system unusable for legitimate requests.

A subtler misuse risk is manipulation of the genre and mood signals that feed the Spotify search query. If the system were connected to a live catalog with ranking incentives (e.g., paid placement), a carefully crafted prompt could reliably steer the Plan step toward specific genres and then let those genre-biased results propagate through the rule engine. The deterministic scoring formula would amplify rather than filter the manipulation. The current system avoids this because the catalog is static and there are no external ranking incentives, but any production deployment with live catalog integration would need to audit this surface carefully. Validating and clamping all LLM-extracted preference fields before they influence catalog queries is the main control in place today.

### Surprises in Reliability Testing

The most surprising reliability finding was how fragile the Check step turned out to be in edge cases. The Plan step is deterministic in structure — it always returns the same fields — but the Check step evaluates qualitative fit, and that judgment varied across runs for the same draft playlist. Two identical inputs sometimes produced different rejection decisions. This is expected behavior for an LLM, but it meant that the same user prompt could produce different final playlists on different runs, even when the rule engine's ranked list was identical. Reliability in the rule engine did not translate to reliability end-to-end.

The second surprise was how the variety re-ranker interacted with edge cases. I designed it to break up monotonous playlists, but in small, genre-poor catalogs it sometimes did the opposite: by penalizing the second pick in a genre, it elevated a lower-quality genre-foreign song into a slot that a better same-genre song deserved. The penalty is calibrated for a reasonably large and balanced catalog — in a thin catalog it becomes a liability. This is a subtle reliability issue that only became visible through targeted profile testing, not through unit tests alone.
