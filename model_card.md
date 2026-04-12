# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**MoodSync 1.0**

---

## 2. Intended Use  

MoodSync 1.0 scores songs from a 35-song catalog and returns a ranked top-5 playlist for a given user.

It assumes the user has a known favorite genre, a favorite mood, and numeric targets for energy, valence, danceability, acousticness, and tempo. It also tracks what the user listened to recently as a separate session signal alongside their all-time preferences.

This system is built for classroom exploration. It is not connected to real streaming data and does not learn from user behavior.

---

## 3. How the Model Works  

Every song gets a score between 0 and 1. Higher is a better match for the user.

The score has two parts. The first part is categorical — it checks how well the song's genre and mood match what the user likes. An exact genre match scores higher than a related genre, which scores higher than a completely different one. Mood works the same way. The user's current session and their long-time history both contribute equally to this part.

The second part is numeric — it measures how close the song's energy, mood tone, danceability, acousticness, and tempo are to the user's targets. If they are far apart, the similarity is low. If they are close, the similarity is high.

Before the numeric part is counted, a genre gate is applied. If the song's genre is completely foreign to the user's preference, its numeric score is cut to 25% of its value. This stops a high-energy hip-hop track from outscoring a rock song for a rock user just because the energy values happen to be similar.

The final score is 40% from the categorical part and 60% from the numeric part.

After scoring, a re-ranker checks consecutive picks and applies a small penalty if two songs in a row share the same genre or mood. This spreads the playlist without changing which songs qualified.

The original system used only one genre and mood signal with binary matching (exact match or zero credit), no session awareness, and no genre gate. Valence, danceability, acousticness, and tempo were not scored.

---

## 4. Data  

The catalog contains 35 songs.

Genres represented include pop, rock, metal, punk, hip-hop, electronic, synthwave, techno, lofi, ambient, indie, folk, classical, r&b, soul, disco, latin, and jazz. Moods include happy, sad, energetic, intense, angry, relaxed, chill, dreamy, focused, moody, melancholic, romantic, and peaceful.

The original dataset was expanded with additional songs. New numeric fields were added to each song: valence, danceability, acousticness, and tempo in BPM.

About 51% of songs are high-energy (energy 0.7–1.0), so low-energy users have fewer candidates. Only 3 songs are in the rock genre. Classical, gospel, and folk are underrepresented. There are no songs that combine high energy with a sad or focused mood, which makes certain user profiles hard to serve well.

---

## 5. Strengths  

The system works best when the user's preferred genre and mood are well-represented in the catalog.

High-energy pop users get clean, intuitive results. All top-5 picks are pop or pop-family songs with matching energy and mood. Chill lofi users also get accurate results — low-energy lofi songs rank above ambient songs once the genre gate is applied.

Proximity matching surfaces related genres and moods instead of returning nothing when an exact match is unavailable. A lofi user will still see ambient and synthwave songs in the list, not just exact lofi tracks.

Session awareness means the system can shift recommendations based on what the user is listening to right now, without fully overriding their long-term taste.

Every recommendation is fully explainable. Each score can be broken down into its exact categorical and numeric contributions by hand.

---

## 6. Limitations and Bias 

### Bias 1 — Energy Dominance

**Observed:** With W_ENERGY = 0.60, energy alone accounted for 36% of the final score. Because 51% of the catalog is high-energy (energy 0.7–1.0), low-energy users were systematically under-served. Songs with high energy could float to the top even when their mood and genre were a poor match.

**Fix:** Reduced W_ENERGY from 0.60 to 0.35. Redistributed weight to valence (0.25) and danceability (0.25). Energy remains the largest single numeric factor but no longer dominates.

---

### Bias 2 — Session Filter Bubble

**Observed:** W_SESSION = 0.70 meant a single off-genre listening session could lock recommendations into that session's genre, effectively erasing the user's long-term taste. One accidental playlist overrode all-time history.

**Fix:** Rebalanced to W_LONG_TERM = 0.50 / W_SESSION = 0.50. Long-term and current session now contribute equally. A single session no longer dominates.

---

### Bias 3 — Genre Bleed (No Genre Gate)

**Observed:** Genre-foreign songs could outscore exact-genre matches purely on energy similarity. A hip-hop song scored higher than a rock song for a rock user when their energy values aligned. The formula had no mechanism to penalize genre mismatch in the numeric component.

**Fix:** Added a `genre_gate` multiplier to the numeric score. Songs with an exact genre match receive full credit (1.00), same-family songs receive half (0.50), and genre-foreign songs are capped at 0.25 (the floor). The floor is intentionally non-zero to preserve cross-genre discovery for users in under-represented genres (e.g., ambient, gospel, folk).

| Genre relationship | Gate | Max numeric contribution |
|---|---|---|
| Exact match | 1.00 | 60% |
| Same family | 0.50 | 30% |
| Foreign genre | 0.25 | 15% |

---

### Bias 4 — Mood Conflation (Energetic vs. Intense)

**Observed:** "energetic," "intense," and "angry" were grouped as equals in the same mood family. A user wanting energetic, upbeat music received angry metal recommendations at the same 0.5 proximity score as upbeat pop — treating fundamentally different emotional states as equivalent.

**Fix:** Split the family. "energetic" is now a singleton with no proximity neighbors. "intense" and "angry" remain in the same family but are completely separate from "energetic." Angry and intense songs no longer receive partial credit toward energetic user profiles.

---

### Bias 5 — Variety Re-ranker Penalty Too Weak

**Observed:** The original max penalty of −0.20 (−0.10 per repeated genre + −0.10 per repeated mood) could not break score gaps exceeding 0.20. In practice, catalog gaps reached 0.25+, meaning the variety re-ranker had no effect on playlists with high-scoring songs of the same genre or mood.

**Fix:** Raised each penalty from −0.10 to −0.15 (max −0.30 combined). Scores are clamped at 0.0 to keep all outputs in [0.0, 1.0].

---

### Residual Limitations

- **Contradictory profiles cannot be resolved.** High energy + sad mood is an inherently conflicting signal. The formula averages conflicting features rather than resolving them. Punk Energy (rock/angry) can still surface for an electronic/sad user when energy similarity is high.
- **Genre floor allows residual bleed.** The GENRE_FLOOR = 0.25 minimum means genre-foreign songs are never fully excluded. A rock user can still see hip-hop at position 4–5 if numeric similarity is strong.
- **Small catalog skew.** 35 songs with 51% high-energy tracks and only 3 rock songs. Under-represented genres have fewer candidates regardless of how well the formula scores them.
- **No content understanding.** Lyrics, instrumentation, time of day, and listening context are fully ignored.

---

## 7. Evaluation  

Six user profiles were tested by running `python -m src.recommender` and inspecting the top-5 output for each. For each profile, the expectation was that the top results would share the user's genre and mood, and that numeric features (energy, valence, danceability) would be close to the user's targets.

### Profiles Tested

| Profile | Genre | Mood | Outcome |
|---|---|---|---|
| High-Energy Pop | pop | happy | Clean results — exact genre/mood matches dominated top 5 |
| Chill Lofi | lofi | relaxed | Clean results — low-energy ambient/lofi songs ranked first |
| Deep Intense Rock | rock | intense | Genre bleed: hip-hop (Urban Beats) scored 0.709 before fix |
| High-Energy Sadness | electronic | sad | Contradictory profile: punk/rock surfaced due to energy match |
| Acoustic Intensity | rock | intense | Exact match ranked last due to acousticness penalty stacking |
| Relaxed Workout | pop | relaxed | Angry metal appeared before mood-family fix |

### Surprising Results

**Exact match ranked last (Acoustic Intensity).** Stone Runner had exact rock/intense genre and mood — the best possible categorical match. Yet it ranked last before fixes. The cause: acousticness 0.18 against a user target of 0.92 produced a similarity of only 0.26, pushing it down in raw scoring. The variety re-ranker then added an additional −0.20 penalty for repeated genre/mood. A perfect categorical match was punished to the bottom because one numeric feature was far off. After fixes, Stone Runner ranks first at score 0.829.

**Energy overrode genre entirely (Deep Intense Rock).** Urban Beats (hip-hop) scored 0.709 for a rock user — higher than most rock songs in the catalog. The genre was completely different, but the energy value was close enough to the target that the numeric component dominated. The formula had no gate to cap how much a genre-foreign song could contribute. After the genre-gate fix, Urban Beats dropped to an effective score of 0.175.

**Contradictory profile cannot be resolved (High-Energy Sadness).** High energy + sad mood is an internally conflicting signal — most high-energy songs are also high-valence (happy). The system surfaces Punk Energy (rock/angry) because its energy similarity (0.98 vs 0.90 target) is very high and the formula has no concept of "this combination doesn't make sense." After the genre-gate fix, rock songs disappeared from this profile's results, but the underlying contradiction remains unresolvable.

**Angry metal for a workout user (Relaxed Workout).** Before the mood-family split, "energetic" and "angry/intense" were grouped as equals. A user wanting upbeat workout music received metal recommendations at 0.5 mood proximity — the same score as upbeat pop. After splitting energetic into a singleton family, angry and intense songs received 0.0 proximity toward energetic users.

### Tests Run

The test suite in `tests/test_recommender.py` contains 8 tests covering both the OOP and functional interfaces. Tests verify: result count equals k, all scores are in [0.0, 1.0], and explanations are non-empty strings. All 8 tests pass after all fixes. No ranking-order assertions exist in the test suite, so weight changes did not break any tests.

---

## 8. Future Work  

Add more songs, especially in underrepresented genres like rock, classical, gospel, and folk.

Allow numeric targets to update automatically from listening history instead of being hardcoded. The current session window logic exists for genre and mood but not for energy, valence, or danceability.

Show per-feature score breakdowns in the output so users can see exactly which features helped or hurt each recommendation.

Add genre and mood diversity quotas across the full top-k results so variety is guaranteed, not just re-ranked.

Add context signals — time of day, activity type, or session energy trend. A user's energy target at 7 AM before a run is different from the same user at 11 PM.

Flag contradictory user profiles before scoring. A profile combining high energy and sad mood is internally conflicting. The system should surface a warning rather than silently averaging the signals.

---

## 9. Personal Reflection  

I learned that small weight choices have large downstream effects. Changing W_ENERGY from 0.35 to 0.60 was enough to let a hip-hop song outscore a rock song for a rock user, purely because their energy values were close. The formula looks precise, but it requires careful calibration to behave as intended.

The most surprising result was Storm Runner ranking last despite being the only exact genre and mood match in the Acoustic Intensity profile. I expected exact categorical matches to always rank near the top. Instead, a single mismatched numeric feature — acousticness — dragged its raw score down, and the variety re-ranker then applied an additional penalty that pushed it to near zero. Fixing the weights and genre gate was enough to move it from rank 5 to rank 1.

This changed how I think about apps like Spotify. Their system likely faces the same tradeoffs between categorical signals and numeric features, but with millions of parameters learned from listening data instead of six hand-coded weights. The biases I could observe and fix by hand in 35 songs are invisible in production systems — buried inside learned parameters that no one can directly read.
