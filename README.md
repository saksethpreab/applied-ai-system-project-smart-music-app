# 🎵 Music Recommender Simulation

![User Preference](user_preference.png)

![Recommendations](recommendations.png)

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

### What Worked Well

- **High-Energy Pop** and **Chill Lofi** produced clean, intuitive results. Top picks had exact genre and mood matches with strong numeric similarity across energy, valence, and danceability.

### Edge Cases and Surprising Results

**Acoustic Intensity — exact match penalized to last place**
Stone Runner has exact rock/intense genre and mood match but acousticness of 0.18 against a target of 0.92 (sim = 0.26). That mismatch pushed it to the bottom of the raw ranking. The variety re-ranker then applied an additional −0.20 penalty for repeated genre/mood, resulting in an effective score of 0.025 — the lowest of any recommendation shown. A perfect categorical match ended up ranked last because one numeric feature was far off.

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

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Real-world recommendation systems like Spotify or YouTube learn their behavior from billions of user interactions — plays, skips, replays, and ratings. The model adjusts millions of internal parameters during training until it gets good at predicting what a user will engage with next. Because those parameters are learned rather than hand-written, no one can point to a single formula and explain exactly why a specific song was recommended. The behavior emerges from the data, not from explicit rules.

Our system is explainable precisely because it works the opposite way. Every score comes from a hard-coded formula with fixed, human-readable weights: 40% from categorical match (genre and mood) and 60% from numeric feature similarity (energy, valence, danceability, acousticness). Every contribution to a final score can be traced and calculated by hand. A variety re-ranking step then applies explicit penalties for consecutive genre or mood repeats. Nothing is learned or hidden. This makes our system fully transparent, at the cost of personalization — it cannot improve with use, but any result can be fully audited and explained.


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

