import csv
import heapq
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ── Genre & Mood proximity families ──────────────────────────────────────────
GENRE_FAMILIES = {
    "indie":      {"indie", "indie pop", "folk", "dream pop", "shoegaze"},
    "electronic": {"electronic", "synthwave", "techno", "lofi", "ambient", "house", "trance", "chillwave"},
    "rock":       {"rock", "metal", "punk", "grunge", "emo", "christian rock"},
    "urban":      {"hip-hop", "r&b", "soul", "funk", "trap"},
    "classical":  {"classical", "country", "blues", "gospel", "jazz", "bossa nova", "hymn", "spiritual", "worship"},
    "pop":        {"pop", "disco", "latin", "reggae", "ballad", "k-pop"},
    "world":      {"afrobeats", "flamenco", "celtic"},
}

MOOD_FAMILIES = {
    "melancholic": {"sad", "melancholic", "moody", "introspective", "nostalgic", "wistful"},
    "calm":        {"chill", "relaxed", "peaceful", "dreamy", "focused"},
    "energetic":   {"energetic", "euphoric", "empowered"},
    "intense":     {"intense", "angry", "anxious"},
    "positive":    {"happy", "joyful", "romantic", "hopeful", "uplifting"},
}

# Top-level score weights
W_CAT = 0.40
W_NUM = 0.60

# Categorical: long-term vs session blend
W_LONG_TERM = 0.50
W_SESSION   = 0.50

# Numeric feature weights (must sum to 1.0)
W_ENERGY      = 0.35
W_VALENCE     = 0.25
W_DANCE       = 0.25
W_ACOUSTICNESS = 0.10
W_TEMPO       = 0.05

# Categorical sub-weights for genre vs mood (must sum to 1.0)
W_GENRE_CAT = 0.50
W_MOOD_CAT  = 0.50

# Fixed BPM scale for stable tempo normalization regardless of catalog size
BPM_MIN   = 60.0
BPM_MAX   = 200.0
BPM_RANGE = BPM_MAX - BPM_MIN  # 140.0

# Minimum genre-gate multiplier applied to numeric score.
# Keeps genre-foreign songs eligible (cross-genre discovery) but caps
# their numeric contribution so exact-genre matches always win on genre.
GENRE_FLOOR = 0.10


def _family_match(a: str, b: str, families: dict) -> float:
    """Return 1.0 for exact match, 0.5 for same family, 0.0 otherwise."""
    if a == b:
        return 1.0
    for members in families.values():
        if a in members and b in members:
            return 0.5
    return 0.0


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre:      str    # long-term: most played genre across all sessions
    favorite_mood:       str    # long-term: most played mood across all sessions
    current_genre:       str    # short-term: most common genre in current session
    current_mood:        str    # short-term: most common mood in current session
    target_energy:       float  # 0–1
    target_valence:      float  # 0–1
    target_danceability: float  # 0–1
    target_acousticness: float  # 0–1
    target_tempo:        float  # raw BPM — normalized at score time using catalog range


def _score_song(song: Song, user: UserProfile) -> float:
    """Score a Song dataclass against a UserProfile."""
    genre_match = (W_LONG_TERM * _family_match(song.genre, user.favorite_genre, GENRE_FAMILIES)
                 + W_SESSION   * _family_match(song.genre, user.current_genre,  GENRE_FAMILIES))
    mood_match  = (W_LONG_TERM * _family_match(song.mood,  user.favorite_mood,  MOOD_FAMILIES)
                 + W_SESSION   * _family_match(song.mood,  user.current_mood,   MOOD_FAMILIES))
    categorical_score = W_GENRE_CAT * genre_match + W_MOOD_CAT * mood_match

    tempo_norm        = (song.tempo_bpm    - BPM_MIN) / BPM_RANGE
    target_tempo_norm = (user.target_tempo - BPM_MIN) / BPM_RANGE

    energy_sim   = 1 - abs(song.energy       - user.target_energy)
    valence_sim  = 1 - abs(song.valence       - user.target_valence)
    dance_sim    = 1 - abs(song.danceability  - user.target_danceability)
    acoustic_sim = 1 - abs(song.acousticness  - user.target_acousticness)
    tempo_sim    = 1 - abs(tempo_norm         - target_tempo_norm)

    genre_gate = max(
        _family_match(song.genre, user.favorite_genre, GENRE_FAMILIES),
        _family_match(song.genre, user.current_genre,  GENRE_FAMILIES),
        GENRE_FLOOR,
    )
    numeric_score = (W_ENERGY       * energy_sim
                   + W_VALENCE      * valence_sim
                   + W_DANCE        * dance_sim
                   + W_ACOUSTICNESS * acoustic_sim
                   + W_TEMPO        * tempo_sim) * genre_gate

    return W_CAT * categorical_score + W_NUM * numeric_score


def _score_song_dict(song: dict, user: dict) -> float:
    """Score a song dict against a user_prefs dict (functional interface)."""
    genre_match = (W_LONG_TERM * _family_match(song["genre"], user["genre"],         GENRE_FAMILIES)
                 + W_SESSION   * _family_match(song["genre"], user["current_genre"],  GENRE_FAMILIES))
    mood_match  = (W_LONG_TERM * _family_match(song["mood"],  user["mood"],           MOOD_FAMILIES)
                 + W_SESSION   * _family_match(song["mood"],  user["current_mood"],   MOOD_FAMILIES))
    categorical_score = W_GENRE_CAT * genre_match + W_MOOD_CAT * mood_match

    tempo_norm        = (song["tempo_bpm"]    - BPM_MIN) / BPM_RANGE
    target_tempo_norm = (user["target_tempo"] - BPM_MIN) / BPM_RANGE

    energy_sim   = 1 - abs(song["energy"]       - user["target_energy"])
    valence_sim  = 1 - abs(song["valence"]       - user["target_valence"])
    dance_sim    = 1 - abs(song["danceability"]  - user["target_danceability"])
    acoustic_sim = 1 - abs(song["acousticness"]  - user["target_acousticness"])
    tempo_sim    = 1 - abs(tempo_norm             - target_tempo_norm)

    genre_gate = max(
        _family_match(song["genre"], user["genre"],        GENRE_FAMILIES),
        _family_match(song["genre"], user["current_genre"], GENRE_FAMILIES),
        GENRE_FLOOR,
    )
    numeric_score = (W_ENERGY       * energy_sim
                   + W_VALENCE      * valence_sim
                   + W_DANCE        * dance_sim
                   + W_ACOUSTICNESS * acoustic_sim
                   + W_TEMPO        * tempo_sim) * genre_gate

    return W_CAT * categorical_score + W_NUM * numeric_score


def _explain_dict(song: dict, user: dict, _score: float) -> str:
    """Natural-language explanation for why a song was recommended."""
    genre_m = _family_match(song["genre"], user["current_genre"], GENRE_FAMILIES)
    mood_m  = _family_match(song["mood"],  user["current_mood"],  MOOD_FAMILIES)

    if genre_m == 1.0:
        genre_phrase = f"fits your {song['genre']} preference"
    elif genre_m == 0.5:
        genre_phrase = f"is a {song['genre']} pick adjacent to your {user['current_genre']} taste"
    else:
        genre_phrase = f"ventures outside your usual {user['current_genre']} into {song['genre']}"

    if mood_m == 1.0:
        mood_phrase = f"matches your {song['mood']} mood"
    elif mood_m == 0.5:
        mood_phrase = f"carries a related {song['mood']} feel"
    else:
        mood_phrase = f"offers a {song['mood']} contrast to your current mood"

    energy = song["energy"]
    energy_word = "high-energy" if energy > 0.66 else ("mid-energy" if energy > 0.33 else "laid-back")

    acoustic_word = (
        "acoustic" if song["acousticness"] > 0.6 else
        ("lightly produced" if song["acousticness"] > 0.3 else "fully produced")
    )

    bpm = song["tempo_bpm"]
    tempo_word = "fast" if bpm > 130 else ("medium-tempo" if bpm > 95 else "slow")

    return (
        f"This {song['genre']} track {genre_phrase} and {mood_phrase}. "
        f"Expect a {energy_word}, {acoustic_word} sound at a {tempo_word} pace ({bpm:.0f} BPM)."
    )


def _explain_numeric(song: dict, user: dict, score: float) -> str:
    """Numeric breakdown explanation for deeper inspection."""
    genre_cur_m = _family_match(song["genre"], user["current_genre"], GENRE_FAMILIES)
    mood_cur_m  = _family_match(song["mood"],  user["current_mood"],  MOOD_FAMILIES)
    genre_label = "exact" if genre_cur_m == 1.0 else ("close" if genre_cur_m == 0.5 else "different")
    mood_label  = "exact" if mood_cur_m  == 1.0 else ("close" if mood_cur_m  == 0.5 else "different")

    energy_sim   = 1 - abs(song["energy"]       - user["target_energy"])
    valence_sim  = 1 - abs(song["valence"]       - user["target_valence"])
    dance_sim    = 1 - abs(song["danceability"]  - user["target_danceability"])
    acoustic_sim = 1 - abs(song["acousticness"]  - user["target_acousticness"])

    return (
        f"Genre {genre_label} ({song['genre']} / {user['current_genre']}), "
        f"mood {mood_label} ({song['mood']} / {user['current_mood']}). "
        f"Energy sim: {energy_sim:.2f}, Valence sim: {valence_sim:.2f}, "
        f"Dance sim: {dance_sim:.2f}, Acoustic sim: {acoustic_sim:.2f}. "
        f"Score: {score:.3f}"
    )


def _apply_variety_ranking(scored: list) -> list:
    """Greedy variety re-ranker: penalise consecutive same-genre/mood picks."""
    pool = list(scored)
    ranked = []
    last_genre = last_mood = None
    while pool:
        best_idx, best_eff = None, -999.0
        for i, (s, sc, ex) in enumerate(pool):
            adj = ((-0.15 if last_genre and s["genre"] == last_genre else 0.0) +
                   (-0.15 if last_mood  and s["mood"]  == last_mood  else 0.0))
            if sc + adj > best_eff:
                best_eff, best_idx = sc + adj, i
        s, sc, ex = pool.pop(best_idx)
        ranked.append((s, sc, ex))
        last_genre, last_mood = s["genre"], s["mood"]
    return ranked


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initialize recommender with a list of songs."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top k songs recommended for the user."""
        song_map = {s.id: s for s in self.songs}
        as_dicts = sorted(
            [({"id": s.id, "genre": s.genre, "mood": s.mood}, _score_song(s, user), "")
             for s in self.songs],
            key=lambda x: x[1], reverse=True,
        )
        ranked = _apply_variety_ranking(as_dicts)
        return [song_map[d["id"]] for d, _, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generate explanation for why a song is recommended."""
        score     = _score_song(song, user)
        genre_m   = _family_match(song.genre, user.current_genre, GENRE_FAMILIES)
        mood_m    = _family_match(song.mood,  user.current_mood,  MOOD_FAMILIES)
        genre_lbl = "exact" if genre_m == 1.0 else ("close" if genre_m == 0.5 else "different")
        mood_lbl  = "exact" if mood_m  == 1.0 else ("close" if mood_m  == 0.5 else "different")

        energy_sim   = 1 - abs(song.energy       - user.target_energy)
        valence_sim  = 1 - abs(song.valence       - user.target_valence)
        dance_sim    = 1 - abs(song.danceability  - user.target_danceability)
        acoustic_sim = 1 - abs(song.acousticness  - user.target_acousticness)

        return (
            f"Genre {genre_lbl} ({song.genre} / {user.current_genre}), "
            f"mood {mood_lbl} ({song.mood} / {user.current_mood}). "
            f"Energy: {energy_sim:.2f}, Valence: {valence_sim:.2f}, "
            f"Dance: {dance_sim:.2f}, Acoustic: {acoustic_sim:.2f}. "
            f"Score: {score:.3f}"
        )


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    score = _score_song_dict(song, user_prefs)
    return score, [_explain_dict(song, user_prefs, score)]

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, seen_ids: set = None) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Returns list of (song_dict, score, explanation) tuples.
    """
    if not songs:
        return []

    catalog = [s for s in songs if seen_ids is None or s["id"] not in seen_ids]

    # Maintain a min-heap of size pool_size while iterating the whole catalog.
    # Each heap op is O(log pool_size), so the full pass is O(n log pool_size)
    # instead of O(n log n) for a full sort. Memory stays O(pool_size).
    # The integer index breaks score ties without falling through to dict
    # comparison (which would error on equal scores).
    pool_size = max(k * 10, 50)
    heap: list = []
    for i, song in enumerate(catalog):
        score = _score_song_dict(song, user_prefs)
        if len(heap) < pool_size:
            heapq.heappush(heap, (score, i, song))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, i, song))

    # Heap holds the top pool_size songs in arbitrary order. Sort descending
    # and attach explanation strings only for the candidates the variety
    # re-ranker will actually see.
    top_pool = [
        (s, sc, _explain_dict(s, user_prefs, sc))
        for sc, _, s in sorted(heap, key=lambda x: -x[0])
    ]
    ranked = _apply_variety_ranking(top_pool)[:k]

    # Numeric explanation is only surfaced for the final k.
    return [
        ({**s, "_numeric_explanation": _explain_numeric(s, user_prefs, sc)}, sc, ex)
        for s, sc, ex in ranked
    ]


if __name__ == "__main__":
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs_full.csv")
    songs = load_songs(csv_path)

    demo_profiles = {
        "High-Energy Pop": {
            "genre": "pop", "mood": "happy",
            "current_genre": "pop", "current_mood": "happy",
            "target_energy": 0.85, "target_valence": 0.75,
            "target_danceability": 0.78, "target_acousticness": 0.15,
            "target_tempo": 128,
        },
        "Chill Lofi": {
            "genre": "lofi", "mood": "relaxed",
            "current_genre": "lofi", "current_mood": "chill",
            "target_energy": 0.25, "target_valence": 0.45,
            "target_danceability": 0.35, "target_acousticness": 0.75,
            "target_tempo": 90,
        },
        "Deep Intense Rock": {
            "genre": "rock", "mood": "intense",
            "current_genre": "rock", "current_mood": "angry",
            "target_energy": 0.88, "target_valence": 0.25,
            "target_danceability": 0.45, "target_acousticness": 0.10,
            "target_tempo": 125,
        },
        "High-Energy Sadness": {
            "genre": "electronic", "mood": "sad",
            "current_genre": "electronic", "current_mood": "sad",
            "target_energy": 0.90, "target_valence": 0.20,
            "target_danceability": 0.85, "target_acousticness": 0.15,
            "target_tempo": 140,
        },
        "Acoustic Intensity": {
            "genre": "rock", "mood": "intense",
            "current_genre": "rock", "current_mood": "intense",
            "target_energy": 0.88, "target_valence": 0.25,
            "target_danceability": 0.15, "target_acousticness": 0.92,
            "target_tempo": 130,
        },
        "Relaxed Workout": {
            "genre": "pop", "mood": "relaxed",
            "current_genre": "pop", "current_mood": "relaxed",
            "target_energy": 0.85, "target_valence": 0.35,
            "target_danceability": 0.88, "target_acousticness": 0.20,
            "target_tempo": 90,
        },
    }

    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    GRAY  = "\033[90m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    for name, prefs in demo_profiles.items():
        print(f"\n{CYAN}{BOLD}=== {name} ==={RESET}")
        for song, score, explanation in recommend_songs(prefs, songs, k=5):
            print(f"  {CYAN}{BOLD}{song['title']}{RESET}  {GREEN}score={score:.3f}{RESET}")
            print(f"  {GRAY}{explanation}{RESET}")
