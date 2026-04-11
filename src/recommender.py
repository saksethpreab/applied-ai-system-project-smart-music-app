import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ── Genre & Mood proximity families ──────────────────────────────────────────
GENRE_FAMILIES = {
    "indie":      {"indie", "indie pop", "folk", "ballad"},
    "electronic": {"electronic", "synthwave", "techno", "lofi", "ambient"},
    "rock":       {"rock", "metal", "punk"},
    "urban":      {"hip-hop", "r&b", "soul", "funk"},
    "classical":  {"classical", "country", "blues", "gospel"},
    "pop":        {"pop", "disco", "latin", "reggae"},
}

MOOD_FAMILIES = {
    "melancholic": {"sad", "melancholic", "moody", "introspective"},
    "calm":        {"chill", "relaxed", "peaceful", "dreamy"},
    "energetic":   {"energetic", "intense", "angry"},
    "positive":    {"happy", "joyful", "romantic"},
    "focused":     {"focused"},
}

# Top-level score weights
W_CAT = 0.40
W_NUM = 0.60

# Categorical: long-term vs session blend
W_LONG_TERM = 0.30
W_SESSION   = 0.70

# Numeric feature weights (must sum to 1.0)
W_ENERGY      = 0.30
W_VALENCE     = 0.25
W_DANCE       = 0.25
W_ACOUSTICNESS = 0.10
W_TEMPO       = 0.10


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


def _score_song(song: Song, user: UserProfile, bpm_min: float, bpm_range: float) -> float:
    """Score a Song dataclass against a UserProfile."""
    genre_match = (W_LONG_TERM * _family_match(song.genre, user.favorite_genre, GENRE_FAMILIES)
                 + W_SESSION   * _family_match(song.genre, user.current_genre,  GENRE_FAMILIES))
    mood_match  = (W_LONG_TERM * _family_match(song.mood,  user.favorite_mood,  MOOD_FAMILIES)
                 + W_SESSION   * _family_match(song.mood,  user.current_mood,   MOOD_FAMILIES))
    categorical_score = (genre_match + mood_match) / 2

    tempo_norm        = (song.tempo_bpm    - bpm_min) / bpm_range if bpm_range > 0 else 0.5
    target_tempo_norm = (user.target_tempo - bpm_min) / bpm_range if bpm_range > 0 else 0.5

    energy_sim   = 1 - abs(song.energy       - user.target_energy)
    valence_sim  = 1 - abs(song.valence       - user.target_valence)
    dance_sim    = 1 - abs(song.danceability  - user.target_danceability)
    acoustic_sim = 1 - abs(song.acousticness  - user.target_acousticness)
    tempo_sim    = 1 - abs(tempo_norm         - target_tempo_norm)

    numeric_score = (W_ENERGY       * energy_sim
                   + W_VALENCE      * valence_sim
                   + W_DANCE        * dance_sim
                   + W_ACOUSTICNESS * acoustic_sim
                   + W_TEMPO        * tempo_sim)

    return W_CAT * categorical_score + W_NUM * numeric_score


def _score_song_dict(song: dict, user: dict, bpm_min: float, bpm_range: float) -> float:
    """Score a song dict against a user_prefs dict (functional interface)."""
    genre_match = (W_LONG_TERM * _family_match(song["genre"], user["genre"],         GENRE_FAMILIES)
                 + W_SESSION   * _family_match(song["genre"], user["current_genre"],  GENRE_FAMILIES))
    mood_match  = (W_LONG_TERM * _family_match(song["mood"],  user["mood"],           MOOD_FAMILIES)
                 + W_SESSION   * _family_match(song["mood"],  user["current_mood"],   MOOD_FAMILIES))
    categorical_score = (genre_match + mood_match) / 2

    tempo_norm        = (song["tempo_bpm"]    - bpm_min) / bpm_range if bpm_range > 0 else 0.5
    target_tempo_norm = (user["target_tempo"] - bpm_min) / bpm_range if bpm_range > 0 else 0.5

    energy_sim   = 1 - abs(song["energy"]       - user["target_energy"])
    valence_sim  = 1 - abs(song["valence"]       - user["target_valence"])
    dance_sim    = 1 - abs(song["danceability"]  - user["target_danceability"])
    acoustic_sim = 1 - abs(song["acousticness"]  - user["target_acousticness"])
    tempo_sim    = 1 - abs(tempo_norm             - target_tempo_norm)

    numeric_score = (W_ENERGY       * energy_sim
                   + W_VALENCE      * valence_sim
                   + W_DANCE        * dance_sim
                   + W_ACOUSTICNESS * acoustic_sim
                   + W_TEMPO        * tempo_sim)

    return W_CAT * categorical_score + W_NUM * numeric_score


def _explain_dict(song: dict, user: dict, score: float) -> str:
    """Generate explanation for song recommendation (functional interface)."""
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
        f"Energy: {energy_sim:.2f}, Valence: {valence_sim:.2f}, "
        f"Dance: {dance_sim:.2f}, Acoustic: {acoustic_sim:.2f}. "
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
            adj = ((-0.10 if last_genre and s["genre"] == last_genre else 0.0) +
                   (-0.10 if last_mood  and s["mood"]  == last_mood  else 0.0))
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
        """Initialize recommender with songs and compute BPM range."""
        self.songs = songs
        self._bpm_min   = min(s.tempo_bpm for s in songs)
        self._bpm_max   = max(s.tempo_bpm for s in songs)
        self._bpm_range = self._bpm_max - self._bpm_min

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top k songs recommended for the user."""
        scored = sorted(
            self.songs,
            key=lambda s: _score_song(s, user, self._bpm_min, self._bpm_range),
            reverse=True,
        )
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generate explanation for why a song is recommended."""
        score     = _score_song(song, user, self._bpm_min, self._bpm_range)
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


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Returns list of (song_dict, score, explanation) tuples.
    """
    if not songs:
        return []

    bpm_min   = min(s["tempo_bpm"] for s in songs)
    bpm_max   = max(s["tempo_bpm"] for s in songs)
    bpm_range = bpm_max - bpm_min

    scored = []
    for song in songs:
        score       = _score_song_dict(song, user_prefs, bpm_min, bpm_range)
        explanation = _explain_dict(song, user_prefs, score)
        scored.append((song, score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = _apply_variety_ranking(scored)
    return ranked[:k]
