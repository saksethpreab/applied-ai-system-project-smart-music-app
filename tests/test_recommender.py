import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.recommender import Song, UserProfile, Recommender, load_songs, recommend_songs

SONGS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

USER_PROFILES = {
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

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        current_genre="pop",
        current_mood="happy",
        target_energy=0.8,
        target_valence=0.7,
        target_danceability=0.7,
        target_acousticness=0.2,
        target_tempo=120,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        current_genre="pop",
        current_mood="happy",
        target_energy=0.8,
        target_valence=0.7,
        target_danceability=0.7,
        target_acousticness=0.2,
        target_tempo=120,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


import pytest

@pytest.mark.parametrize("profile_name,user_prefs", USER_PROFILES.items())
def test_profile_recommendations(profile_name, user_prefs):
    songs = load_songs(SONGS_CSV)
    results = recommend_songs(user_prefs, songs, k=5)

    print(f"\n{'='*50}")
    print(f"  {profile_name}")
    print(f"{'='*50}")
    for song, score, explanation in results:
        print(f"  {song['title']} ({song['artist']})  score={score:.3f}")
        print(f"    {explanation}")

    assert len(results) == 5
    for song, score, explanation in results:
        assert 0.0 <= score <= 1.0
        assert isinstance(explanation, str)
