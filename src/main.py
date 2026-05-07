"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    """Run the music recommender simulation and display top recommendations."""
    songs = load_songs("data/songs_full.csv")

    # Define three distinct user profiles
    user_profiles = {
        "High-Energy Pop": {
            "genre":               "pop",
            "mood":                "happy",
            "current_genre":       "pop",
            "current_mood":        "happy",
            "target_energy":       0.85,
            "target_valence":      0.75,
            "target_danceability": 0.78,
            "target_acousticness": 0.15,
            "target_tempo":        128,
        },
        "Chill Lofi": {
            "genre":               "lofi",
            "mood":                "relaxed",
            "current_genre":       "lofi",
            "current_mood":        "chill",
            "target_energy":       0.25,
            "target_valence":      0.45,
            "target_danceability": 0.35,
            "target_acousticness": 0.75,
            "target_tempo":        90,
        },
        "Deep Intense Rock": {
            "genre":               "rock",
            "mood":                "intense",
            "current_genre":       "rock",
            "current_mood":        "angry",
            "target_energy":       0.88,
            "target_valence":      0.25,
            "target_danceability": 0.45,
            "target_acousticness": 0.10,
            "target_tempo":        125,
        },
        "High-Energy Sadness": {
            "genre":               "electronic",
            "mood":                "sad",
            "current_genre":       "electronic",
            "current_mood":        "sad",
            "target_energy":       0.90,
            "target_valence":      0.20,
            "target_danceability": 0.85,
            "target_acousticness": 0.15,
            "target_tempo":        140,
        },
        "Acoustic Intensity": {
            "genre":               "rock",
            "mood":                "intense",
            "current_genre":       "rock",
            "current_mood":        "intense",
            "target_energy":       0.88,
            "target_valence":      0.25,
            "target_danceability": 0.15,
            "target_acousticness": 0.92,
            "target_tempo":        130,
        },
        "Relaxed Workout": {
            "genre":               "pop",
            "mood":                "relaxed",
            "current_genre":       "pop",
            "current_mood":        "relaxed",
            "target_energy":       0.85,
            "target_valence":      0.35,
            "target_danceability": 0.88,
            "target_acousticness": 0.20,
            "target_tempo":        90,
        },
    }

    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    for profile_name, user_prefs in user_profiles.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print(f"\n{CYAN}{BOLD}=== {profile_name} ==={RESET}\n")
        for rec in recommendations:
            # You decide the structure of each returned item.
            # A common pattern is: (song, score, explanation)
            song, score, explanation = rec
            print(f"{CYAN}{BOLD}{song['title']}{RESET} - {GREEN}Score: {score:.2f}{RESET}")
            print(f"{GRAY}Because: {explanation}{RESET}")
            print()


if __name__ == "__main__":
    main()
