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
    songs = load_songs("data/songs.csv") 

    user_prefs = {
        "genre":               "indie pop",
        "mood":                "moody",
        "current_genre":       "indie pop",
        "current_mood":        "moody",
        "target_energy":       0.43,
        "target_valence":      0.40,
        "target_danceability": 0.41,
        "target_acousticness": 0.54,
        "target_tempo":        85,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{CYAN}{BOLD}Top recommendations:{RESET}\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{CYAN}{BOLD}{song['title']}{RESET} - {GREEN}Score: {score:.2f}{RESET}")
        print(f"{GRAY}Because: {explanation}{RESET}")
        print()


if __name__ == "__main__":
    main()
