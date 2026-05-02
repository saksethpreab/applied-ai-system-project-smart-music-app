import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from agent import run_agent

st.set_page_config(
    page_title="Smart Music Recommender",
    page_icon="🎵",
    layout="centered",
)

for key, default in [("result", None), ("last_prompt", None), ("error", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎵 Smart Music Recommender")
st.caption("Powered by Claude AI + a rule-based scoring engine")

# ── Input ─────────────────────────────────────────────────────────────────────

user_prompt = st.text_area(
    "What music do you want?",
    placeholder=(
        "e.g. I'm driving home at night through a rainy city, feeling nostalgic "
        "but oddly at peace. Think Blade Runner vibes — something slow, atmospheric, "
        "and a little melancholic. Like if Portishead met an old jazz bar."
    ),
    height=120,
)
run_button = st.button("Get Recommendations", type="primary")

# ── Run logic ─────────────────────────────────────────────────────────────────

if run_button and user_prompt.strip():
    st.session_state["error"] = None
    try:
        with st.spinner("Analyzing your request and building your playlist..."):
            st.session_state["result"] = run_agent(user_prompt.strip())
            st.session_state["last_prompt"] = user_prompt.strip()
    except EnvironmentError as e:
        st.session_state["error"] = f"API key error: {e}"
        st.session_state["result"] = None
    except RuntimeError as e:
        st.session_state["error"] = f"Pipeline error: {e}"
        st.session_state["result"] = None
    except Exception as e:
        st.session_state["error"] = f"Unexpected error: {e}"
        st.session_state["result"] = None
elif run_button:
    st.warning("Please enter a prompt first.")

# ── Error display ─────────────────────────────────────────────────────────────

if st.session_state["error"]:
    st.error(st.session_state["error"])

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state["result"] is not None:
    result = st.session_state["result"]

    # Metrics row
    st.subheader("Pipeline Metrics")
    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{m['analysis_confidence']:.0%}")
    c2.metric("Draft Score", f"{m['draft_score']:.3f}")
    c3.metric("Corrections", m["correction_count"])
    c4.metric(
        "Final Score",
        f"{m['final_avg_score']:.3f}",
        delta=f"{m['final_avg_score'] - m['draft_score']:+.3f}",
    )

    st.divider()

    # Final playlist
    st.subheader("Your Playlist")
    for i, song in enumerate(result["final_playlist"], 1):
        with st.container():
            col_num, col_main, col_score = st.columns([0.5, 6, 1.5])
            col_num.markdown(f"**{i}.**")
            col_main.markdown(f"**{song['title']}** — {song['artist']}")
            col_main.caption(
                f"{song['genre'].title()} · {song['mood'].title()} · {song['tempo_bpm']:.0f} BPM"
            )
            col_score.metric(label="Score", value=f"{song['score']:.3f}")
            with st.expander("Why this song?"):
                st.write(song["explanation"])
            st.divider()

    # Analysis
    with st.expander("Analysis — How we interpreted your request", expanded=False):
        analysis = result["analysis"]
        prefs = analysis["user_prefs"]
        st.markdown(f"**Reasoning:** {analysis['reasoning']}")
        st.markdown(f"**Confidence:** {analysis['confidence']:.0%}")
        st.markdown("**Detected Preferences:**")
        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            st.markdown(f"- Genre: `{prefs['genre']}`")
            st.markdown(f"- Mood: `{prefs['mood']}`")
            st.markdown(f"- Session genre: `{prefs['current_genre']}`")
            st.markdown(f"- Session mood: `{prefs['current_mood']}`")
        with pref_col2:
            st.markdown(f"- Energy: `{prefs['target_energy']:.2f}`")
            st.markdown(f"- Valence: `{prefs['target_valence']:.2f}`")
            st.markdown(f"- Danceability: `{prefs['target_danceability']:.2f}`")
            st.markdown(f"- Acousticness: `{prefs['target_acousticness']:.2f}`")
            st.markdown(f"- Tempo: `{prefs['target_tempo']:.0f} BPM`")

    # Draft vs. Final comparison
    with st.expander("Draft vs. Final Comparison", expanded=False):
        draft = result["draft_playlist"]
        final = result["final_playlist"]

        draft_ids = {s["id"] for s in draft}
        final_ids = {s["id"] for s in final}
        removed_ids = draft_ids - final_ids
        added_ids = final_ids - draft_ids

        corrections = result["metrics"]["correction_count"]
        if corrections == 0:
            st.success("No changes — the draft playlist was approved as-is.")
        else:
            st.info(f"{corrections} song(s) replaced during the self-correction step.")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("**Draft Playlist**")
            for song in draft:
                if song["id"] in removed_ids:
                    st.markdown(
                        f":red[Removed] ~~{song['title']}~~ — {song['artist']} `{song['score']:.3f}`"
                    )
                else:
                    st.markdown(f"{song['title']} — {song['artist']} `{song['score']:.3f}`")

        with comp_col2:
            st.markdown("**Final Playlist**")
            for song in final:
                if song["id"] in added_ids:
                    st.markdown(
                        f":green[New] **{song['title']}** — {song['artist']} `{song['score']:.3f}`"
                    )
                else:
                    st.markdown(f"{song['title']} — {song['artist']} `{song['score']:.3f}`")

    # Footer
    st.caption(f'Prompt: "{st.session_state["last_prompt"]}"')
