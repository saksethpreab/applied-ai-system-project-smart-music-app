import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from agent import refresh_playlist

logger = logging.getLogger(__name__)
_MAX_REQUESTS_PER_MINUTE = 10

st.set_page_config(
    page_title="MoodSync",
    page_icon="🎵",
    layout="centered",
)

for key, default in [
    ("result", None),
    ("last_prompt", None),
    ("error", None),
    ("session", {}),   # maps prompt string -> set of seen song IDs
    ("request_times", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎵 MoodSync")
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
col_run, col_refresh = st.columns([2, 1])
run_button     = col_run.button("Get Recommendations", type="primary",   use_container_width=True)
refresh_button = col_refresh.button(
    "🔄 Refresh",
    type="secondary",
    use_container_width=True,
    disabled=st.session_state["last_prompt"] is None,
)

# ── Run logic ─────────────────────────────────────────────────────────────────

def _check_rate_limit() -> bool:
    now = time.time()
    window = [t for t in st.session_state.get("request_times", []) if now - t < 60]
    if len(window) >= _MAX_REQUESTS_PER_MINUTE:
        return False
    window.append(now)
    st.session_state["request_times"] = window
    return True


def _run(prompt: str, fresh: bool) -> None:
    if not _check_rate_limit():
        st.session_state["error"] = "Too many requests — please wait a moment before trying again."
        return
    st.session_state["error"] = None
    if fresh:
        st.session_state["session"].pop(prompt, None)   # clear history → new search
    try:
        with st.spinner("Analyzing your request and building your playlist..."):
            result = refresh_playlist(prompt, st.session_state["session"])
            st.session_state["result"] = result
            st.session_state["last_prompt"] = prompt
    except EnvironmentError as e:
        st.session_state["error"] = f"API key error: {e}"
        st.session_state["result"] = None
    except RuntimeError as e:
        st.session_state["error"] = f"Pipeline error: {e}"
        st.session_state["result"] = None
    except Exception:
        logger.exception("Unexpected pipeline error for prompt: %r", prompt)
        st.session_state["error"] = "An unexpected error occurred. Please try again."
        st.session_state["result"] = None

if run_button and user_prompt.strip():
    _run(user_prompt.strip(), fresh=True)
    st.rerun()
elif run_button:
    st.warning("Please enter a prompt first.")
elif refresh_button:
    _run(st.session_state["last_prompt"], fresh=False)
    st.rerun()

# ── Error display ─────────────────────────────────────────────────────────────

if st.session_state["error"]:
    st.error(st.session_state["error"])

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state["result"] is not None:
    result   = st.session_state["result"]
    analysis = result["analysis"]

    # Metrics row
    st.subheader("Pipeline Metrics")
    m = result["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence", f"{m['analysis_confidence']:.0%}")
    c2.metric("Draft Score", f"{m['draft_score']:.3f}")
    c3.metric("Corrections", m["correction_count"])
    c4.metric(
        "Final Score",
        f"{m['final_avg_score']:.3f}",
        delta=f"{m['final_avg_score'] - m['draft_score']:+.3f}",
    )
    _intent_labels = {"match": "Match", "uplift": "Uplift", "energize": "Energize", "calm": "Calm", "contrast": "Contrast"}
    c5.metric("Intent", _intent_labels.get(analysis.get("emotion_intent", "match"), "Match"))

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
                if song.get("explanation_numeric"):
                    with st.expander("Deeper Dive — numeric breakdown"):
                        st.caption(song["explanation_numeric"])
            st.divider()

    # ── Emotional Space Map ───────────────────────────────────────────────────
    with st.expander("🎭 Emotional Space Map", expanded=True):
        prefs          = analysis["user_prefs"]
        emotion_intent = analysis.get("emotion_intent", "match")
        intent_display = {
            "match": "Match mood", "uplift": "Uplift mood",
            "energize": "Energize", "calm": "Calm down", "contrast": "Contrast",
        }.get(emotion_intent, emotion_intent.title())

        quadrant_shapes = [
            dict(type="rect", x0=0,   x1=0.5, y0=0,   y1=0.5, fillcolor="rgba(100,120,200,0.12)", line_width=0, layer="below"),
            dict(type="rect", x0=0,   x1=0.5, y0=0.5, y1=1.0, fillcolor="rgba(200,80,80,0.12)",   line_width=0, layer="below"),
            dict(type="rect", x0=0.5, x1=1.0, y0=0,   y1=0.5, fillcolor="rgba(80,180,130,0.12)",  line_width=0, layer="below"),
            dict(type="rect", x0=0.5, x1=1.0, y0=0.5, y1=1.0, fillcolor="rgba(255,200,50,0.12)",  line_width=0, layer="below"),
        ]
        quadrant_annotations = [
            dict(x=0.25, y=0.93, text="Intense / Angry",      showarrow=False, font=dict(size=11, color="rgba(180,60,60,0.7)"),   xref="x", yref="y"),
            dict(x=0.75, y=0.93, text="Energetic / Euphoric", showarrow=False, font=dict(size=11, color="rgba(180,150,20,0.7)"),  xref="x", yref="y"),
            dict(x=0.25, y=0.07, text="Melancholic / Sad",    showarrow=False, font=dict(size=11, color="rgba(60,80,180,0.7)"),   xref="x", yref="y"),
            dict(x=0.75, y=0.07, text="Calm / Peaceful",      showarrow=False, font=dict(size=11, color="rgba(40,140,90,0.7)"),   xref="x", yref="y"),
        ]

        target_trace = go.Scatter(
            x=[prefs["target_valence"]],
            y=[prefs["target_energy"]],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color="#FFD700", line=dict(color="#333", width=1.5)),
            text=["Target"],
            textposition="top center",
            textfont=dict(size=12, color="#333"),
            name="Your Target",
            hovertemplate=(
                f"<b>Your Target</b><br>Intent: {intent_display}<br>"
                "Valence: %{x:.2f}<br>Energy: %{y:.2f}<extra></extra>"
            ),
        )

        songs_data   = result["final_playlist"]
        hover_texts  = [
            f"<b>{i}. {s['title']}</b><br>Artist: {s['artist']}<br>"
            f"Genre: {s['genre'].title()} · Mood: {s['mood'].title()}<br>"
            f"Valence: {s['valence']:.2f} · Energy: {s['energy']:.2f}<br>Score: {s['score']:.3f}"
            for i, s in enumerate(songs_data, 1)
        ]
        songs_trace = go.Scatter(
            x=[s["valence"] for s in songs_data],
            y=[s["energy"]  for s in songs_data],
            mode="markers+text",
            marker=dict(symbol="circle", size=28, color="#4A90D9", line=dict(color="#1a5fa8", width=1.5), opacity=0.85),
            text=[str(i) for i in range(1, len(songs_data) + 1)],
            textposition="middle center",
            textfont=dict(size=11, color="white", family="Arial Black"),
            name="Recommended Songs",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
        )

        fig = go.Figure(data=[songs_trace, target_trace])
        fig.update_layout(
            shapes=quadrant_shapes,
            annotations=quadrant_annotations,
            xaxis=dict(title="Valence (negative → positive)", range=[0, 1], tickvals=[0, .25, .5, .75, 1], gridcolor="rgba(200,200,200,0.3)", zeroline=False),
            yaxis=dict(title="Energy (calm → intense)",       range=[0, 1], tickvals=[0, .25, .5, .75, 1], gridcolor="rgba(200,200,200,0.3)", zeroline=False),
            plot_bgcolor="rgba(250,250,252,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=60, b=60),
            height=460,
        )
        fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1,   line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0,   x1=1,   y0=0.5, y1=0.5, line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"))

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"⭐ = your target (intent: **{intent_display}**) · numbered circles = recommended songs")

    # Analysis
    with st.expander("Analysis — How we interpreted your request", expanded=False):
        prefs = analysis["user_prefs"]
        st.markdown(f"**Reasoning:** {analysis['reasoning']}")
        st.markdown(f"**Confidence:** {analysis['confidence']:.0%}")
        st.markdown(f"**Emotion Intent:** `{analysis.get('emotion_intent', 'match')}`")
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
