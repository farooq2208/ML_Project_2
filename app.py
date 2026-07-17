"""
Personality Type Classifier — Streamlit Web Application
=========================================================
Predicts whether a person is an Introvert or Extrovert based on
seven behavioural and lifestyle features using a pre-trained
Logistic Regression pipeline.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personality Type Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "personality_classifier_model.pkl"

# Label map: model outputs 0 = Introvert, 1 = Extrovert
LABEL_MAP = {0: "Introvert", 1: "Extrovert"}

RESULT_CONFIG = {
    "Extrovert": {
        "emoji": "",
        "color": "#1565c0",
        "bg": "#e3f2fd",
        "border": "#90caf9",
        "tagline": "You thrive in social settings and draw energy from the world around you.",
        "traits": [
            "Energised by social interaction",
            "Comfortable in large groups and public speaking",
            "Expressive, outgoing, and action-oriented",
            "Seeks external stimulation and variety",
            "Builds a wide circle of friends and acquaintances",
        ],
        "tip": (
            " **Tip:** Channel your social energy into leadership, "
            "networking, or collaborative projects."
        ),
    },
    "Introvert": {
        "emoji": "",
        "color": "#4a148c",
        "bg": "#f3e5f5",
        "border": "#ce93d8",
        "tagline": "You recharge through solitude and prefer depth over breadth in relationships.",
        "traits": [
            "Energised by quiet time and reflection",
            "Prefers small groups and meaningful one-on-one conversations",
            "Thoughtful, observant, and deeply focused",
            "May feel drained after prolonged social events",
            "Builds a small but close-knit circle of friends",
        ],
        "tip": (
            " **Tip:** Lean into your ability to focus deeply — creative, "
            "analytical, and research-driven work often suits introverts well."
        ),
    },
}

# Feature range reference (from dataset describe())
FEATURE_INFO = {
    "Time_spent_Alone":         {"min": 0,   "max": 11,  "label": "Time Spent Alone (hrs/day)",     "step": 0.5},
    "Social_event_attendance":  {"min": 0,   "max": 10,  "label": "Social Event Attendance (0–10)",  "step": 0.5},
    "Going_outside":            {"min": 0,   "max": 10,  "label": "Going Outside Frequency (0–10)",  "step": 0.5},
    "Friends_circle_size":      {"min": 0,   "max": 15,  "label": "Friends Circle Size (0–15)",      "step": 0.5},
    "Post_frequency":           {"min": 0,   "max": 10,  "label": "Social Media Post Frequency (0–10)", "step": 0.5},
}


# ── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the pre-trained scikit-learn pipeline from disk."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f" Model file `{MODEL_PATH}` not found. "
            "Make sure it lives in the same directory as `app.py`."
        )
        st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("Personality Type Classifier")
        st.markdown("---")

        st.markdown(
            """
            ### About
            This app uses a **Logistic Regression** model trained on
            behavioural survey data to classify personality types as
            **Introvert** or **Extrovert**.
            
            st.markdown("---")

            ### Prediction Classes
            | Class | Description |
            |-------|-------------|
            |  Extrovert | Energised by social interaction |
            |  Introvert | Energised by solitude & reflection |
         
            st.markdown("---")

            ### Disclaimer
            > This classifier is for **educational
            > purposes only**. Personality is complex and multidimensional —
            > no single model can fully capture it.
            """
        )

        st.markdown("---")
        st.markdown(
            """
            ### Model Details
            - **Algorithm:** Logistic Regression
            - **Tuning:** 5-fold GridSearchCV
            - **Metric:** Accuracy
            - **Test Accuracy:** ~93 %
            - **Dataset:** 2900 behavioural survey responses
            """
        )

        st.markdown("---")
        st.caption("Developed by: Farooq Hassnain Sheikh")


# ── Input form ───────────────────────────────────────────────────────────────
def render_input_form() -> dict:
    """Render the user-input form and return raw values as a dict."""

    st.header(" Tell Us About Your Habits & Preferences")
    st.markdown(
        "Move the sliders and select the options that best reflect your "
        "typical behaviour. There are no right or wrong answers."
    )

    col1, col2 = st.columns(2, gap="large")

    # ── Left column: behavioural sliders ─────────────────────────────────────
    with col1:
        st.subheader("Daily Habits")

        time_alone = st.slider(
            "How many hours per day do you typically spend alone?",
            min_value=0.0,
            max_value=11.0,
            value=4.0,
            step=0.5,
            format="%.1f hrs",
            help=(
                "Include reading, working independently, or any activity "
                "where you are not actively socialising. Range: 0–11 hrs."
            ),
        )

        going_outside = st.slider(
            "How often do you go outside for leisure? (0 = Never, 10 = Daily)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            help="Rate how frequently you leave home for non-work activities.",
        )

        post_frequency = st.slider(
            "How often do you post on social media? (0 = Never, 10 = Very Often)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            format="%.1f",
            help="Consider all platforms combined (Instagram, X, TikTok, etc.).",
        )

    # ── Right column: social sliders + yes/no ────────────────────────────────
    with col2:
        st.subheader(" Social Preferences")

        social_events = st.slider(
            "How often do you attend social events? (0 = Never, 10 = Very Often)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            help="Parties, meetups, gatherings, dinners with friends, etc.",
        )

        friends_circle = st.slider(
            "How large is your active friends circle?",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            format="%.0f people",
            help=(
                "Count people you interact with at least once a month. "
                "Range: 0–15."
            ),
        )

        st.markdown("&nbsp;")  # small spacer

        stage_fear = st.radio(
            "Do you experience stage fright or anxiety when speaking in public?",
            options=["No", "Yes"],
            index=0,
            horizontal=True,
            help="Public speaking, presentations, performing in front of a group.",
        )

        drained = st.radio(
            "Do you feel drained or exhausted after socialising for a long time?",
            options=["No", "Yes"],
            index=0,
            horizontal=True,
            help=(
                "This is the classic introvert/extrovert energy question — "
                "introverts typically need alone time to recharge after "
                "social events."
            ),
        )

    return {
        "Time_spent_Alone":        time_alone,
        "Stage_fear":              stage_fear,
        "Social_event_attendance": social_events,
        "Going_outside":           going_outside,
        "Drained_after_socializing": drained,
        "Friends_circle_size":     friends_circle,
        "Post_frequency":          post_frequency,
    }


# ── Build DataFrame safe for the pipeline ────────────────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with explicit numpy-backed dtypes.
    Using pd.DataFrame([dict]) on pandas 3.x with PyArrow backend produces
    ArrowStringDtype for string columns, which breaks sklearn's ColumnTransformer.
    Explicit dtype assignment guarantees plain object columns.
    """
    return pd.DataFrame({
        "Time_spent_Alone":          pd.array([float(inputs["Time_spent_Alone"])],          dtype="float64"),
        "Stage_fear":                pd.array([str(inputs["Stage_fear"])],                  dtype="object"),
        "Social_event_attendance":   pd.array([float(inputs["Social_event_attendance"])],   dtype="float64"),
        "Going_outside":             pd.array([float(inputs["Going_outside"])],             dtype="float64"),
        "Drained_after_socializing": pd.array([str(inputs["Drained_after_socializing"])],  dtype="object"),
        "Friends_circle_size":       pd.array([float(inputs["Friends_circle_size"])],       dtype="float64"),
        "Post_frequency":            pd.array([float(inputs["Post_frequency"])],            dtype="float64"),
    })


# ── Result display ────────────────────────────────────────────────────────────
def render_result(label: str, proba: np.ndarray, inputs: dict):
    """Render the prediction card, confidence gauge, traits, and input summary."""
    cfg = RESULT_CONFIG[label]

    # ── Prediction card ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: {cfg['bg']};
            border: 2px solid {cfg['border']};
            border-radius: 14px;
            padding: 28px 32px;
            margin-top: 12px;
        ">
            <h2 style="color: {cfg['color']}; margin: 0 0 6px 0; font-size: 2rem;">
                {cfg['emoji']} You are most likely an <u>{label}</u>
            </h2>
            <p style="color: #444; font-size: 1.05rem; margin: 0;">
                {cfg['tagline']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    # ── Confidence scores ─────────────────────────────────────────────────────
    with res_col1:
        st.subheader(" Prediction Confidence")
        st.caption("How confident the model is for each personality type.")

        # proba order: [Introvert (0), Extrovert (1)]
        conf_map = {"Introvert": float(proba[0]), "Extrovert": float(proba[1])}
        for ptype, prob in conf_map.items():
            pcfg = RESULT_CONFIG[ptype]
            st.markdown(
                f"""
                <div style="margin-bottom: 14px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-weight:600;">{pcfg['emoji']} {ptype}</span>
                        <span style="font-weight:700; color:{pcfg['color']};">{prob*100:.1f}%</span>
                    </div>
                    <div style="background:#e0e0e0; border-radius:8px; height:20px; overflow:hidden;">
                        <div style="
                            width:{prob*100:.1f}%;
                            background:{pcfg['color']};
                            height:100%;
                            border-radius:8px;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Personality traits ────────────────────────────────────────────────────
    with res_col2:
        st.subheader(f" Key Traits of an {label}")
        for trait in cfg["traits"]:
            st.markdown(f"- {trait}")
        st.markdown(cfg["tip"])

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander(" Review your submitted answers"):
        display_map = {
            "Time Spent Alone (hrs/day)":         inputs["Time_spent_Alone"],
            "Stage Fright":                       inputs["Stage_fear"],
            "Social Event Attendance (0–10)":     inputs["Social_event_attendance"],
            "Goes Outside for Leisure (0–10)":    inputs["Going_outside"],
            "Feels Drained After Socialising":    inputs["Drained_after_socializing"],
            "Friends Circle Size":                inputs["Friends_circle_size"],
            "Social Media Post Frequency (0–10)": inputs["Post_frequency"],
        }
        summary_df = pd.DataFrame(
            [{"Feature": k, "Your Answer": str(v)} for k, v in display_map.items()]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Main application ──────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d47a1 0%, #7b1fa2 100%);
            border-radius: 14px;
            padding: 32px 36px;
            margin-bottom: 28px;
        ">
            <h1 style="color: white; margin: 0 0 8px 0; font-size: 2.3rem;">
                 Personality Type Classifier
            </h1>
            <p style="color: #e1bee7; font-size: 1.05rem; margin: 0;">
                Answer seven quick questions about your everyday habits and
                social preferences. Our AI will predict whether you lean
                <strong>Introvert</strong> or <strong>Extrovert</strong> —
                with a confidence breakdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    inputs = render_input_form()

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_clicked = st.button(
            " Predict My Personality",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        input_df = build_input_df(inputs)

        with st.spinner("Analysing your personality profile…"):
            prediction_idx = int(model.predict(input_df)[0])
            probabilities  = model.predict_proba(input_df)[0]

        label = LABEL_MAP[prediction_idx]
        render_result(label, probabilities, inputs)


if __name__ == "__main__":
    main()
