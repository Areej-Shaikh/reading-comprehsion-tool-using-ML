import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_b_train import get_distractor_candidates, get_hints

# ── Page Config ──────────────────────────
st.set_page_config(
    page_title="Reading Comprehension Quiz",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Models ──────────────────────────
@st.cache_resource
def load_models():
    lr    = joblib.load("models/model_a/traditional/logistic_regression_verifier.pkl")
    svm   = joblib.load("models/model_a/traditional/svm_verifier.pkl")
    vec_a = joblib.load("models/model_a/traditional/onehot_vectorizer.pkl")
    distractor_model = joblib.load("models/model_b/traditional/distractor_ranker.pkl")
    vec_b = joblib.load("models/model_b/traditional/distractor_vectorizer.pkl")
    return lr, svm, vec_a, distractor_model, vec_b

@st.cache_data
def load_sample_data():
    df = pd.read_csv("data/processed/test.csv")
    return df

# ── Session State Init ───────────────────
for key, val in {
    "result": None,
    "hints_used": 0,
    "answer_revealed": False,
    "selected": None,
    "checked": False,
    "session_log": [],
    "current_sample": None,
    "page": "🏠 Article Input"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load everything ──────────────────────
try:
    lr, svm, vec_a, distractor_model, vec_b = load_models()
    df = load_sample_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")

# ── Sidebar ──────────────────────────────
with st.sidebar:
    st.title("📚 RC Quiz System")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Article Input",
        "❓ Quiz",
        "💡 Hints",
        "📊 Dashboard"
    ], key="page")
    st.markdown("---")
    st.info("Built with Classical ML\n\nModel A: Answer Verifier\n\nModel B: Distractor & Hint Generator")

# ════════════════════════════════════════
# SCREEN 1 — Article Input
# ════════════════════════════════════════
if page == "🏠 Article Input":
    st.title("📖 Article Input")
    st.markdown("Paste a reading passage below or load a random sample from the RACE dataset.")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🎲 Load Random Sample", use_container_width=True):
            if models_loaded:
                sample = df.sample(1).iloc[0]
                st.session_state.current_sample  = sample
                st.session_state.result          = None
                st.session_state.hints_used      = 0
                st.session_state.answer_revealed = False
                st.session_state.selected        = None
                st.session_state.checked         = False

    with col1:
        if st.session_state.current_sample is not None:
            article_input  = st.text_area("Reading Passage", value=st.session_state.current_sample["article"], height=200)
            question_input = st.text_input("Question",       value=st.session_state.current_sample["question"])
            correct_answer = st.session_state.current_sample[st.session_state.current_sample["answer"]]
        else:
            article_input  = st.text_area("Reading Passage", height=200, placeholder="Paste your article here...")
            question_input = st.text_input("Question", placeholder="Enter a question about the passage...")
            correct_answer = st.text_input("Correct Answer", placeholder="Enter the correct answer...")

    if st.button("🚀 Generate Quiz", type="primary", use_container_width=True):
        if not article_input or not question_input:
            st.error("Please enter both a passage and a question!")
        else:
            with st.spinner("Generating distractors and hints..."):
                time.sleep(0.5)
                distractors = get_distractor_candidates(article_input, str(correct_answer), vec_b)
                hints       = get_hints(article_input, question_input, vec_b)

                all_options = [str(correct_answer)] + distractors[:3]
                np.random.shuffle(all_options)

                st.session_state.result = {
                    "article":        article_input,
                    "question":       question_input,
                    "correct_answer": str(correct_answer),
                    "options":        all_options,
                    "hints":          hints,
                }
                st.session_state.hints_used      = 0
                st.session_state.answer_revealed = False
                st.session_state.selected        = None
                st.session_state.checked         = False

            st.success("✅ Quiz generated! Now click **❓ Quiz** in the sidebar.")
            st.balloons()

# ════════════════════════════════════════
# SCREEN 2 — Quiz
# ════════════════════════════════════════
elif page == "❓ Quiz":
    st.title("❓ Quiz")

    if st.session_state.result is None:
        st.warning("⚠️ No quiz loaded yet! Go to 🏠 Article Input, click **Load Random Sample**, then click **Generate Quiz**.")
        st.stop()

    r = st.session_state.result

    st.markdown("### 📄 Passage")
    with st.expander("Click to read the passage"):
        st.write(r["article"])

    st.markdown(f"### ❓ Question")
    st.markdown(f"**{r['question']}**")
    st.markdown("### Choose your answer:")

    options = r["options"]
    labels  = ["A", "B", "C", "D"]

    selected = st.radio(
        "Options",
        options,
        format_func=lambda x: f"{labels[options.index(x)]})  {x}",
        label_visibility="collapsed"
    )
    st.session_state.selected = selected

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Check Answer", type="primary", use_container_width=True):
            st.session_state.checked = True
            start = time.time()
            combined = r["article"] + " " + r["question"] + " " + selected
            X        = vec_a.transform([combined])
            elapsed  = round(time.time() - start, 3)
            is_correct = (selected == r["correct_answer"])

            if is_correct:
                st.success("✅ Correct! Well done!")
            else:
                st.error(f"❌ Incorrect. The correct answer was: **{r['correct_answer']}**")

            st.session_state.session_log.append({
                "question":   r["question"],
                "selected":   selected,
                "correct":    r["correct_answer"],
                "is_correct": is_correct,
                "latency_s":  elapsed
            })

    with col2:
        if st.button("💡 Need a hint?", use_container_width=True):
            st.session_state.page = "💡 Hints"
            st.rerun()

# ════════════════════════════════════════
# SCREEN 3 — Hints
# ════════════════════════════════════════
elif page == "💡 Hints":
    st.title("💡 Hints")

    if st.session_state.result is None:
        st.warning("⚠️ No quiz loaded yet! Go to 🏠 Article Input first.")
        st.stop()

    hints = st.session_state.result["hints"]
    st.markdown("Use hints one at a time. Try to answer before revealing the next one!")
    st.markdown("---")

    if st.button("Show Hint 1 — General Clue"):
        st.session_state.hints_used = max(st.session_state.hints_used, 1)
    if st.session_state.hints_used >= 1:
        st.warning(f"💡 **Hint 1:** {hints[0]}")

    if st.session_state.hints_used >= 1:
        if st.button("Show Hint 2 — More Specific"):
            st.session_state.hints_used = max(st.session_state.hints_used, 2)
    if st.session_state.hints_used >= 2:
        st.warning(f"💡 **Hint 2:** {hints[1]}")

    if st.session_state.hints_used >= 2:
        if st.button("Show Hint 3 — Nearly There"):
            st.session_state.hints_used = max(st.session_state.hints_used, 3)
    if st.session_state.hints_used >= 3:
        st.warning(f"💡 **Hint 3:** {hints[2]}")

    if st.session_state.hints_used >= 3:
        st.markdown("---")
        if st.button("🔓 Reveal Answer", type="primary"):
            st.session_state.answer_revealed = True
        if st.session_state.answer_revealed:
            st.success(f"✅ The correct answer is: **{st.session_state.result['correct_answer']}**")

# ════════════════════════════════════════
# SCREEN 4 — Dashboard
# ════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")

    if not st.session_state.session_log:
        st.info("📭 No quiz attempts yet. Answer some questions first!")
        st.stop()

    log_df = pd.DataFrame(st.session_state.session_log)

    total    = len(log_df)
    correct  = log_df["is_correct"].sum()
    accuracy = round(correct / total * 100, 1)
    avg_lat  = round(log_df["latency_s"].mean(), 3)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Attempts", total)
    col2.metric("Correct",        correct)
    col3.metric("Accuracy",       f"{accuracy}%")
    col4.metric("Avg Latency",    f"{avg_lat}s")

    st.markdown("---")

    import plotly.express as px
    fig = px.pie(
        names=["Correct", "Incorrect"],
        values=[correct, total - correct],
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        title="Correct vs Incorrect"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Session Log")
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export to CSV", csv, "session_log.csv", "text/csv")