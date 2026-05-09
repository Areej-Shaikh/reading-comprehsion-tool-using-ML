import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import re
import joblib
import html

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from inference import generate_quiz, verify_answer
from model_b_train import get_distractor_candidates, get_hints

# ── Page Config ──────────────────────────
st.set_page_config(
    page_title="RC Quiz System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

OPTIONS = ["A", "B", "C", "D"]


def safe_html(value):
    return html.escape(str(value))


# ── CSS ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #09090c;
    --panel: #101016;
    --panel-soft: #14141d;
    --border: #242435;
    --border-soft: #1b1b28;
    --text: #ececf5;
    --muted: #a0a0b8;
    --faint: #66667d;
    --accent: #7c83ff;
    --accent-soft: #171833;
    --success: #38d996;
    --danger: #ff7b7b;
    --warning: #f5bf4f;
}

#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

.main .block-container {
    max-width: 1180px !important;
    padding: 2rem 2.2rem 3rem !important;
}

h1, h2, h3, h4, h5, h6, p, li, label {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    font-weight: 650 !important;
    letter-spacing: -0.02em !important;
}

p, li { color: var(--muted) !important; }

[data-testid="stSidebar"] {
    background: #0d0d12 !important;
    border-right: 1px solid var(--border-soft) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

[data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"] {
    display: none !important;
}

.rc-sidebar-brand {
    padding: 26px 22px 18px;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 10px;
}

.rc-brand-name {
    color: var(--text) !important;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.rc-brand-sub {
    color: var(--faint) !important;
    font: 12px 'JetBrains Mono', monospace !important;
    margin-top: 5px;
}

.rc-sidebar-footer {
    margin: 22px 16px 16px;
    padding: 14px;
    border-radius: 12px;
    background: var(--panel);
    border: 1px solid var(--border-soft);
    color: var(--faint) !important;
    font: 11px/1.75 'JetBrains Mono', monospace !important;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 6px !important;
}

[data-testid="stSidebar"] .stRadio label {
    min-height: 42px !important;
    padding: 10px 18px !important;
    margin: 0 12px !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    color: var(--muted) !important;
    background: transparent !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--panel) !important;
    border-color: var(--border-soft) !important;
}

[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] .stRadio label p {
    color: inherit !important;
    font-size: 14px !important;
    margin: 0 !important;
}

.rc-page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 2px 0 22px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--border-soft);
}

.rc-page-title {
    color: var(--text);
    font-size: 28px;
    line-height: 1.15;
    font-weight: 720;
    letter-spacing: -0.035em;
    margin: 0;
}

.rc-page-sub {
    color: var(--faint);
    font: 12px 'JetBrains Mono', monospace;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.rc-page-tag {
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid #262859;
    border-radius: 999px;
    padding: 7px 12px;
    font: 11px 'JetBrains Mono', monospace;
    white-space: nowrap;
}

.rc-section {
    color: var(--faint);
    font: 12px 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 26px 0 12px;
    padding-bottom: 9px;
    border-bottom: 1px solid var(--border-soft);
}

.rc-card, [data-testid="metric-container"] {
    background: var(--panel) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px !important;
    box-shadow: 0 14px 28px rgba(0,0,0,0.18) !important;
}

.rc-card {
    padding: 18px 20px;
    margin-bottom: 16px;
}

.rc-card-title {
    color: var(--faint);
    font: 12px 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

.rc-question-card {
    background: linear-gradient(135deg, #10101c, #111122);
    border: 1px solid #28284b;
    border-radius: 18px;
    padding: 22px 24px;
    margin: 6px 0 20px;
}

.rc-q-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}

.rc-q-badge {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 4px 9px;
    border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid #282a5c;
    color: var(--accent);
    font: 11px 'JetBrains Mono', monospace;
}

.rc-q-text {
    color: var(--text) !important;
    font-size: 17px;
    line-height: 1.65;
    margin: 0 !important;
    font-weight: 600;
}

.rc-source {
    margin-top: 16px;
    padding: 15px 18px;
    border-radius: 14px;
    background: #0f1020;
    border: 1px solid #272850;
    border-left: 4px solid var(--accent);
    color: var(--muted);
    line-height: 1.7;
}

.rc-source-label {
    color: var(--accent);
    font: 11px 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.rc-gold {
    display: inline-flex;
    align-items: center;
    margin-top: 10px;
    padding: 7px 11px;
    border-radius: 999px;
    background: #17130a;
    border: 1px solid #2c2412;
    color: var(--warning);
    font: 12px 'JetBrains Mono', monospace;
}

.rc-verdict-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 18px 0 14px;
}

.rc-verdict-chip {
    padding: 14px 12px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid var(--border-soft);
    background: var(--panel);
}

.rc-vc-label {
    color: var(--faint);
    font: 11px 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.rc-vc-val {
    font-size: 15px;
    font-weight: 650;
}

.rc-chip-ok { background: #081611; border-color: #1b3b2b; }
.rc-chip-ok .rc-vc-val { color: var(--success); }
.rc-chip-bad { background: #180b0d; border-color: #3d2023; }
.rc-chip-bad .rc-vc-val { color: var(--danger); }
.rc-hint-intro {
    color: var(--text);
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 16px;
}

.rc-hint-item {
    display: grid;
    grid-template-columns: 38px 1fr;
    gap: 16px;
    align-items: flex-start;
    padding: 18px 20px;
    border-radius: 18px;
    border: 1px solid var(--border-soft);
    background: var(--panel);
    margin-bottom: 14px;
}

.rc-hint-num {
    width: 34px;
    height: 34px;
    border-radius: 12px;
    background: var(--accent-soft);
    border: 1px solid #282a5c;
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font: 12px 'JetBrains Mono', monospace;
    margin-top: 2px;
}

.rc-hint-title {
    color: var(--text);
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
}

.rc-hint-body {
    color: var(--muted);
    line-height: 1.65;
    font-size: 14px;
}

.rc-hint-locked {
    opacity: 0.55;
}

.rc-hint-locked .rc-hint-num {
    background: #11111a;
    border-color: #1b1b28;
    color: var(--faint);
}

.rc-hint-locked .rc-hint-title,
.rc-hint-locked .rc-hint-body {
    color: var(--faint);
}

.rc-reveal {
    padding: 16px 18px;
    border-radius: 16px;
    background: #081611;
    border: 1px solid #1b3b2b;
    color: var(--success);
    margin-top: 16px;
    font-size: 15px;
}

.stTextArea textarea, .stTextInput input {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124, 131, 255, 0.18) !important;
}

.stTextArea label, .stTextInput label, .stCheckbox label, .stRadio > label {
    color: var(--faint) !important;
    font: 12px 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 14px !important;
    min-height: 44px !important;
    font-weight: 650 !important;
    border: 1px solid var(--border) !important;
    background: var(--panel-soft) !important;
    color: var(--text) !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: white !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--accent) !important;
}

.stRadio > div {
    gap: 10px !important;
}

.stRadio > div > label {
    background: var(--panel) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    padding: 13px 16px !important;
}

.stRadio > div > label:hover {
    border-color: var(--accent) !important;
    background: var(--panel-soft) !important;
}

.stRadio > div > label p {
    color: var(--muted) !important;
    font-size: 15px !important;
    margin: 0 !important;
}

details[data-testid="stExpander"] {
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    background: var(--panel) !important;
    overflow: hidden !important;
}
            
details[data-testid="stExpander"] summary {
    color: var(--muted) !important;
    padding: 12px 14px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    min-height: 42px !important;
}

details[data-testid="stExpander"] summary svg {
    width: 16px !important;
    height: 16px !important;
    margin: 0 !important;
    flex-shrink: 0 !important;
    color: var(--muted) !important;
}

details[data-testid="stExpander"] summary p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}


details[data-testid="stExpander"] summary {
    position: relative !important;
    padding-left: 16px !important;
}

[data-testid="metric-container"] {
    padding: 16px 18px !important;
}

[data-testid="stMetricLabel"] p {
    color: var(--faint) !important;
    font: 11px 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 25px !important;
    font-weight: 720 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

[data-testid="stNotification"] {
    border-radius: 14px !important;
    border: 1px solid var(--border-soft) !important;
}

[data-testid="stNotification"] p {
    color: inherit !important;
}

.block-spacer-small { height: 10px; }
.block-spacer-medium { height: 18px; }

@media (max-width: 900px) {
    .main .block-container { padding: 1.3rem 1rem 2rem !important; }
    .rc-page-header { align-items: flex-start; flex-direction: column; }
    .rc-verdict-row { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
# MODEL A HELPERS
# ════════════════════════════════════════
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(article):
    article = clean_text(article)
    sentences = re.split(r"(?<=[.!?])\s+", article)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def tokenize(text):
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))

def sentence_score(sentence, answer):
    s_tokens = tokenize(sentence)
    a_tokens = tokenize(answer)
    if not s_tokens:
        return 0
    overlap = len(s_tokens.intersection(a_tokens))
    answer_bonus   = 3 if str(answer).lower() in sentence.lower() else 0
    length_penalty = abs(len(sentence.split()) - 18) / 30
    return overlap + answer_bonus - length_penalty

def choose_candidate_sentence(article, answer):
    sentences = split_sentences(article)
    if not sentences:
        return clean_text(article)[:250]
    scored = [(sentence_score(s, answer), s) for s in sentences]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]

def guess_wh_word(answer):
    answer = str(answer).strip()
    if re.search(r"\b(19|20)\d{2}\b", answer) or re.search(r"\b\d+\b", answer):
        return "When"
    if answer[:1].isupper() and len(answer.split()) <= 4:
        return "Who"
    if any(w in answer.lower() for w in ["city","country","school","home","village","park","room","river","mountain"]):
        return "Where"
    if any(w in answer.lower() for w in ["because","reason","due to","so that"]):
        return "Why"
    return "What"

def generate_questions_from_sentence(sentence, answer):
    sentence = clean_text(sentence)
    answer   = clean_text(answer)
    wh = guess_wh_word(answer)
    generated = []
    if answer and answer.lower() in sentence.lower():
        blanked = re.sub(re.escape(answer), "_____", sentence, flags=re.IGNORECASE)
        generated.append(f"{wh} best completes this sentence: {blanked}")
        generated.append(f"{wh} is referred to in the sentence: {blanked}")
    else:
        generated.append(f"What is the best answer based on this part of the passage: {sentence}")
        generated.append(f"What can be inferred from this sentence: {sentence}")
    generated.append("According to the passage, which option is correct?")
    return generated

def rank_generated_questions(questions, article, answer, source_sentence,
                              question_ranker=None, question_ranker_vectorizer=None):
    if question_ranker is not None and question_ranker_vectorizer is not None:
        feature_texts = [
            clean_text(q) + " " + clean_text(answer) + " " + clean_text(source_sentence)
            for q in questions
        ]
        X = question_ranker_vectorizer.transform(feature_texts)
        scores = (question_ranker.predict_proba(X)[:, 1]
                  if hasattr(question_ranker, "predict_proba")
                  else question_ranker.decision_function(X))
        return questions[int(np.argmax(scores))], "Trained Random Forest ranker"

    ranked = []
    for q in questions:
        q_tokens       = tokenize(q)
        a_tokens       = tokenize(answer)
        article_tokens = tokenize(article)
        overlap_with_answer  = len(q_tokens.intersection(a_tokens))
        overlap_with_article = len(q_tokens.intersection(article_tokens))
        length       = len(q.split())
        wh_bonus     = 2 if q.split()[0].lower() in ["who","what","where","when","why","according"] else 0
        length_score = 2 if 8 <= length <= 35 else -1
        blank_bonus  = 1 if "_____" in q else 0
        score = wh_bonus + length_score + blank_bonus + (0.05*overlap_with_article) - (0.2*overlap_with_answer)
        ranked.append((score, q))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked[0][1], "Heuristic fallback ranker"

def model_a_generate_question(article, correct_answer,
                               question_ranker=None, question_ranker_vectorizer=None):
    candidate_sentence = choose_candidate_sentence(article, correct_answer)
    questions = generate_questions_from_sentence(candidate_sentence, correct_answer)
    best_question, ranker_used = rank_generated_questions(
        questions, article, correct_answer, candidate_sentence,
        question_ranker, question_ranker_vectorizer
    )
    return best_question, candidate_sentence, ranker_used


# ── Load Models ──────────────────────────
@st.cache_resource
def build_model_b_eval_dataset(df):
    rows = []
    for _, row in df.iterrows():
        correct = row["answer"]
        for opt in ["A", "B", "C", "D"]:
            text  = str(row["article"]) + " " + str(row[opt])
            label = 0 if opt == correct else 1
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)

@st.cache_data
def evaluate_model_b_static(_model, _vectorizer, df):
    eval_df = build_model_b_eval_dataset(df)
    X      = _vectorizer.transform(eval_df["text"])
    y_true = eval_df["label"]
    y_pred = _model.predict(X)
    acc       = round(accuracy_score(y_true, y_pred) * 100, 1)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 3)
    recall    = round(recall_score(y_true, y_pred, zero_division=0), 3)
    macro_f1  = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm,
        index=["Actual Correct Answer", "Actual Distractor"],
        columns=["Predicted Correct Answer", "Predicted Distractor"])
    return acc, precision, recall, macro_f1, cm_df

def load_models():
    lr    = joblib.load("models/model_a/traditional/logistic_regression_verifier.pkl")
    svm   = joblib.load("models/model_a/traditional/svm_verifier.pkl")
    vec_a = joblib.load("models/model_a/traditional/onehot_vectorizer.pkl")
    question_ranker            = None
    question_ranker_vectorizer = None
    ranker_path     = "models/model_a/traditional/question_ranker.pkl"
    ranker_vec_path = "models/model_a/traditional/question_ranker_vectorizer.pkl"
    if os.path.exists(ranker_path) and os.path.exists(ranker_vec_path):
        question_ranker            = joblib.load(ranker_path)
        question_ranker_vectorizer = joblib.load(ranker_vec_path)
    distractor_model = joblib.load("models/model_b/traditional/distractor_ranker.pkl")
    vec_b            = joblib.load("models/model_b/traditional/distractor_vectorizer.pkl")
    return lr, svm, vec_a, question_ranker, question_ranker_vectorizer, distractor_model, vec_b

@st.cache_data
def load_sample_data():
    return pd.read_csv("data/processed/test.csv")


# ── Session State ─────────────────────────
for key, val in {
    "result": None, "hints_used": 0, "answer_revealed": False,
    "selected": None, "checked": False, "session_log": [],
    "current_sample": None, "page": "Article input",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Load everything ──────────────────────
try:
    lr, svm, vec_a, question_ranker, question_ranker_vectorizer, distractor_model, vec_b = load_models()
    df = load_sample_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")


# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="rc-sidebar-brand">
        <div class="rc-brand-name">RC Quiz System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Article input", "Quiz", "Hints", "Dashboard"],
        label_visibility="collapsed",
        key="page"
    )

    st.markdown("""
    <div class="rc-sidebar-footer">
model_a<br>
&nbsp; question_generator<br>
&nbsp; lr_verifier<br>
&nbsp; svm_verifier<br>
&nbsp; hard_vote_ensemble<br>
<br>
model_b<br>
&nbsp; distractor_ranker<br>
&nbsp; hint_generator
    </div>
    """, unsafe_allow_html=True)


# ── Page header ───────────────────────────
PAGE_META = {
    "Article input": ("Input", "Paste or load a reading passage"),
    "Quiz": ("Quiz", "Answer the generated question"),
    "Hints": ("Hints", "Progressive hints to guide you"),
    "Dashboard": ("Stats", "Model metrics and session analytics"),
}
tag, sub = PAGE_META[page]
st.markdown(f"""
<div class="rc-page-header">
    <div>
        <div class="rc-page-title">{safe_html(page)}</div>
        <div class="rc-page-sub">{safe_html(sub)}</div>
    </div>
    <div class="rc-page-tag">{safe_html(tag)}</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
# SCREEN 1 — Article Input
# ════════════════════════════════════════
if page == "Article input":
    col_main, col_ctrl = st.columns([3, 1.2], gap="large")

    with col_ctrl:
        use_generated_question = st.checkbox(
            "Use Model A generated question",
            value=True,
            help="Model A generates a template-based question from the passage."
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Load random sample", use_container_width=True):
            if models_loaded:
                sample = df.sample(1).iloc[0]
                st.session_state.current_sample  = sample
                st.session_state.result          = None
                st.session_state.hints_used      = 0
                st.session_state.answer_revealed = False
                st.session_state.selected        = None
                st.session_state.checked         = False
                st.rerun()

    with col_main:
        if st.session_state.current_sample is not None:
            s = st.session_state.current_sample
            article_input     = st.text_area("Reading passage", value=s["article"], height=200)
            original_question = s["question"]
            correct_answer    = s[s["answer"]]
            if use_generated_question:
                question_input = ""
                st.info("Model A will generate the question automatically from the passage.")
                with st.expander("View original RACE question", expanded=False):
                    st.write(original_question)
            else:
                question_input = st.text_input("Question", value=original_question)
            st.markdown(
                f'<div class="rc-gold">Gold answer — {safe_html(correct_answer)}</div>',
                unsafe_allow_html=True
            )
        else:
            article_input = st.text_area("Reading passage", height=200,
                                          placeholder="Paste your article here…")
            if use_generated_question:
                question_input = ""
                st.info("Model A will generate the question automatically from the passage.")
            else:
                question_input = st.text_input("Question",
                                                placeholder="Enter a question about the passage…")
            correct_answer = st.text_input("Correct answer",
                                            placeholder="Enter the correct answer…")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Generate quiz", type="primary", use_container_width=True):
        if not models_loaded:
            st.error("Models not loaded — check your model paths.")
        elif not article_input:
            st.error("Please enter a passage.")
        elif not correct_answer:
            st.error("Please enter or load the correct answer.")
        else:
            with st.spinner("Running Model A and Model B…"):
                start = time.time()
                if use_generated_question or not clean_text(question_input):
                    generated_question, source_sentence, ranker_used = model_a_generate_question(
                        article_input, str(correct_answer),
                        question_ranker, question_ranker_vectorizer)
                    final_question  = generated_question
                    question_source = "Generated by Model A"
                else:
                    final_question  = question_input
                    source_sentence = choose_candidate_sentence(article_input, str(correct_answer))
                    ranker_used     = "Not used — question provided"
                    question_source = "Provided / RACE original"

                distractors = get_distractor_candidates(article_input, str(correct_answer), vec_b)
                hints       = get_hints(article_input, final_question, vec_b)
                all_options = [str(correct_answer)] + distractors[:3]
                while len(all_options) < 4:
                    all_options.append(f"Option {len(all_options)+1}")
                all_options = all_options[:4]
                np.random.shuffle(all_options)
                elapsed = round(time.time() - start, 3)

                st.session_state.result = {
                    "article":              article_input,
                    "question":             final_question,
                    "question_source":      question_source,
                    "source_sentence":      source_sentence,
                    "correct_answer":       str(correct_answer),
                    "options":              all_options,
                    "hints":                (hints[:3] if len(hints) >= 3
                                             else hints + ["Review the relevant sentence again."]*(3-len(hints))),
                    "generation_latency_s": elapsed,
                    "ranker_used":          ranker_used,
                }
                st.session_state.hints_used      = 0
                st.session_state.answer_revealed = False
                st.session_state.selected        = None
                st.session_state.checked         = False

            st.success("Quiz generated. Go to the Quiz screen in the sidebar.")
            st.balloons()


# ════════════════════════════════════════
# SCREEN 2 — Quiz
# ════════════════════════════════════════
elif page == "Quiz":
    if st.session_state.result is None:
        st.warning("No quiz loaded yet — go to Article input and click Generate quiz.")
        st.stop()

    r = st.session_state.result

    st.markdown('<div class="rc-section">Passage</div>', unsafe_allow_html=True)
    with st.expander("Click to read the passage"):
        st.write(r["article"])

    question_source_html = safe_html(r["question_source"])
    ranker_html = safe_html(r.get("ranker_used", "N/A"))
    question_html = safe_html(r["question"])

    st.markdown(f"""
    <div class="rc-question-card">
        <div class="rc-q-meta">
            <span class="rc-q-badge">{question_source_html}</span>
            <span class="rc-q-badge">{ranker_html}</span>
        </div>
        <p class="rc-q-text">{question_html}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rc-section">Choose your answer</div>', unsafe_allow_html=True)
    options = r["options"]
    labels  = ["A", "B", "C", "D"]
    selected = st.radio(
        "Options",
        options,
        format_func=lambda x: f"{labels[options.index(x)]}   {x}",
        label_visibility="collapsed",
        key="quiz_radio"
    )
    st.session_state.selected = selected

    if st.button("Check answer", type="primary", use_container_width=True):
        st.session_state.checked = True
        start = time.time()
        models_dict = {"lr": lr, "svm": svm, "vec_a": vec_a}
        verdict = verify_answer(r["article"], r["question"], selected, models_dict)
        elapsed = round(time.time() - start, 3)

        gold_correct = int(selected == r["correct_answer"])

        if gold_correct:
            st.success("Correct. Well done.")
        else:
            st.error(f"Incorrect. The correct answer was: **{r['correct_answer']}**")

        def cc(v): return "rc-chip-ok" if v == 1 else "rc-chip-bad"
        def ct(v): return "Correct"   if v == 1 else "Incorrect"

        source_sentence_html = safe_html(r["source_sentence"])

        st.markdown(f"""
        <div class="rc-verdict-row">
            <div class="rc-verdict-chip {cc(verdict['lr_pred'])}">
                <div class="rc-vc-label">Logistic Regression</div>
                <div class="rc-vc-val">{ct(verdict['lr_pred'])}</div>
            </div>
            <div class="rc-verdict-chip {cc(verdict['svm_pred'])}">
                <div class="rc-vc-label">SVM</div>
                <div class="rc-vc-val">{ct(verdict['svm_pred'])}</div>
            </div>
            <div class="rc-verdict-chip {cc(verdict['ensemble_pred'])}">
                <div class="rc-vc-label">Hard-vote ensemble</div>
                <div class="rc-vc-val">{ct(verdict['ensemble_pred'])}</div>
            </div>
        </div>
        <div class="rc-source">
            <div class="rc-source-label">Supporting sentence from passage</div>
            {source_sentence_html}
        </div>
        """, unsafe_allow_html=True)

        st.session_state.session_log.append({
            "question":        r["question"],
            "question_source": r["question_source"],
            "selected":        selected,
            "correct":         r["correct_answer"],
            "gold_label":      gold_correct,
            "lr_pred":         verdict["lr_pred"],
            "svm_pred":        verdict["svm_pred"],
            "ensemble_pred":   verdict["ensemble_pred"],
            "exact_match":     gold_correct,
            "latency_s":       elapsed,
        })


# ════════════════════════════════════════
# SCREEN 3 — Hints
# ════════════════════════════════════════
elif page == "Hints":
    if st.session_state.result is None:
        st.warning("No quiz loaded yet — go to Article input first.")
        st.stop()

    hints = st.session_state.result["hints"]
    hint_labels = ["General clue", "More specific", "Nearly there"]

    st.markdown(
        '<div class="rc-card rc-hint-intro">'
        'Reveal hints one at a time. Try to answer before moving to the next.'
        '</div>',
        unsafe_allow_html=True
    )

    for i in range(3):
        unlocked = st.session_state.hints_used >= i + 1
        cls = "rc-hint-item" if unlocked else "rc-hint-item rc-hint-locked"
        body = hints[i] if unlocked else "Locked. Reveal the previous hint first."

        st.markdown(f"""
        <div class="{cls}">
            <div class="rc-hint-num">{i + 1}</div>
            <div>
                <div class="rc-hint-title">{safe_html(hint_labels[i])}</div>
                <div class="rc-hint-body">{safe_html(body)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    next_hint = st.session_state.hints_used + 1

    if next_hint <= 3:
        if st.button(f"Show hint {next_hint}", use_container_width=False):
            st.session_state.hints_used = next_hint
            st.rerun()
    else:
        if st.button("Reveal answer", type="primary", use_container_width=False):
            st.session_state.answer_revealed = True
            st.rerun()

    if st.session_state.answer_revealed:
        st.markdown(
            f'<div class="rc-reveal">The correct answer is: '
            f'<strong>{safe_html(st.session_state.result["correct_answer"])}</strong></div>',
            unsafe_allow_html=True
        )
# ════════════════════════════════════════
# SCREEN 4 — Dashboard
# ════════════════════════════════════════
elif page == "Dashboard":
    MODEL_A_RESULTS = {
        "Logistic Regression":  [51.60, 0.4801, 0.2599, 0.5065],
        "SVM":                  [51.60, 0.4802, 0.2600, 0.5069],
        "Hard Voting Ensemble": [51.63, 0.4803, 0.2599, 0.5062],
    }
    UNSUPERVISED_RESULTS = {
        "K-Means Silhouette":         0.0095,
        "K-Means Purity":             0.7913,
        "Label Propagation Accuracy": 0.7793,
        "Label Propagation Macro F1": 0.4552,
    }
    ENSEMBLE_CM = pd.DataFrame(
        [[13698, 12663], [4339, 4448]],
        index=["Actual Incorrect", "Actual Correct"],
        columns=["Predicted Incorrect", "Predicted Correct"]
    )

    # ── Live session ──
    st.markdown('<div class="rc-section">Live session</div>', unsafe_allow_html=True)
    if st.session_state.session_log:
        log_df  = pd.DataFrame(st.session_state.session_log)
        total   = len(log_df)
        exact   = int(log_df["exact_match"].sum())
        acc     = round(exact / total * 100, 1)
        avg_lat = round(log_df["latency_s"].mean(), 3)
    else:
        total, exact, acc, avg_lat = 0, 0, 0.0, 0.0
        log_df = pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total attempts",     total)
    c2.metric("Exact matches",      exact)
    c3.metric("User accuracy",      f"{acc}%")
    c4.metric("Avg verify latency", f"{avg_lat}s")

    if log_df.empty:
        st.info("No quiz attempts yet — complete a quiz to see live metrics.")

    # ── Model A ──
    st.markdown('<div class="rc-section">Model A — verifier metrics</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Ensemble accuracy", "51.63%")
    a2.metric("Macro F1",          "0.480")
    a3.metric("Precision",         "0.260")
    a4.metric("Recall",            "0.506")

    model_a_df = pd.DataFrame(MODEL_A_RESULTS,
                               index=["Accuracy (%)", "Macro F1", "Precision", "Recall"]).T
    st.markdown("**Model comparison**")
    st.dataframe(model_a_df, use_container_width=True)
    st.markdown("**Ensemble confusion matrix**")
    st.dataframe(ENSEMBLE_CM, use_container_width=True)

    # ── Unsupervised ──
    st.markdown('<div class="rc-section">Model A — unsupervised / semi-supervised</div>',
                unsafe_allow_html=True)
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("K-Means silhouette",  UNSUPERVISED_RESULTS["K-Means Silhouette"])
    u2.metric("K-Means purity",      UNSUPERVISED_RESULTS["K-Means Purity"])
    u3.metric("Label prop accuracy", UNSUPERVISED_RESULTS["Label Propagation Accuracy"])
    u4.metric("Label prop macro F1", UNSUPERVISED_RESULTS["Label Propagation Macro F1"])

    # ── Model B ──
    st.markdown('<div class="rc-section">Model B — distractor ranker</div>', unsafe_allow_html=True)
    try:
        b_acc, b_precision, b_recall, b_macro_f1, b_cm_df = evaluate_model_b_static(
            distractor_model, vec_b, df)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Accuracy",  f"{b_acc}%")
        b2.metric("Macro F1",  b_macro_f1)
        b3.metric("Precision", b_precision)
        b4.metric("Recall",    b_recall)
        st.markdown("**Model B confusion matrix**")
        st.dataframe(b_cm_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not calculate Model B metrics: {e}")

    # ── Session log ──
    st.markdown('<div class="rc-section">Session log</div>', unsafe_allow_html=True)
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True)
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export session log to CSV", csv, "session_log.csv", "text/csv")
    else:
        st.info("Session log will appear after quiz attempts.")