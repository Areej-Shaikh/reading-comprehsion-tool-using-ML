import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import re
import joblib

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from inference import generate_quiz, verify_answer
from model_b_train import get_distractor_candidates, get_hints

# ── Page Config ──────────────────────────
st.set_page_config(
    page_title="Reading Comprehension Quiz",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

OPTIONS = ["A", "B", "C", "D"]


# ════════════════════════════════════════
# MODEL A HELPERS
# Question generation + answer verification
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
    answer_bonus = 3 if str(answer).lower() in sentence.lower() else 0
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

    place_words = ["city", "country", "school", "home", "village", "park", "room", "river", "mountain"]
    if any(w in answer.lower() for w in place_words):
        return "Where"

    reason_words = ["because", "reason", "due to", "so that"]
    if any(w in answer.lower() for w in reason_words):
        return "Why"

    return "What"


def generate_questions_from_sentence(sentence, answer):
    sentence = clean_text(sentence)
    answer = clean_text(answer)
    wh = guess_wh_word(answer)

    generated = []

    if answer and answer.lower() in sentence.lower():
        blanked = re.sub(re.escape(answer), "_____", sentence, flags=re.IGNORECASE)
        generated.append(f"{wh} best completes this sentence: {blanked}")
        generated.append(f"{wh} is referred to in the sentence: {blanked}")
    else:
        generated.append(f"What is the best answer based on this part of the passage: {sentence}")
        generated.append(f"What can be inferred from this sentence: {sentence}")

    generated.append(f"According to the passage, which option is correct?")
    return generated


def rank_generated_questions(questions, article, answer, source_sentence, question_ranker=None, question_ranker_vectorizer=None):
    """
    Prefer the trained Random Forest question ranker when available.
    Otherwise, use a safe heuristic fallback.
    """
    if question_ranker is not None and question_ranker_vectorizer is not None:
        feature_texts = [
            clean_text(q) + " " + clean_text(answer) + " " + clean_text(source_sentence)
            for q in questions
        ]
        X = question_ranker_vectorizer.transform(feature_texts)

        if hasattr(question_ranker, "predict_proba"):
            scores = question_ranker.predict_proba(X)[:, 1]
        else:
            scores = question_ranker.decision_function(X)

        best_index = int(np.argmax(scores))
        return questions[best_index], "Trained Random Forest ranker"

    ranked = []

    for q in questions:
        q_tokens = tokenize(q)
        a_tokens = tokenize(answer)
        article_tokens = tokenize(article)

        overlap_with_answer = len(q_tokens.intersection(a_tokens))
        overlap_with_article = len(q_tokens.intersection(article_tokens))
        length = len(q.split())

        wh_bonus = 2 if q.split()[0].lower() in ["who", "what", "where", "when", "why", "according"] else 0
        length_score = 2 if 8 <= length <= 35 else -1
        blank_bonus = 1 if "_____" in q else 0

        score = wh_bonus + length_score + blank_bonus + (0.05 * overlap_with_article) - (0.2 * overlap_with_answer)
        ranked.append((score, q))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked[0][1], "Heuristic fallback ranker"


def model_a_generate_question(article, correct_answer, question_ranker=None, question_ranker_vectorizer=None):
    candidate_sentence = choose_candidate_sentence(article, correct_answer)
    questions = generate_questions_from_sentence(candidate_sentence, correct_answer)
    best_question, ranker_used = rank_generated_questions(
        questions,
        article,
        correct_answer,
        candidate_sentence,
        question_ranker,
        question_ranker_vectorizer
    )

    return best_question, candidate_sentence, ranker_used


def verify_answer_with_model_a(article, question, selected_option, lr, svm, vec_a):
    combined = clean_text(article) + " " + clean_text(question) + " " + clean_text(selected_option)
    X = vec_a.transform([combined])

    lr_pred = int(lr.predict(X)[0])
    svm_pred = int(svm.predict(X)[0])

    # Hard-voting ensemble of the two supervised Model A verifiers.
    ensemble_pred = 1 if (lr_pred + svm_pred) >= 1 else 0

    return {
        "lr_pred": lr_pred,
        "svm_pred": svm_pred,
        "ensemble_pred": ensemble_pred
    }


# ── Load Models ──────────────────────────
@st.cache_resource
def build_model_b_eval_dataset(df):
    rows = []
    options = ["A", "B", "C", "D"]

    for _, row in df.iterrows():
        correct = row["answer"]

        for opt in options:
            text = str(row["article"]) + " " + str(row[opt])
            label = 0 if opt == correct else 1

            rows.append({
                "text": text,
                "label": label
            })

    return pd.DataFrame(rows)


@st.cache_data
def evaluate_model_b_static(_model, _vectorizer, df):
    eval_df = build_model_b_eval_dataset(df)

    X_text = eval_df["text"]
    y_true = eval_df["label"]

    X = _vectorizer.transform(X_text)
    y_pred = _model.predict(X)

    acc = round(accuracy_score(y_true, y_pred) * 100, 1)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 3)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 3)
    macro_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Correct Answer", "Actual Distractor"],
        columns=["Predicted Correct Answer", "Predicted Distractor"]
    )

    return acc, precision, recall, macro_f1, cm_df


def load_models():
    lr    = joblib.load("models/model_a/traditional/logistic_regression_verifier.pkl")
    svm   = joblib.load("models/model_a/traditional/svm_verifier.pkl")
    vec_a = joblib.load("models/model_a/traditional/onehot_vectorizer.pkl")

    # Optional trained Model A question ranker.
    # If these files do not exist yet, the app falls back to heuristic ranking.
    question_ranker = None
    question_ranker_vectorizer = None
    ranker_path = "models/model_a/traditional/question_ranker.pkl"
    ranker_vec_path = "models/model_a/traditional/question_ranker_vectorizer.pkl"

    if os.path.exists(ranker_path) and os.path.exists(ranker_vec_path):
        question_ranker = joblib.load(ranker_path)
        question_ranker_vectorizer = joblib.load(ranker_vec_path)

    distractor_model = joblib.load("models/model_b/traditional/distractor_ranker.pkl")
    vec_b = joblib.load("models/model_b/traditional/distractor_vectorizer.pkl")

    return lr, svm, vec_a, question_ranker, question_ranker_vectorizer, distractor_model, vec_b


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
    lr, svm, vec_a, question_ranker, question_ranker_vectorizer, distractor_model, vec_b = load_models()
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
    st.info(
        "Built with Classical ML\n\n"
        "Model A: Question Generator + Answer Verifier + Hard-Vote Ensemble\n\n"
        "Model B: Distractor & Hint Generator"
    )


# ════════════════════════════════════════
# SCREEN 1 — Article Input
# ════════════════════════════════════════
if page == "🏠 Article Input":
    st.title("📖 Article Input")
    st.markdown("Paste a reading passage below or load a random sample from the RACE dataset.")

    col1, col2 = st.columns([3, 1])

    with col2:
        use_generated_question = st.checkbox(
            "Use Model A generated question",
            value=True,
            help="If checked, Model A generates a template-based question from the passage."
        )

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
            article_input  = st.text_area(
                "Reading Passage",
                value=st.session_state.current_sample["article"],
                height=220
            )

            original_question = st.session_state.current_sample["question"]
            correct_answer = st.session_state.current_sample[st.session_state.current_sample["answer"]]

            if use_generated_question:
                question_input = ""
                st.info("Model A will generate the question automatically from the passage.")

                with st.expander("View original RACE question"):
                    st.write(original_question)
            else:
                question_input = st.text_input(
                    "Question",
                    value=original_question,
                    placeholder="Enter a question about the passage..."
                )

            st.caption(f"Gold answer from RACE sample: {correct_answer}")

        else:
            article_input = st.text_area("Reading Passage", height=220, placeholder="Paste your article here...")

            if use_generated_question:
                question_input = ""
                st.info("Model A will generate the question automatically from the passage.")
            else:
                question_input = st.text_input(
                    "Question",
                    placeholder="Enter a question about the passage..."
                )

            correct_answer = st.text_input("Correct Answer", placeholder="Enter the correct answer...")

    if st.button("🚀 Generate Quiz", type="primary", use_container_width=True):
        if not models_loaded:
            st.error("Models are not loaded. Check your model paths.")
        elif not article_input:
            st.error("Please enter a passage.")
        elif not correct_answer:
            st.error("Please enter or load the correct answer.")
        else:
            with st.spinner("Running Model A and Model B..."):
                start = time.time()

                if use_generated_question or not clean_text(question_input):
                    generated_question, source_sentence, ranker_used = model_a_generate_question(
                        article_input,
                        str(correct_answer),
                        question_ranker,
                        question_ranker_vectorizer
                    )
                    final_question = generated_question
                    question_source = "Generated by Model A"
                else:
                    final_question = question_input
                    source_sentence = choose_candidate_sentence(article_input, str(correct_answer))
                    ranker_used = "Not used; question was provided"
                    question_source = "Provided / RACE original"

                distractors = get_distractor_candidates(article_input, str(correct_answer), vec_b)
                hints       = get_hints(article_input, final_question, vec_b)

                all_options = [str(correct_answer)] + distractors[:3]

                # Make sure exactly 4 options exist.
                while len(all_options) < 4:
                    all_options.append(f"Option {len(all_options) + 1}")

                all_options = all_options[:4]
                np.random.shuffle(all_options)

                elapsed = round(time.time() - start, 3)

                st.session_state.result = {
                    "article": article_input,
                    "question": final_question,
                    "question_source": question_source,
                    "source_sentence": source_sentence,
                    "correct_answer": str(correct_answer),
                    "options": all_options,
                    "hints": hints[:3] if len(hints) >= 3 else hints + ["Review the relevant sentence again."] * (3 - len(hints)),
                    "generation_latency_s": elapsed,
                    "ranker_used": ranker_used
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
        st.warning("⚠️ No quiz loaded yet! Go to 🏠 Article Input, load/enter data, then click **Generate Quiz**.")
        st.stop()

    r = st.session_state.result

    st.markdown("### 📄 Passage")
    with st.expander("Click to read the passage"):
        st.write(r["article"])

    st.markdown("### ❓ Question")
    st.caption(f"Question source: {r['question_source']}")
    st.caption(f"Question ranker: {r.get('ranker_used', 'N/A')}")
    st.markdown(f"**{r['question']}**")

    with st.expander("Model A source sentence"):
        st.write(r["source_sentence"])

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

    # FIX 1: col1 and col2 are now properly separated at the same level
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Check Answer", type="primary", use_container_width=True):
            st.session_state.checked = True

            start = time.time()

            models_dict = {
                "lr": lr,
                "svm": svm,
                "vec_a": vec_a
            }

            verdict = verify_answer(
                r["article"],
                r["question"],
                selected,
                models_dict
            )

            elapsed = round(time.time() - start, 3)

            gold_correct = int(selected == r["correct_answer"])
            model_correct = verdict["ensemble_pred"]

            if gold_correct:
                st.success("✅ Correct! Well done!")
            else:
                st.error(f"❌ Incorrect. The correct answer was: **{r['correct_answer']}**")

            st.markdown("#### Explanation")

            st.info(
                f"The answer is supported by this sentence from the passage:\n\n"
                f"**{r['source_sentence']}**"
            )

            st.markdown("#### Model A Verifier Output")

            st.write(
                f"Logistic Regression prediction: "
                f"**{'Correct' if verdict['lr_pred'] == 1 else 'Incorrect'}**"
            )

            st.write(
                f"SVM prediction: "
                f"**{'Correct' if verdict['svm_pred'] == 1 else 'Incorrect'}**"
            )

            st.write(
                f"Hard-vote ensemble prediction: "
                f"**{'Correct' if model_correct == 1 else 'Incorrect'}**"
            )

            st.session_state.session_log.append({
                "question": r["question"],
                "question_source": r["question_source"],
                "selected": selected,
                "correct": r["correct_answer"],
                "gold_label": gold_correct,
                "lr_pred": verdict["lr_pred"],
                "svm_pred": verdict["svm_pred"],
                "ensemble_pred": verdict["ensemble_pred"],
                "exact_match": gold_correct,
                "latency_s": elapsed
            })

    # FIX 2: col2 is now properly outside col1, at the same indentation level
    with col2:
        if st.button("💡 Need a hint?", use_container_width=True):
            st.session_state.page = "💡 Hints"
            st.rerun()


# ════════════════════════════════════════
# SCREEN 3 — Hints
# ════════════════════════════════════════
# FIX 3: elif is now at the top level, not nested inside the Quiz block
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

    y_true = log_df["gold_label"].astype(int)
    y_pred = log_df["ensemble_pred"].astype(int)

    total = len(log_df)
    exact_matches = int(log_df["exact_match"].sum())
    user_accuracy = round(exact_matches / total * 100, 1)

    model_accuracy = round(accuracy_score(y_true, y_pred) * 100, 1)
    model_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)
    model_precision = round(precision_score(y_true, y_pred, zero_division=0), 3)
    model_recall = round(recall_score(y_true, y_pred, zero_division=0), 3)
    avg_lat = round(log_df["latency_s"].mean(), 3)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Attempts", total)
    col2.metric("Exact Matches", exact_matches)
    col3.metric("User Accuracy", f"{user_accuracy}%")
    col4.metric("Avg Verify Latency", f"{avg_lat}s")

    st.markdown("### Model A Verifier Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{model_accuracy}%")
    c2.metric("Macro F1", model_f1)
    c3.metric("Precision", model_precision)
    c4.metric("Recall", model_recall)

    st.markdown("### Model B Distractor Ranker Metrics")

    try:
        b_acc, b_precision, b_recall, b_macro_f1, b_cm_df = evaluate_model_b_static(
            distractor_model,
            vec_b,
            df
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Accuracy", f"{b_acc}%")
        b2.metric("Macro F1", b_macro_f1)
        b3.metric("Precision", b_precision)
        b4.metric("Recall", b_recall)

        st.markdown("#### Model B Confusion Matrix")
        st.dataframe(b_cm_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not calculate Model B metrics: {e}")

    st.markdown("---")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Incorrect", "Actual Correct"],
        columns=["Predicted Incorrect", "Predicted Correct"]
    )

    st.markdown("### Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Session Log")
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Export to CSV",
        csv,
        "session_log.csv",
        "text/csv"
    )