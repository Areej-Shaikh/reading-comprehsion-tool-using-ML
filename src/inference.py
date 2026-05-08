import os
import re
import sys
import joblib
import numpy as np

sys.path.append(os.path.dirname(__file__))

from model_b_train import get_distractor_candidates, get_hints


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

    place_words = [
        "city", "country", "school", "home", "village",
        "park", "room", "river", "mountain", "place"
    ]

    if any(w in answer.lower() for w in place_words):
        return "Where"

    reason_words = ["because", "reason", "due to", "so that"]

    if any(w in answer.lower() for w in reason_words):
        return "Why"

    return "What"


def generate_candidate_questions(sentence, answer):
    sentence = clean_text(sentence)
    answer = clean_text(answer)
    wh = guess_wh_word(answer)

    questions = []

    if answer and answer.lower() in sentence.lower():
        blanked = re.sub(re.escape(answer), "_____", sentence, flags=re.IGNORECASE)
        questions.append(f"{wh} best completes this sentence: {blanked}")
        questions.append(f"{wh} is referred to in the sentence: {blanked}")
        questions.append("According to the passage, which option is correct?")
    else:
        questions.append(f"What is the best answer based on this part of the passage: {sentence}")
        questions.append(f"What can be inferred from this sentence: {sentence}")
        questions.append("According to the passage, which option is correct?")

    return questions


def load_models():
    lr = joblib.load("models/model_a/traditional/logistic_regression_verifier.pkl")
    svm = joblib.load("models/model_a/traditional/svm_verifier.pkl")
    vec_a = joblib.load("models/model_a/traditional/onehot_vectorizer.pkl")

    distractor_model = joblib.load("models/model_b/traditional/distractor_ranker.pkl")
    vec_b = joblib.load("models/model_b/traditional/distractor_vectorizer.pkl")

    question_ranker = None
    question_ranker_vectorizer = None

    ranker_path = "models/model_a/traditional/question_ranker.pkl"
    ranker_vec_path = "models/model_a/traditional/question_ranker_vectorizer.pkl"

    if os.path.exists(ranker_path) and os.path.exists(ranker_vec_path):
        question_ranker = joblib.load(ranker_path)
        question_ranker_vectorizer = joblib.load(ranker_vec_path)

    return {
        "lr": lr,
        "svm": svm,
        "vec_a": vec_a,
        "distractor_model": distractor_model,
        "vec_b": vec_b,
        "question_ranker": question_ranker,
        "question_ranker_vectorizer": question_ranker_vectorizer
    }


def generate_question(article, correct_answer, question_ranker=None, question_ranker_vectorizer=None):
    source_sentence = choose_candidate_sentence(article, correct_answer)
    questions = generate_candidate_questions(source_sentence, correct_answer)

    if question_ranker is not None and question_ranker_vectorizer is not None:
        feature_texts = []

        for q in questions:
            feature_texts.append(
                clean_text(q) + " " +
                clean_text(correct_answer) + " " +
                clean_text(source_sentence)
            )

        X = question_ranker_vectorizer.transform(feature_texts)

        if hasattr(question_ranker, "predict_proba"):
            scores = question_ranker.predict_proba(X)[:, 1]
        else:
            scores = question_ranker.decision_function(X)

        best_index = int(np.argmax(scores))
        return questions[best_index], source_sentence, "Trained Random Forest ranker"

    return questions[0], source_sentence, "Template fallback ranker"


def generate_quiz(article, correct_answer, models, provided_question=None):
    if provided_question is None or clean_text(provided_question) == "":
        question, source_sentence, ranker_used = generate_question(
            article,
            correct_answer,
            models.get("question_ranker"),
            models.get("question_ranker_vectorizer")
        )
        question_source = "Generated by Model A"
    else:
        question = provided_question
        source_sentence = choose_candidate_sentence(article, correct_answer)
        ranker_used = "Not used; question was provided"
        question_source = "Provided / RACE original"

    distractors = get_distractor_candidates(article, str(correct_answer), models["vec_b"])
    hints = get_hints(article, question, models["vec_b"])

    options = [str(correct_answer)] + distractors[:3]

    while len(options) < 4:
        options.append(f"Option {len(options) + 1}")

    options = options[:4]
    np.random.shuffle(options)

    while len(hints) < 3:
        hints.append("Review the relevant sentence again.")

    return {
        "article": article,
        "question": question,
        "question_source": question_source,
        "source_sentence": source_sentence,
        "correct_answer": str(correct_answer),
        "options": options,
        "hints": hints[:3],
        "ranker_used": ranker_used
    }


def verify_answer(article, question, selected_option, models):
    combined = clean_text(article) + " " + clean_text(question) + " " + clean_text(selected_option)
    X = models["vec_a"].transform([combined])

    lr_pred = int(models["lr"].predict(X)[0])
    svm_pred = int(models["svm"].predict(X)[0])

    ensemble_pred = 1 if (lr_pred + svm_pred) >= 1 else 0

    return {
        "lr_pred": lr_pred,
        "svm_pred": svm_pred,
        "ensemble_pred": ensemble_pred
    }