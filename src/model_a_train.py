import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

DATA_TRAIN = "data/processed/train.csv"
DATA_DEV   = "data/processed/dev.csv"
SAVE_DIR   = "models/model_a/traditional"

OPTIONS = ["A", "B", "C", "D"]

os.makedirs(SAVE_DIR, exist_ok=True)

#clean tect
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#tokenize text into set of words
def tokenize(text):
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))

#split article into sentences
def split_sentences(article):
    article = clean_text(article)
    sentences = re.split(r"(?<=[.!?])\s+", article)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

#score sentences based on overlap with answer and length
def make_verification_dataset(df):
    dfs = []

    for option in OPTIONS:
        temp = pd.DataFrame()
        temp["text"] = (
    "article " + df["article"].fillna("") + " " +
    "question " + df["question"].fillna("") + " " +
    "option " + df[option].fillna("")
            )
        temp["label"] = (df["answer"] == option).astype(int)
        dfs.append(temp)

    return pd.concat(dfs, ignore_index=True)

#train supervised verifiers
def clustering_purity(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)
    return cm.max(axis=0).sum() / cm.sum()

#hard voting training
def train_supervised_verifiers():
    print("\n" + "=" * 70)
    print("MODEL A: SUPERVISED ANSWER VERIFIERS")
    print("=" * 70)

    print("Loading train/dev data...")
    train = pd.read_csv(DATA_TRAIN)
    dev   = pd.read_csv(DATA_DEV)

    print("Building verification datasets...")
    train_verify = make_verification_dataset(train)
    dev_verify   = make_verification_dataset(dev)

    X_train_text = train_verify["text"]
    y_train      = train_verify["label"]
    X_dev_text   = dev_verify["text"]
    y_dev        = dev_verify["label"]

    print("Vectorizing with One-Hot / binary CountVectorizer...")
    vectorizer = CountVectorizer(
    binary=True,
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3,
    max_df=0.90
)
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev   = vectorizer.transform(X_dev_text)

    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "onehot_vectorizer.pkl"))
    print("Saved vectorizer: onehot_vectorizer.pkl")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            C=2.0,
            random_state=42
        ),
    "SVM": LinearSVC(
    max_iter=1500,
    dual=False,
    class_weight="balanced",
    C=0.3,
    tol=1e-3,
    random_state=42
    )
    }

    supervised_results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)

        acc = accuracy_score(y_dev, preds)
        f1  = f1_score(y_dev, preds, average="macro")
        precision = precision_score(y_dev, preds, zero_division=0)
        recall    = recall_score(y_dev, preds, zero_division=0)

        supervised_results[name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "precision": precision,
            "recall": recall
        }

        print(f"\n── {name} Results ──")
        print("Accuracy :", acc)
        print("Macro F1 :", f1)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("\nClassification Report:")
        print(classification_report(y_dev, preds, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_dev, preds))

        filename = name.lower().replace(" ", "_") + "_verifier.pkl"
        joblib.dump(model, os.path.join(SAVE_DIR, filename))
        print(f"Saved: {filename}")

    return vectorizer, X_train, y_train, X_dev, y_dev, supervised_results

#candidate sentence selection
def train_ensemble_verifier(X_train, y_train, X_dev, y_dev):
    print("\n" + "=" * 70)
    print("MODEL A: HARD-VOTING ENSEMBLE")
    print("=" * 70)

    lr =LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced",
    C=2.0,
    random_state=42
    )
    svm = LinearSVC(   
    max_iter=1500,
    dual=False,
    class_weight="balanced",
    C=0.3,
    tol=1e-3,
    random_state=42
)

    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("svm", svm)
        ],
        voting="hard"
    )

    print("Training hard-voting ensemble...")
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_dev)

    acc = accuracy_score(y_dev, preds)
    f1  = f1_score(y_dev, preds, average="macro")
    precision = precision_score(y_dev, preds, zero_division=0)
    recall    = recall_score(y_dev, preds, zero_division=0)

    print("\n── Ensemble Results ──")
    print("Accuracy :", acc)
    print("Macro F1 :", f1)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("\nClassification Report:")
    print(classification_report(y_dev, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_dev, preds))

    joblib.dump(ensemble, os.path.join(SAVE_DIR, "hard_voting_ensemble_verifier.pkl"))
    print("Saved: hard_voting_ensemble_verifier.pkl")

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall
    }

#k-means clustering
def run_kmeans(X_dev, y_dev):
    print("\n" + "=" * 70)
    print("MODEL A: K-MEANS CLUSTERING")
    print("=" * 70)

    X_sample = X_dev[:3000]
    y_sample = np.array(y_dev)[:3000]

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_sample)

    sil = silhouette_score(X_sample, cluster_labels, metric="cosine")
    purity = clustering_purity(y_sample, cluster_labels)

    print("Silhouette Score :", round(sil, 4))
    print("Clustering Purity:", round(purity, 4))

    joblib.dump(model, os.path.join(SAVE_DIR, "kmeans.pkl"))
    print("Saved: kmeans.pkl")

    return sil, purity
#label propagation
def run_label_propagation(X_train, y_train, X_dev, y_dev):
    print("\n" + "=" * 70)
    print("MODEL A: LABEL PROPAGATION")
    print("=" * 70)

    X_lp = X_train[:5000].toarray()
    y_lp = np.array(y_train)[:5000].copy()

    rng = np.random.default_rng(42)
    mask = rng.random(len(y_lp)) > 0.20
    y_lp[mask] = -1

    labeled_count = (y_lp != -1).sum()
    print(f"Labeled samples: {labeled_count} / {len(y_lp)}")

    model = LabelPropagation(
        kernel="knn",
        n_neighbors=7,
        max_iter=1000
    )

    model.fit(X_lp, y_lp)

    X_dev_dense = X_dev[:3000].toarray()
    y_dev_arr = np.array(y_dev)[:3000]

    preds = model.predict(X_dev_dense)

    acc = accuracy_score(y_dev_arr, preds)
    f1 = f1_score(y_dev_arr, preds, average="macro")

    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_dev_arr, preds, zero_division=0))

    joblib.dump(model, os.path.join(SAVE_DIR, "label_propagation.pkl"))
    print("Saved: label_propagation.pkl")

    return acc, f1
#candidate sentence selection
def sentence_score(sentence, answer):
    s_tokens = tokenize(sentence)
    a_tokens = tokenize(answer)

    if not s_tokens:
        return 0

    overlap = len(s_tokens.intersection(a_tokens))
    answer_bonus = 3 if str(answer).lower() in sentence.lower() else 0
    length_penalty = abs(len(sentence.split()) - 18) / 30

    return overlap + answer_bonus - length_penalty

#choose candidate sentence from article based on answer
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
        questions.append(f"According to the passage, what does _____ refer to?")
    else:
        questions.append(f"What is the best answer based on this part of the passage: {sentence}")
        questions.append(f"What can be inferred from this sentence: {sentence}")
        questions.append("According to the passage, which option is correct?")

    return questions

#train question generation ranker
def build_question_ranker_dataset(df, max_rows=8000):
    """
    Positive examples:
        Original RACE human-written questions.

    Negative examples:
        Template-generated questions.

    The classifier learns to rank/select the more human-like and relevant question.
    """
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    rows = []

    for _, row in df.iterrows():
        article = clean_text(row.get("article", ""))
        gold_question = clean_text(row.get("question", ""))

        answer_label = str(row.get("answer", "")).strip()

        if answer_label not in OPTIONS:
            continue

        correct_answer = clean_text(row.get(answer_label, ""))

        if not article or not gold_question or not correct_answer:
            continue

        source_sentence = choose_candidate_sentence(article, correct_answer)

        rows.append({
            "question_text": gold_question,
            "answer": correct_answer,
            "source_sentence": source_sentence,
            "label": 1
        })

        generated_questions = generate_candidate_questions(source_sentence, correct_answer)

        for q in generated_questions:
            rows.append({
                "question_text": clean_text(q),
                "answer": correct_answer,
                "source_sentence": source_sentence,
                "label": 0
            })

    return pd.DataFrame(rows)

#make feature text for question ranker
def make_question_ranker_feature_text(df):
    return (
        df["question_text"].fillna("") + " " +
        df["answer"].fillna("") + " " +
        df["source_sentence"].fillna("")
    )


def train_question_ranker():
    print("\n" + "=" * 70)
    print("MODEL A: QUESTION GENERATION RANKER")
    print("=" * 70)

    print("Loading train data...")
    train_df = pd.read_csv(DATA_TRAIN)

    print("Building question-ranker dataset...")
    rank_df = build_question_ranker_dataset(train_df, max_rows=8000)

    print("Ranker dataset shape:", rank_df.shape)
    print("Label distribution:")
    print(rank_df["label"].value_counts())

    X_text = make_question_ranker_feature_text(rank_df)
    y = rank_df["label"].astype(int)

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Vectorizing ranker data...")
    vectorizer = CountVectorizer(
    binary=True,
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3,
    max_df=0.90
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    print("Training Random Forest question ranker...")
    ranker = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    ranker.fit(X_train, y_train)

    preds = ranker.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)

    print("\n── Question Ranker Results ──")
    print("Accuracy :", acc)
    print("Macro F1 :", f1)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("\nClassification Report:")
    print(classification_report(y_val, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))

    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "question_ranker_vectorizer.pkl"))
    joblib.dump(ranker, os.path.join(SAVE_DIR, "question_ranker.pkl"))

    print("Saved: question_ranker_vectorizer.pkl")
    print("Saved: question_ranker.pkl")

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall
    }

def print_final_summary(supervised_results, ensemble_results, kmeans_results, label_prop_results, question_ranker_results):
    print("\n" + "=" * 90)
    print("FINAL MODEL A SUMMARY")
    print("=" * 90)

    print("\nSUPERVISED VERIFIERS")
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)

    for name, result in supervised_results.items():
        print(
            f"{name:<25} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_f1']:>10.4f} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>10.4f}"
        )

    print(
        f"{'Hard Voting Ensemble':<25} "
        f"{ensemble_results['accuracy']:>10.4f} "
        f"{ensemble_results['macro_f1']:>10.4f} "
        f"{ensemble_results['precision']:>10.4f} "
        f"{ensemble_results['recall']:>10.4f}"
    )

    print("\nUNSUPERVISED / SEMI-SUPERVISED")
    print(f"{'Model':<25} {'Metric 1':>15} {'Metric 2':>15}")
    print("-" * 60)
    print(f"{'K-Means':<25} {'Silhouette':>15} {kmeans_results[0]:>15.4f}")
    print(f"{'K-Means':<25} {'Purity':>15} {kmeans_results[1]:>15.4f}")
    print(f"{'Label Propagation':<25} {'Accuracy':>15} {label_prop_results[0]:>15.4f}")
    print(f"{'Label Propagation':<25} {'Macro F1':>15} {label_prop_results[1]:>15.4f}")

    print("\nQUESTION GENERATION RANKER")
    print(f"{'Accuracy':<15}: {question_ranker_results['accuracy']:.4f}")
    print(f"{'Macro F1':<15}: {question_ranker_results['macro_f1']:.4f}")
    print(f"{'Precision':<15}: {question_ranker_results['precision']:.4f}")
    print(f"{'Recall':<15}: {question_ranker_results['recall']:.4f}")

    print("\nSaved artifacts location:")
    print(SAVE_DIR)
    print("=" * 90)

if __name__ == "__main__":
    vectorizer, X_train, y_train, X_dev, y_dev, supervised_results = train_supervised_verifiers()

    ensemble_results = train_ensemble_verifier(X_train, y_train, X_dev, y_dev)

    kmeans_results = run_kmeans(X_dev, y_dev)

    label_prop_results = run_label_propagation(X_train, y_train, X_dev, y_dev)

    question_ranker_results = train_question_ranker()

    print_final_summary(
        supervised_results,
        ensemble_results,
        kmeans_results,
        label_prop_results,
        question_ranker_results
    )
