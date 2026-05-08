import pandas as pd
import numpy as np
import joblib
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required nltk data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

# ── Import Model B functions ──────────────
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_b_train import get_distractor_candidates, get_hints

# ── Paths ─────────────────────────────────
DATA_TEST = "data/processed/test.csv"
SAVE_DIR  = "models/model_b/traditional"

# ── Load Model B artifacts ─────────────────
def load_model_b():
    model     = joblib.load(os.path.join(SAVE_DIR, "distractor_ranker.pkl"))
    vectorizer = joblib.load(os.path.join(SAVE_DIR, "distractor_vectorizer.pkl"))
    return model, vectorizer

def build_distractor_eval_dataset(df):
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


def evaluate_distractor_ranker(df, model, vectorizer):
    print("\n" + "=" * 60)
    print("EVALUATING MODEL B DISTRACTOR RANKER")
    print("=" * 60)

    eval_df = build_distractor_eval_dataset(df)

    X_text = eval_df["text"]
    y_true = eval_df["label"]

    X = vectorizer.transform(X_text)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n── Model B Ranker Results ──")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm
    }
# ── BLEU Score ────────────────────────────
def compute_bleu(reference, hypothesis):
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    smoothie    = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

# ── ROUGE Score ───────────────────────────
def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# ── METEOR Score ──────────────────────────
def compute_meteor(reference, hypothesis):
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    return meteor_score([ref_tokens], hyp_tokens)

# ── Evaluate Distractors ──────────────────
def evaluate_distractors(df, vectorizer, n_samples=100):
    print("\n" + "="*60)
    print("EVALUATING DISTRACTORS — BLEU, ROUGE, METEOR")
    print("="*60)

    bleu_scores   = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []

    options = ["A", "B", "C", "D"]
    sample  = df.sample(min(n_samples, len(df)), random_state=42)

    for _, row in sample.iterrows():
        article        = str(row["article"])
        correct_answer = str(row[row["answer"]])

        # Get wrong options from dataset as reference distractors
        wrong_options = [str(row[o]) for o in options if o != row["answer"]]

        # Generate distractors using Model B
        generated = get_distractor_candidates(article, correct_answer, vectorizer)

        # Compare each generated distractor to reference wrong options
        for gen, ref in zip(generated, wrong_options):
            bleu_scores.append(compute_bleu(ref, gen))
            rouge = compute_rouge(ref, gen)
            rouge1_scores.append(rouge["rouge1"])
            rouge2_scores.append(rouge["rouge2"])
            rougeL_scores.append(rouge["rougeL"])
            meteor_scores.append(compute_meteor(ref, gen))

    print(f"\nEvaluated on {len(sample)} samples")
    print(f"\n── Distractor Evaluation Results ──")
    print(f"BLEU Score   : {np.mean(bleu_scores):.4f}")
    print(f"ROUGE-1      : {np.mean(rouge1_scores):.4f}")
    print(f"ROUGE-2      : {np.mean(rouge2_scores):.4f}")
    print(f"ROUGE-L      : {np.mean(rougeL_scores):.4f}")
    print(f"METEOR Score : {np.mean(meteor_scores):.4f}")

    return {
        "bleu":   round(np.mean(bleu_scores), 4),
        "rouge1": round(np.mean(rouge1_scores), 4),
        "rouge2": round(np.mean(rouge2_scores), 4),
        "rougeL": round(np.mean(rougeL_scores), 4),
        "meteor": round(np.mean(meteor_scores), 4),
    }

# ── Evaluate Hints ────────────────────────
def evaluate_hints(df, vectorizer, n_samples=100):
    print("\n" + "="*60)
    print("EVALUATING HINTS — BLEU, ROUGE, METEOR")
    print("="*60)

    bleu_scores   = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []

    sample = df.sample(min(n_samples, len(df)), random_state=99)

    for _, row in sample.iterrows():
        article  = str(row["article"])
        question = str(row["question"])

        # Use the correct answer as the reference
        reference = str(row[row["answer"]])

        # Generate hints using Model B
        hints = get_hints(article, question, vectorizer)

        # Evaluate the most specific hint (hint 3) against reference
        best_hint = hints[-1]

        bleu_scores.append(compute_bleu(reference, best_hint))
        rouge = compute_rouge(reference, best_hint)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])
        meteor_scores.append(compute_meteor(reference, best_hint))

    print(f"\nEvaluated on {len(sample)} samples")
    print(f"\n── Hint Evaluation Results ──")
    print(f"BLEU Score   : {np.mean(bleu_scores):.4f}")
    print(f"ROUGE-1      : {np.mean(rouge1_scores):.4f}")
    print(f"ROUGE-2      : {np.mean(rouge2_scores):.4f}")
    print(f"ROUGE-L      : {np.mean(rougeL_scores):.4f}")
    print(f"METEOR Score : {np.mean(meteor_scores):.4f}")

    return {
        "bleu":   round(np.mean(bleu_scores), 4),
        "rouge1": round(np.mean(rouge1_scores), 4),
        "rouge2": round(np.mean(rouge2_scores), 4),
        "rougeL": round(np.mean(rougeL_scores), 4),
        "meteor": round(np.mean(meteor_scores), 4),
    }

# ── Main ──────────────────────────────────
if __name__ == "__main__":
    print("Loading test data...")
    df = pd.read_csv(DATA_TEST)

    print("Loading Model B...")
    model, vectorizer = load_model_b()

    ranker_results = evaluate_distractor_ranker(df, model, vectorizer)

    distractor_results = evaluate_distractors(df, vectorizer, n_samples=100)
    hint_results       = evaluate_hints(df, vectorizer, n_samples=100)

    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)

    print("\nModel B Distractor Ranker:")
    for k, v in ranker_results.items():
        if k != "confusion_matrix":
            print(f"  {k.upper():<10}: {v}")

    print("\nDistractor Generation:")
    for k, v in distractor_results.items():
        print(f"  {k.upper():<10}: {v}")

    print("\nHint Generation:")
    for k, v in hint_results.items():
        print(f"  {k.upper():<10}: {v}")