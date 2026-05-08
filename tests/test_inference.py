import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model_b_train import get_distractor_candidates, get_hints


def test_model_files_exist():
    assert os.path.exists("models/model_a/traditional/logistic_regression_verifier.pkl")
    assert os.path.exists("models/model_a/traditional/svm_verifier.pkl")
    assert os.path.exists("models/model_a/traditional/onehot_vectorizer.pkl")
    assert os.path.exists("models/model_b/traditional/distractor_ranker.pkl")
    assert os.path.exists("models/model_b/traditional/distractor_vectorizer.pkl")


def test_processed_data_exists():
    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/dev.csv")
    assert os.path.exists("data/processed/test.csv")


def test_distractor_generation_basic():
    import joblib

    vectorizer = joblib.load("models/model_b/traditional/distractor_vectorizer.pkl")

    article = (
        "Ali went to the library. He borrowed a science book. "
        "Then he returned home and studied for his exam."
    )
    correct_answer = "science book"

    distractors = get_distractor_candidates(article, correct_answer, vectorizer)

    assert isinstance(distractors, list)
    assert len(distractors) == 3
    assert all(isinstance(d, str) for d in distractors)


def test_hint_generation_basic():
    import joblib

    vectorizer = joblib.load("models/model_b/traditional/distractor_vectorizer.pkl")

    article = (
        "Sara planted flowers in her garden. She watered them every morning. "
        "After two weeks, the flowers started blooming."
    )
    question = "What did Sara plant in her garden?"

    hints = get_hints(article, question, vectorizer)

    assert isinstance(hints, list)
    assert len(hints) == 3
    assert all(isinstance(h, str) for h in hints)