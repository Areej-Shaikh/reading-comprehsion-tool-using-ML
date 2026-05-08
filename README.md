# Intelligent Reading Comprehension and Quiz Generation System

## Overview

This project is an AI-based Reading Comprehension and Quiz Generation System built using the RACE dataset. The system can:

- Generate comprehension questions from passages
- Verify answers using machine learning models
- Generate distractor options
- Provide hints
- Display quizzes through an interactive Streamlit interface

---

# Main Features

## Model A — Question Generation & Answer Verification

Implemented models:
- Logistic Regression
- Support Vector Machine (SVM)
- Hard Voting Ensemble
- K-Means
- Gaussian Mixture Model (GMM)
- Label Propagation

Functionalities:
- Template-based question generation
- Question ranking using Random Forest
- Answer verification

---

## Model B — Distractor & Hint Generation

Functionalities:
- Distractor generation using cosine similarity
- Distractor ranking with Logistic Regression
- Extractive hint generation
- Multi-level hints from general to specific

---

# User Interface

The application includes:
1. Article Input Screen
2. Quiz Screen
3. Hint Panel
4. Analytics Dashboard

---

# Dataset

The project uses the RACE dataset, which contains:
- Reading passages
- Multiple-choice questions
- Four answer options (A/B/C/D)
- Correct answer labels

---

# Installation & Setup

## Clone Repository

```bash
git clone <repository-url>
cd reading-comprehsion-tool-using-ML
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Preprocess Dataset

```bash
python src/preprocessing.py
```

This creates processed train, dev, and test datasets.

---

## 2. Train Model A

```bash
python src/model_a_train.py
```

This trains:
- Logistic Regression
- SVM
- Ensemble models
- K-Means
- GMM
- Label Propagation
- Question Ranker

---

## 3. Train Model B

```bash
python src/model_b_train.py
```

This trains:
- Distractor Ranker
- Hint Generation pipeline

---

## 4. Run Evaluation

```bash
python src/evaluate.py
```

Evaluation includes:
- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix
- BLEU
- ROUGE
- METEOR

---

## 5. Run Unit Tests

```bash
python -m pytest .\tests\test_inference.py
```

---

## 6. Launch Streamlit App

```bash
streamlit run ui/app.py
```

---

# Dashboard Features

The dashboard displays:
- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix
- Session Logs
- Latency Tracking
- Quiz Analytics

---
# Authors

- Navaira Azmat
- Areej Shaikh
- FAST-NUCES Islamabad  
