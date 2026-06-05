"""
=============================================================
modelling.py  —  Baseline Model Training (FIXED VERSION)
Nama Siswa : Valina Puspita Sari
Kriteria   : 2 — Membangun Model Machine Learning
MLflow     : autolog() | Tracking via DagsHub
Dataset    : Heart Failure Prediction
=============================================================
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DAGSHUB_USERNAME = "Valina22"
DAGSHUB_REPO     = "Workflow-CI"
EXPERIMENT_NAME  = "Heart_Disease_Baseline"

TRAIN_PATH = "dataset_preprocessing/train.csv"
VAL_PATH   = "dataset_preprocessing/val.csv"
TEST_PATH  = "dataset_preprocessing/test.csv"
TARGET_COL = "HeartDisease"

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# INIT DAGSUB + MLFLOW
# ─────────────────────────────────────────────
import os

if os.getenv("RUNNING_IN_DOCKER", "0") != "1":
    dagshub.init(DAGSHUB_REPO, DAGSHUB_USERNAME)
    
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
mlflow.set_experiment(EXPERIMENT_NAME)

# IMPORTANT: autolog hanya sekali (FIX)
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=False,
    silent=True,
)


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
def load_splits():
    logger.info("Loading dataset splits...")

    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    logger.info(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)

    metrics = {
        f"{split_name}_accuracy": accuracy_score(y, y_pred),
        f"{split_name}_f1": f1_score(y, y_pred),
        f"{split_name}_precision": precision_score(y, y_pred),
        f"{split_name}_recall": recall_score(y, y_pred),
    }

    # ROC AUC SAFE HANDLING
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
            metrics[f"{split_name}_roc_auc"] = roc_auc_score(y, y_proba)
        except Exception:
            pass

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return metrics


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
def save_confusion_matrix(model, X, y, model_name: str, output_dir="artifacts"):
    Path(output_dir).mkdir(exist_ok=True)

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    path = f"{output_dir}/{model_name}_cm.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
BASELINE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING STARTED")
    logger.info("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    results = []
    Path("artifacts").mkdir(exist_ok=True)

    for name, model in BASELINE_MODELS.items():
        logger.info(f"\nTraining {name}")

        # ─────────────────────────────
        # TRAIN
        # ─────────────────────────────
        model.fit(X_train, y_train)

        # ─────────────────────────────
        # EVALUATION
        # ─────────────────────────────
        logger.info("Validation")
        val_metrics = evaluate(model, X_val, y_val, "val")

        logger.info("Test")
        test_metrics = evaluate(model, X_test, y_test, "test")

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="f1"
        )

        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # ─────────────────────────────
        # LOG METRICS (MLflow Projects SAFE MODE)
        # ─────────────────────────────
        mlflow.log_metrics({
            **val_metrics,
            **test_metrics,
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std
        })

        # ─────────────────────────────
        # CLASSIFICATION REPORT
        # ─────────────────────────────
        report = classification_report(y_val, model.predict(X_val))
        report_path = f"artifacts/{name}_report.txt"

        with open(report_path, "w") as f:
            f.write(f"Model: {name}\n\n")
            f.write(report)

        mlflow.log_artifact(report_path)

        # ─────────────────────────────
        # CONFUSION MATRIX
        # ─────────────────────────────
        cm_path = save_confusion_matrix(model, X_val, y_val, name)
        mlflow.log_artifact(cm_path)

        # ─────────────────────────────
        # TAGS
        # ─────────────────────────────
        mlflow.set_tags({
            "model": name,
            "student": "Valina Puspita Sari",
            "stage": "baseline"
        })

        # ─────────────────────────────
        # SAVE SUMMARY
        # ─────────────────────────────
        results.append({
            "model": name,
            "val_f1": val_metrics["val_f1"],
            "test_f1": test_metrics["test_f1"],
            "val_auc": val_metrics.get("val_roc_auc", 0)
        })

    # ─────────────────────────────
    # LEADERBOARD
    # ─────────────────────────────
    board = pd.DataFrame(results).sort_values("val_f1", ascending=False)

    logger.info("\n" + "=" * 60)
    logger.info("LEADERBOARD")
    logger.info("=" * 60)
    logger.info("\n" + board.to_string(index=False))

    best = board.iloc[0]
    logger.info(
        f"\nBest Model: {best['model']} | F1: {best['val_f1']:.4f}"
    )

    logger.info("DONE")