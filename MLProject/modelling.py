print("🚀 SCRIPT STARTED")

import shutil
import os
import sys
import logging
import warnings
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

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

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXPERIMENT_NAME = "Heart_Disease_Baseline"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_preprocessing"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET_COL = "HeartDisease"

# ─────────────────────────────────────────────
# SAFE DAGS HUB INIT
# ─────────────────────────────────────────────
def init_dagshub():
    if os.getenv("RUNNING_IN_DOCKER") == "1":
        logger.info("Running in Docker → skip DagsHub OAuth")
        return

    try:
        import dagshub
        def init_dagshub():
            logger.info("Skip DagsHub configuration")
            logger.info("DagsHub initialized")
    except Exception as e:
        logger.warning(f"DagsHub init skipped: {e}")

init_dagshub()

# ─────────────────────────────────────────────
# MLFLOW CONFIG
# ─────────────────────────────────────────────
logger.info("Using local MLflow tracking")

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=True
)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def load_splits():
    logger.info("Loading dataset...")

    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X, y, prefix):
    pred = model.predict(X)

    metrics = {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
    }

    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X)[:, 1]
            metrics[f"{prefix}_roc_auc"] = roc_auc_score(y, prob)
        except:
            pass

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return metrics


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
def save_cm(model, X, y, name):
    Path("artifacts").mkdir(exist_ok=True)

    pred = model.predict(X)
    cm = confusion_matrix(y, pred)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    path = f"artifacts/{name}_cm.png"
    plt.savefig(path)
    plt.close()

    return path


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
MODELS = {
    "LogReg": LogisticRegression(max_iter=1000),
    "Tree": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(n_estimators=100),
    "GB": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}


# ─────────────────────────────────────────────
# MAIN (FIX: MLflow RUN CONTEXT)
# ─────────────────────────────────────────────
def main():
    logger.info("=" * 50)
    logger.info("START TRAINING")
    logger.info("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    results = []
    trained_models = {}
):

        for name, model in MODELS.items():
            logger.info(f"\nTraining {name}")

            model.fit(X_train, y_train)
            trained_models[name] = model

            val_metrics = evaluate(model, X_val, y_val, "val")
            test_metrics = evaluate(model, X_test, y_test, "test")

            cv = cross_val_score(
                model,
                X_train,
                y_train,
                cv=3,
                scoring="f1"
            )

            mlflow.log_metrics({
                **val_metrics,
                **test_metrics,
                f"{name}_cv_f1_mean": cv.mean()
            })

            report = classification_report(
                y_val,
                model.predict(X_val)
            )

            Path("artifacts").mkdir(exist_ok=True)

            report_path = f"artifacts/{name}_report.txt"

            with open(report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(report_path)
            mlflow.log_artifact(
                save_cm(model, X_val, y_val, name)
            )

            results.append({
                "model": name,
                "val_f1": val_metrics["val_f1"],
                "test_f1": test_metrics["test_f1"]
            })

    df = pd.DataFrame(results).sort_values(
        "val_f1",
        ascending=False
    )

    logger.info("\nLEADERBOARD")
    logger.info(df.to_string(index=False))

    best_model_name = df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    Path("models").mkdir(exist_ok=True)

    MODEL_DIR = "models/best_model"

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    mlflow.sklearn.save_model(
        sk_model=best_model,
        path=MODEL_DIR
    )
   

    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
