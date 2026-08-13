"""
Model Training Module - Production Code
Handles model instantiation, training, evaluation, and persistence.
"""

import logging
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             log_loss)
from sklearn.pipeline import Pipeline

from src.feature_engineering import create_feature_pipeline, save_pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/iris_pipeline.joblib"


def build_model(
    n_estimators: int = 100, random_state: int = 42
) -> RandomForestClassifier:
    """Instantiate a RandomForestClassifier with the given hyperparameters."""
    logger.info(
        "Building RandomForestClassifier: n_estimators=%d, random_state=%d.",
        n_estimators,
        random_state,
    )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """
    Build a Pipeline, fit it on training data, save it to disk, and return it.
    Logs training progress and raises on failure.
    """
    logger.info("Starting model training with %d samples.", len(X_train))
    try:
        clf = build_model(n_estimators=n_estimators, random_state=random_state)
        pipeline = create_feature_pipeline(clf)
        pipeline.fit(X_train, y_train)
        logger.info("Model training complete.")
        save_pipeline(pipeline, model_path)
        return pipeline
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate a fitted pipeline on test data.
    Returns a dict of model quality metrics.
    """
    logger.info("Evaluating model on %d test samples.", len(X_test))
    try:
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        classes = pipeline.classes_
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        logloss = log_loss(y_test, y_proba, labels=classes)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "log_loss": round(logloss, 4),
            "classification_report": report,
        }
        logger.info(
            "Evaluation complete. Accuracy=%.4f, F1-macro=%.4f, Log-loss=%.4f.",
            accuracy,
            f1_macro,
            logloss,
        )
        return metrics
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise


def overfit_check(
    X_small: pd.DataFrame,
    y_small: pd.Series,
    n_estimators: int = 10,
) -> float:
    """
    Fit model on a tiny batch and return training accuracy.
    Accuracy should approach 1.0 (overfit) for a sanity check.
    """
    logger.info("Running overfit sanity check on %d samples.", len(X_small))
    clf = build_model(n_estimators=n_estimators, random_state=0)
    pipeline = create_feature_pipeline(clf)
    pipeline.fit(X_small, y_small)
    train_acc = accuracy_score(y_small, pipeline.predict(X_small))
    logger.info("Overfit check training accuracy: %.4f.", train_acc)
    return train_acc
