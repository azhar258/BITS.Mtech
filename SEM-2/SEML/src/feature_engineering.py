"""
Feature Engineering Module - Production Code
Handles scaling and building sklearn pipelines for the Iris classifier.
"""

import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """Fit StandardScaler on training data and transform both splits."""
    logger.info("Fitting StandardScaler on %d training samples.", len(X_train))
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Feature scaling complete.")
        return X_train_scaled, X_test_scaled, scaler
    except Exception as exc:
        logger.error("Feature scaling failed: %s", exc)
        raise


def create_feature_pipeline(model) -> Pipeline:
    """
    Wrap a classifier with a StandardScaler into an sklearn Pipeline.
    The pipeline ensures identical preprocessing during training and inference.
    """
    logger.info(
        "Creating sklearn Pipeline: [StandardScaler -> %s].", type(model).__name__
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", model),
        ]
    )
    logger.info("Pipeline created successfully.")
    return pipeline


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Persist a fitted pipeline to disk using joblib."""
    logger.info("Saving pipeline to '%s'.", path)
    try:
        joblib.dump(pipeline, path)
        logger.info("Pipeline saved successfully.")
    except Exception as exc:
        logger.error("Failed to save pipeline: %s", exc)
        raise


def load_pipeline(path: str) -> Pipeline:
    """Load a persisted pipeline from disk."""
    logger.info("Loading pipeline from '%s'.", path)
    try:
        pipeline = joblib.load(path)
        logger.info("Pipeline loaded successfully.")
        return pipeline
    except FileNotFoundError:
        logger.error("Pipeline file not found: '%s'.", path)
        raise
    except Exception as exc:
        logger.error("Failed to load pipeline: %s", exc)
        raise
