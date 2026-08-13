"""
Inference Module - Production Code
Exposes predict() and predict_batch() for single and bulk predictions.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering import load_pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/iris_pipeline.joblib"
FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


class IrisClassifier:
    """Wraps a fitted sklearn Pipeline for thread-safe, logged inference."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._pipeline: Pipeline | None = None

    def load(self) -> None:
        """Load the pipeline from disk."""
        logger.info("Loading model from '%s'.", self.model_path)
        self._pipeline = load_pipeline(self.model_path)
        logger.info("Model loaded. Classes: %s.", list(self._pipeline.classes_))

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        return self._pipeline

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict the species for a single sample.

        Args:
            features: dict with keys matching FEATURE_NAMES.

        Returns:
            dict with 'predicted_class' and 'probabilities'.
        """
        logger.info("Predicting single sample: %s", features)
        try:
            df = _dict_to_dataframe(features)
            pred_class = self.pipeline.predict(df)[0]
            proba = self.pipeline.predict_proba(df)[0]
            result = {
                "predicted_class": pred_class,
                "probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(self.pipeline.classes_, proba)
                },
            }
            logger.info(
                "Prediction: %s (confidence %.2f%%)", pred_class, max(proba) * 100
            )
            return result
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            raise

    def predict_batch(self, samples: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Predict species for a list of samples.

        Args:
            samples: list of feature dicts.

        Returns:
            list of prediction dicts.
        """
        logger.info("Predicting batch of %d samples.", len(samples))
        if not samples:
            logger.warning("Empty batch received.")
            return []
        try:
            df = pd.DataFrame(samples)[FEATURE_NAMES]
            pred_classes = self.pipeline.predict(df)
            probas = self.pipeline.predict_proba(df)
            results = []
            for pred, proba in zip(pred_classes, probas):
                results.append(
                    {
                        "predicted_class": pred,
                        "probabilities": {
                            cls: round(float(p), 4)
                            for cls, p in zip(self.pipeline.classes_, proba)
                        },
                    }
                )
            logger.info("Batch prediction complete: %d results.", len(results))
            return results
        except Exception as exc:
            logger.error("Batch inference failed: %s", exc)
            raise


def _dict_to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """Convert a single feature dict to a one-row DataFrame with correct column order."""
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
