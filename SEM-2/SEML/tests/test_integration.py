"""
Integration Tests - Test the full ML pipeline end-to-end.
Run with: pytest tests/test_integration.py -v
"""

import os

import pandas as pd
import pytest

from src.data_ingestion import load_data, split_data, validate_data
from src.feature_engineering import load_pipeline
from src.inference import IrisClassifier
from src.model_training import evaluate_model, train_model


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("models") / "integration_model.joblib")


@pytest.fixture(scope="module")
def full_pipeline(model_path):
    """Run the full training pipeline once for all integration tests."""
    df = load_data()
    validate_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = train_model(X_train, y_train, n_estimators=20, model_path=model_path)
    return pipeline, X_test, y_test, model_path


class TestFullTrainingPipeline:
    def test_pipeline_trains_without_error(self, full_pipeline):
        pipeline, X_test, y_test, _ = full_pipeline
        assert pipeline is not None

    def test_model_file_created(self, full_pipeline):
        _, _, _, model_path = full_pipeline
        assert os.path.exists(model_path)

    def test_accuracy_above_threshold(self, full_pipeline):
        pipeline, X_test, y_test, _ = full_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert (
            metrics["accuracy"] >= 0.85
        ), f"Accuracy {metrics['accuracy']:.4f} below acceptable threshold 0.85"

    def test_f1_above_threshold(self, full_pipeline):
        pipeline, X_test, y_test, _ = full_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert metrics["f1_macro"] >= 0.85


class TestInferencePipeline:
    def test_single_predict_returns_valid_class(self, full_pipeline):
        _, _, _, model_path = full_pipeline
        clf = IrisClassifier(model_path=model_path)
        clf.load()
        result = clf.predict(
            {
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            }
        )
        assert result["predicted_class"] in ["setosa", "versicolor", "virginica"]

    def test_batch_predict_length_matches_input(self, full_pipeline):
        _, _, _, model_path = full_pipeline
        clf = IrisClassifier(model_path=model_path)
        clf.load()
        samples = [
            {
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
            {
                "sepal_length_cm": 6.7,
                "sepal_width_cm": 3.0,
                "petal_length_cm": 5.2,
                "petal_width_cm": 2.3,
            },
        ]
        results = clf.predict_batch(samples)
        assert len(results) == 2

    def test_probabilities_sum_to_one(self, full_pipeline):
        _, _, _, model_path = full_pipeline
        clf = IrisClassifier(model_path=model_path)
        clf.load()
        result = clf.predict(
            {
                "sepal_length_cm": 6.3,
                "sepal_width_cm": 2.5,
                "petal_length_cm": 5.0,
                "petal_width_cm": 1.9,
            }
        )
        total_prob = sum(result["probabilities"].values())
        assert abs(total_prob - 1.0) < 1e-4

    def test_missing_feature_raises(self, full_pipeline):
        _, _, _, model_path = full_pipeline
        clf = IrisClassifier(model_path=model_path)
        clf.load()
        with pytest.raises(ValueError, match="Missing features"):
            clf.predict({"sepal_length_cm": 5.1, "sepal_width_cm": 3.5})

    def test_load_nonexistent_model_raises(self):
        clf = IrisClassifier(model_path="nonexistent/path.joblib")
        with pytest.raises(FileNotFoundError):
            clf.load()


class TestPipelineConsistency:
    def test_reload_gives_same_prediction(self, full_pipeline):
        """Loading the saved model should give identical predictions as the in-memory pipeline."""
        pipeline, X_test, _, model_path = full_pipeline
        reloaded = load_pipeline(model_path)
        original_preds = pipeline.predict(X_test)
        reloaded_preds = reloaded.predict(X_test)
        assert (original_preds == reloaded_preds).all()
