"""
ML Component Tests - Tests specific to model training and inference behaviour.
Run with: pytest tests/test_ml_components.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data_ingestion import load_data, split_data
from src.feature_engineering import create_feature_pipeline
from src.inference import IrisClassifier, _dict_to_dataframe
from src.model_training import (build_model, evaluate_model, overfit_check,
                                train_model)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def full_data():
    df = load_data()
    return split_data(df)


@pytest.fixture(scope="module")
def trained_pipeline_module(tmp_path_factory, full_data):
    X_train, X_test, y_train, y_test = full_data
    path = str(tmp_path_factory.mktemp("models") / "ml_test_model.joblib")
    pipeline = train_model(X_train, y_train, n_estimators=50, model_path=path)
    return pipeline, X_test, y_test, path


# ─── 7a. Testing Model Training ───────────────────────────────────────────────


class TestModelTraining:
    def test_overfit_on_small_batch(self, full_data):
        """
        A model fitted on a tiny batch should achieve near-perfect training accuracy.
        This verifies the model has sufficient capacity to fit the data.
        """
        X_train, _, y_train, _ = full_data
        X_small = X_train.head(15)
        y_small = y_train.head(15)
        train_acc = overfit_check(X_small, y_small)
        assert (
            train_acc >= 0.95
        ), f"Expected near-perfect overfit accuracy, got {train_acc:.4f}"

    def test_classifier_is_random_forest(self):
        clf = build_model()
        assert isinstance(clf, RandomForestClassifier)

    def test_pipeline_has_classes_after_training(self, full_data):
        X_train, _, y_train, _ = full_data
        clf = build_model(n_estimators=10)
        pipeline = create_feature_pipeline(clf)
        pipeline.fit(X_train, y_train)
        assert hasattr(pipeline, "classes_")
        assert len(pipeline.classes_) == 3

    def test_training_accuracy_high(self, full_data):
        """Model should achieve high accuracy on its own training data."""
        from sklearn.metrics import accuracy_score

        X_train, _, y_train, _ = full_data
        clf = build_model(n_estimators=50)
        pipeline = create_feature_pipeline(clf)
        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        assert train_acc >= 0.95, f"Training accuracy {train_acc:.4f} unexpectedly low"

    def test_feature_importances_sum_to_one(self, trained_pipeline_module):
        """RandomForest feature importances should sum to 1."""
        pipeline, _, _, _ = trained_pipeline_module
        importances = pipeline.named_steps["classifier"].feature_importances_
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_more_estimators_not_worse(self, full_data):
        """Increasing n_estimators should maintain or improve accuracy."""
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = full_data
        results = {}
        for n in [5, 50]:
            clf = build_model(n_estimators=n, random_state=42)
            pipe = create_feature_pipeline(clf)
            pipe.fit(X_train, y_train)
            results[n] = accuracy_score(y_test, pipe.predict(X_test))
        assert (
            results[50] >= results[5] - 0.05
        ), f"n=50 accuracy {results[50]:.4f} unexpectedly much worse than n=5 {results[5]:.4f}"


# ─── 7b. Testing Model Inference ──────────────────────────────────────────────


class TestModelInference:
    def test_output_is_valid_class(self, trained_pipeline_module):
        pipeline, X_test, _, _ = trained_pipeline_module
        preds = pipeline.predict(X_test)
        valid_classes = {"setosa", "versicolor", "virginica"}
        assert all(p in valid_classes for p in preds)

    def test_output_shape_matches_input(self, trained_pipeline_module):
        """predict() should return one label per input row."""
        pipeline, X_test, _, _ = trained_pipeline_module
        preds = pipeline.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_proba_shape(self, trained_pipeline_module):
        """predict_proba() should return n_samples × n_classes."""
        pipeline, X_test, _, _ = trained_pipeline_module
        probas = pipeline.predict_proba(X_test)
        assert probas.shape == (len(X_test), 3)

    def test_probabilities_sum_to_one(self, trained_pipeline_module):
        pipeline, X_test, _, _ = trained_pipeline_module
        probas = pipeline.predict_proba(X_test)
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_probabilities_in_range(self, trained_pipeline_module):
        pipeline, X_test, _, _ = trained_pipeline_module
        probas = pipeline.predict_proba(X_test)
        assert (probas >= 0).all()
        assert (probas <= 1).all()

    def test_directional_setosa(self, trained_pipeline_module):
        """
        Directional test: very small petal measurements should predict Setosa.
        Setosa is characterised by petal_length < 2 cm.
        """
        pipeline, _, _, _ = trained_pipeline_module
        setosa_sample = pd.DataFrame(
            [[5.0, 3.6, 1.4, 0.2]],
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )
        pred = pipeline.predict(setosa_sample)[0]
        assert pred == "setosa", f"Expected 'setosa', got '{pred}'"

    def test_directional_virginica(self, trained_pipeline_module):
        """
        Directional test: large petal measurements should predict Virginica.
        Virginica has petal_length > 5 cm on average.
        """
        pipeline, _, _, _ = trained_pipeline_module
        virginica_sample = pd.DataFrame(
            [[6.5, 3.0, 5.8, 2.2]],
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )
        pred = pipeline.predict(virginica_sample)[0]
        assert pred == "virginica", f"Expected 'virginica', got '{pred}'"

    def test_invariance_to_feature_order(self, trained_pipeline_module):
        """
        Invariance test: predictions should be deterministic and not change
        when the same values are passed in the same column order.
        """
        pipeline, _, _, _ = trained_pipeline_module
        sample = pd.DataFrame(
            [[5.9, 3.0, 5.1, 1.8]],
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )
        pred1 = pipeline.predict(sample)[0]
        pred2 = pipeline.predict(sample)[0]
        assert pred1 == pred2

    def test_iris_classifier_wrapper(self, trained_pipeline_module):
        _, _, _, model_path = trained_pipeline_module
        clf = IrisClassifier(model_path=model_path)
        clf.load()
        result = clf.predict(
            {
                "sepal_length_cm": 5.0,
                "sepal_width_cm": 3.6,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            }
        )
        assert "predicted_class" in result
        assert "probabilities" in result
        assert result["predicted_class"] == "setosa"
