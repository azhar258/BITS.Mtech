"""
Unit Tests - Test individual functions in each module.
Run with: pytest tests/test_unit.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_ingestion import (CLASS_NAMES, FEATURE_NAMES,
                                compute_data_quality_metrics, load_data,
                                split_data, validate_data)
from src.feature_engineering import create_feature_pipeline, scale_features
from src.model_training import build_model, evaluate_model, train_model

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def iris_df():
    return load_data()


@pytest.fixture
def split_iris(iris_df):
    return split_data(iris_df)


@pytest.fixture
def trained_pipeline(tmp_path, split_iris):
    X_train, X_test, y_train, y_test = split_iris
    model_path = str(tmp_path / "test_model.joblib")
    pipeline = train_model(X_train, y_train, n_estimators=10, model_path=model_path)
    return pipeline, X_test, y_test


# ─── data_ingestion tests ──────────────────────────────────────────────────────


class TestLoadData:
    def test_returns_dataframe(self, iris_df):
        assert isinstance(iris_df, pd.DataFrame)

    def test_correct_columns(self, iris_df):
        assert all(c in iris_df.columns for c in FEATURE_NAMES + ["species"])

    def test_correct_row_count(self, iris_df):
        assert len(iris_df) == 150

    def test_valid_class_labels(self, iris_df):
        assert set(iris_df["species"].unique()) == set(CLASS_NAMES)


class TestValidateData:
    def test_valid_data_passes(self, iris_df):
        assert validate_data(iris_df) is True

    def test_missing_column_raises(self, iris_df):
        bad_df = iris_df.drop(columns=["sepal_length_cm"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_data(bad_df)

    def test_missing_values_raises(self, iris_df):
        bad_df = iris_df.copy()
        bad_df.loc[0, "sepal_length_cm"] = np.nan
        with pytest.raises(ValueError, match="Missing values"):
            validate_data(bad_df)

    def test_invalid_label_raises(self, iris_df):
        bad_df = iris_df.copy()
        bad_df.loc[0, "species"] = "unknown_flower"
        with pytest.raises(ValueError, match="Invalid class labels"):
            validate_data(bad_df)


class TestSplitData:
    def test_split_sizes(self, iris_df):
        X_train, X_test, y_train, y_test = split_data(iris_df, test_size=0.2)
        assert len(X_train) == 120
        assert len(X_test) == 30

    def test_no_overlap(self, iris_df):
        X_train, X_test, _, _ = split_data(iris_df, test_size=0.2)
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_invalid_test_size_raises(self, iris_df):
        with pytest.raises(ValueError):
            split_data(iris_df, test_size=1.5)


class TestDataQualityMetrics:
    def test_returns_dict(self, iris_df):
        metrics = compute_data_quality_metrics(iris_df)
        assert isinstance(metrics, dict)

    def test_keys_present(self, iris_df):
        metrics = compute_data_quality_metrics(iris_df)
        for key in ["total_rows", "missing_value_counts", "class_distribution"]:
            assert key in metrics

    def test_no_missing_values_in_clean_data(self, iris_df):
        metrics = compute_data_quality_metrics(iris_df)
        for col, count in metrics["missing_value_counts"].items():
            assert count == 0


# ─── feature_engineering tests ────────────────────────────────────────────────


class TestScaleFeatures:
    def test_output_shapes(self, split_iris):
        X_train, X_test, _, _ = split_iris
        X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)
        assert X_tr_s.shape == X_train.shape
        assert X_te_s.shape == X_test.shape

    def test_train_mean_near_zero(self, split_iris):
        X_train, X_test, _, _ = split_iris
        X_tr_s, _, _ = scale_features(X_train, X_test)
        assert np.allclose(X_tr_s.mean(axis=0), 0, atol=1e-10)

    def test_train_std_near_one(self, split_iris):
        X_train, X_test, _, _ = split_iris
        X_tr_s, _, _ = scale_features(X_train, X_test)
        assert np.allclose(X_tr_s.std(axis=0), 1, atol=1e-10)


class TestCreatePipeline:
    def test_returns_pipeline(self, split_iris):
        X_train, _, y_train, _ = split_iris
        clf = build_model(n_estimators=5)
        pipeline = create_feature_pipeline(clf)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_scaler_and_classifier(self, split_iris):
        clf = build_model(n_estimators=5)
        pipeline = create_feature_pipeline(clf)
        assert "scaler" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


# ─── model_training tests ─────────────────────────────────────────────────────


class TestBuildModel:
    def test_returns_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = build_model()
        assert isinstance(clf, RandomForestClassifier)

    def test_custom_params(self):
        clf = build_model(n_estimators=50, random_state=7)
        assert clf.n_estimators == 50
        assert clf.random_state == 7


class TestTrainModel:
    def test_returns_fitted_pipeline(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        assert isinstance(pipeline, Pipeline)
        assert hasattr(pipeline, "classes_")

    def test_model_saved_to_disk(self, tmp_path, split_iris):
        X_train, _, y_train, _ = split_iris
        model_path = str(tmp_path / "saved_model.joblib")
        train_model(X_train, y_train, n_estimators=5, model_path=model_path)
        import os

        assert os.path.exists(model_path)


class TestEvaluateModel:
    def test_returns_expected_keys(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)
        for key in ["accuracy", "f1_macro", "f1_weighted", "log_loss"]:
            assert key in metrics

    def test_accuracy_in_range(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_in_range(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert 0.0 <= metrics["f1_macro"] <= 1.0
