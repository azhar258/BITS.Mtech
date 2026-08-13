"""
Data Validation Tests - Verify schema, quality, and distribution of the dataset.
Run with: pytest tests/test_data_validation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import (CLASS_NAMES, FEATURE_NAMES, FEATURE_RANGES,
                                compute_data_quality_metrics, load_data)


@pytest.fixture(scope="module")
def iris_df():
    return load_data()


class TestSchemaValidation:
    def test_all_feature_columns_present(self, iris_df):
        for col in FEATURE_NAMES:
            assert col in iris_df.columns, f"Column '{col}' missing"

    def test_target_column_present(self, iris_df):
        assert "species" in iris_df.columns

    def test_feature_dtypes_numeric(self, iris_df):
        for col in FEATURE_NAMES:
            assert pd.api.types.is_numeric_dtype(
                iris_df[col]
            ), f"Column '{col}' should be numeric, got {iris_df[col].dtype}"

    def test_no_extra_unexpected_columns(self, iris_df):
        expected = set(FEATURE_NAMES) | {"species"}
        assert set(iris_df.columns) == expected


class TestMissingValueChecks:
    def test_no_missing_features(self, iris_df):
        null_counts = iris_df[FEATURE_NAMES].isnull().sum()
        assert (
            null_counts.sum() == 0
        ), f"Missing values detected:\n{null_counts[null_counts > 0]}"

    def test_no_missing_target(self, iris_df):
        assert iris_df["species"].isnull().sum() == 0

    def test_no_infinite_values(self, iris_df):
        for col in FEATURE_NAMES:
            assert not np.isinf(iris_df[col]).any(), f"Inf values in column '{col}'"


class TestValueRanges:
    def test_feature_values_within_expected_ranges(self, iris_df):
        for col, (low, high) in FEATURE_RANGES.items():
            assert (
                iris_df[col].min() >= low * 0.8
            ), f"'{col}' min {iris_df[col].min():.2f} unexpectedly low"
            assert (
                iris_df[col].max() <= high * 1.2
            ), f"'{col}' max {iris_df[col].max():.2f} unexpectedly high"

    def test_no_negative_measurements(self, iris_df):
        for col in FEATURE_NAMES:
            assert (iris_df[col] > 0).all(), f"Non-positive values in '{col}'"


class TestClassDistribution:
    def test_three_classes_present(self, iris_df):
        assert set(iris_df["species"].unique()) == set(CLASS_NAMES)

    def test_balanced_classes(self, iris_df):
        counts = iris_df["species"].value_counts()
        assert counts.min() >= 40, "Class severely underrepresented"
        assert counts.max() <= 60, "Class severely overrepresented"

    def test_correct_total_samples(self, iris_df):
        assert len(iris_df) == 150


class TestDuplicateDetection:
    def test_duplicate_rows_within_threshold(self, iris_df):
        """
        The Iris dataset is known to contain exactly 1 duplicate feature row
        (rows 34 and 37 have identical measurements). We allow up to 1% duplicates.
        """
        n_dupes = iris_df[FEATURE_NAMES].duplicated().sum()
        pct = n_dupes / len(iris_df) * 100
        assert pct < 1.0, f"Duplicate rows ({n_dupes}, {pct:.1f}%) exceed 1% threshold"


class TestDataQualityMetricsOutput:
    def test_missing_pct_all_zero(self, iris_df):
        metrics = compute_data_quality_metrics(iris_df)
        for col, pct in metrics["missing_value_pct"].items():
            assert pct == 0.0, f"Column '{col}' has {pct}% missing"

    def test_class_distribution_has_three_classes(self, iris_df):
        metrics = compute_data_quality_metrics(iris_df)
        assert len(metrics["class_distribution"]) == 3

    def test_outlier_counts_are_integers(self, iris_df):
        metrics = compute_data_quality_metrics(iris_df)
        for col in FEATURE_NAMES:
            key = f"{col}_outliers_z3"
            assert isinstance(metrics[key], int)
