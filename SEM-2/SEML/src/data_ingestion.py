"""
Data Ingestion Module - Production Code
Handles loading, validating, and splitting the Iris dataset.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_NAME = "species"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

FEATURE_RANGES = {
    "sepal_length_cm": (4.0, 8.0),
    "sepal_width_cm": (1.5, 5.0),
    "petal_length_cm": (0.5, 8.0),
    "petal_width_cm": (0.1, 3.0),
}


def load_data() -> pd.DataFrame:
    """Load Iris dataset and return as a labelled DataFrame."""
    logger.info("Loading Iris dataset from sklearn.")
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
        df[TARGET_NAME] = [CLASS_NAMES[t] for t in iris.target]
        logger.info("Dataset loaded: %d rows, %d columns.", len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate dataset integrity: checks schema, missing values, and value ranges.
    Returns True if valid, raises ValueError otherwise.
    """
    logger.info("Validating dataset with %d rows.", len(df))

    # Schema check
    missing_cols = [c for c in FEATURE_NAMES + [TARGET_NAME] if c not in df.columns]
    if missing_cols:
        logger.error("Schema validation failed. Missing columns: %s", missing_cols)
        raise ValueError(f"Missing columns: {missing_cols}")

    # Missing values check
    null_counts = df[FEATURE_NAMES].isnull().sum()
    if null_counts.any():
        logger.warning("Missing values detected:\n%s", null_counts[null_counts > 0])
        raise ValueError(
            f"Missing values found: {null_counts[null_counts > 0].to_dict()}"
        )

    # Range validation
    for col, (low, high) in FEATURE_RANGES.items():
        out_of_range = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_range > 0:
            logger.warning(
                "Column '%s': %d values outside expected range [%.1f, %.1f].",
                col,
                out_of_range,
                low,
                high,
            )

    # Class label validation
    invalid_labels = set(df[TARGET_NAME].unique()) - set(CLASS_NAMES)
    if invalid_labels:
        logger.error("Invalid class labels found: %s", invalid_labels)
        raise ValueError(f"Invalid class labels: {invalid_labels}")

    logger.info("Dataset validation passed.")
    return True


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test feature and label sets."""
    logger.info(
        "Splitting data: test_size=%.2f, random_state=%d.", test_size, random_state
    )
    if not (0.0 < test_size < 1.0):
        logger.error("Invalid test_size: %f. Must be between 0 and 1.", test_size)
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split complete: %d train, %d test samples.", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def compute_data_quality_metrics(df: pd.DataFrame) -> dict:
    """Return a dict of data quality metrics for reporting."""
    logger.info("Computing data quality metrics.")
    metrics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_value_counts": df[FEATURE_NAMES].isnull().sum().to_dict(),
        "missing_value_pct": (df[FEATURE_NAMES].isnull().mean() * 100)
        .round(2)
        .to_dict(),
        "class_distribution": df[TARGET_NAME].value_counts().to_dict(),
        "feature_stats": df[FEATURE_NAMES].describe().round(4).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Detect feature drift proxy: z-score outliers (|z| > 3)
    for col in FEATURE_NAMES:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            outliers = ((df[col] - mean).abs() / std > 3).sum()
            metrics[f"{col}_outliers_z3"] = int(outliers)

    logger.info("Data quality metrics computed.")
    return metrics
