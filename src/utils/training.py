"""
Training utilities for benchmark notebooks (non-FLAML).

Contains only helpers used by the LogReg/XGBoost/CatBoost notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

# Dataset name prefixes for which targets are regression by default
# (except group_task_label).
REGRESSION_DATASET_PREFIXES = ("Cognitive_load", "Emotions_",)


def load_split_info(split_info_path: Union[str, Path]) -> Dict[str, Any]:
    """Load split-info JSON file."""
    with open(split_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_task_type(y: pd.Series) -> str:
    """Infer classification/regression from target values."""
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        n_samples = len(y)

        if pd.api.types.is_integer_dtype(y):
            if n_unique < 20 and n_unique < n_samples * 0.1:
                return "classification"
            if n_unique <= 10:
                return "classification"
        else:
            if n_unique < 10:
                return "classification"

        return "regression"
    return "classification"


def get_task_type_for_dataset_label(dataset_name: str, label_column: str, y: pd.Series) -> str:
    """
    Effective task type for a (dataset, label) pair.
    Surgical/Cognitive/Emotion datasets are regression by default,
    except group_task_label.
    """
    if label_column == "group_task_label":
        return get_task_type(y)
    if any(dataset_name.startswith(prefix) for prefix in REGRESSION_DATASET_PREFIXES):
        return "regression"
    return get_task_type(y)


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], task_type: str
) -> Dict[str, float]:
    """Compute metrics for classification or regression."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    metrics: Dict[str, Any] = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if task_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
        except Exception:
            metrics["roc_auc"] = np.nan
        metrics["n_classes"] = len(np.unique(y_true))
        metrics["class_distribution"] = str(dict(zip(*np.unique(y_true, return_counts=True))))
    else:
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mean_target"] = np.mean(y_true)
        metrics["std_target"] = np.std(y_true)

    return metrics


def save_results_incremental(new_result: Dict[str, Any], results_file: Union[str, Path]) -> None:
    """Append one result row to CSV and keep it sorted."""
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([new_result])
    if results_file.exists():
        existing_df = pd.read_csv(results_file)
        results_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        results_df = new_df

    sort_cols = ["dataset", "label", "feature_battery"]
    if "aoi_column" in results_df.columns:
        sort_cols.append("aoi_column")
    if all(col in results_df.columns for col in sort_cols):
        results_df = results_df.sort_values(sort_cols)

    results_df.to_csv(results_file, index=False)


def print_results_summary(results_df: pd.DataFrame):
    """Print compact summary for training result tables."""
    if results_df.empty:
        print("No results to summarize.")
        return

    print("\nSummary Statistics:")
    print(f"  Total results: {len(results_df)}")
    if "dataset" in results_df.columns:
        print(f"  Datasets: {results_df['dataset'].nunique()}")
    if "feature_battery" in results_df.columns:
        print(f"  Feature batteries: {results_df['feature_battery'].nunique()}")
    if "label" in results_df.columns:
        print(f"  Labels: {results_df['label'].nunique()}")

