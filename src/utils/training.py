"""
Training utilities for benchmark notebooks (non-FLAML).

Contains only helpers used by the LogReg/XGBoost/CatBoost notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

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


def _sanitize_filename_component(value: str) -> str:
    for char in ("/", "\\", ":", "*", "?", '"', "<", ">", "|", "+"):
        value = str(value).replace(char, "_")
    return value


def feature_importances_path(
    results_dir: Union[str, Path],
    dataset: str,
    label: str,
    feature_battery: str,
) -> Path:
    """Path for per-experiment feature-importance CSV under a results directory."""
    results_dir = Path(results_dir)
    safe_dataset = _sanitize_filename_component(dataset)
    safe_label = _sanitize_filename_component(label)
    safe_battery = _sanitize_filename_component(feature_battery)
    return results_dir / "feature_importances" / f"{safe_dataset}__{safe_label}__{safe_battery}.csv"


def extract_feature_importances(
    model: Any,
    feature_cols: List[str],
    model_family: str = "xgboost",
) -> pd.DataFrame:
    """Extract feature importances from a trained XGBoost or CatBoost model."""
    n_features = len(feature_cols)
    importances = np.zeros(n_features, dtype=float)

    if model_family == "xgboost":
        if hasattr(model, "get_score") and not hasattr(model, "fit"):
            scores = model.get_score(importance_type="gain")
            for key, value in scores.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if 0 <= idx < n_features:
                        importances[idx] = float(value)
        elif hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "get_booster"):
            scores = model.get_booster().get_score(importance_type="gain")
            for key, value in scores.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if 0 <= idx < n_features:
                        importances[idx] = float(value)
    elif model_family == "catboost":
        if hasattr(model, "get_feature_importance"):
            importances = np.asarray(model.get_feature_importance(), dtype=float)
        elif hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    if len(importances) != n_features:
        raise ValueError(
            f"Importance length {len(importances)} != n_features {n_features}"
        )

    imp_df = pd.DataFrame({"feature": list(feature_cols), "importance": importances})
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importances(
    model: Any,
    feature_cols: List[str],
    results_dir: Union[str, Path],
    dataset: str,
    label: str,
    feature_battery: str,
    model_family: str = "xgboost",
    timestamp: str | None = None,
) -> Path:
    """Save feature importances for one experiment to CSV."""
    from datetime import datetime

    output_path = feature_importances_path(results_dir, dataset, label, feature_battery)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imp_df = extract_feature_importances(model, feature_cols, model_family=model_family)
    imp_df.insert(0, "dataset", dataset)
    imp_df.insert(1, "label", label)
    imp_df.insert(2, "feature_battery", feature_battery)
    imp_df["timestamp"] = timestamp or datetime.now().isoformat()
    imp_df.to_csv(output_path, index=False)
    return output_path


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

