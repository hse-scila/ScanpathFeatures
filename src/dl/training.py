"""Timeseries LSTM training pipeline (3 layers, 128 hidden units)."""

from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dl.datasets import DatasetTimeSeries
from src.dl.models import Classifier, DictWrapperRNN, Regressor, SimpleRNN
from src.utils.dataset_utils import find_all_datasets, load_and_preprocess_dataset
from src.utils.eye_utils import _split_dataframe
from src.utils.feature_extraction_utils import get_split_info_paths_for_dataset
from src.utils.split_utils import load_split_info
from src.utils.training import (
    compute_metrics,
    get_task_type_for_dataset_label,
    save_results_incremental,
)

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 3
DEFAULT_TIMESERIES_FEATURES = ["duration"]
DEFAULT_MAX_LENGTH = 300
DEFAULT_LEARNING_RATE = 1e-3
FEATURE_BATTERY = "timeseries_lstm"
SKIP_DATASET_SUBSTRINGS = ("label_Anger",)


def _keep_dataset(dataset_name: str) -> bool:
    if dataset_name.endswith("_gaze") or dataset_name.endswith("_gazes"):
        return False
    if dataset_name.startswith("Cognitive_load_ready_data_gazes_") or dataset_name.startswith(
        "Emotions_ready_data_gazes_"
    ):
        return "_0.02" in dataset_name
    return True


def discover_fixation_datasets(
    datasets_dir: Path,
    splits_dir: Path,
) -> list[Path]:
    """Return fixation CSV paths that have split info in splits_dir."""
    all_datasets = find_all_datasets(datasets_dir)
    candidates = all_datasets.get("fixation", []) + all_datasets.get("saccade", [])

    dataset_paths: list[Path] = []
    for path in candidates:
        if not _keep_dataset(path.stem):
            continue
        if get_split_info_paths_for_dataset(splits_dir, path.stem):
            dataset_paths.append(path)

    return sorted(dataset_paths, key=lambda p: p.stem)


def prepare_labels(
    df: pd.DataFrame, col_info: dict, pk: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    label_cols = col_info.get("label_cols", [])
    if not label_cols:
        raise ValueError("label_cols must be provided")
    if not pk:
        raise ValueError("pk (group_cols) must be provided")

    label_col = label_cols[0]
    cols_to_select = pk.copy()
    if label_col not in cols_to_select:
        cols_to_select.append(label_col)

    Y = df[cols_to_select].drop_duplicates()
    if Y.columns.tolist().count(label_col) > 1:
        Y = Y.loc[:, ~Y.columns.duplicated()].copy()
    if label_col != "label":
        Y = Y.rename(columns={label_col: "label"})

    if isinstance(Y["label"], pd.DataFrame):
        label_series = Y["label"].iloc[:, 0]
        Y = Y.drop(columns="label")
        Y["label"] = label_series

    for col in pk:
        if col not in Y.columns:
            raise ValueError(f"Primary key column '{col}' is missing from Y")

    if Y["label"].dtype == "object" or Y["label"].dtype.name == "object":
        unique_labels = sorted(Y["label"].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        Y["label"] = Y["label"].map(label_to_id).astype(int)
    elif not pd.api.types.is_numeric_dtype(Y["label"]):
        try:
            Y["label"] = pd.to_numeric(Y["label"], errors="coerce").astype(int)
        except Exception:
            unique_labels = sorted(Y["label"].unique())
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            Y["label"] = Y["label"].map(label_to_id).astype(int)
    else:
        Y["label"] = Y["label"].astype(int)

    return Y, pk


def get_split_indices(
    df: pd.DataFrame, pk_cols: list[str], split_info: dict[str, Any]
) -> tuple[list[int], list[int], list[int]]:
    train_indexes = set(split_info.get("train", split_info.get("train_indexes", [])))
    val_indexes = set(split_info.get("val", split_info.get("val_indexes", [])))
    test_indexes = set(split_info.get("test", split_info.get("test_indexes", [])))

    groups = list(_split_dataframe(df, pk_cols, encode=True))
    group_to_idx = {}
    for idx, (group_id, _) in enumerate(groups):
        composite_pk = (
            group_id
            if isinstance(group_id, str)
            else "_".join(str(g) for g in group_id)
        )
        group_to_idx[composite_pk] = idx

    train_indices = [group_to_idx[pk] for pk in train_indexes if pk in group_to_idx]
    val_indices = [group_to_idx[pk] for pk in val_indexes if pk in group_to_idx]
    test_indices = [group_to_idx[pk] for pk in test_indexes if pk in group_to_idx]
    return train_indices, val_indices, test_indices


def create_timeseries_dataset(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    x_col: str,
    y_col: str,
    pk: list[str],
    features: list[str] | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> DatasetTimeSeries:
    if features is not None:
        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) != len(features):
            missing = set(features) - set(valid_features)
            warnings.warn(
                f"Features {missing} not found in DataFrame, using only {valid_features or 'coordinates'}"
            )
        features = valid_features if valid_features else None

    return DatasetTimeSeries(
        df, Y, x=x_col, y=y_col, pk=pk, features=features, max_length=max_length
    )


def create_lstm_model(
    train_dataset,
    task_type: str,
    n_classes: int | None = None,
    hidden_size: int = LSTM_HIDDEN_SIZE,
    num_layers: int = LSTM_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> pl.LightningModule:
    sample = train_dataset[0]
    sample_x = sample if isinstance(sample, dict) else sample[0]
    sequences = sample_x["sequences"]
    input_size = sequences.shape[-1] if len(sequences.shape) > 1 else 1

    rnn = SimpleRNN(
        "LSTM",
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    backbone = DictWrapperRNN(rnn)

    if task_type == "classification":
        return Classifier(backbone, n_classes=n_classes, learning_rate=learning_rate)
    return Regressor(backbone, output_dim=1, learning_rate=learning_rate)


def get_collate_fn(dataset):
    if hasattr(dataset, "collate_fn"):
        return dataset.collate_fn
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "collate_fn"):
        return dataset.dataset.collate_fn
    return None


def get_all_labels_from_dataset(dataset) -> list[int]:
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["y"] if isinstance(sample, dict) else sample[1]
        labels.append(int(label.item()) if torch.is_tensor(label) else int(label))
    return labels


def create_collate_with_label_remap(
    base_collate_fn,
    label_encoder: LabelEncoder | None,
    task_type: str = "classification",
):
    def collate_with_label_remap(batch):
        if base_collate_fn is not None:
            result = base_collate_fn(batch)
        else:
            result = {
                "sequences": torch.stack([x["sequences"] for x in batch]),
                "y": torch.tensor([x["y"] for x in batch]),
            }

        for key in ["sequences"]:
            if key in result and result[key] is not None and result[key].dtype == torch.float64:
                result[key] = result[key].float()

        if label_encoder is not None and "y" in result:
            y = result["y"]
            if torch.is_tensor(y):
                y_np = y.cpu().numpy()
                y_remapped = label_encoder.transform(y_np)
                result["y"] = torch.tensor(y_remapped, dtype=torch.long)
        elif task_type == "regression" and "y" in result:
            if result["y"].dtype in (torch.long, torch.int, torch.int32, torch.int64):
                result["y"] = result["y"].float()

        return result

    return collate_with_label_remap


class EpochTqdmCallback(pl.Callback):
    def __init__(self):
        self._pbar: tqdm | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._pbar = tqdm(total=trainer.max_epochs, desc="Epoch", unit="epoch")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._pbar is not None:
            self._pbar.update(1)
            if trainer.current_epoch + 1 < trainer.max_epochs:
                self._pbar.set_postfix({"epoch": trainer.current_epoch + 1})

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def train_model(
    train_dataset,
    val_dataset,
    test_dataset,
    task_type: str,
    n_classes: int | None = None,
    max_epochs: int = 50,
    batch_size: int = 32,
    label_encoder: LabelEncoder | None = None,
    hidden_size: int = LSTM_HIDDEN_SIZE,
    num_layers: int = LSTM_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, Any]:
    model = create_lstm_model(
        train_dataset,
        task_type,
        n_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
    )

    base_collate_fn = get_collate_fn(train_dataset)
    collate_fn = create_collate_with_label_remap(
        base_collate_fn, label_encoder, task_type=task_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        callbacks=[EpochTqdmCallback()],
    )
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    y_true_train, y_pred_train = [], []
    y_true_val, y_pred_val = [], []
    y_true_test, y_pred_test = [], []

    with torch.no_grad():
        for loader, y_true_list, y_pred_list in [
            (train_loader, y_true_train, y_pred_train),
            (val_loader, y_true_val, y_pred_val),
            (test_loader, y_true_test, y_pred_test),
        ]:
            for batch in loader:
                y = batch.pop("y")
                pred = model(batch)
                if task_type == "classification":
                    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
                y_true_list.extend(y.cpu().numpy())
                y_pred_list.extend(pred.cpu().numpy().flatten())

    return {
        "task_type": task_type,
        "n_classes": n_classes,
        "lstm_hidden_size": hidden_size,
        "lstm_num_layers": num_layers,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_metrics": compute_metrics(y_true_train, y_pred_train, task_type),
        "val_metrics": compute_metrics(y_true_val, y_pred_val, task_type),
        "test_metrics": compute_metrics(y_true_test, y_pred_test, task_type),
    }


def process_dataset_label(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    x_col: str,
    y_col: str,
    pk: list[str],
    split_info: dict[str, Any],
    max_epochs: int = 50,
    batch_size: int = 32,
    timeseries_features: list[str] | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_name: str | None = None,
    dataset_name: str | None = None,
    hidden_size: int = LSTM_HIDDEN_SIZE,
    num_layers: int = LSTM_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, Any] | None:
    try:
        full_dataset = create_timeseries_dataset(
            df, Y, x_col, y_col, pk, timeseries_features, max_length
        )

        train_indices, val_indices, test_indices = get_split_indices(df, pk, split_info)
        if not train_indices or not val_indices or not test_indices:
            print("  Empty split indices, skipping")
            return None

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        train_labels = get_all_labels_from_dataset(train_dataset)
        val_labels = get_all_labels_from_dataset(val_dataset)
        test_labels = get_all_labels_from_dataset(test_dataset)
        all_labels = train_labels + val_labels + test_labels
        all_labels_series = pd.Series(all_labels)

        if dataset_name is not None and label_name is not None:
            task_type = get_task_type_for_dataset_label(
                dataset_name, label_name, all_labels_series
            )
        else:
            from src.utils.training import get_task_type

            task_type = get_task_type(all_labels_series)

        label_encoder = None
        n_classes = None
        if task_type == "classification":
            unique_labels = sorted(set(all_labels))
            n_classes = len(unique_labels)
            if n_classes < 2:
                print("  Less than 2 classes, skipping")
                return None
            label_encoder = LabelEncoder()
            label_encoder.fit(unique_labels)

        results = train_model(
            train_dataset,
            val_dataset,
            test_dataset,
            task_type,
            n_classes,
            max_epochs=max_epochs,
            batch_size=batch_size,
            label_encoder=label_encoder,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
        )

        del full_dataset, train_dataset, val_dataset, test_dataset
        gc.collect()
        return results

    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def _result_to_row(
    dataset_name: str,
    label_col: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "dataset": dataset_name,
        "label": label_col,
    }
    test_metrics = result.get("test_metrics", {})
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            row[f"test_{k}"] = v
    return row


def run_training_battery(
    splits_dir: str | Path,
    results_file: str | Path,
    *,
    datasets_dir: Path | None = None,
    max_epochs: int = 50,
    batch_size: int = 32,
    timeseries_features: list[str] | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    skip_existing: bool = True,
    test_mode: bool = False,
    test_max_samples: int = 100,
    hidden_size: int = LSTM_HIDDEN_SIZE,
    num_layers: int = LSTM_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> pd.DataFrame:
    splits_dir = Path(splits_dir)
    results_file = Path(results_file)
    datasets_dir = Path(datasets_dir) if datasets_dir is not None else Path("data")
    if timeseries_features is None:
        timeseries_features = DEFAULT_TIMESERIES_FEATURES

    results_file.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[tuple[str, str]] = set()
    if results_file.exists() and skip_existing:
        existing_df = pd.read_csv(results_file)
        if all(col in existing_df.columns for col in ["dataset", "label"]):
            existing_keys = set(
                zip(existing_df["dataset"], existing_df["label"], strict=False)
            )

    dataset_paths = discover_fixation_datasets(datasets_dir, splits_dir)
    print(f"Found {len(dataset_paths)} datasets with splits to process")

    results: list[dict[str, Any]] = []

    for dataset_path in tqdm(dataset_paths, desc="Datasets"):
        dataset_name = dataset_path.stem
        split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
        if not split_paths:
            continue

        try:
            df, col_info, _ = load_and_preprocess_dataset(dataset_path)
        except FileNotFoundError:
            print(f"\nFailed to load {dataset_name}, skipping...")
            continue

        if df is None or len(df) == 0:
            continue

        pk = col_info.get("group_cols", [])
        x_col, y_col = col_info["x_col"], col_info["y_col"]
        if not x_col or not y_col:
            print(f"\nMissing coordinates for {dataset_name}, skipping...")
            continue

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name} ({len(df):,} rows, {len(split_paths)} split(s))")
        print(f"{'=' * 60}")

        for split_path in split_paths:
            split_id = split_path.stem.replace("_split_info", "")
            if any(skip in split_id for skip in SKIP_DATASET_SUBSTRINGS):
                continue

            split_info = load_split_info(split_path)

            label_col = split_info.get("label_col")
            if not label_col and len(split_id) > len(dataset_name) + 1:
                label_col = split_id[len(dataset_name) + 1 :]
            if not label_col:
                print(f"\nCould not infer label for {split_id}, skipping...")
                continue

            if test_mode:
                for split_key in ["train", "val", "test"]:
                    if (
                        split_key in split_info
                        and len(split_info[split_key]) > test_max_samples
                    ):
                        split_info[split_key] = split_info[split_key][:test_max_samples]
                all_split_samples = set(
                    split_info.get("train", [])
                    + split_info.get("val", [])
                    + split_info.get("test", [])
                )
                if len(pk) == 1:
                    df_composite_idx = df[pk[0]].astype(str)
                else:
                    df_composite_idx = df[pk].astype(str).agg("_".join, axis=1)
                df_work = df[df_composite_idx.isin(all_split_samples)].copy()
            else:
                df_work = df.copy()

            pk_work = pk.copy()
            if label_col in pk_work:
                label_col_copy = f"{label_col}_label"
                if label_col_copy not in df_work.columns:
                    df_work[label_col_copy] = df_work[label_col].copy()
                label_col_to_use = label_col_copy
            else:
                label_col_to_use = label_col

            col_info_single = col_info.copy()
            col_info_single["label_cols"] = [label_col_to_use]

            try:
                Y, pk_work = prepare_labels(df_work, col_info_single, pk_work)
            except Exception as e:
                print(f"  Can't prepare labels for '{label_col}': {e}")
                continue

            if Y["label"].nunique() < 2:
                print(f"  Constant label '{label_col}', skipping...")
                continue

            key = (dataset_name, label_col)
            if key in existing_keys:
                print(f"Skipping {label_col} (already in results)")
                continue

            print(f"\n  Label: {label_col}")
            print("    Training timeseries LSTM...", end=" ")

            result = process_dataset_label(
                df_work,
                Y,
                x_col,
                y_col,
                pk_work,
                split_info,
                max_epochs=max_epochs,
                batch_size=batch_size,
                timeseries_features=timeseries_features,
                max_length=max_length,
                label_name=label_col,
                dataset_name=dataset_name,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
            )

            if result and "error" not in result:
                result_dict = _result_to_row(dataset_name, label_col, result)
                results.append(result_dict)
                save_results_incremental(result_dict, results_file)
                existing_keys.add(key)

                if result.get("task_type") == "classification":
                    acc = result_dict.get("test_accuracy", 0)
                    f1 = result_dict.get("test_f1", 0)
                    print(f"acc={acc:.4f}, f1_macro={f1:.4f}")
                else:
                    r2 = result_dict.get("test_r2", 0)
                    print(f"r2={r2:.4f}")
            elif result and "error" in result:
                print(f"error: {result['error'][:50]}")
                error_row = {
                    "dataset": dataset_name,
                    "label": label_col,
                    "error": result["error"],
                }
                results.append(error_row)
                save_results_incremental(error_row, results_file)
            else:
                print("skipped")

            del df_work, Y
            gc.collect()

        del df, col_info
        gc.collect()

    if results:
        print(f"\nSaved {len(results)} results to {results_file}")
    elif results_file.exists():
        return pd.read_csv(results_file)
    return pd.DataFrame()
