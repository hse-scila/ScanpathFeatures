"""Time-series dataset for fixation sequences."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.eye_utils import _split_dataframe


def _get_features(
    X: pd.DataFrame,
    features: list[str] | None,
    x: str,
    y: str,
    pk: list[str],
) -> list[np.ndarray]:
    columns = [x, y] + (features or [])
    output: list[np.ndarray] = []
    for _, group_X in tqdm(_split_dataframe(X, pk), desc="Building sequences"):
        output.append(group_X[columns].values)
    return output


class DatasetTimeSeries(Dataset):
    """Custom dataset for time-series fixation data."""

    def __init__(
        self,
        X: pd.DataFrame,
        Y: ArrayLike,
        x: str,
        y: str,
        pk: list[str],
        features: list[str] | None = None,
        transforms: Callable | None = None,
        max_length: int = 300,
    ):
        self.pmk = pk
        self.X = _get_features(X, features, x, y, pk)
        if not isinstance(Y, pd.Series):
            Y = Y.set_index(pk).squeeze(axis=0)
        self.Y = Y.sort_index().values
        if np.issubdtype(self.Y.dtype, np.integer):
            self.Y = torch.tensor(self.Y, dtype=torch.long)
        else:
            self.Y = torch.tensor(self.Y, dtype=torch.float)

        self.n_features = 2 + (len(features) if features is not None else 0)
        self.transforms = transforms
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        X = self.X[idx]
        label = self.Y[idx]

        if self.transforms:
            X = self.transforms(X)

        return {
            "sequences": torch.tensor(X, dtype=torch.float),
            "y": label,
        }

    def collate_fn(self, batch):
        max_len = self.max_length
        lengths = [min(x["sequences"].shape[0], max_len) for x in batch]
        padded_batch = [
            torch.cat(
                [
                    x["sequences"][: self.max_length],
                    torch.zeros(
                        max_len - x["sequences"][: self.max_length].shape[0],
                        self.n_features,
                    ),
                ],
                axis=0,
            )
            for x in batch
        ]

        y = torch.tensor([x["y"] for x in batch])
        return {
            "sequences": torch.stack(padded_batch),
            "lengths": torch.tensor(lengths),
            "y": y,
        }
