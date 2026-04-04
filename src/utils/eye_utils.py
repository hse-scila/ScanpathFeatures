from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _get_id(elements: Iterable[Any]) -> str:
    """Stable mapping between pk/group key elements and a single string id."""

    return "_".join(str(e) for e in elements)


def _get_objs(id_: str) -> list[str]:
    """Inverse of `_get_id` (splits by underscore)."""

    # In this benchmark code, group keys are encoded using underscore-separated strings.
    return id_.split("_")


def _split_dataframe(
    df: pd.DataFrame, pk: list[str], encode: bool = True
) -> list[tuple[str, pd.DataFrame]] | list[tuple[tuple[Any, ...] | Any, pd.DataFrame]]:
    """
    Split dataframe by primary-key columns.

    When `encode=True` (default), group keys are encoded into a string id using `_get_id`.
    """

    assert set(pk).issubset(set(df.columns)), "Some key columns in df are missing"

    grouped = list(df.groupby(by=pk, sort=False))
    if not encode:
        return grouped

    # Note: pandas returns tuple keys even for a single grouping column when `by=[col]`.
    return [(_get_id(grouped[i][0]), grouped[i][1]) for i in range(len(grouped))]

