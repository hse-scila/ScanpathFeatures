"""
Split utilities for consistent train/val/test splitting across feature batteries.

This module provides functions for:
- Loading and saving split info
- Applying splits to features and labels
- Creating composite indexes for consistent splitting
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split

# Default split configuration
SPLIT_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.2,  # 25% of remaining after test split
    'random_state': 42
}


def load_split_info(split_info_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load split info from JSON file.
    
    Args:
        split_info_path: Path to the split info JSON file
    
    Returns:
        Dictionary containing split information
    """
    with open(split_info_path, 'r') as f:
        return json.load(f)


def save_split_info(split_info: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save split info to JSON file.
    
    Args:
        split_info: Dictionary containing split information
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=2, default=str)


def create_composite_index(df: pd.DataFrame, pk_cols: List[str]) -> pd.Series:
    """
    Create a composite index from multiple pk columns.
    
    Joins pk column values with '_' separator, matching the format used
    when saving DataFrames with MultiIndex.
    
    Args:
        df: DataFrame with pk columns
        pk_cols: List of column names to join
    
    Returns:
        Series with composite index strings
    """
    if len(pk_cols) == 1:
        return df[pk_cols[0]].astype(str)
    else:
        return df[pk_cols].astype(str).agg('_'.join, axis=1)


def apply_split_to_features(
    features_df: pd.DataFrame,
    split_info: Dict[str, Any],
    index_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply pre-defined split to features DataFrame using split info.
    
    This function uses the 'train', 'val', 'test' fields from split_info,
    which contain composite index strings (pk values joined by '_').
    
    Args:
        features_df: Features DataFrame (should have composite index as index or in a column)
        split_info: Split info dictionary containing 'train', 'val', 'test' lists
        index_column: Name of the column containing the index values. If None, will try:
                     1. DataFrame index
                     2. 'index' column if present
                     3. 'Unnamed: 0' column (common when CSV saved with index=True)
    
    Returns:
        (train_features, val_features, test_features)
    """
    # Get indexes from split_info (support both old and new key names)
    if 'train' in split_info:
        train_indexes = set(split_info['train'])
        val_indexes = set(split_info['val'])
        test_indexes = set(split_info['test'])
    elif 'train_indexes' in split_info:
        # Backward compatibility with old format
        train_indexes = set(split_info['train_indexes'])
        val_indexes = set(split_info['val_indexes'])
        test_indexes = set(split_info['test_indexes'])
    else:
        raise ValueError(
            "split_info does not contain 'train' or 'train_indexes'. "
            "Use apply_split_from_info for subject-based splitting, "
            "or regenerate split_info with index-based format."
        )
    
    # Determine which column/index to use for matching
    if index_column is not None:
        if index_column not in features_df.columns:
            raise ValueError(f"Index column '{index_column}' not found in dataframe")
        match_values = features_df[index_column].astype(str)
    elif 'index' in features_df.columns:
        # Composite pk index column (preferred)
        match_values = features_df['index'].astype(str)
    elif 'Unnamed: 0' in features_df.columns:
        # CSV was saved with index=True, creating 'Unnamed: 0' column
        match_values = features_df['Unnamed: 0'].astype(str)
    else:
        # Use the DataFrame's actual index
        match_values = features_df.index.astype(str)
    
    # Create masks for each split
    train_mask = match_values.isin(train_indexes)
    val_mask = match_values.isin(val_indexes)
    test_mask = match_values.isin(test_indexes)
    
    train_features = features_df[train_mask].copy()
    val_features = features_df[val_mask].copy()
    test_features = features_df[test_mask].copy()
    
    # Validate that we found some matches
    n_matched = train_mask.sum() + val_mask.sum() + test_mask.sum()
    n_total = len(features_df)
    if n_matched == 0:
        raise ValueError(
            f"No rows matched the split indexes. "
            f"Check that the index format matches between split_info and the dataframe. "
            f"Sample split indexes: {list(train_indexes)[:3]}, "
            f"Sample dataframe values: {match_values.head(3).tolist()}"
        )
    
    if n_matched < n_total:
        print(f"Warning: {n_total - n_matched} rows did not match any split index")
    
    return train_features, val_features, test_features


def apply_split_to_labels(
    labels_df: pd.DataFrame,
    split_info: Dict[str, Any],
    index_column: str = 'index'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply pre-defined split to labels DataFrame using split info.
    
    Args:
        labels_df: Labels DataFrame (should have 'index' column with composite index)
        split_info: Split info dictionary containing 'train', 'val', 'test' lists
        index_column: Name of the column containing the index values (default: 'index')
    
    Returns:
        (train_labels, val_labels, test_labels)
    """
    if index_column not in labels_df.columns:
        raise ValueError(
            f"Index column '{index_column}' not found in labels dataframe. "
            f"Available columns: {list(labels_df.columns)}"
        )
    
    # Get indexes from split_info
    if 'train' in split_info:
        train_indexes = set(split_info['train'])
        val_indexes = set(split_info['val'])
        test_indexes = set(split_info['test'])
    elif 'train_indexes' in split_info:
        train_indexes = set(split_info['train_indexes'])
        val_indexes = set(split_info['val_indexes'])
        test_indexes = set(split_info['test_indexes'])
    else:
        raise ValueError(
            "split_info does not contain 'train' or 'train_indexes'. "
            "Regenerate split_info with index-based format."
        )
    
    # Create masks
    match_values = labels_df[index_column].astype(str)
    train_mask = match_values.isin(train_indexes)
    val_mask = match_values.isin(val_indexes)
    test_mask = match_values.isin(test_indexes)
    
    train_labels = labels_df[train_mask].copy()
    val_labels = labels_df[val_mask].copy()
    test_labels = labels_df[test_mask].copy()
    
    return train_labels, val_labels, test_labels


def get_split_pk_for_dataset(
    dataset_name: str,
    col_info: Dict[str, Any],
    default_split_pk: Optional[str] = None
) -> str:
    """
    Determine the split_pk (subject column) for a dataset.
    
    This function tries to identify the subject/participant column that should
    be used for splitting. Common names include:
    - group_subject
    - group_participant
    - group_Participant (capital P)
    
    Args:
        dataset_name: Name of the dataset
        col_info: Column info dictionary from load_and_preprocess_dataset
        default_split_pk: Default column name to use if not found (optional)
    
    Returns:
        Name of the split_pk column
    """
    group_cols = col_info.get('group_cols', [])
    
    # Common subject column names (in order of preference)
    subject_names = [
        'group_subject',
        'group_participant',
        'group_Participant',  # Capital P variant
        'subject',
        'participant'
    ]
    
    # Check if any subject column exists in group_cols
    for name in subject_names:
        if name in group_cols:
            return name
    
    # If no subject column found, use first group column or default
    if group_cols:
        if default_split_pk:
            print(f"Warning: Using default split_pk '{default_split_pk}' for {dataset_name}")
            return default_split_pk
        else:
            print(f"Warning: No subject column found for {dataset_name}, using first group column: {group_cols[0]}")
            return group_cols[0]
    else:
        raise ValueError(
            f"No group columns found for {dataset_name}. "
            f"Cannot determine split_pk for subject-based splitting."
        )
