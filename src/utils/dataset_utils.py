"""
Helper functions for loading and preprocessing datasets according to README_data.md rules.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def classify_dataset(filename: Path) -> str:
    """Classify dataset based on filename rules.
    
    Default: If no clear classification, assume it's fixations.
    """
    name = filename.stem
    
    # Skip datasets ending with _skip
    if name.endswith('_skip'):
        return 'skip'
    
    # Gaze datasets (need fixation extraction)
    if name.endswith('_gaze') or name.endswith('_gazes'):
        return 'gaze'
    
    # Fixation datasets (already have fixations)
    if name.endswith('_fixations') or name.endswith('_fixation'):
        return 'fixation'
    
    # Saccade datasets
    if name.endswith('_saccades') or name.endswith('_saccade'):
        return 'saccade'
    
    # Default: assume fixations if no clear classification
    return 'fixation'


def find_all_datasets(directory: Path) -> Dict[str, List[Path]]:
    """Find all datasets in a directory, categorized by type.
    
    Args:
        directory: Directory with all dataset CSV files
    """
    csv_files = [f for f in directory.iterdir() if f.suffix == '.csv' and f.name != 'Datasets_description.csv']
    
    datasets = {
        'skip': [],
        'gaze': [],
        'fixation': [],
        'saccade': [],
        'unknown': []
    }
    
    for csv_file in csv_files:
        dataset_type = classify_dataset(csv_file)
        datasets[dataset_type].append(csv_file)
    
    # Sort each category
    for key in datasets:
        datasets[key].sort()
    
    return datasets


def identify_columns(df: pd.DataFrame) -> Dict:
    """Identify column types in the dataset."""
    group_cols = [col for col in df.columns if col.startswith('group_')]
    meta_cols = [col for col in df.columns if col.startswith('meta_')]
    label_cols = [col for col in df.columns if col.endswith('_label')]
    
    # Find coordinate columns
    has_norm_pos_x = 'norm_pos_x' in df.columns
    has_norm_pos_y = 'norm_pos_y' in df.columns
    has_x = 'x' in df.columns
    has_y = 'y' in df.columns
    has_x_left = 'x_left' in df.columns
    has_y_left = 'y_left' in df.columns
    has_x_right = 'x_right' in df.columns
    has_y_right = 'y_right' in df.columns
    
    # Find timestamp columns
    has_timestamp = 'timestamp' in df.columns
    has_timestamp_start = 'timestamp_start' in df.columns
    has_timestamp_end = 'timestamp_end' in df.columns
    
    # Determine which columns to use
    if has_norm_pos_x and has_norm_pos_y:
        x_col = 'norm_pos_x'
        y_col = 'norm_pos_y'
        needs_normalization = False
    elif has_x and has_y:
        x_col = 'x'
        y_col = 'y'
        needs_normalization = True
    elif has_x_left and has_y_left and has_x_right and has_y_right:
        # Calculate mean of left and right
        df['x'] = (df['x_left'] + df['x_right']) / 2
        df['y'] = (df['y_left'] + df['y_right']) / 2
        x_col = 'x'
        y_col = 'y'
        needs_normalization = True
    else:
        x_col = None
        y_col = None
        needs_normalization = False
    
    # Use timestamp if available, otherwise timestamp_start
    if has_timestamp:
        t_col = 'timestamp'
    elif has_timestamp_start:
        t_col = 'timestamp_start'
    else:
        t_col = None
    
    return {
        'group_cols': group_cols,
        'meta_cols': meta_cols,
        'label_cols': label_cols,
        'x_col': x_col,
        'y_col': y_col,
        't_col': t_col,
        'has_norm_pos': has_norm_pos_x and has_norm_pos_y,
        'needs_normalization': needs_normalization,
        'has_duration': 'duration' in df.columns or (has_timestamp_start and has_timestamp_end),
        'has_dispersion': 'dispersion' in df.columns or 'diameters' in df.columns
    }


def normalize_coordinates(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Normalize coordinates by dividing by max(x) and max(y)."""
    df = df.copy()
    max_x = df[x_col].max()
    max_y = df[y_col].max()
    
    if max_x > 0:
        df['norm_pos_x'] = df[x_col] / max_x
    else:
        df['norm_pos_x'] = df[x_col]
    
    if max_y > 0:
        df['norm_pos_y'] = df[y_col] / max_y
    else:
        df['norm_pos_y'] = df[y_col]
    
    return df


def convert_coordinates_to_numeric(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Convert coordinate columns to numeric format (handles comma-separated decimals)."""
    df = df.copy()
    
    if x_col in df.columns:
        df[x_col] = pd.to_numeric(df[x_col].astype(str).str.replace(',', '.'), errors='coerce')
    if y_col in df.columns:
        df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(',', '.'), errors='coerce')
    
    return df


def load_and_preprocess_dataset(
    file_path: Path,
) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Load and preprocess a dataset according to README_data.md rules.
    
    Args:
        file_path: Path to dataset CSV file
    
    Returns:
        (dataframe, column_info, dataset_type)
    """
    dataset_type = classify_dataset(file_path)
    
    # Read dataset
    df = pd.read_csv(file_path)
    
    # Convert coordinates to numeric if needed
    if 'x' in df.columns or 'norm_pos_x' in df.columns:
        x_col = 'norm_pos_x' if 'norm_pos_x' in df.columns else 'x'
        y_col = 'norm_pos_y' if 'norm_pos_y' in df.columns else 'y'
        df = convert_coordinates_to_numeric(df, x_col, y_col)
    
    # Identify columns
    col_info = identify_columns(df)
    
    # Normalize coordinates if needed
    if col_info['needs_normalization'] and col_info['x_col'] and col_info['y_col']:
        df = normalize_coordinates(df, col_info['x_col'], col_info['y_col'])
        col_info['x_col'] = 'norm_pos_x'
        col_info['y_col'] = 'norm_pos_y'
        col_info['has_norm_pos'] = True
        col_info['needs_normalization'] = False
    
    # Compute duration from timestamp_start and timestamp_end if duration is missing
    if 'duration' not in df.columns:
        if 'timestamp_start' in df.columns and 'timestamp_end' in df.columns:
            df['duration'] = df['timestamp_end'] - df['timestamp_start']
            col_info['has_duration'] = True
            print(f"Computed duration from timestamp_start and timestamp_end")
        elif 'start_time' in df.columns and 'end_time' in df.columns:
            df['duration'] = df['end_time'] - df['start_time']
            col_info['has_duration'] = True
            print(f"Computed duration from start_time and end_time")
    
    # Ensure we have required columns for feature extraction
    if col_info['x_col'] is None or col_info['y_col'] is None:
        print(f"Warning: Missing coordinate columns in {file_path.name}")
        return None, col_info, dataset_type
    
    return df, col_info, dataset_type
