"""
Common utilities for feature extraction notebooks.

This module provides shared functions for:
- Setting up paths and discovering datasets
- Preparing features DataFrames for splitting
- Applying splits and saving results
- Generating summary reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

from src.utils.dataset_utils import find_all_datasets, load_and_preprocess_dataset
from src.utils.split_utils import (
    load_split_info,
    apply_split_to_features,
    apply_split_to_labels
)


def setup_paths(
    datasets_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    splits_dir: Optional[str] = None
) -> Dict[str, Path]:
    """
    Set up common paths for feature extraction notebooks.
    
    Args:
        datasets_dir: Path to datasets directory (default: local 'data' directory)
        output_dir: Path to output directory (default: 'extensive_features')
        splits_dir: Path to splits directory (default: output_dir / 'splits')
    
    Returns:
        Dictionary with 'datasets_dir', 'output_dir', 'splits_dir' Path objects
    """
    if datasets_dir is None:
        datasets_dir = Path('data')
    else:
        datasets_dir = Path(datasets_dir)
    
    if output_dir is None:
        output_dir = Path('extensive_features')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    if splits_dir is None:
        splits_dir = output_dir / 'splits'
    else:
        splits_dir = Path(splits_dir)
    
    splits_dir.mkdir(exist_ok=True)
    
    return {
        'datasets_dir': datasets_dir,
        'output_dir': output_dir,
        'splits_dir': splits_dir
    }


def get_split_info_paths_for_dataset(splits_dir: Path, dataset_name: str) -> List[Path]:
    """
    Return list of split info JSON paths for a dataset.
    Uses single file {dataset}_split_info.json if present, otherwise all
    label-based files {dataset}_*_split_info.json from the alternative splits notebook.
    """
    exact = splits_dir / f"{dataset_name}_split_info.json"
    if exact.exists():
        return [exact]
    return sorted(splits_dir.glob(f"{dataset_name}_*_split_info.json"))


def apply_splits_and_save(
    features_df: pd.DataFrame,
    dataset_name: str,
    feature_type: str,
    col_info: Dict[str, Any],
    paths: Dict[str, Path],
    split_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply splits to features and save train/val/test files.
    When multiple label-based splits exist, saves one set of train/val/test per label.
    Also saves labels if available (per split_id when using label-based splits).
    
    Args:
        features_df: Features DataFrame
        dataset_name: Name of the dataset
        feature_type: Type of features (e.g., 'simple_features', 'complex_features')
        col_info: Column info dictionary
        paths: Dictionary with 'output_dir' and 'splits_dir' Path objects
        split_info: Split info dictionary (if None, will try to load from file(s))
    
    Returns:
        Dictionary with results including paths and counts
    """
    output_dir = paths['output_dir']
    splits_dir = paths['splits_dir']
    
    # Prepare features index
    features_df['index'] = features_df.index.astype(str)
    
    # Resolve which split(s) to use
    if split_info is not None:
        split_infos = [(None, split_info)]
    else:
        split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
        if not split_paths:
            print(f"⚠️  Warning: Split info not found for {dataset_name}")
            print(f"    Run notebook 0_create_splits.ipynb or 0_create_splits_alternative.ipynb first to create splits")
            output_path = output_dir / f"{dataset_name}_{feature_type}.csv"
            features_df.to_csv(output_path, index=True)
            print(f"✅ Saved features to {output_path}")
            return {
                'status': 'success',
                'num_scanpaths': len(features_df),
                'num_features': features_df.shape[1],
                'output_path': str(output_path),
                'note': 'No split info available'
            }
        split_infos = [(p, load_split_info(p)) for p in split_paths]
    
    features_df_indexed = features_df.set_index('index')
    labels_path = splits_dir / f"{dataset_name}_labels.csv"
    labels_df = pd.read_csv(labels_path) if labels_path.exists() else None
    
    split_results = []
    for idx, (split_path, si) in enumerate(split_infos):
        if split_path is not None:
            split_id = split_path.stem.replace('_split_info', '')
            if len(split_infos) > 1:
                print(f"  Split [{idx + 1}/{len(split_infos)}] {split_id} (split by: {si.get('split_pk', 'unknown')})")
            else:
                print(f"  Loaded split info (split by: {si.get('split_pk', 'unknown')})")
        else:
            split_id = dataset_name
        
        train_features, val_features, test_features = apply_split_to_features(
            features_df_indexed, si, index_column=None
        )
        print(f"  Split features: Train {len(train_features)}, Val {len(val_features)}, Test {len(test_features)}")
        
        train_output_path = output_dir / f"{split_id}_{feature_type}_train.csv"
        val_output_path = output_dir / f"{split_id}_{feature_type}_val.csv"
        test_output_path = output_dir / f"{split_id}_{feature_type}_test.csv"
        train_features.to_csv(train_output_path, index=True)
        val_features.to_csv(val_output_path, index=True)
        test_features.to_csv(test_output_path, index=True)
        print(f"  ✅ Saved: {train_output_path.name}, {val_output_path.name}, {test_output_path.name}")
        
        if labels_df is not None:
            train_labels, val_labels, test_labels = apply_split_to_labels(
                labels_df, si, index_column='index'
            )
            train_labels_path = splits_dir / f"{split_id}_labels_train.csv"
            val_labels_path = splits_dir / f"{split_id}_labels_val.csv"
            test_labels_path = splits_dir / f"{split_id}_labels_test.csv"
            train_labels.to_csv(train_labels_path, index=False)
            val_labels.to_csv(val_labels_path, index=False)
            test_labels.to_csv(test_labels_path, index=False)
            print(f"  ✅ Saved labels: {train_labels_path.name}, {val_labels_path.name}, {test_labels_path.name}")
        
        split_results.append({
            'split_id': split_id,
            'n_train_scanpaths': len(train_features),
            'n_val_scanpaths': len(val_features),
            'n_test_scanpaths': len(test_features),
            'num_features': train_features.shape[1],
            'train_output_path': str(train_output_path),
            'val_output_path': str(val_output_path),
            'test_output_path': str(test_output_path),
        })
    
    if labels_path and not labels_path.exists():
        print(f"⚠️  Labels file not found at {labels_path}")
    
    first = split_results[0]
    result = {
        'status': 'success',
        'n_train_scanpaths': first['n_train_scanpaths'],
        'n_val_scanpaths': first['n_val_scanpaths'],
        'n_test_scanpaths': first['n_test_scanpaths'],
        'num_features': first['num_features'],
        'train_output_path': first['train_output_path'],
        'val_output_path': first['val_output_path'],
        'test_output_path': first['test_output_path'],
    }
    if len(split_results) > 1:
        result['splits'] = split_results
    return result


def check_cache(
    dataset_name: str,
    feature_type: str,
    paths: Dict[str, Path]
) -> Optional[Dict[str, Any]]:
    """
    Check if features already exist in cache (for train/val/test splits).
    Supports both single split ({dataset_name}_split_info.json) and label-based splits
    (extensive_features: {dataset_name}_{label}_split_info.json -> files named {split_id}_{feature_type}_*.csv).
    
    Args:
        dataset_name: Name of the dataset
        feature_type: Type of features (e.g., 'simple_features', 'complex_features')
        paths: Dictionary with 'output_dir' and 'splits_dir' Path objects
    
    Returns:
        Dictionary with cached results if found, None otherwise
    """
    output_dir = paths['output_dir']
    splits_dir = paths['splits_dir']
    
    # Resolve split_id(s) the same way apply_splits_and_save does (label-based vs single split)
    split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
    if split_paths:
        split_ids = [p.stem.replace('_split_info', '') for p in split_paths]
        all_cached = True
        for split_id in split_ids:
            t = output_dir / f"{split_id}_{feature_type}_train.csv"
            v = output_dir / f"{split_id}_{feature_type}_val.csv"
            s = output_dir / f"{split_id}_{feature_type}_test.csv"
            if not (t.exists() and v.exists() and s.exists()):
                all_cached = False
                break
        if all_cached:
            # Use first split for counts/paths in return dict
            split_id = split_ids[0]
            train_output_path = output_dir / f"{split_id}_{feature_type}_train.csv"
            val_output_path = output_dir / f"{split_id}_{feature_type}_val.csv"
            test_output_path = output_dir / f"{split_id}_{feature_type}_test.csv"
            train_features = pd.read_csv(train_output_path, index_col=0)
            val_features = pd.read_csv(val_output_path, index_col=0)
            test_features = pd.read_csv(test_output_path, index_col=0)
            note = 'Loaded from cache' + (f' ({len(split_ids)} label splits)' if len(split_ids) > 1 else '')
            return {
                'status': 'cached',
                'n_train_scanpaths': len(train_features),
                'n_val_scanpaths': len(val_features),
                'n_test_scanpaths': len(test_features),
                'num_features': train_features.shape[1],
                'train_output_path': str(train_output_path),
                'val_output_path': str(val_output_path),
                'test_output_path': str(test_output_path),
                'note': note
            }
    
    # Fallback: single split naming {dataset_name}_{feature_type}_train/val/test.csv
    train_output_path = output_dir / f"{dataset_name}_{feature_type}_train.csv"
    val_output_path = output_dir / f"{dataset_name}_{feature_type}_val.csv"
    test_output_path = output_dir / f"{dataset_name}_{feature_type}_test.csv"
    if train_output_path.exists() and val_output_path.exists() and test_output_path.exists():
        train_features = pd.read_csv(train_output_path, index_col=0)
        val_features = pd.read_csv(val_output_path, index_col=0)
        test_features = pd.read_csv(test_output_path, index_col=0)
        return {
            'status': 'cached',
            'n_train_scanpaths': len(train_features),
            'n_val_scanpaths': len(val_features),
            'n_test_scanpaths': len(test_features),
            'num_features': train_features.shape[1],
            'train_output_path': str(train_output_path),
            'val_output_path': str(val_output_path),
            'test_output_path': str(test_output_path),
            'note': 'Loaded from cache'
        }
    
    # Check if full features file exists (no split info case)
    full_output_path = output_dir / f"{dataset_name}_{feature_type}.csv"
    if full_output_path.exists():
        features_df = pd.read_csv(full_output_path, index_col=0)
        return {
            'status': 'cached',
            'num_scanpaths': len(features_df),
            'num_features': features_df.shape[1],
            'output_path': str(full_output_path),
            'note': 'Loaded from cache (no split info)'
        }
    
    return None


def print_summary(results: List[Dict[str, Any]], feature_type: str = "features") -> None:
    """
    Print summary of feature extraction results.
    
    Args:
        results: List of result dictionaries from feature extraction
        feature_type: Type of features extracted (for display purposes)
    """
    successful = [r for r in results if r['status'] == 'success']
    cached = [r for r in results if r['status'] == 'cached']
    failed = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    print("\n" + "="*80)
    print(f"{feature_type.upper()} EXTRACTION SUMMARY")
    print("="*80)
    
    print(f"\n✅ Successfully processed: {len(successful)} datasets")
    if len(cached) > 0:
        print(f"💾 Loaded from cache: {len(cached)} datasets")
    print(f"❌ Failed: {len(failed)} datasets")
    print(f"⚠️  Skipped: {len(skipped)} datasets")
    
    if len(successful) > 0:
        print("\nSuccessfully processed datasets:")
        print(f"{'Dataset':<45} {'Train':>8} {'Val':>8} {'Test':>8} {'Features':>10}")
        print("-" * 85)
        for r in successful:
            if 'n_train_scanpaths' in r:
                n_train = r.get('n_train_scanpaths', 0)
                n_val = r.get('n_val_scanpaths', 0)
                n_test = r.get('n_test_scanpaths', 0)
                n_features = r.get('num_features', 0)
                print(f"  {r['dataset']:<43} {n_train:>8} {n_val:>8} {n_test:>8} {n_features:>10}")
            else:
                # Fallback for datasets without split info
                print(f"  {r['dataset']}: {r.get('num_scanpaths', 0)} scanpaths, {r.get('num_features', 0)} features {r.get('note', '')}")
    
    if len(cached) > 0:
        print("\n💾 Loaded from cache:")
        print(f"{'Dataset':<45} {'Train':>8} {'Val':>8} {'Test':>8} {'Features':>10}")
        print("-" * 85)
        for r in cached:
            if 'n_train_scanpaths' in r:
                n_train = r.get('n_train_scanpaths', 0)
                n_val = r.get('n_val_scanpaths', 0)
                n_test = r.get('n_test_scanpaths', 0)
                n_features = r.get('num_features', 0)
                print(f"  {r['dataset']:<43} {n_train:>8} {n_val:>8} {n_test:>8} {n_features:>10}")
            else:
                print(f"  {r['dataset']}: {r.get('num_scanpaths', 0)} scanpaths, {r.get('num_features', 0)} features {r.get('note', '')}")
    
    if len(failed) > 0:
        print("\nFailed datasets:")
        for r in failed:
            print(f"  {r['dataset']}: {r.get('error', 'Unknown error')}")
    
    if len(skipped) > 0:
        print("\nSkipped datasets:")
        for r in skipped:
            print(f"  {r['dataset']}: {r.get('reason', 'Unknown reason')}")


def extract_and_save_features(
    df: pd.DataFrame,
    dataset_name: str,
    feature_type: str,
    extractor,
    col_info: Dict[str, Any],
    paths: Dict[str, Path],
    check_cache_first: bool = True
) -> Dict[str, Any]:
    """
    Extract features from a loaded DataFrame and save results.
    
    This function handles the common workflow:
    1. Check cache (optional)
    2. Extract features using extractor
    3. Apply splits and save
    
    Args:
        df: Loaded DataFrame
        dataset_name: Name of the dataset
        feature_type: Type of features (e.g., 'simple_features', 'complex_features')
        extractor: Extractor instance (will call fit_transform)
        col_info: Column info dictionary
        paths: Dictionary with path Path objects
        check_cache_first: If True, check for cached results first
    
    Returns:
        Dictionary with processing results
    """
    # Check cache if requested
    if check_cache_first:
        cached_result = check_cache(dataset_name, feature_type, paths)
        if cached_result:
            print(f"✅ Features already exist - skipping computation")
            if 'train_output_path' in cached_result:
                print(f"    Train: {cached_result['train_output_path']}")
                print(f"    Val:   {cached_result['val_output_path']}")
                print(f"    Test:  {cached_result['test_output_path']}")
            else:
                print(f"    Full: {cached_result['output_path']}")
            cached_result['dataset'] = dataset_name
            return cached_result
    
    # Extract features
    print("Extracting features...")
    features_df = extractor.fit_transform(df)
    print(f"Extracted {features_df.shape[1]} features for {features_df.shape[0]} scanpaths")
    
    # Apply splits and save
    result = apply_splits_and_save(
        features_df,
        dataset_name,
        feature_type,
        col_info,
        paths
    )
    result['dataset'] = dataset_name
    
    return result

