"""
Dataset Loader and Feature Caching Module.
Handles loading, preprocessing, and feature caching for all three datasets
in the MLP2 workspace:
1. DS1: Rice Leaf Disease Images (4 classes, 5932 images)
2. DS2: Dataset (Full_Train_Data & Test_Data_SVM, 5 classes, 2800 images)
3. DS3: rice leaf diseases dataset (3 classes, 4684 images)
"""

import os
import glob
import numpy as np
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

from src.config import DATASETS, CACHE_DIR, ML_LEVELS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_fused_features


def get_image_files_by_class(dataset_dir: str) -> Dict[str, List[str]]:
    """
    Scans a directory where subdirectories represent disease classes
    and returns a sorted dictionary mapping class name to image paths.
    """
    classes_map = {}
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    subdirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
    ])
    
    for cls in subdirs:
        class_folder = os.path.join(dataset_dir, cls)
        files = sorted([
            os.path.join(class_folder, f)
            for f in os.listdir(class_folder)
            if f.endswith(valid_exts) and not f.startswith('.')
        ])
        if files:
            classes_map[cls] = files
            
    return classes_map


def load_dataset(
    dataset_key: str = 'DS1',
    split: str = 'all',
    max_samples_per_class: Optional[int] = None,
    level: int = ML_LEVELS,
    use_cache: bool = True,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Loads images for the specified dataset, extracts fused handcrafted features,
    and returns feature matrix X, labels y, class names, and image file paths.

    Features are cached to disk so subsequent calls are near instantaneous.

    Args:
        dataset_key: 'DS1', 'DS2', or 'DS3'.
        split: For DS2, can be 'train', 'test', or 'all'. Ignored for DS1/DS3.
        max_samples_per_class: Optional limit on samples per class (for quick tests).
        level: Multi-level spatial block decomposition level L.
        use_cache: Whether to use disk cache for extracted features.
        force_recompute: Force re-extraction even if cache exists.

    Returns:
        X: Feature matrix of shape (N, feature_dim).
        y: Numeric labels array of shape (N,).
        class_names: List of class label strings.
        image_paths: List of file paths corresponding to each row in X.
    """
    dataset_info = DATASETS.get(dataset_key.upper())
    if not dataset_info:
        raise ValueError(f"Unknown dataset_key: {dataset_key}. Choose from {list(DATASETS.keys())}")

    # Determine source directory / directories
    if dataset_key.upper() == 'DS2':
        if split == 'train':
            classes_map = get_image_files_by_class(dataset_info['train_path'])
        elif split == 'test':
            classes_map = get_image_files_by_class(dataset_info['test_path'])
        else:
            # Combine train and test
            train_map = get_image_files_by_class(dataset_info['train_path'])
            test_map = get_image_files_by_class(dataset_info['test_path'])
            classes_map = {}
            all_classes = sorted(set(list(train_map.keys()) + list(test_map.keys())))
            for cls in all_classes:
                classes_map[cls] = train_map.get(cls, []) + test_map.get(cls, [])
    else:
        classes_map = get_image_files_by_class(dataset_info['path'])

    class_names = sorted(classes_map.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    # Subsample if max_samples_per_class is requested
    filtered_map = {}
    for cls in class_names:
        files = classes_map[cls]
        if max_samples_per_class is not None and max_samples_per_class > 0:
            files = files[:max_samples_per_class]
        filtered_map[cls] = files

    total_images = sum(len(f) for f in filtered_map.values())
    limit_suffix = f"_max{max_samples_per_class}" if max_samples_per_class else "_full"
    split_suffix = f"_{split}" if dataset_key.upper() == 'DS2' else ""
    cache_filename = f"{dataset_key.upper()}{split_suffix}_L{level}{limit_suffix}_rich_features.npz"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # Check cache
    if use_cache and not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y'], list(data['class_names']), list(data['image_paths'])

    print(f"Extracting features for {dataset_key} (Total images: {total_images}, Level={level})...")
    X_list = []
    y_list = []
    paths_list = []

    for cls in class_names:
        files = filtered_map[cls]
        print(f"  Class '{cls}': {len(files)} images")
        for img_path in tqdm(files, desc=f"  {cls}", leave=False):
            try:
                # Preprocess: resize to 224x224 and apply log enhancement (c=2.0)
                img = preprocess_image(img_path, enhance=True, augment=False)
                # Feature extraction: ML-MCLBP + Color Correlogram
                feat = extract_fused_features(img, level=level)
                X_list.append(feat)
                y_list.append(class_to_idx[cls])
                paths_list.append(img_path)
            except Exception as e:
                print(f"Warning: skipping {img_path} due to error: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Save to cache
    if use_cache:
        print(f"Saving features to cache: {cache_path}")
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            class_names=np.array(class_names),
            image_paths=np.array(paths_list)
        )

    return X, y, class_names, paths_list


def split_dataset_3way(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a 3-way stratified partition of features X and labels y into:
    - Training set: 70% (train_ratio = 0.70)
    - Validation set: 10% (val_ratio = 0.10)
    - Testing set: 20% (test_ratio = 0.20)

    Args:
        X: Feature matrix of shape (N, D).
        y: Target label vector of shape (N,).
        train_ratio: Fraction of data for training (default 0.70).
        val_ratio: Fraction of data for validation (default 0.10).
        test_ratio: Fraction of data for testing (default 0.20).
        random_state: Random seed for stratified reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    val_and_test = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_and_test, random_state=random_state, stratify=y
    )
    test_share_of_temp = test_ratio / val_and_test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_share_of_temp, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

