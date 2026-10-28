"""
Model Training and Comparison Script.
Trains and compares Machine Learning models from ML ALL.pdf on the rice disease
datasets (DS1, DS2, DS3) according to Research.pdf.

Usage examples:
    python train.py --dataset DS1 --model all --samples 100
    python train.py --dataset DS1 --model svm_linear
    python train.py --dataset DS2 --model all --levels 3
    python train.py --dataset DS3 --model svm_linear
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# Ensure local modules are accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SAVED_MODELS_DIR, REPORTS_DIR, ML_LEVELS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from src.dataset_loader import load_dataset, split_dataset_3way
from src.models import get_model, get_all_models
from src.evaluation import evaluate_predictions, print_evaluation_summary


def train_and_evaluate_model(
    model_name: str,
    pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
) -> dict:
    """
    Fits a model pipeline on the training set (70%) and evaluates its performance
    on both validation (10%) and test (20%) sets.
    """
    print(f"\nTraining [{model_name}] on 70% training partition...")
    pipeline.fit(X_train, y_train)

    # 1. Test set evaluation (20%)
    preds_test = pipeline.predict(X_test)
    eval_res = evaluate_predictions(y_test, preds_test, class_names)

    # 2. Accuracies across all three partitions
    eval_res['train_accuracy'] = float(pipeline.score(X_train, y_train) * 100.0)
    eval_res['val_accuracy'] = float(pipeline.score(X_val, y_val) * 100.0)
    eval_res['test_accuracy'] = float(pipeline.score(X_test, y_test) * 100.0)

    return eval_res


def main():
    parser = argparse.ArgumentParser(description="Train Rice Leaf Disease Detection ML Models")
    parser.add_argument('--dataset', type=str, default='DS1', choices=['DS1', 'DS2', 'DS3'],
                        help="Which dataset to train on: DS1, DS2, or DS3")
    parser.add_argument('--model', type=str, default='all',
                        help="Model to train: 'all', 'svm_linear', 'svm_rbf', 'svm_poly', 'decision_tree', 'random_forest', 'extra_trees', 'adaboost', 'knn', 'mlp', 'softmax'")
    parser.add_argument('--samples', type=int, default=None,
                        help="Max samples per class (leave empty to use full dataset)")
    parser.add_argument('--levels', type=int, default=ML_LEVELS,
                        help=f"Multi-Level spatial blocks grid level L (default: {ML_LEVELS})")
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO,
                        help=f"Train partition fraction (default: {TRAIN_RATIO} for 70%)")
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO,
                        help=f"Validation partition fraction (default: {VAL_RATIO} for 10%)")
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO,
                        help=f"Test partition fraction (default: {TEST_RATIO} for 20%)")
    parser.add_argument('--recompute', action='store_true',
                        help="Force recompute features instead of loading cache")
    args = parser.parse_args()

    print("=" * 80)
    print("      RICE CROP DISEASE DETECTION - ML 3-WAY PARTITION TRAINING PIPELINE")
    print(f"      Dataset: {args.dataset} | ML Levels: {args.levels} | Split: 70% Train / 10% Val / 20% Test")
    print("=" * 80)

    # 1. Load Dataset and extract/load features
    X, y, class_names, paths = load_dataset(
        dataset_key=args.dataset,
        max_samples_per_class=args.samples,
        level=args.levels,
        force_recompute=args.recompute
    )

    print(f"\nDataset Loaded: {len(X)} samples, {X.shape[1]} features per sample.")
    print(f"Classes ({len(class_names)}): {class_names}")

    # 2. 3-Way Stratified Split: 70% Train, 10% Validation, 20% Test
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_3way(
        X, y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=42
    )
    print(f"\nStratified 3-Way Data Partition:")
    print(f"  Training Samples   (70%): {len(X_train)}")
    print(f"  Validation Samples (10%): {len(X_val)}")
    print(f"  Testing Samples    (20%): {len(X_test)}")

    # 3. Determine models to train
    if args.model.lower() == 'all':
        models_dict = get_all_models()
    else:
        models_dict = {args.model: get_model(args.model)}

    results_table = []
    best_acc = -1.0
    best_train_acc = -1.0
    best_val_acc = -1.0
    best_model_name = None
    best_pipeline = None

    for m_name, pipeline in models_dict.items():
        try:
            eval_res = train_and_evaluate_model(
                m_name, pipeline,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                class_names
            )
            macro = eval_res['macro_metrics']
            print(f"Partition Accuracies for [{m_name}]:")
            print(f"  -> Train (70%): {eval_res['train_accuracy']:.2f}% | Val (10%): {eval_res['val_accuracy']:.2f}% | Test (20%): {eval_res['test_accuracy']:.2f}%")
            print_evaluation_summary(eval_res, title=f"Model: {m_name}")

            results_table.append({
                'Model': m_name,
                'Train Acc (%)': eval_res['train_accuracy'],
                'Val Acc (%)': eval_res['val_accuracy'],
                'Test Acc (%)': macro['Accuracy (%)'],
                'Precision (%)': macro['Precision (%)'],
                'Recall (%)': macro['Recall (%)'],
                'F1-Score (%)': macro['F1-Score (%)'],
                'Specificity (%)': macro['Specificity (%)'],
                'MCC (%)': macro['MCC (%)']
            })

            if macro['Accuracy (%)'] > best_acc:
                best_acc = macro['Accuracy (%)']
                best_train_acc = eval_res['train_accuracy']
                best_val_acc = eval_res['val_accuracy']
                best_model_name = m_name
                best_pipeline = pipeline

        except Exception as e:
            print(f"Error training {m_name}: {e}")

    # 4. Save and print comparison summary
    if results_table:
        df_results = pd.DataFrame(results_table).sort_values(by='Test Acc (%)', ascending=False)
        print("\n" + "=" * 90)
        print(f"      FINAL MODEL COMPARISON SUMMARY (70% Train / 10% Val / 20% Test) - DATASET: {args.dataset}")
        print("=" * 90)
        print(df_results.to_string(index=False))
        print("=" * 90)

        # Save results to CSV
        report_path = os.path.join(REPORTS_DIR, f"model_comparison_{args.dataset.lower()}.csv")
        df_results.to_csv(report_path, index=False)
        print(f"\nSaved comparison report to: {report_path}")

        # 5. Save the best performing model
        if best_pipeline is not None:
            save_path = os.path.join(SAVED_MODELS_DIR, f"best_model_{args.dataset.lower()}.joblib")
            save_payload = {
                'model_name': best_model_name,
                'pipeline': best_pipeline,
                'class_names': class_names,
                'levels': args.levels,
                'dataset': args.dataset,
                'split': {
                    'train_ratio': args.train_ratio,
                    'val_ratio': args.val_ratio,
                    'test_ratio': args.test_ratio
                },
                'train_accuracy': best_train_acc,
                'val_accuracy': best_val_acc,
                'test_accuracy': best_acc
            }
            joblib.dump(save_payload, save_path)
            print(f"Saved best model [{best_model_name}] (Test: {best_acc:.2f}%, Val: {best_val_acc:.2f}%) to: {save_path}")


if __name__ == '__main__':
    main()
