"""
Performance Evaluation Module.
Implements the exact evaluation metrics formulated in Research.pdf:
- Accuracy (Eq. 17)
- Precision (Eq. 18)
- Sensitivity / Recall (Eq. 19)
- F1-Score / Dice Similarity Coefficient (Eq. 20)
- Specificity (Eq. 21)
- Matthews Correlation Coefficient / MCC (Eq. 22)
- Multi-class Confusion Matrix
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import confusion_matrix, classification_report


def compute_class_metrics(cm: np.ndarray, class_idx: int) -> Dict[str, float]:
    """
    Computes binary metrics for a specific class using one-against-all confusion matrix:
    TP: True Positives (cm[i, i])
    FP: False Positives (sum of column i minus cm[i, i])
    FN: False Negatives (sum of row i minus cm[i, i])
    TN: True Negatives (all remaining entries)

    Mathematical formulations: Research.pdf Eq. (17)-(22)
    """
    total = np.sum(cm)
    tp = float(cm[class_idx, class_idx])
    fp = float(np.sum(cm[:, class_idx]) - tp)
    fn = float(np.sum(cm[class_idx, :]) - tp)
    tn = float(total - (tp + fp + fn))

    # Eq. 17: Accuracy
    acc = (tp + tn) / total if total > 0 else 0.0

    # Eq. 18: Precision
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Eq. 19: Sensitivity / Recall
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Eq. 20: F1-Score
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0

    # Eq. 21: Specificity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Eq. 22: Matthews Correlation Coefficient (MCC)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Accuracy': acc * 100.0,
        'Precision': prec * 100.0,
        'Recall': rec * 100.0,
        'F1-Score': f1 * 100.0,
        'Specificity': spec * 100.0,
        'MCC': mcc * 100.0
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes all Research.pdf evaluation metrics across all classes,
    both per-class and macro-averaged across the dataset.

    Returns:
        Dictionary containing:
        - 'macro_metrics': overall macro-averaged metrics (Accuracy, Precision, Recall, F1, Specificity, MCC)
        - 'per_class_metrics': dictionary of metrics for each individual class
        - 'confusion_matrix': full 2D confusion matrix
        - 'class_names': list of class names
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    num_classes = len(class_names)

    per_class = {}
    macro_acc = []
    macro_prec = []
    macro_rec = []
    macro_f1 = []
    macro_spec = []
    macro_mcc = []

    for i, cls in enumerate(class_names):
        metrics = compute_class_metrics(cm, i)
        per_class[cls] = metrics
        macro_acc.append(metrics['Accuracy'])
        macro_prec.append(metrics['Precision'])
        macro_rec.append(metrics['Recall'])
        macro_f1.append(metrics['F1-Score'])
        macro_spec.append(metrics['Specificity'])
        macro_mcc.append(metrics['MCC'])

    # Overall dataset accuracy is diagonal sum over total
    overall_accuracy = float(np.trace(cm)) / float(np.sum(cm)) * 100.0

    macro_metrics = {
        'Accuracy (%)': overall_accuracy,
        'Precision (%)': float(np.mean(macro_prec)),
        'Recall (%)': float(np.mean(macro_rec)),
        'F1-Score (%)': float(np.mean(macro_f1)),
        'Specificity (%)': float(np.mean(macro_spec)),
        'MCC (%)': float(np.mean(macro_mcc))
    }

    return {
        'macro_metrics': macro_metrics,
        'per_class_metrics': per_class,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def print_evaluation_summary(results: Dict[str, Any], title: str = "Evaluation Report"):
    """
    Prints a formatted evaluation table matching Table 11 & 13 of Research.pdf.
    """
    print(f"\n{'='*70}")
    print(f"  {title.upper()}")
    print(f"{'='*70}")
    
    # Macro metrics
    m = results['macro_metrics']
    print(f"Overall Accuracy:  {m['Accuracy (%)']:.2f}%")
    print(f"Macro Precision:   {m['Precision (%)']:.2f}%")
    print(f"Macro Recall:      {m['Recall (%)']:.2f}%")
    print(f"Macro F1-Score:    {m['F1-Score (%)']:.2f}%")
    print(f"Macro Specificity: {m['Specificity (%)']:.2f}%")
    print(f"Macro MCC:         {m['MCC (%)']:.2f}%")
    print(f"{'-'*70}")

    # Per-class breakdown
    print(f"{'Disease Class':<18} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'F1 (%)':<8} | {'Spec (%)':<8} | {'MCC (%)':<8}")
    print(f"{'-'*70}")
    for cls, val in results['per_class_metrics'].items():
        print(
            f"{cls:<18} | "
            f"{val['Precision']:<8.2f} | "
            f"{val['Recall']:<8.2f} | "
            f"{val['F1-Score']:<8.2f} | "
            f"{val['Specificity']:<8.2f} | "
            f"{val['MCC']:<8.2f}"
        )
    print(f"{'='*70}\n")
