"""
Machine Learning Models Module.
Implements the classification algorithms detailed in ML ALL.pdf and evaluated
in Table 11 of Research.pdf:
1. Linear Support Vector Machine (One-vs-Rest / OAA) [Proposed Winner] - ML ALL.pdf p.28-36
2. RBF Kernel SVM - ML ALL.pdf p.43, 47
3. Polynomial Kernel SVM - ML ALL.pdf p.38
4. Decision Tree (CART / Gini Index) - ML ALL.pdf p.57-77
5. Random Forest (Ensemble Bagging) - ML ALL.pdf p.85-92
6. AdaBoost Classifier - ML ALL.pdf p.98
7. K-Nearest Neighbors (KNN Local Proximity) - ML ALL.pdf p.112-114
8. Multi-Layer Perceptron (MLP Neural Network with Adam) - ML ALL.pdf p.194-219
9. Softmax Regression (Multinomial Cross-Entropy) - ML ALL.pdf p.22-25
"""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def get_model(model_name: str, random_state: int = 42) -> Pipeline:
    """
    Factory function creating an end-to-end ML pipeline consisting of:
    1. StandardScaler: standardizes features (mean=0, variance=1)
    2. Classifier: chosen from ML ALL.pdf and Research.pdf

    Args:
        model_name: One of:
            - 'svm_linear'  : One-Against-All Linear SVM (Research.pdf proposed model)
            - 'svm_rbf'     : Gaussian Radial Basis Function Kernel SVM
            - 'svm_poly'    : Polynomial Kernel (degree 3) SVM
            - 'decision_tree': CART Decision Tree using Gini impurity
            - 'random_forest': Bagging Ensemble of Decision Trees
            - 'adaboost'    : AdaBoost adaptive boosting ensemble
            - 'knn'         : K-Nearest Neighbors (Euclidean distance)
            - 'mlp'         : Multi-Layer Perceptron Neural Network (Adam optimizer)
            - 'softmax'     : Multinomial Logistic / Softmax Regression
        random_state: Random seed for reproducibility.

    Returns:
        Configured scikit-learn Pipeline ready to fit and predict.
    """
    key = model_name.lower().strip()

    if key in ['svm_linear', 'linear_svm', 'svm']:
        # Proposed Classifier in Research.pdf (Eq. 14, 16) & ML ALL.pdf (p.28-36)
        # Uses One-Against-All strategy with linear hyperplane
        classifier = SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            decision_function_shape='ovr',
            random_state=random_state
        )

    elif key in ['svm_rbf', 'rbf_svm']:
        # Gaussian RBF Kernel SVM (ML ALL.pdf p.43, 47 & Research.pdf Table 11)
        classifier = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            decision_function_shape='ovr',
            random_state=random_state
        )

    elif key in ['svm_poly', 'poly_svm']:
        # Polynomial Kernel SVM (ML ALL.pdf p.38 & Research.pdf Table 11)
        classifier = SVC(
            kernel='poly',
            degree=3,
            C=1.0,
            probability=True,
            decision_function_shape='ovr',
            random_state=random_state
        )

    elif key in ['decision_tree', 'dt', 'cart']:
        # CART Decision Tree using Gini Impurity (ML ALL.pdf p.57-77)
        classifier = DecisionTreeClassifier(
            criterion='gini',
            max_depth=15,
            min_samples_split=4,
            random_state=random_state
        )

    elif key in ['random_forest', 'rf', 'ensemble_bag']:
        # Random Forest Bagging Ensemble (ML ALL.pdf p.85-92 & Research.pdf Table 11)
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            n_jobs=-1,
            random_state=random_state
        )

    elif key in ['adaboost', 'ada']:
        # AdaBoost with adaptive weighting (ML ALL.pdf p.98 & Research.pdf Table 11)
        classifier = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=random_state
        )

    elif key in ['knn', 'k_nearest_neighbors']:
        # K-Nearest Neighbors (ML ALL.pdf p.112-114 & Research.pdf Table 11)
        classifier = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        )

    elif key in ['mlp', 'neural_network', 'ann']:
        # Multi-Layer Perceptron with Backpropagation & Adam (ML ALL.pdf p.194-219)
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=300,
            early_stopping=True,
            random_state=random_state
        )

    elif key in ['softmax', 'softmax_regression', 'logistic_regression']:
        # Multinomial Softmax Regression (ML ALL.pdf p.22-25)
        classifier = LogisticRegression(
            solver='lbfgs',
            C=1.0,
            max_iter=500,
            random_state=random_state
        )

    elif key in ['extra_trees', 'et', 'extremely_randomized_trees']:
        # Extremely Randomized Trees Ensemble (ML ALL.pdf p.85-92)
        classifier = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=random_state
        )

    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available models: "
            "svm_linear, svm_rbf, svm_poly, decision_tree, random_forest, extra_trees, adaboost, knn, mlp, softmax"
        )

    return Pipeline([
        ('variance_selector', VarianceThreshold(threshold=1e-5)),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


def get_all_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Returns a dictionary of all comparison models from ML ALL.pdf and Research.pdf Table 11.
    """
    model_keys = [
        'svm_linear',
        'svm_rbf',
        'svm_poly',
        'decision_tree',
        'random_forest',
        'extra_trees',
        'adaboost',
        'knn',
        'mlp',
        'softmax'
    ]
    return {k: get_model(k, random_state=random_state) for k in model_keys}
