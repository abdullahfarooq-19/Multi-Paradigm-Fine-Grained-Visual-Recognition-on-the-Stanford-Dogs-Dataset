"""
SVM Classifier for Dog Breed Recognition
=========================================

Implements Support Vector Machine classification with:
- Multiple kernel options (RBF, Linear, Polynomial)
- Cross-validation for hyperparameter tuning
- Performance metrics and confusion matrix
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
import pickle
import os


class SVMClassifier:
    """
    SVM-based classifier for dog breed recognition.
    
    SVM finds optimal hyperplane that maximizes margin between classes.
    For multi-class, uses one-vs-one strategy.
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 class_weight: str = 'balanced'):
        """
        Args:
            kernel: 'rbf', 'linear', 'poly', or 'sigmoid'
            C: Regularization parameter
            gamma: Kernel coefficient ('scale' or 'auto')
            class_weight: 'balanced' or None
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Train SVM classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (can be strings or integers)
            
        Returns:
            self
        """
        # Encode labels if they are strings
        if isinstance(y[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            self.label_encoder.fit(y)
        
        self.classes_ = self.label_encoder.classes_
        self.svm.fit(X, y_encoded)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted.")
        
        predictions = self.svm.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted.")
        return self.svm.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Comprehensive evaluation with multiple metrics.
        
        Returns:
            Dict with accuracy, top-5 accuracy, classification report
        """
        predictions = self.predict(X)
        proba = self.predict_proba(X)
        
        # Encode true labels
        y_encoded = self.label_encoder.transform(y)
        pred_encoded = self.label_encoder.transform(predictions)
        
        # Metrics
        accuracy = accuracy_score(y, predictions)
        
        # Top-5 accuracy (if more than 5 classes)
        n_classes = len(self.classes_)
        if n_classes > 5:
            top5_acc = top_k_accuracy_score(y_encoded, proba, k=5)
        else:
            top5_acc = accuracy
        
        # Per-class report
        report = classification_report(y, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'n_classes': n_classes,
            'report': report
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> Tuple[float, float]:
        """
        Perform cross-validation.
        
        Returns:
            (mean_accuracy, std_accuracy)
        """
        if isinstance(y[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        scores = cross_val_score(self.svm, X, y_encoded, cv=cv)
        return scores.mean(), scores.std()
    
    def save(self, path: str) -> None:
        """Save trained model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'label_encoder': self.label_encoder,
                'classes': self.classes_
            }, f)
        print(f"Model saved to: {path}")
    
    def load(self, path: str) -> 'SVMClassifier':
        """Load trained model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.svm = data['svm']
        self.label_encoder = data['label_encoder']
        self.classes_ = data['classes']
        self.is_fitted = True
        return self


def tune_svm_hyperparameters(X: np.ndarray, y: np.ndarray,
                              cv: int = 3) -> Dict:
    """
    Grid search for optimal SVM hyperparameters.
    
    Returns:
        Dict with best parameters and score
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVC(class_weight='balanced', random_state=42)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Tuning SVM hyperparameters...")
    grid_search = GridSearchCV(svm, param_grid, cv=cv, 
                                scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y_encoded)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list = None,
                          normalize: bool = True,
                          title: str = "Confusion Matrix",
                          save_path: str = None) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        normalize: If True, show percentages
        title: Plot title
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Limit to 20 classes for readability
    if cm.shape[0] > 20:
        print(f"Note: Showing only first 20 classes (out of {cm.shape[0]})")
        cm = cm[:20, :20]
        if class_names:
            class_names = class_names[:20]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    plt.show()


def print_classification_report(results: Dict) -> None:
    """Print formatted classification results."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Number of classes: {results['n_classes']}")
    print(f"Top-1 Accuracy: {results['accuracy']:.2%}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2%}")
    print("=" * 60)
    
    # Print per-class metrics for top/bottom performers
    report = results['report']
    
    # Get class-wise f1-scores
    class_scores = []
    for cls, metrics in report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            class_scores.append((cls, metrics['f1-score']))
    
    class_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 best classified breeds:")
    for cls, score in class_scores[:5]:
        print(f"  {cls}: F1={score:.3f}")
    
    print("\nTop 5 worst classified breeds:")
    for cls, score in class_scores[-5:]:
        print(f"  {cls}: F1={score:.3f}")


if __name__ == "__main__":
    # Quick test with random data
    print("SVM Classifier Test")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    classes = ['Beagle', 'Poodle', 'Husky', 'Bulldog', 'Retriever']
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(classes, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train classifier
    clf = SVMClassifier(kernel='rbf')
    clf.fit(X_train, y_train)
    
    # Evaluate
    results = clf.evaluate(X_test, y_test)
    print_classification_report(results)
