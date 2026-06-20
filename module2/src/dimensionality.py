"""
Dimensionality Reduction with PCA
=================================

Implements Principal Component Analysis for:
- Reducing feature dimensionality
- Visualizing feature clusters
- Analyzing explained variance
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class DimensionalityReducer:
    """
    PCA-based dimensionality reduction for feature vectors.
    
    PCA finds orthogonal directions of maximum variance in the data,
    allowing us to:
    1. Reduce dimensionality while preserving information
    2. Visualize high-dimensional features in 2D/3D
    3. Remove noise (minor components)
    """
    
    def __init__(self, n_components: int = 100, whiten: bool = False):
        """
        Args:
            n_components: Number of principal components to keep
            whiten: If True, normalize components to unit variance
        """
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> 'DimensionalityReducer':
        """
        Fit PCA on training features.
        
        Args:
            features: Training features (n_samples, n_features)
            
        Returns:
            self
        """
        # Standardize features first
        scaled = self.scaler.fit_transform(features)
        
        # Fit PCA
        self.pca = PCA(n_components=min(self.n_components, scaled.shape[1]),
                       whiten=self.whiten)
        self.pca.fit(scaled)
        self.is_fitted = True
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.
        
        Args:
            features: Features to transform
            
        Returns:
            Reduced features
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("PCA not fitted.")
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative explained variance."""
        return np.cumsum(self.get_explained_variance())
    
    def print_variance_summary(self) -> None:
        """Print variance explanation summary."""
        if not self.is_fitted:
            raise ValueError("PCA not fitted.")
        
        cumvar = self.get_cumulative_variance()
        
        print("\nPCA Variance Summary:")
        print("-" * 40)
        print(f"Components used: {self.pca.n_components_}")
        print(f"Total variance explained: {cumvar[-1]:.2%}")
        
        # Find components needed for different variance thresholds
        thresholds = [0.80, 0.90, 0.95, 0.99]
        for thresh in thresholds:
            n_needed = np.argmax(cumvar >= thresh) + 1
            print(f"Components for {thresh:.0%} variance: {n_needed}")


def visualize_pca_2d(features: np.ndarray, labels: np.ndarray, 
                     label_names: list = None,
                     title: str = "PCA Visualization",
                     save_path: str = None) -> None:
    """
    Visualize features in 2D using PCA.
    
    Args:
        features: High-dimensional features
        labels: Class labels (numeric)
        label_names: Optional list of class names
        title: Plot title
        save_path: Path to save figure
    """
    # Reduce to 2D
    reducer = DimensionalityReducer(n_components=2)
    features_2d = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Use colormap
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))
    
    for i, label in enumerate(unique_labels[:20]):  # Limit to 20 classes for visibility
        mask = labels == label
        name = label_names[label] if label_names else f"Class {label}"
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=name[:15], alpha=0.6, s=30)
    
    plt.xlabel(f"PC1 ({reducer.get_explained_variance()[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({reducer.get_explained_variance()[1]:.1%} variance)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def visualize_pca_3d(features: np.ndarray, labels: np.ndarray,
                     label_names: list = None,
                     title: str = "PCA 3D Visualization",
                     save_path: str = None) -> None:
    """
    Visualize features in 3D using PCA.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reduce to 3D
    reducer = DimensionalityReducer(n_components=3)
    features_3d = reducer.fit_transform(features)
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))
    
    for i, label in enumerate(unique_labels[:20]):
        mask = labels == label
        name = label_names[label] if label_names else f"Class {label}"
        ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                  c=[colors[i]], label=name[:15], alpha=0.6, s=30)
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_explained_variance(features: np.ndarray, 
                            max_components: int = 50,
                            save_path: str = None) -> None:
    """
    Plot cumulative explained variance vs number of components.
    
    Useful for choosing optimal number of components.
    """
    reducer = DimensionalityReducer(n_components=max_components)
    reducer.fit(features)
    
    cumvar = reducer.get_cumulative_variance()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumvar) + 1), cumvar, 'b-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% variance')
    
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Explained Variance vs Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary
    reducer.print_variance_summary()


if __name__ == "__main__":
    # Quick test with random data
    print("Dimensionality Reduction Test")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 1000
    n_classes = 10
    
    # Create synthetic features
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Test PCA
    reducer = DimensionalityReducer(n_components=50)
    reduced = reducer.fit_transform(features)
    
    print(f"Original shape: {features.shape}")
    print(f"Reduced shape: {reduced.shape}")
    reducer.print_variance_summary()
