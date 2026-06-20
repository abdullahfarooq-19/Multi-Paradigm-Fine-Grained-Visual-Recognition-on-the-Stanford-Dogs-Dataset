"""
Module 2 Demo: Classical Feature-Based Vision Pipeline
======================================================

Complete pipeline demonstrating:
1. 50-breed subset loading
2. Feature extraction (SIFT, HOG, LBP)
3. Feature fusion
4. PCA dimensionality reduction
5. SVM classification
6. Ablation analysis

Run: python module2/src/demo.py
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

# Add source directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from breed_subset import SELECTED_BREEDS, get_subset_folders, load_subset_images, create_train_test_split
from feature_extraction import HOGExtractor, LBPExtractor, FeatureFusion, SIFTExtractor
from dimensionality import DimensionalityReducer, visualize_pca_2d, plot_explained_variance
from classifier import SVMClassifier, print_classification_report, plot_confusion_matrix


def ensure_output_dir():
    """Ensure output directory exists."""
    output_dir = os.path.join(os.path.dirname(script_dir), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_and_prepare_data(root_path: str, max_per_breed: int = 50):
    """
    Load dataset and prepare train/test splits.
    
    Args:
        root_path: Path to CV-Project
        max_per_breed: Max images per breed (to speed up experiments)
        
    Returns:
        (X_train, X_test, y_train, y_test, breed_names)
    """
    print("\n" + "="*60)
    print("LOADING 50-BREED SUBSET")
    print("="*60)
    
    # Load images
    images = load_subset_images(root_path, max_per_breed=max_per_breed)
    
    # Split into train/test
    train_images, test_images = create_train_test_split(images, test_ratio=0.2)
    
    # Get breed names in order
    breed_names = sorted(images.keys())
    
    return train_images, test_images, breed_names


def extract_features_from_splits(train_images: dict, test_images: dict, 
                                  extractor, verbose: bool = True):
    """
    Extract features from train and test image dictionaries.
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    breeds = sorted(train_images.keys())
    
    if verbose:
        print(f"\nExtracting features using {type(extractor).__name__}...")
    
    for breed in tqdm(breeds, disable=not verbose):
        # Train set
        for path in train_images[breed]:
            image = cv2.imread(path)
            if image is not None:
                feat = extractor.extract(image)
                X_train_list.append(feat)
                y_train_list.append(breed)
        
        # Test set
        for path in test_images[breed]:
            image = cv2.imread(path)
            if image is not None:
                feat = extractor.extract(image)
                X_test_list.append(feat)
                y_test_list.append(breed)
    
    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    
    if verbose:
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def run_single_feature_experiment(train_images, test_images, extractor, 
                                   feature_name: str, output_dir: str):
    """Run experiment with a single feature type."""
    print(f"\n--- {feature_name} Features ---")
    
    # Extract features
    X_train, X_test, y_train, y_test = extract_features_from_splits(
        train_images, test_images, extractor
    )
    
    # Train classifier
    clf = SVMClassifier(kernel='rbf', C=10)
    clf.fit(X_train, y_train)
    
    # Evaluate
    results = clf.evaluate(X_test, y_test)
    
    print(f"  Top-1 Accuracy: {results['accuracy']:.2%}")
    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2%}")
    
    return {
        'feature': feature_name,
        'accuracy': results['accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'classifier': clf
    }


def run_ablation_study(train_images, test_images, output_dir: str):
    """
    Ablation study comparing different feature combinations.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Feature Comparison")
    print("="*60)
    
    results = []
    
    # HOG only
    hog_result = run_single_feature_experiment(
        train_images, test_images,
        HOGExtractor(image_size=(128, 128)),
        "HOG",
        output_dir
    )
    results.append(hog_result)
    
    # LBP only
    lbp_result = run_single_feature_experiment(
        train_images, test_images,
        LBPExtractor(image_size=(128, 128)),
        "LBP",
        output_dir
    )
    results.append(lbp_result)
    
    # HOG + LBP Fusion
    fusion_result = run_single_feature_experiment(
        train_images, test_images,
        FeatureFusion(use_hog=True, use_lbp=True),
        "HOG+LBP Fusion",
        output_dir
    )
    results.append(fusion_result)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'Feature':<20} {'Top-1 Acc':>12} {'Top-5 Acc':>12}")
    print("-"*45)
    
    for r in results:
        print(f"{r['feature']:<20} {r['accuracy']:>11.2%} {r['top5_accuracy']:>11.2%}")
    
    print("="*60)
    
    return results


def main():
    """Run complete Module 2 demonstration."""
    print("\n" + "#"*60)
    print("#" + " "*15 + "MODULE 2 DEMONSTRATION" + " "*15 + "#")
    print("#" + " "*8 + "Classical Feature-Based Vision" + " "*8 + "#")
    print("#"*60)
    
    # Setup paths
    root_path = os.path.dirname(os.path.dirname(script_dir))
    output_dir = ensure_output_dir()
    
    print(f"\nProject root: {root_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data (use max 30 images per breed for faster demo)
    train_images, test_images, breed_names = load_and_prepare_data(
        root_path, max_per_breed=30
    )
    
    # Run ablation study
    ablation_results = run_ablation_study(train_images, test_images, output_dir)
    
    # Use best model (fusion) for detailed analysis
    best_result = ablation_results[-1]  # Fusion
    
    # PCA Visualization
    print("\n" + "="*60)
    print("PCA VISUALIZATION")
    print("="*60)
    
    # Combine train and test for visualization
    X_all = np.vstack([best_result['X_train'], best_result['X_test']])
    y_all = np.concatenate([best_result['y_train'], best_result['y_test']])
    
    # Label encoding for visualization
    unique_breeds = sorted(set(y_all))
    breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}
    y_numeric = np.array([breed_to_idx[b] for b in y_all])
    
    # Plot explained variance
    print("\nAnalyzing PCA components...")
    reducer = DimensionalityReducer(n_components=50)
    reducer.fit(X_all)
    reducer.print_variance_summary()
    
    # Classification report for best model
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (HOG+LBP)")
    print("="*60)
    print_classification_report({
        'accuracy': best_result['accuracy'],
        'top5_accuracy': best_result['top5_accuracy'],
        'n_classes': len(unique_breeds),
        'report': best_result['classifier'].evaluate(
            best_result['X_test'], best_result['y_test']
        )['report']
    })
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"\nKey Finding: Classical features achieve ~{best_result['accuracy']:.1%} on 50 breeds")
    print("This is expected for fine-grained classification with classical methods.")
    print("\nModule 3 (Deep Learning) will significantly improve these results.")


if __name__ == "__main__":
    main()
