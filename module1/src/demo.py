"""
Module 1 Demo: Comprehensive Image Processing Pipeline
======================================================

This script demonstrates all Module 1 capabilities:
1. Dataset exploration
2. Geometric and intensity transformations
3. Edge detection
4. Noise modeling and restoration
5. Quality metrics (PSNR, SSIM)
6. Comparative study on feature stability

Run: python module1/src/demo.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add source directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from dataset_utils import StanfordDogsDataset, resize_for_display
from transforms import (
    GeometricTransforms, IntensityTransforms,
    rotate, scale, flip_horizontal, histogram_eq, gamma_correct
)
from edge_detection import EdgeDetector, compute_edge_density
from noise_restoration import NoiseGenerator, ImageRestoration
from metrics import psnr, ssim, compare_filters, print_comparison_table


def ensure_output_dir():
    """Ensure output directory exists."""
    output_dir = os.path.join(os.path.dirname(script_dir), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def demo_dataset_exploration(dataset):
    """Demonstrate dataset exploration."""
    print("\n" + "="*70)
    print("1. DATASET EXPLORATION")
    print("="*70)
    
    dataset.print_dataset_summary()
    
    # Get sample image
    image, path = dataset.get_sample_image()
    breed_name = os.path.basename(os.path.dirname(path)).split("-")[1].replace("_", " ")
    
    print(f"\nSample image: {breed_name}")
    print(f"Path: {path}")
    print(f"Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"Channels: {image.shape[2] if len(image.shape) > 2 else 1}")
    
    return image, path


def demo_geometric_transforms(image, output_dir):
    """Demonstrate geometric transformations."""
    print("\n" + "="*70)
    print("2. GEOMETRIC TRANSFORMATIONS")
    print("="*70)
    
    transforms = {
        'original': image,
        'rotated_15': rotate(image, 15),
        'rotated_45': rotate(image, 45),
        'scaled_0.7': scale(image, 0.7),
        'scaled_1.3': scale(image, 1.3),
        'flipped_h': flip_horizontal(image),
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, img) in zip(axes, transforms.items()):
        display_img = cv2.cvtColor(resize_for_display(img, 300), cv2.COLOR_BGR2RGB)
        ax.imshow(display_img)
        ax.set_title(name.replace('_', ' ').title())
        ax.axis('off')
    
    plt.suptitle("Geometric Transformations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geometric_transforms.png"), dpi=150)
    plt.close()
    
    print("Transformations applied:")
    for name in transforms:
        if name != 'original':
            print(f"  - {name}: Shape {transforms[name].shape[:2]}")
    
    print(f"\nVisualization saved to: {output_dir}/geometric_transforms.png")


def demo_intensity_transforms(image, output_dir):
    """Demonstrate intensity transformations."""
    print("\n" + "="*70)
    print("3. INTENSITY TRANSFORMATIONS")
    print("="*70)
    
    transforms = {
        'original': image,
        'hist_eq_global': histogram_eq(image, 'global'),
        'hist_eq_clahe': histogram_eq(image, 'clahe'),
        'gamma_0.5_bright': gamma_correct(image, 0.5),
        'gamma_1.5_dark': gamma_correct(image, 1.5),
        'gamma_2.0_darker': gamma_correct(image, 2.0),
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, img) in zip(axes, transforms.items()):
        display_img = cv2.cvtColor(resize_for_display(img, 300), cv2.COLOR_BGR2RGB)
        ax.imshow(display_img)
        ax.set_title(name.replace('_', ' ').title())
        ax.axis('off')
    
    plt.suptitle("Intensity Transformations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intensity_transforms.png"), dpi=150)
    plt.close()
    
    print("Transformations applied:")
    print("  - Global Histogram Equalization: Uniform intensity distribution")
    print("  - CLAHE: Contrast-limited adaptive equalization (preserves texture)")
    print("  - Gamma 0.5: Brightens image (enhances dark regions)")
    print("  - Gamma 1.5/2.0: Darkens image (enhances bright regions)")
    
    print(f"\nVisualization saved to: {output_dir}/intensity_transforms.png")


def demo_edge_detection(image, output_dir):
    """Demonstrate edge detection methods."""
    print("\n" + "="*70)
    print("4. EDGE DETECTION")
    print("="*70)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = {
        'original': gray,
        'sobel': EdgeDetector.sobel(image),
        'canny_low': EdgeDetector.canny(image, 30, 90),
        'canny_high': EdgeDetector.canny(image, 100, 200),
        'laplacian': EdgeDetector.laplacian(image),
        'auto_canny': EdgeDetector.auto_canny(image),
    }
    
    # Compute edge densities
    print("\nEdge detection results:")
    for name, edge_img in edges.items():
        if name != 'original':
            density = compute_edge_density(edge_img)
            print(f"  - {name}: Edge density = {density:.2%}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, img) in zip(axes, edges.items()):
        ax.imshow(resize_for_display(img, 300), cmap='gray')
        ax.set_title(name.replace('_', ' ').title())
        ax.axis('off')
    
    plt.suptitle("Edge Detection Methods", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edge_detection.png"), dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}/edge_detection.png")
    print("\nObservations:")
    print("  - Sobel: Captures gradients in all directions")
    print("  - Canny (low threshold): Captures fine texture like fur")
    print("  - Canny (high threshold): Captures only strong edges (body outline)")
    print("  - Auto Canny: Adaptive thresholds based on image statistics")


def demo_noise_restoration(image, output_dir):
    """Demonstrate noise modeling and restoration."""
    print("\n" + "="*70)
    print("5. NOISE MODELING AND RESTORATION")
    print("="*70)
    
    # Add Gaussian noise and restore
    print("\nGaussian Noise (sigma=25):")
    noisy_gaussian = NoiseGenerator.gaussian(image, sigma=25)
    
    restored_gaussian = {
        'gaussian_blur': ImageRestoration.gaussian_blur(noisy_gaussian),
        'median': ImageRestoration.median_filter(noisy_gaussian),
        'bilateral': ImageRestoration.bilateral_filter(noisy_gaussian),
        'nlm': ImageRestoration.nlm_denoise(noisy_gaussian)
    }
    
    results_gaussian = compare_filters(image, noisy_gaussian, restored_gaussian)
    print_comparison_table(results_gaussian)
    
    # Add Salt-and-Pepper noise and restore
    print("\nSalt-and-Pepper Noise (p=0.02):")
    noisy_sp = NoiseGenerator.salt_and_pepper(image, prob=0.02)
    
    restored_sp = {
        'gaussian_blur': ImageRestoration.gaussian_blur(noisy_sp),
        'median': ImageRestoration.median_filter(noisy_sp),
        'bilateral': ImageRestoration.bilateral_filter(noisy_sp),
    }
    
    results_sp = compare_filters(image, noisy_sp, restored_sp)
    print_comparison_table(results_sp)
    
    # Create visualization for Gaussian noise
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    images_to_show = [
        ('Original', image),
        ('Noisy (Gaussian)', noisy_gaussian),
        ('Gaussian Blur', restored_gaussian['gaussian_blur']),
        ('Median Filter', restored_gaussian['median']),
        ('Bilateral Filter', restored_gaussian['bilateral']),
        ('NLM Denoise', restored_gaussian['nlm']),
    ]
    
    for ax, (name, img) in zip(axes, images_to_show):
        display_img = cv2.cvtColor(resize_for_display(img, 300), cv2.COLOR_BGR2RGB)
        ax.imshow(display_img)
        ax.set_title(name)
        ax.axis('off')
    
    plt.suptitle("Noise Restoration Comparison (Gaussian Noise)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_restoration.png"), dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}/noise_restoration.png")
    print("\nConclusion:")
    print("  - For Gaussian noise: Bilateral or NLM filters work best (edge-preserving)")
    print("  - For Salt-and-Pepper noise: Median filter is superior (outlier removal)")


def demo_comparative_study(image, output_dir):
    """Demonstrate preprocessing impact on image quality."""
    print("\n" + "="*70)
    print("6. COMPARATIVE STUDY: PREPROCESSING IMPACT ON IMAGE QUALITY")
    print("="*70)
    
    from comparative_study import PreprocessingAnalyzer
    
    analyzer = PreprocessingAnalyzer()
    results = analyzer.analyze_preprocessing_impact(image)
    analyzer.print_analysis_report(results)


def main():
    """Run complete Module 1 demonstration."""
    print("\n" + "#"*70)
    print("#" + " "*20 + "MODULE 1 DEMONSTRATION" + " "*20 + "#")
    print("#" + " "*10 + "Foundations of Vision and Image Analysis" + " "*10 + "#")
    print("#"*70)
    
    # Setup
    root_path = os.path.dirname(os.path.dirname(script_dir))
    output_dir = ensure_output_dir()
    
    print(f"\nProject root: {root_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize dataset
    try:
        dataset = StanfordDogsDataset(root_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you're running from the CV-Project directory")
        return
    
    # Run demonstrations
    image, path = demo_dataset_exploration(dataset)
    demo_geometric_transforms(image, output_dir)
    demo_intensity_transforms(image, output_dir)
    demo_edge_detection(image, output_dir)
    demo_noise_restoration(image, output_dir)
    demo_comparative_study(image, output_dir)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
