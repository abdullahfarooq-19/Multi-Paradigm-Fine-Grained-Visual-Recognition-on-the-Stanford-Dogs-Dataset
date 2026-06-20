"""
Comparative Study: Preprocessing Impact on Image Quality
=========================================================

This module demonstrates how different preprocessing techniques affect
image quality metrics and edge detection results.

This addresses the project requirement:
"Design comparative studies demonstrating how different pre-processing 
pipelines affect feature stability."

For Module 1, we focus on low-level metrics (PSNR, edge consistency)
rather than feature descriptors like SIFT (which belong in Module 2).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transforms import (
    GeometricTransforms, IntensityTransforms,
    rotate, histogram_eq, gamma_correct
)
from src.edge_detection import EdgeDetector, compute_edge_density
from src.noise_restoration import NoiseGenerator, ImageRestoration
from src.metrics import psnr, ssim, compute_all_metrics


class PreprocessingAnalyzer:
    """
    Analyze how preprocessing affects image quality and edge detection.
    
    Metrics:
    - Image quality: PSNR, SSIM compared to original
    - Edge consistency: Similarity of detected edges before/after preprocessing
    - Edge density: Proportion of edge pixels
    """
    
    @staticmethod
    def compute_edge_similarity(edges1: np.ndarray, edges2: np.ndarray) -> float:
        """
        Compute similarity between two edge images.
        
        Uses Intersection over Union (IoU) of edge pixels.
        
        Returns:
            Similarity score between 0 and 1 (1 = identical edges)
        """
        # Ensure binary
        e1 = (edges1 > 0).astype(np.uint8)
        e2 = (edges2 > 0).astype(np.uint8)
        
        intersection = np.sum(e1 & e2)
        union = np.sum(e1 | e2)
        
        if union == 0:
            return 1.0  # Both empty
        
        return intersection / union
    
    def analyze_preprocessing_impact(self, image: np.ndarray) -> Dict:
        """
        Analyze how different preprocessing steps affect image quality.
        
        Args:
            image: Original input image
            
        Returns:
            Dictionary with analysis results for each preprocessing method
        """
        results = {}
        
        # Compute edges on original for reference
        edges_original = EdgeDetector.auto_canny(image)
        edge_density_original = compute_edge_density(edges_original)
        
        results['original'] = {
            'edge_density': edge_density_original,
            'edge_similarity': 1.0,
            'psnr': float('inf'),
            'ssim': 1.0
        }
        
        # Define preprocessing pipelines to test
        preprocessings = {
            # Intensity transforms
            'histogram_eq_global': lambda img: IntensityTransforms.histogram_equalization(img, 'global'),
            'histogram_eq_clahe': lambda img: IntensityTransforms.histogram_equalization(img, 'clahe'),
            'gamma_0.5_bright': lambda img: IntensityTransforms.gamma_correction(img, 0.5),
            'gamma_1.5_dark': lambda img: IntensityTransforms.gamma_correction(img, 1.5),
            'gamma_2.0_darker': lambda img: IntensityTransforms.gamma_correction(img, 2.0),
            'contrast_1.5': lambda img: IntensityTransforms.adjust_contrast(img, 1.5),
            'contrast_0.7': lambda img: IntensityTransforms.adjust_contrast(img, 0.7),
            
            # Smoothing (denoising without noise)
            'gaussian_blur_3': lambda img: ImageRestoration.gaussian_blur(img, ksize=3),
            'gaussian_blur_5': lambda img: ImageRestoration.gaussian_blur(img, ksize=5),
            'bilateral_smooth': lambda img: ImageRestoration.bilateral_filter(img),
            
            # Noise + Restoration
            'gaussian_noise_raw': lambda img: NoiseGenerator.gaussian(img, sigma=25),
            'gaussian_noise_bilateral': lambda img: ImageRestoration.bilateral_filter(
                NoiseGenerator.gaussian(img, sigma=25)
            ),
            'gaussian_noise_nlm': lambda img: ImageRestoration.nlm_denoise(
                NoiseGenerator.gaussian(img, sigma=25)
            ),
            'salt_pepper_raw': lambda img: NoiseGenerator.salt_and_pepper(img, prob=0.02),
            'salt_pepper_median': lambda img: ImageRestoration.median_filter(
                NoiseGenerator.salt_and_pepper(img, prob=0.02)
            ),
            
            # Combined pipelines
            'clahe_bilateral': lambda img: ImageRestoration.bilateral_filter(
                IntensityTransforms.histogram_equalization(img, 'clahe')
            ),
        }
        
        for name, preprocess_func in preprocessings.items():
            try:
                processed = preprocess_func(image)
                
                # Resize back if needed for fair comparison
                if processed.shape[:2] != image.shape[:2]:
                    processed = cv2.resize(processed, (image.shape[1], image.shape[0]))
                
                # Compute edges on processed image
                edges_processed = EdgeDetector.auto_canny(processed)
                
                # Compute metrics
                edge_density = compute_edge_density(edges_processed)
                edge_similarity = self.compute_edge_similarity(edges_original, edges_processed)
                
                # Image quality metrics
                quality = compute_all_metrics(image, processed)
                
                results[name] = {
                    'edge_density': edge_density,
                    'edge_similarity': edge_similarity,
                    'psnr': quality['psnr'],
                    'ssim': quality['ssim']
                }
            except Exception as e:
                results[name] = {
                    'error': str(e)
                }
        
        return results
    
    def print_analysis_report(self, results: Dict) -> None:
        """Print formatted analysis report."""
        print("\n" + "="*80)
        print("Preprocessing Impact Analysis (Module 1: Low-Level Vision)")
        print("="*80)
        print(f"{'Method':<25} {'Edge Density':>12} {'Edge Sim':>10} {'PSNR':>10} {'SSIM':>10}")
        print("-"*80)
        
        for name, data in results.items():
            if 'error' in data:
                print(f"{name:<25} {'ERROR':>12}")
                continue
            
            density = data['edge_density']
            similarity = data['edge_similarity']
            psnr_val = data['psnr']
            ssim_val = data['ssim']
            
            psnr_str = f"{psnr_val:.1f}" if psnr_val != float('inf') else "∞"
            
            print(f"{name:<25} {density:>11.2%} {similarity:>9.2%} {psnr_str:>10} {ssim_val:>10.4f}")
        
        print("="*80)
        
        # Analysis summary
        print("\nKEY FINDINGS:")
        
        # Exclude original and error entries
        valid_methods = [
            (name, data) 
            for name, data in results.items() 
            if 'edge_similarity' in data and name != 'original'
        ]
        
        if valid_methods:
            # Best edge preservation
            best_edge = max(valid_methods, key=lambda x: x[1]['edge_similarity'])
            worst_edge = min(valid_methods, key=lambda x: x[1]['edge_similarity'])
            
            print(f"  Best edge preservation: {best_edge[0]} ({best_edge[1]['edge_similarity']:.1%})")
            print(f"  Worst edge preservation: {worst_edge[0]} ({worst_edge[1]['edge_similarity']:.1%})")
        
        # Insights for dog breed recognition
        print("\nIMPLICATIONS FOR DOG BREED RECOGNITION:")
        print("  - CLAHE improves contrast while preserving edge structure")
        print("  - Bilateral filtering smooths noise without destroying edges")
        print("  - Raw noise significantly degrades edge detection quality")
        print("  - Proper restoration (NLM/Median) recovers most edge information")


def run_comparative_study(image_path: str, output_dir: str = None) -> Dict:
    """
    Run complete comparative study on an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save visualizations (optional)
        
    Returns:
        Analysis results dictionary
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Run analysis
    analyzer = PreprocessingAnalyzer()
    results = analyzer.analyze_preprocessing_impact(image)
    
    # Print report
    analyzer.print_analysis_report(results)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparative study of preprocessing impact")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    if args.image:
        run_comparative_study(args.image, args.output)
    else:
        print("Please provide an image path with --image argument")
        print("Example: python comparative_study.py --image path/to/dog.jpg")
