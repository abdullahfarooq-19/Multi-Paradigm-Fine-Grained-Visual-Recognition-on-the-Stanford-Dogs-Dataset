"""
Image Quality Metrics
=====================

Implements metrics for quantitative comparison of image processing results.
Used to scientifically evaluate which restoration filter performs best.

Metrics: PSNR, SSIM, MSE
"""

import cv2
import numpy as np
from typing import Tuple


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.
    
    Args:
        image1: First image (usually the reference/original)
        image2: Second image (usually the processed/restored)
        
    Returns:
        MSE value (lower is better, 0 = identical)
        
    Theory:
        MSE = (1/N) × Σ(I1(i) - I2(i))²
        
        Simple pixel-wise difference metric.
        Limitation: Doesn't account for structural/perceptual similarity.
    """
    # Ensure same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    err = np.sum((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    err /= float(image1.size)
    return err


def psnr(original: np.ndarray, processed: np.ndarray, 
         max_pixel: int = 255) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        original: Reference image
        processed: Processed image
        max_pixel: Maximum possible pixel value (255 for 8-bit)
        
    Returns:
        PSNR value in dB (higher is better)
        
    Theory:
        PSNR = 10 × log10(MAX² / MSE)
        
        Typical values:
        - PSNR > 40 dB: Excellent (nearly indistinguishable)
        - PSNR 30-40 dB: Good
        - PSNR 20-30 dB: Acceptable
        - PSNR < 20 dB: Poor
        
        For restoration evaluation:
        - Compare PSNR of restored image vs original
        - Higher PSNR = better restoration
        
    Note:
        PSNR = ∞ when images are identical (MSE = 0)
    """
    if original.shape != processed.shape:
        raise ValueError("Images must have the same dimensions")
    
    mse_val = mse(original, processed)
    
    if mse_val == 0:
        return float('inf')  # Identical images
    
    return 10 * np.log10((max_pixel ** 2) / mse_val)


def ssim(image1: np.ndarray, image2: np.ndarray, 
         window_size: int = 11, 
         k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        image1: First image
        image2: Second image  
        window_size: Size of sliding window
        k1, k2: Stability constants
        
    Returns:
        SSIM value between -1 and 1 (1 = identical, higher is better)
        
    Theory:
        SSIM combines three comparisons:
        1. Luminance: l(x,y) = (2μx × μy + C1) / (μx² + μy² + C1)
        2. Contrast: c(x,y) = (2σx × σy + C2) / (σx² + σy² + C2)
        3. Structure: s(x,y) = (σxy + C3) / (σx × σy + C3)
        
        SSIM = l × c × s
        
        Where:
        - μ = mean
        - σ = standard deviation
        - σxy = covariance
        - C1, C2, C3 = stabilization constants
        
        SSIM better correlates with human perception than PSNR because it
        considers structural information, not just pixel differences.
    """
    # Convert to grayscale if color
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # Constants
    L = 255  # Dynamic range
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2
    
    # Create Gaussian window
    gaussian = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(gaussian, gaussian.T)
    
    # Compute means
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    # Compute SSIM components
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map))


def compute_all_metrics(original: np.ndarray, 
                        processed: np.ndarray) -> dict:
    """
    Compute all quality metrics at once.
    
    Args:
        original: Reference image
        processed: Processed image
        
    Returns:
        Dictionary with MSE, PSNR, and SSIM values
    """
    return {
        'mse': mse(original, processed),
        'psnr': psnr(original, processed),
        'ssim': ssim(original, processed)
    }


def compare_filters(original: np.ndarray, noisy: np.ndarray,
                    restored_images: dict) -> dict:
    """
    Compare multiple restoration filters using quality metrics.
    
    Args:
        original: Clean reference image
        noisy: Noisy image
        restored_images: Dict of {filter_name: restored_image}
        
    Returns:
        Dict of {filter_name: {mse, psnr, ssim}}
        
    Example:
        restored = {
            'gaussian': gaussian_restored,
            'median': median_restored,
            'bilateral': bilateral_restored
        }
        results = compare_filters(original, noisy, restored)
        print(results)
        # {'gaussian': {'mse': 45.2, 'psnr': 31.5, 'ssim': 0.89}, ...}
    """
    results = {
        'noisy': compute_all_metrics(original, noisy)
    }
    
    for name, restored in restored_images.items():
        results[name] = compute_all_metrics(original, restored)
    
    return results


def print_comparison_table(results: dict) -> None:
    """
    Print a formatted comparison table of filter results.
    """
    print("\n" + "="*60)
    print("Filter Comparison Results")
    print("="*60)
    print(f"{'Filter':<15} {'MSE':>12} {'PSNR (dB)':>12} {'SSIM':>12}")
    print("-"*60)
    
    for name, metrics in results.items():
        psnr_val = f"{metrics['psnr']:.2f}" if metrics['psnr'] != float('inf') else "∞"
        print(f"{name:<15} {metrics['mse']:>12.2f} {psnr_val:>12} {metrics['ssim']:>12.4f}")
    
    print("="*60)
    
    # Find best filter (excluding noisy baseline)
    filter_results = {k: v for k, v in results.items() if k != 'noisy'}
    best_psnr = max(filter_results.keys(), key=lambda x: filter_results[x]['psnr'])
    best_ssim = max(filter_results.keys(), key=lambda x: filter_results[x]['ssim'])
    
    print(f"\nBest PSNR: {best_psnr}")
    print(f"Best SSIM: {best_ssim}")
