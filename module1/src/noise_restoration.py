"""
Noise Modeling and Image Restoration
=====================================

This module simulates real-world image degradation (noise) and implements
restoration filters to recover original image quality.

Noise Types: Gaussian, Salt-and-Pepper, Speckle
Restoration: Gaussian, Median, Bilateral, Non-Local Means
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class NoiseGenerator:
    """
    Generate various types of noise to simulate real-world degradation.
    
    Common noise sources in dog photography:
    - Low light conditions → Gaussian noise
    - Sensor defects → Salt-and-pepper noise
    - Movement/vibration → Motion blur (not noise, but degradation)
    """
    
    @staticmethod
    def gaussian(image: np.ndarray, mean: float = 0, 
                 sigma: float = 25) -> np.ndarray:
        """
        Add Gaussian (normal) noise to image.
        
        Args:
            image: Input image
            mean: Mean of noise distribution (usually 0)
            sigma: Standard deviation (higher = more noise)
                   Typical values: 10 (mild), 25 (moderate), 50 (severe)
                   
        Returns:
            Noisy image
            
        Theory:
            Gaussian noise models thermal and electronic sensor noise.
            Each pixel gets additive noise: I_noisy = I_original + N(mean, σ²)
            
            This is the most common noise model in image processing research.
        """
        noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def salt_and_pepper(image: np.ndarray, prob: float = 0.02) -> np.ndarray:
        """
        Add salt-and-pepper (impulse) noise.
        
        Args:
            image: Input image
            prob: Probability of noise at each pixel (0 to 1)
                  Typical values: 0.01 (mild), 0.05 (moderate), 0.1 (severe)
                  
        Returns:
            Noisy image
            
        Theory:
            Models dead pixels or transmission errors:
            - Salt (white spots): Pixel set to 255
            - Pepper (black spots): Pixel set to 0
            
            Each occurs with probability prob/2.
            Median filter is particularly effective against this noise type.
        """
        noisy = image.copy()
        
        # Generate random numbers for each pixel
        rnd = np.random.random(image.shape[:2])
        
        # Salt (white)
        salt_mask = rnd < prob / 2
        if len(image.shape) == 3:
            noisy[salt_mask] = [255, 255, 255]
        else:
            noisy[salt_mask] = 255
            
        # Pepper (black)
        pepper_mask = (rnd >= prob / 2) & (rnd < prob)
        if len(image.shape) == 3:
            noisy[pepper_mask] = [0, 0, 0]
        else:
            noisy[pepper_mask] = 0
            
        return noisy
    
    @staticmethod
    def speckle(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Add speckle (multiplicative) noise.
        
        Args:
            image: Input image
            sigma: Noise intensity factor
            
        Returns:
            Noisy image
            
        Theory:
            Speckle is multiplicative: I_noisy = I × (1 + N(0, σ²))
            Common in radar/ultrasound imaging, less common in regular photos.
        """
        noise = np.random.randn(*image.shape) * sigma
        noisy = image.astype(np.float32) * (1 + noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def poisson(image: np.ndarray) -> np.ndarray:
        """
        Add Poisson (shot) noise.
        
        Returns:
            Noisy image
            
        Theory:
            Poisson noise models photon counting statistics in cameras.
            Variance equals signal: Var(I) = I
            Thus, brighter regions have more absolute noise but less relative noise.
        """
        # Scale to reasonable photon counts
        noisy = np.random.poisson(image.astype(np.float32))
        return np.clip(noisy, 0, 255).astype(np.uint8)


class ImageRestoration:
    """
    Image restoration filters to remove noise while preserving details.
    
    Trade-off: Noise reduction ↔ Detail preservation (especially edges)
    """
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, ksize: int = 5, 
                      sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.
        
        Args:
            image: Input image
            ksize: Kernel size (must be odd)
            sigma: Gaussian standard deviation (0 = auto from ksize)
            
        Returns:
            Blurred image
            
        Theory:
            Gaussian kernel: G(x,y) = (1/2πσ²) × exp(-(x²+y²)/(2σ²))
            
            Pros: Effectively reduces Gaussian noise
            Cons: Blurs edges along with noise
            
            Not ideal for fine fur texture preservation.
        """
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    @staticmethod
    def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Apply median filter for noise reduction.
        
        Args:
            image: Input image
            ksize: Kernel size (must be odd)
            
        Returns:
            Filtered image
            
        Theory:
            Replaces each pixel with median of neighboring pixels.
            
            Pros: 
            - Excellent for salt-and-pepper noise (outlier removal)
            - Preserves edges better than Gaussian blur
            
            Cons: 
            - Can remove fine texture details
            - Computationally more expensive than linear filters
            
            Good choice for removing impulse noise in dog images while
            keeping body contours sharp.
        """
        return cv2.medianBlur(image, ksize)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, 
                         sigma_color: float = 75, 
                         sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood (use -1 for auto from sigma_space)
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            Filtered image
            
        Theory:
            Bilateral filter combines spatial and intensity domain filtering:
            
            w(i,j,k,l) = exp(-(||p_ij - p_kl||²)/(2σ_s²)) × 
                         exp(-(|I(i,j) - I(k,l)|²)/(2σ_c²))
            
            First term: Spatial proximity (like Gaussian blur)
            Second term: Intensity similarity (preserves edges)
            
            Edges are preserved because dissimilar pixels get low weight.
            
            Excellent for dog images: smooths fur texture while keeping
            clear boundaries around eyes, nose, and body contour.
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def nlm_denoise(image: np.ndarray, h: float = 10, 
                    template_size: int = 7, 
                    search_size: int = 21) -> np.ndarray:
        """
        Apply Non-Local Means denoising.
        
        Args:
            image: Input image
            h: Filter strength. Higher = more noise removal but less detail
            template_size: Size of template patch
            search_size: Size of area to search for similar patches
            
        Returns:
            Denoised image
            
        Theory:
            NLM exploits self-similarity in images:
            - For each pixel, find similar patches elsewhere in image
            - Average weighted by patch similarity
            
            w(i,j) = exp(-||P(i) - P(j)||² / h²)
            
            Pros: State-of-the-art noise reduction quality
            Cons: Computationally expensive
            
            Excellent for preserving repetitive texture patterns like fur.
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_size, search_size
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, None, h, template_size, search_size
            )
    
    @staticmethod
    def adaptive_median(image: np.ndarray, max_ksize: int = 7) -> np.ndarray:
        """
        Apply adaptive median filter.
        
        Args:
            image: Input image
            max_ksize: Maximum kernel size
            
        Returns:
            Filtered image
            
        Theory:
            Adaptive median increases window size until it finds
            a median that is not itself an impulse.
            
            Better than fixed median for varying noise density.
        """
        # This is a simplified implementation
        # Full adaptive median requires pixel-by-pixel processing
        result = cv2.medianBlur(image, 3)
        
        # If still noisy, apply larger kernels
        for ksize in range(5, max_ksize + 1, 2):
            result = cv2.medianBlur(result, ksize)
            
        return result


def add_noise_and_restore(image: np.ndarray, noise_type: str = 'gaussian',
                          noise_params: dict = None,
                          filter_type: str = 'bilateral',
                          filter_params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise and then restore image. Returns both for comparison.
    
    Args:
        image: Clean input image
        noise_type: 'gaussian', 'salt_pepper', 'speckle', or 'poisson'
        noise_params: Parameters for noise generation
        filter_type: 'gaussian', 'median', 'bilateral', or 'nlm'
        filter_params: Parameters for restoration filter
        
    Returns:
        (noisy_image, restored_image) tuple
    """
    noise_params = noise_params or {}
    filter_params = filter_params or {}
    
    # Add noise
    noise_funcs = {
        'gaussian': NoiseGenerator.gaussian,
        'salt_pepper': NoiseGenerator.salt_and_pepper,
        'speckle': NoiseGenerator.speckle,
        'poisson': NoiseGenerator.poisson
    }
    noisy = noise_funcs[noise_type](image, **noise_params)
    
    # Restore
    filter_funcs = {
        'gaussian': ImageRestoration.gaussian_blur,
        'median': ImageRestoration.median_filter,
        'bilateral': ImageRestoration.bilateral_filter,
        'nlm': ImageRestoration.nlm_denoise
    }
    restored = filter_funcs[filter_type](noisy, **filter_params)
    
    return noisy, restored


# Convenience functions
def add_gaussian_noise(image, sigma=25):
    return NoiseGenerator.gaussian(image, sigma=sigma)

def add_salt_pepper_noise(image, prob=0.02):
    return NoiseGenerator.salt_and_pepper(image, prob)

def denoise_bilateral(image):
    return ImageRestoration.bilateral_filter(image)

def denoise_nlm(image):
    return ImageRestoration.nlm_denoise(image)
