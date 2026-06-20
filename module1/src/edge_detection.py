"""
Edge Detection Methods for Feature Boundary Analysis
=====================================================

This module implements edge detection algorithms to analyze feature boundaries
in dog images (fur texture, body contours, facial features).

Implements: Sobel, Canny, Laplacian operators with tunable parameters.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class EdgeDetector:
    """
    Edge detection using various differential operators.
    
    Edges represent significant local changes in image intensity,
    corresponding to boundaries between objects or texture patterns.
    """
    
    @staticmethod
    def sobel(image: np.ndarray, ksize: int = 3, 
              direction: str = 'both') -> np.ndarray:
        """
        Apply Sobel edge detection.
        
        Args:
            image: Input image (grayscale or BGR)
            ksize: Kernel size (1, 3, 5, or 7). Larger = smoother gradients
            direction: 'x' (vertical edges), 'y' (horizontal edges), or 'both'
            
        Returns:
            Edge magnitude image
            
        Theory:
            Sobel operator approximates image gradient using convolution:
            
            Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
                  [-2, 0, 2],            [ 0,  0,  0],
                  [-1, 0, 1]]            [ 1,  2,  1]]
            
            Gradient magnitude: G = sqrt(Gx² + Gy²)
            
            For dog images, Sobel detects:
            - Fur texture boundaries (fine edges)
            - Body silhouette (strong edges)
            - Facial features (eyes, nose, ears)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if direction == 'x':
            return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
        elif direction == 'y':
            return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
        else:  # both
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return cv2.convertScaleAbs(magnitude)
    
    @staticmethod
    def canny(image: np.ndarray, low_threshold: int = 50, 
              high_threshold: int = 150, aperture_size: int = 3) -> np.ndarray:
        """
        Apply Canny edge detection.
        
        Args:
            image: Input image
            low_threshold: Lower threshold for hysteresis
            high_threshold: Upper threshold for hysteresis
            aperture_size: Sobel kernel size
            
        Returns:
            Binary edge image (0 or 255)
            
        Theory:
            Canny is a multi-stage algorithm:
            1. Gaussian blur to reduce noise
            2. Gradient calculation (Sobel)
            3. Non-maximum suppression (thin edges)
            4. Hysteresis thresholding:
               - Pixels > high_threshold: Strong edges (kept)
               - Pixels < low_threshold: Discarded
               - In between: Kept if connected to strong edge
            
            Recommended ratio: high_threshold = 2-3 × low_threshold
            
            For dog breed recognition:
            - Low thresholds capture fur texture (useful for texture analysis)
            - High thresholds capture body outline (useful for shape analysis)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        edges = cv2.Canny(blurred, low_threshold, high_threshold, 
                          apertureSize=aperture_size)
        return edges
    
    @staticmethod
    def laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Apply Laplacian edge detection (second derivative).
        
        Args:
            image: Input image
            ksize: Kernel size
            
        Returns:
            Edge image
            
        Theory:
            Laplacian = ∂²f/∂x² + ∂²f/∂y²
            
            Detects edges in all directions (isotropic) but is sensitive to noise.
            Zero-crossings of Laplacian indicate edge locations.
            
            Less commonly used than Canny for edge detection, but useful
            for finding regions of rapid intensity change.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Blur first to reduce noise sensitivity
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
        return cv2.convertScaleAbs(laplacian)
    
    @staticmethod
    def log(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Laplacian of Gaussian (LoG) edge detection.
        
        Args:
            image: Input image
            sigma: Gaussian blur standard deviation
            
        Returns:
            Edge image
            
        Theory:
            LoG combines Gaussian smoothing with Laplacian:
            - Gaussian: Reduces noise
            - Laplacian: Detects edges
            
            LoG(x,y) = -1/(πσ⁴) × [1 - (x² + y²)/(2σ²)] × exp(-(x² + y²)/(2σ²))
            
            Also known as the "Mexican hat" filter due to its shape.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Generate LoG kernel
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
            
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        log_result = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.convertScaleAbs(log_result)
    
    @staticmethod
    def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        Automatic Canny edge detection with adaptive thresholds.
        
        Args:
            image: Input image
            sigma: Threshold spread factor (higher = wider range)
            
        Returns:
            Binary edge image
            
        Theory:
            Automatically selects thresholds based on median pixel intensity:
            - lower = (1 - sigma) × median
            - upper = (1 + sigma) × median
            
            This adaptive approach works well across varying image conditions.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        median = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        return cv2.Canny(gray, lower, upper)


def compute_edge_density(edge_image: np.ndarray) -> float:
    """
    Compute the edge density (proportion of edge pixels).
    
    Useful for characterizing image complexity.
    Higher density = more texture/detail.
    """
    return np.sum(edge_image > 0) / edge_image.size


def compute_edge_orientation_histogram(image: np.ndarray, 
                                        num_bins: int = 8) -> np.ndarray:
    """
    Compute histogram of edge orientations.
    
    Args:
        image: Input image
        num_bins: Number of orientation bins
        
    Returns:
        Normalized orientation histogram
        
    Theory:
        Edge orientation = arctan(Gy / Gx)
        Distribution of orientations characterizes texture patterns.
        
        For dog images:
        - Random fur: Uniform orientation distribution
        - Directional fur: Peaked orientation distribution
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)
    
    # Convert to degrees [0, 180) - edges are undirected
    orientation_deg = np.degrees(orientation) % 180
    
    # Compute magnitude-weighted histogram
    hist, _ = np.histogram(orientation_deg, bins=num_bins, range=(0, 180),
                           weights=magnitude)
    
    # Normalize
    hist = hist / (np.sum(hist) + 1e-7)
    return hist


# Convenience functions
def sobel_edges(image, ksize=3):
    return EdgeDetector.sobel(image, ksize)

def canny_edges(image, low=50, high=150):
    return EdgeDetector.canny(image, low, high)

def auto_canny_edges(image):
    return EdgeDetector.auto_canny(image)
