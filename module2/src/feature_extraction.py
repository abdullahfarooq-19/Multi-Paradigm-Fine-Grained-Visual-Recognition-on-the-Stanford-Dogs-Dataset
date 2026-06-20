"""
Feature Extraction: SIFT, HOG, and LBP
======================================

Implements classical feature descriptors for dog breed recognition:
- SIFT: Scale-Invariant Feature Transform (keypoint-based)
- HOG: Histogram of Oriented Gradients (shape descriptor)
- LBP: Local Binary Patterns (texture descriptor)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from skimage.feature import hog, local_binary_pattern


class SIFTExtractor:
    """
    SIFT (Scale-Invariant Feature Transform) feature extraction.
    
    SIFT detects keypoints and computes descriptors that are invariant to:
    - Scale changes
    - Rotation
    - Illumination changes (partially)
    
    For classification, we use Bag of Visual Words (BoVW) approach:
    1. Extract SIFT descriptors from all training images
    2. Cluster descriptors to create visual vocabulary
    3. Represent each image as histogram of visual words
    """
    
    def __init__(self, n_features: int = 500):
        """
        Args:
            n_features: Maximum number of keypoints to detect
        """
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.n_features = n_features
    
    def extract(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            (keypoints, descriptors) where descriptors is Nx128 array
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        
        return keypoints, descriptors
    
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """Extract SIFT descriptors from image file."""
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros((0, 128), dtype=np.float32)
        
        _, descriptors = self.extract(image)
        return descriptors


class HOGExtractor:
    """
    HOG (Histogram of Oriented Gradients) feature extraction.
    
    HOG captures shape information by computing gradient orientations
    in local cells and normalizing within blocks.
    
    Good for capturing:
    - Body silhouette
    - Ear shape
    - Snout shape
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (128, 128),
                 orientations: int = 9,
                 pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Args:
            image_size: Resize images to this size
            orientations: Number of orientation bins
            pixels_per_cell: Size of each cell in pixels
            cells_per_block: Number of cells per block
        """
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
        # Calculate feature vector length
        n_cells = (image_size[0] // pixels_per_cell[0], 
                   image_size[1] // pixels_per_cell[1])
        n_blocks = (n_cells[0] - cells_per_block[0] + 1,
                    n_cells[1] - cells_per_block[1] + 1)
        self.feature_length = (n_blocks[0] * n_blocks[1] * 
                               cells_per_block[0] * cells_per_block[1] * 
                               orientations)
    
    def extract(self, image: np.ndarray, visualize: bool = False):
        """
        Extract HOG features from image.
        
        Args:
            image: Input image
            visualize: If True, also return HOG visualization
            
        Returns:
            HOG feature vector (1D array) or (features, hog_image) if visualize=True
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to fixed size
        resized = cv2.resize(gray, self.image_size)
        
        # Extract HOG
        if visualize:
            features, hog_image = hog(
                resized,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=True,
                feature_vector=True
            )
            return features, hog_image
        else:
            features = hog(
                resized,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                feature_vector=True
            )
            return features
    
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """Extract HOG features from image file."""
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(self.feature_length, dtype=np.float32)
        
        return self.extract(image)


class LBPExtractor:
    """
    LBP (Local Binary Patterns) feature extraction.
    
    LBP captures texture information by comparing each pixel to its neighbors.
    The resulting binary pattern encodes local texture structure.
    
    Good for capturing:
    - Fur texture patterns
    - Coat type (fluffy vs smooth)
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (128, 128),
                 n_points: int = 24,
                 radius: int = 3,
                 method: str = 'uniform',
                 n_bins: int = 26):
        """
        Args:
            image_size: Resize images to this size
            n_points: Number of circularly symmetric neighbor points
            radius: Radius of circle for neighbor sampling
            method: LBP method ('uniform' reduces histogram size)
            n_bins: Number of histogram bins
        """
        self.image_size = image_size
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.n_bins = n_bins
        
        # For spatial pyramid, we compute LBP for regions
        self.n_regions = 4  # 2x2 grid
        self.feature_length = n_bins * self.n_regions
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP histogram features from image.
        
        Uses spatial pyramid: image divided into 2x2 grid,
        LBP histogram computed for each region and concatenated.
        
        Args:
            image: Input image
            
        Returns:
            LBP histogram feature vector
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to fixed size
        resized = cv2.resize(gray, self.image_size)
        
        # Compute LBP
        lbp = local_binary_pattern(resized, self.n_points, self.radius, 
                                    method=self.method)
        
        # Spatial pyramid - 2x2 regions
        h, w = resized.shape
        regions = [
            lbp[:h//2, :w//2],       # Top-left
            lbp[:h//2, w//2:],       # Top-right
            lbp[h//2:, :w//2],       # Bottom-left
            lbp[h//2:, w//2:],       # Bottom-right
        ]
        
        # Compute histogram for each region
        histograms = []
        for region in regions:
            hist, _ = np.histogram(region.ravel(), bins=self.n_bins, 
                                   range=(0, self.n_bins))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)  # Normalize
            histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """Extract LBP features from image file."""
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(self.feature_length, dtype=np.float32)
        
        return self.extract(image)


class FeatureFusion:
    """
    Combines multiple feature extractors for improved representation.
    
    Strategy: Concatenate normalized feature vectors from HOG and LBP.
    SIFT is handled separately via Bag of Visual Words.
    """
    
    def __init__(self, 
                 use_hog: bool = True,
                 use_lbp: bool = True,
                 hog_params: dict = None,
                 lbp_params: dict = None):
        """
        Args:
            use_hog: Include HOG features
            use_lbp: Include LBP features
            hog_params: Parameters for HOG extractor
            lbp_params: Parameters for LBP extractor
        """
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        
        if use_hog:
            self.hog = HOGExtractor(**(hog_params or {}))
        if use_lbp:
            self.lbp = LBPExtractor(**(lbp_params or {}))
        
        # Calculate total feature length
        self.feature_length = 0
        if use_hog:
            self.feature_length += self.hog.feature_length
        if use_lbp:
            self.feature_length += self.lbp.feature_length
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract fused features from image.
        
        Returns:
            Concatenated feature vector
        """
        features = []
        
        if self.use_hog:
            hog_feat = self.hog.extract(image)
            # L2 normalize
            hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-7)
            features.append(hog_feat)
        
        if self.use_lbp:
            lbp_feat = self.lbp.extract(image)
            # Already normalized in LBP extractor
            features.append(lbp_feat)
        
        return np.concatenate(features)
    
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """Extract fused features from image file."""
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(self.feature_length, dtype=np.float32)
        
        return self.extract(image)


def extract_features_batch(image_paths: list, extractor, verbose: bool = True):
    """
    Extract features from multiple images.
    
    Args:
        image_paths: List of image file paths
        extractor: Feature extractor instance (HOG, LBP, or FeatureFusion)
        verbose: Print progress
        
    Returns:
        numpy array of shape (n_images, feature_length)
    """
    features = []
    
    for i, path in enumerate(image_paths):
        feat = extractor.extract_from_path(path)
        features.append(feat)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} images")
    
    return np.array(features)


if __name__ == "__main__":
    # Quick test
    import os
    
    print("Feature Extraction Test")
    print("=" * 50)
    
    # Test with a sample image
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(root_path, "images", "Images", "n02085620-Chihuahua")
    
    if os.path.exists(sample_path):
        sample_image = os.path.join(sample_path, os.listdir(sample_path)[0])
        image = cv2.imread(sample_image)
        
        print(f"Sample image: {sample_image}")
        print(f"Image shape: {image.shape}")
        
        # Test SIFT
        sift = SIFTExtractor()
        kps, descs = sift.extract(image)
        print(f"\nSIFT: {len(kps)} keypoints, descriptor shape: {descs.shape}")
        
        # Test HOG
        hog_ext = HOGExtractor()
        hog_feat = hog_ext.extract(image)
        print(f"HOG: feature length: {len(hog_feat)}")
        
        # Test LBP
        lbp_ext = LBPExtractor()
        lbp_feat = lbp_ext.extract(image)
        print(f"LBP: feature length: {len(lbp_feat)}")
        
        # Test Fusion
        fusion = FeatureFusion()
        fused_feat = fusion.extract(image)
        print(f"Fused (HOG+LBP): feature length: {len(fused_feat)}")
    else:
        print("Sample path not found. Please check dataset location.")
