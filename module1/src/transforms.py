"""
Geometric and Intensity Transformations for Image Processing
=============================================================

This module implements various image transformations to study their effects
on image interpretability and downstream vision tasks.

Geometric Transforms: Rotation, Scaling, Flipping
Intensity Transforms: Histogram Equalization, Gamma Correction, Contrast Adjustment
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class GeometricTransforms:
    """
    Geometric transformations that modify spatial arrangement of pixels.
    These help study invariance properties of feature detectors.
    """
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image (BGR or grayscale)
            angle: Rotation angle in degrees (positive = counter-clockwise)
            center: Center of rotation. If None, uses image center.
            
        Returns:
            Rotated image with same dimensions
            
        Theory:
            Rotation matrix R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
            Used to study rotation invariance of descriptors like SIFT.
        """
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    @staticmethod
    def scale(image: np.ndarray, scale_x: float, scale_y: Optional[float] = None) -> np.ndarray:
        """
        Scale image by specified factors.
        
        Args:
            image: Input image
            scale_x: Horizontal scaling factor (>1 enlarges, <1 shrinks)
            scale_y: Vertical scaling factor. If None, uses scale_x (uniform scaling)
            
        Returns:
            Scaled image
            
        Theory:
            Scaling matrix S = [[sx, 0], [0, sy]]
            Non-uniform scaling (sx ≠ sy) tests descriptor robustness to aspect changes.
        """
        if scale_y is None:
            scale_y = scale_x
            
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale_x), int(h * scale_y)
        
        # Use appropriate interpolation based on scaling direction
        interpolation = cv2.INTER_CUBIC if scale_x > 1 else cv2.INTER_AREA
        scaled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return scaled
    
    @staticmethod
    def flip(image: np.ndarray, mode: str = 'horizontal') -> np.ndarray:
        """
        Flip image horizontally or vertically.
        
        Args:
            image: Input image
            mode: 'horizontal', 'vertical', or 'both'
            
        Returns:
            Flipped image
            
        Theory:
            Flipping creates mirror images. SIFT achieves flip invariance
            through gradient orientation normalization.
        """
        flip_codes = {
            'horizontal': 1,
            'vertical': 0,
            'both': -1
        }
        if mode not in flip_codes:
            raise ValueError(f"Mode must be one of {list(flip_codes.keys())}")
        
        return cv2.flip(image, flip_codes[mode])
    
    @staticmethod
    def translate(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """
        Translate (shift) image by specified pixels.
        
        Args:
            image: Input image
            tx: Horizontal translation (positive = right)
            ty: Vertical translation (positive = down)
            
        Returns:
            Translated image
        """
        h, w = image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, translation_matrix, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)
        return translated
    
    @staticmethod
    def affine_transform(image: np.ndarray, src_points: np.ndarray, 
                         dst_points: np.ndarray) -> np.ndarray:
        """
        Apply general affine transformation defined by point correspondences.
        
        Args:
            image: Input image
            src_points: 3x2 array of source points
            dst_points: 3x2 array of destination points
            
        Returns:
            Affine transformed image
            
        Theory:
            Affine transformations preserve parallel lines but not angles.
            Combines rotation, scaling, shearing, and translation.
        """
        h, w = image.shape[:2]
        affine_matrix = cv2.getAffineTransform(
            src_points.astype(np.float32), 
            dst_points.astype(np.float32)
        )
        transformed = cv2.warpAffine(image, affine_matrix, (w, h),
                                      borderMode=cv2.BORDER_REFLECT)
        return transformed


class IntensityTransforms:
    """
    Intensity transformations that modify pixel values without changing geometry.
    These help study illumination invariance and enhance image quality.
    """
    
    @staticmethod
    def histogram_equalization(image: np.ndarray, method: str = 'global') -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image: Input image (BGR or grayscale)
            method: 'global' or 'clahe' (Contrast Limited Adaptive HE)
            
        Returns:
            Contrast-enhanced image
            
        Theory:
            Global HE: Maps histogram to uniform distribution using CDF transform.
            CLAHE: Applies HE locally with contrast limiting to prevent noise amplification.
            
            For dog images, CLAHE often works better as it preserves local texture details
            while enhancing overall contrast.
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image.copy()
        
        if method == 'global':
            equalized = cv2.equalizeHist(l_channel)
        elif method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(l_channel)
        else:
            raise ValueError("Method must be 'global' or 'clahe'")
        
        if len(image.shape) == 3:
            lab[:, :, 0] = equalized
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return equalized
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction for brightness adjustment.
        
        Args:
            image: Input image
            gamma: Gamma value. 
                   gamma < 1: Brightens image (enhances dark regions)
                   gamma > 1: Darkens image (enhances bright regions)
                   gamma = 1: No change
                   
        Returns:
            Gamma-corrected image
            
        Theory:
            Output = 255 * (Input / 255)^(1/gamma)
            
            This is a non-linear transformation that mimics human visual perception.
            Useful for correcting images taken under different lighting conditions.
        """
        # Build lookup table for efficiency
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in np.arange(256)
        ]).astype(np.uint8)
        
        return cv2.LUT(image, table)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, beta: int) -> np.ndarray:
        """
        Adjust image brightness by adding constant value.
        
        Args:
            image: Input image
            beta: Brightness offset (-255 to 255)
            
        Returns:
            Brightness-adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
        """
        Adjust image contrast by scaling pixel values.
        
        Args:
            image: Input image
            alpha: Contrast factor.
                   alpha < 1: Reduces contrast
                   alpha > 1: Increases contrast
                   
        Returns:
            Contrast-adjusted image
            
        Theory:
            Output = alpha * (Input - mean) + mean
            This preserves mean intensity while modifying dynamic range.
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    @staticmethod
    def normalize(image: np.ndarray, norm_type: str = 'minmax') -> np.ndarray:
        """
        Normalize image intensities.
        
        Args:
            image: Input image
            norm_type: 'minmax' (0-255 range) or 'zscore' (zero mean, unit std)
            
        Returns:
            Normalized image
        """
        if norm_type == 'minmax':
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        elif norm_type == 'zscore':
            mean, std = cv2.meanStdDev(image)
            normalized = (image.astype(np.float32) - mean) / (std + 1e-7)
            # Scale back to 0-255 range for visualization
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)
        else:
            raise ValueError("norm_type must be 'minmax' or 'zscore'")


def apply_transform_pipeline(image: np.ndarray, transforms: list) -> np.ndarray:
    """
    Apply a sequence of transformations to an image.
    
    Args:
        image: Input image
        transforms: List of (transform_func, kwargs) tuples
        
    Returns:
        Transformed image
        
    Example:
        pipeline = [
            (GeometricTransforms.rotate, {'angle': 15}),
            (IntensityTransforms.gamma_correction, {'gamma': 1.2}),
        ]
        result = apply_transform_pipeline(image, pipeline)
    """
    result = image.copy()
    for transform_func, kwargs in transforms:
        result = transform_func(result, **kwargs)
    return result


# Convenience functions for quick access
def rotate(image, angle): 
    return GeometricTransforms.rotate(image, angle)

def scale(image, factor): 
    return GeometricTransforms.scale(image, factor)

def flip_horizontal(image): 
    return GeometricTransforms.flip(image, 'horizontal')

def flip_vertical(image): 
    return GeometricTransforms.flip(image, 'vertical')

def histogram_eq(image, method='clahe'): 
    return IntensityTransforms.histogram_equalization(image, method)

def gamma_correct(image, gamma): 
    return IntensityTransforms.gamma_correction(image, gamma)
