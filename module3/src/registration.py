"""
Module 3: Image Registration Utilities
Geometric alignment for pose-invariant breed comparison
"""

import cv2
import numpy as np


def center_crop_square(image):
    """
    Center crop image to square.
    
    Args:
        image: Input image [H, W, C]
    
    Returns:
        Square cropped image
    """
    h, w = image.shape[:2]
    size = min(h, w)
    y = (h - size) // 2
    x = (w - size) // 2
    return image[y:y+size, x:x+size]


def resize_image(image, target_size=(224, 224)):
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target (width, height)
    
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def align_image(image, target_size=(224, 224)):
    """
    Align image: center crop to square then resize.
    
    Args:
        image: Input image
        target_size: Target size
    
    Returns:
        Aligned image
    """
    cropped = center_crop_square(image)
    resized = resize_image(cropped, target_size)
    return resized


def compute_affine_transform(src_points, dst_points):
    """
    Compute affine transformation matrix.
    
    Args:
        src_points: Source points (3 points, 2D)
        dst_points: Destination points (3 points, 2D)
    
    Returns:
        2x3 affine transformation matrix
    """
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    return cv2.getAffineTransform(src, dst)


def apply_affine_transform(image, matrix, output_size):
    """
    Apply affine transformation to image.
    
    Args:
        image: Input image
        matrix: 2x3 affine transformation matrix
        output_size: Output (width, height)
    
    Returns:
        Transformed image
    """
    return cv2.warpAffine(image, matrix, output_size)


def normalize_pose(image, keypoints=None, target_size=(224, 224)):
    """
    Normalize dog pose using keypoints (if available) or simple alignment.
    
    Args:
        image: Input image
        keypoints: Optional keypoints dict with 'left_eye', 'right_eye', 'nose'
        target_size: Target output size
    
    Returns:
        Pose-normalized image
    """
    if keypoints is None:
        # Simple alignment without keypoints
        return align_image(image, target_size)
    
    # Keypoint-based alignment
    # Define canonical positions (normalized)
    w, h = target_size
    canonical = {
        'left_eye': (w * 0.35, h * 0.35),
        'right_eye': (w * 0.65, h * 0.35),
        'nose': (w * 0.5, h * 0.6)
    }
    
    src_pts = np.float32([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose']
    ])
    dst_pts = np.float32([
        canonical['left_eye'],
        canonical['right_eye'],
        canonical['nose']
    ])
    
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(image, matrix, target_size)
    
    return aligned


def batch_align_images(images, target_size=(224, 224)):
    """
    Align a batch of images.
    
    Args:
        images: List of images
        target_size: Target size
    
    Returns:
        List of aligned images
    """
    return [align_image(img, target_size) for img in images]
