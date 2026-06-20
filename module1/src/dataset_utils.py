"""
Dataset Utilities for Stanford Dogs Dataset
============================================

Utilities for loading, exploring, and sampling from the Stanford Dogs dataset.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import random


class StanfordDogsDataset:
    """
    Utility class for working with Stanford Dogs dataset.
    
    Dataset Structure:
        images/Images/
            n02085620-Chihuahua/
                n02085620_10074.jpg
                n02085620_10131.jpg
                ...
            n02085782-Japanese_spaniel/
                ...
    """
    
    def __init__(self, root_path: str):
        """
        Initialize dataset loader.
        
        Args:
            root_path: Path to CV-Project directory (e.g., D:/CV-Project)
        """
        self.root_path = root_path
        self.images_path = os.path.join(root_path, "images", "Images")
        
        if not os.path.exists(self.images_path):
            raise ValueError(f"Images directory not found: {self.images_path}")
        
        # Load breed information
        self.breeds = self._load_breeds()
        self.breed_names = [b['name'] for b in self.breeds]
        
    def _load_breeds(self) -> List[Dict]:
        """Load list of breeds from directory structure."""
        breeds = []
        
        for folder in sorted(os.listdir(self.images_path)):
            folder_path = os.path.join(self.images_path, folder)
            if os.path.isdir(folder_path):
                # Extract breed name from folder (e.g., "n02085620-Chihuahua" -> "Chihuahua")
                parts = folder.split("-", 1)
                breed_id = parts[0]
                breed_name = parts[1].replace("_", " ") if len(parts) > 1 else folder
                
                # Count images
                images = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                breeds.append({
                    'id': breed_id,
                    'name': breed_name,
                    'folder': folder,
                    'path': folder_path,
                    'num_images': len(images)
                })
        
        return breeds
    
    def get_dataset_stats(self) -> Dict:
        """Get overall dataset statistics."""
        total_images = sum(b['num_images'] for b in self.breeds)
        
        return {
            'num_breeds': len(self.breeds),
            'total_images': total_images,
            'avg_images_per_breed': total_images / len(self.breeds) if self.breeds else 0,
            'min_images': min(b['num_images'] for b in self.breeds) if self.breeds else 0,
            'max_images': max(b['num_images'] for b in self.breeds) if self.breeds else 0
        }
    
    def get_breed_by_name(self, name: str) -> Optional[Dict]:
        """Find breed by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for breed in self.breeds:
            if name_lower in breed['name'].lower():
                return breed
        return None
    
    def get_sample_image(self, breed_name: str = None) -> Tuple[np.ndarray, str]:
        """
        Get a sample image from the dataset.
        
        Args:
            breed_name: Specific breed to sample from (optional)
            
        Returns:
            (image, image_path) tuple
        """
        if breed_name:
            breed = self.get_breed_by_name(breed_name)
            if not breed:
                raise ValueError(f"Breed not found: {breed_name}")
        else:
            breed = random.choice(self.breeds)
        
        # Get random image from breed folder
        images = [f for f in os.listdir(breed['path']) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            raise ValueError(f"No images found for breed: {breed['name']}")
        
        image_file = random.choice(images)
        image_path = os.path.join(breed['path'], image_file)
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image, image_path
    
    def get_sample_images(self, n: int = 5, breed_name: str = None) -> List[Tuple[np.ndarray, str]]:
        """
        Get multiple sample images.
        
        Args:
            n: Number of images to return
            breed_name: If specified, sample only from this breed
            
        Returns:
            List of (image, image_path) tuples
        """
        samples = []
        for _ in range(n):
            try:
                sample = self.get_sample_image(breed_name)
                samples.append(sample)
            except ValueError:
                continue
        return samples
    
    def get_images_from_breed(self, breed_name: str, 
                               max_images: int = None) -> List[str]:
        """
        Get all image paths from a specific breed.
        
        Args:
            breed_name: Breed name to get images from
            max_images: Maximum number of images to return
            
        Returns:
            List of image paths
        """
        breed = self.get_breed_by_name(breed_name)
        if not breed:
            raise ValueError(f"Breed not found: {breed_name}")
        
        images = [os.path.join(breed['path'], f) 
                 for f in os.listdir(breed['path'])
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images:
            images = images[:max_images]
        
        return images
    
    def print_dataset_summary(self) -> None:
        """Print dataset summary to console."""
        stats = self.get_dataset_stats()
        
        print("\n" + "="*60)
        print("Stanford Dogs Dataset Summary")
        print("="*60)
        print(f"Number of breeds: {stats['num_breeds']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Average images per breed: {stats['avg_images_per_breed']:.1f}")
        print(f"Min images per breed: {stats['min_images']}")
        print(f"Max images per breed: {stats['max_images']}")
        print("="*60)
        
        print("\nSample breeds:")
        for breed in self.breeds[:5]:
            print(f"  - {breed['name']}: {breed['num_images']} images")
        print(f"  ... and {len(self.breeds) - 5} more breeds")


def load_image(path: str) -> np.ndarray:
    """Load an image from path."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Save an image to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def resize_for_display(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """Resize image for display while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    return cv2.resize(image, (new_w, new_h))


if __name__ == "__main__":
    # Quick test
    import sys
    
    # Adjust path as needed
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        dataset = StanfordDogsDataset(root_path)
        dataset.print_dataset_summary()
        
        # Get a sample image
        image, path = dataset.get_sample_image()
        print(f"\nSample image: {path}")
        print(f"Image shape: {image.shape}")
    except Exception as e:
        print(f"Error: {e}")
