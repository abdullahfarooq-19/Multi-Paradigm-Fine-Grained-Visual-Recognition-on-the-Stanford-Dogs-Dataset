"""
50-Breed Subset Selection for Module 2
======================================

Selects 50 diverse breeds from the Stanford Dogs Dataset for classical
feature-based classification. The subset is chosen to include:
- Visual diversity (different sizes, colors, coat types)
- Similar breed pairs (to test fine-grained discrimination)
- Sufficient samples per class
"""

# 50 breeds selected for maximum visual diversity and challenge
# Includes similar pairs: (Husky vs Malamute), (Golden vs Labrador), etc.

SELECTED_BREEDS = [
    # Small dogs (10)
    "Chihuahua",
    "Japanese_spaniel",
    "Maltese_dog",
    "Pomeranian",
    "toy_poodle",
    "Shih-Tzu",
    "papillon",          # Fixed: lowercase
    "Yorkshire_terrier",
    "miniature_pinscher",
    "pug",
    
    # Medium dogs (15)
    "beagle",
    "basset",
    "cocker_spaniel",
    "Border_collie",
    "Shetland_sheepdog",
    "Australian_terrier",
    "Welsh_springer_spaniel",
    "English_springer",
    "Brittany_spaniel",
    "French_bulldog",
    "Boston_bull",
    "Staffordshire_bullterrier",
    "Pembroke",   # Corgi
    "Cardigan",   # Cardigan Corgi
    "basenji",
    
    # Large dogs (15)
    "golden_retriever",
    "Labrador_retriever",
    "German_shepherd",
    "Rottweiler",
    "Doberman",
    "boxer",
    "Great_Dane",
    "Saint_Bernard",
    "Siberian_husky",
    "malamute",
    "Samoyed",
    "collie",
    "Old_English_sheepdog",
    "Afghan_hound",
    "borzoi",
    
    # Distinctive breeds (10)
    "whippet",           # Replaced dalmatian (not in Stanford Dogs)
    "chow",
    "Newfoundland",
    "Great_Pyrenees",
    "komondor",
    "Weimaraner",
    "vizsla",
    "Rhodesian_ridgeback",
    "bloodhound",
    "Irish_setter",
]

# Verify count
assert len(SELECTED_BREEDS) == 50, f"Expected 50 breeds, got {len(SELECTED_BREEDS)}"


def get_breed_folder_mapping(dataset_path: str) -> dict:
    """
    Map breed names to actual folder names in the dataset.
    
    Returns:
        Dict of {breed_name: folder_name}
    """
    import os
    
    breed_folders = {}
    images_path = os.path.join(dataset_path, "images", "Images")
    
    for folder in os.listdir(images_path):
        # Extract breed name from folder (e.g., "n02085620-Chihuahua" -> "Chihuahua")
        if "-" in folder:
            breed_name = folder.split("-", 1)[1]
            breed_folders[breed_name] = folder
    
    return breed_folders


def get_subset_folders(dataset_path: str) -> list:
    """
    Get list of folder paths for the 50-breed subset.
    
    Returns:
        List of full paths to breed folders
    """
    import os
    
    mapping = get_breed_folder_mapping(dataset_path)
    images_path = os.path.join(dataset_path, "images", "Images")
    
    subset_folders = []
    missing = []
    
    for breed in SELECTED_BREEDS:
        if breed in mapping:
            folder_path = os.path.join(images_path, mapping[breed])
            subset_folders.append({
                'name': breed,
                'folder': mapping[breed],
                'path': folder_path
            })
        else:
            missing.append(breed)
    
    if missing:
        print(f"Warning: {len(missing)} breeds not found: {missing[:5]}...")
    
    print(f"Loaded {len(subset_folders)} breeds for Module 2 subset")
    return subset_folders


def load_subset_images(dataset_path: str, max_per_breed: int = None) -> dict:
    """
    Load all images from the 50-breed subset.
    
    Args:
        dataset_path: Path to CV-Project root
        max_per_breed: Maximum images per breed (None = all)
        
    Returns:
        Dict of {breed_name: [list of image paths]}
    """
    import os
    
    subset_folders = get_subset_folders(dataset_path)
    images = {}
    
    for breed_info in subset_folders:
        breed_name = breed_info['name']
        breed_path = breed_info['path']
        
        breed_images = [
            os.path.join(breed_path, f) 
            for f in os.listdir(breed_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if max_per_breed:
            breed_images = breed_images[:max_per_breed]
        
        images[breed_name] = breed_images
    
    total = sum(len(v) for v in images.values())
    print(f"Loaded {total} images from {len(images)} breeds")
    
    return images


def create_train_test_split(images: dict, test_ratio: float = 0.2, seed: int = 42):
    """
    Split images into train and test sets.
    
    Returns:
        (train_images, test_images) - Both dicts of {breed: [paths]}
    """
    import random
    random.seed(seed)
    
    train_images = {}
    test_images = {}
    
    for breed, paths in images.items():
        shuffled = paths.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train_images[breed] = shuffled[:split_idx]
        test_images[breed] = shuffled[split_idx:]
    
    train_total = sum(len(v) for v in train_images.values())
    test_total = sum(len(v) for v in test_images.values())
    
    print(f"Train: {train_total} images | Test: {test_total} images")
    
    return train_images, test_images


if __name__ == "__main__":
    import os
    
    # Test with actual dataset
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("50-Breed Subset for Module 2")
    print("=" * 50)
    
    for i, breed in enumerate(SELECTED_BREEDS, 1):
        print(f"{i:2}. {breed}")
    
    print("\nTesting subset loading...")
    folders = get_subset_folders(root_path)
