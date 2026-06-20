"""Extract metrics and save to JSON"""
import os, sys, cv2, json
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from dataset_utils import StanfordDogsDataset
from edge_detection import EdgeDetector, compute_edge_density
from noise_restoration import NoiseGenerator, ImageRestoration
from metrics import psnr

root_path = os.path.dirname(os.path.dirname(script_dir))
dataset = StanfordDogsDataset(root_path)

breeds = [
    ('Fluffy', 'Samoyed'),
    ('Dark', 'Rottweiler'),
    ('Light', 'golden_retriever'),
    ('Smooth', 'beagle'),
    ('Long-haired', 'Afghan')
]

results = {'edge_density': [], 'psnr': []}

for dog_type, breed_name in breeds:
    breed = dataset.get_breed_by_name(breed_name)
    if not breed:
        continue
    
    image_path = dataset.get_images_from_breed(breed_name, max_images=1)[0]
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    # Edge density
    canny = EdgeDetector.canny(image, 50, 150)
    density = compute_edge_density(canny)
    results['edge_density'].append({
        'type': dog_type, 
        'breed': breed['name'], 
        'density': round(density * 100, 2)
    })
    
    # PSNR
    noisy = NoiseGenerator.gaussian(image, sigma=25)
    psnr_data = {
        'type': dog_type,
        'noisy': round(psnr(image, noisy), 2),
        'median': round(psnr(image, ImageRestoration.median_filter(noisy)), 2),
        'bilateral': round(psnr(image, ImageRestoration.bilateral_filter(noisy)), 2),
        'nlm': round(psnr(image, ImageRestoration.nlm_denoise(noisy)), 2)
    }
    results['psnr'].append(psnr_data)

# Save to JSON
output_path = os.path.join(os.path.dirname(script_dir), 'outputs', 'metrics.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_path}")
print(json.dumps(results, indent=2))
