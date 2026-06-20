"""
Extract metrics for Module 1 Report
"""
import os
import sys
import cv2
import numpy as np

# Add source directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from dataset_utils import StanfordDogsDataset
from edge_detection import EdgeDetector, compute_edge_density
from noise_restoration import NoiseGenerator, ImageRestoration
from metrics import psnr, ssim

# Initialize dataset
root_path = os.path.dirname(os.path.dirname(script_dir))
dataset = StanfordDogsDataset(root_path)

# Select 5 diverse breeds
breeds_to_test = [
    ('Fluffy', 'Samoyed'),
    ('Dark', 'Rottweiler'),
    ('Light', 'golden_retriever'),
    ('Smooth', 'beagle'),
    ('Long-haired', 'Afghan')
]

print("=" * 80)
print("MODULE 1 METRICS EXTRACTION FOR REPORT")
print("=" * 80)

# Store results
edge_results = []
noise_results = []

for dog_type, breed_name in breeds_to_test:
    breed = dataset.get_breed_by_name(breed_name)
    if not breed:
        print(f"WARNING: {breed_name} not found")
        continue
    
    # Load image
    image_path = dataset.get_images_from_breed(breed_name, max_images=1)[0]
    image = cv2.imread(image_path)
    
    if image is None:
        continue
    
    actual_breed = breed['name']
    print(f"\n--- {dog_type}: {actual_breed} ---")
    
    # Edge Detection
    canny = EdgeDetector.canny(image, 50, 150)
    sobel = EdgeDetector.sobel(image)
    
    canny_density = compute_edge_density(canny)
    sobel_density = compute_edge_density(sobel)
    
    edge_results.append({
        'type': dog_type,
        'breed': actual_breed,
        'canny_density': canny_density,
        'sobel_density': sobel_density
    })
    
    print(f"  Edge Density (Canny): {canny_density:.2%}")
    print(f"  Edge Density (Sobel): {sobel_density:.2%}")
    
    # Noise & Restoration
    noisy = NoiseGenerator.gaussian(image, sigma=25)
    
    restored_median = ImageRestoration.median_filter(noisy)
    restored_bilateral = ImageRestoration.bilateral_filter(noisy)
    restored_nlm = ImageRestoration.nlm_denoise(noisy)
    
    psnr_noisy = psnr(image, noisy)
    psnr_median = psnr(image, restored_median)
    psnr_bilateral = psnr(image, restored_bilateral)
    psnr_nlm = psnr(image, restored_nlm)
    
    noise_results.append({
        'type': dog_type,
        'breed': actual_breed,
        'noisy': psnr_noisy,
        'median': psnr_median,
        'bilateral': psnr_bilateral,
        'nlm': psnr_nlm
    })
    
    print(f"  PSNR Noisy: {psnr_noisy:.2f} dB")
    print(f"  PSNR Median: {psnr_median:.2f} dB")
    print(f"  PSNR Bilateral: {psnr_bilateral:.2f} dB")
    print(f"  PSNR NLM: {psnr_nlm:.2f} dB")

# Print summary tables
print("\n" + "=" * 80)
print("EDGE DENSITY TABLE (for report)")
print("=" * 80)
print(f"| {'Type':<15} | {'Breed':<20} | {'Canny Density':>15} |")
print("|" + "-"*17 + "|" + "-"*22 + "|" + "-"*17 + "|")
for r in edge_results:
    print(f"| {r['type']:<15} | {r['breed']:<20} | {r['canny_density']:>14.2%} |")

print("\n" + "=" * 80)
print("PSNR TABLE (for report) - Gaussian Noise sigma=25")
print("=" * 80)
print(f"| {'Type':<15} | {'Noisy':>10} | {'Median':>10} | {'Bilateral':>10} | {'NLM':>10} |")
print("|" + "-"*17 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
for r in noise_results:
    print(f"| {r['type']:<15} | {r['noisy']:>9.2f} | {r['median']:>9.2f} | {r['bilateral']:>9.2f} | {r['nlm']:>9.2f} |")

print("\n" + "=" * 80)
print("COPY THESE VALUES TO THE REPORT")
print("=" * 80)
