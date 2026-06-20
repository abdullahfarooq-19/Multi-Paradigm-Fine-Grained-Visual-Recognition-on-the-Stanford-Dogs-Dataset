# Dog Breed Recognition Pipeline: Classical CV to Deep Learning

A modular computer vision pipeline for fine-grained dog breed classification 
using the Stanford Dogs Dataset (120 breeds, 20,580 images). The project 
progresses through three paradigms — image processing, classical ML, and 
deep learning — making it a complete end-to-end CV study.

---

## Modules

### Module 1 — Low-Level Image Processing
Fundamental image operations and quality analysis on raw pixel data.
- **Geometric Transforms:** rotation, scaling, flipping, affine warping
- **Intensity Transforms:** CLAHE, gamma correction, brightness/contrast, normalization
- **Edge Detection:** Sobel, Canny, Laplacian, LoG (Mexican hat), Auto-Canny
- **Noise Modeling:** Gaussian, Salt-and-Pepper, Speckle, Poisson
- **Restoration Filters:** Gaussian blur, Median, Bilateral, Non-Local Means (NLM)
- **Quality Metrics:** PSNR and SSIM for quantitative evaluation

### Module 2 — Classical Feature-Based Classification (50 Breeds)
Hand-crafted visual features with a traditional ML classifier — no neural networks.
- **Features:** SIFT (Bag of Visual Words), HOG (shape), LBP (texture), HOG+LBP fusion
- **Dimensionality Reduction:** PCA with 2D/3D scatter visualization
- **Classifier:** SVM (RBF kernel) with GridSearchCV hyperparameter tuning
- **Evaluation:** Top-1/Top-5 accuracy, per-class F1-score, confusion matrix

### Module 3 — Deep Learning Transfer Learning (120 Breeds)
High-accuracy classification with a pretrained CNN and visual explainability.
- **Model:** ResNet50 (ImageNet pretrained) with custom fine-tuned head
- **Pre-processing:** YOLOv8 dog detection and crop before classification
- **Explainability:** Grad-CAM heatmaps + saliency maps via PyTorch hooks
- **Head Architecture:** Dropout(0.5) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→120)

---

## Dataset

**Stanford Dogs Dataset**
- 20,580 images across 120 dog breeds
- Train/test splits provided as `.mat` files
- Module 1: qualitative subset | Module 2: 50-breed subset | Module 3: full 120 breeds

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8+ |
| Computer Vision | OpenCV, scikit-image |
| Classical ML | scikit-learn (SVM, PCA, GridSearchCV) |
| Deep Learning | PyTorch, torchvision (ResNet50) |
| Object Detection | Ultralytics YOLOv8 |
| Numerical | NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter / Google Colab |

---

## Project Structure

```
CV-Project/
├── images/                    # Stanford Dogs Dataset (120 breed folders)
├── lists/                     # Train/test splits (.mat format)
├── requirements.txt
│
├── module1/src/
│   ├── transforms.py          # Geometric & intensity transforms
│   ├── edge_detection.py      # Sobel, Canny, LoG, Auto-Canny
│   ├── noise_restoration.py   # Noise models + denoising filters
│   ├── metrics.py             # PSNR & SSIM
│   ├── comparative_study.py   # Batch comparison across configs
│   └── demo.py
│
├── module2/src/
│   ├── feature_extraction.py  # SIFT, HOG, LBP, FeatureFusion
│   ├── classifier.py          # SVM + cross-validation
│   ├── dimensionality.py      # PCA + scatter plots
│   ├── breed_subset.py        # Stratified 50-breed sampling
│   └── demo.py
│
└── module3/src/
    ├── model.py               # ResNet50 custom head
    ├── gradcam.py             # Grad-CAM + saliency maps
    ├── yolo_detection.py      # YOLOv8 detection & crop pipeline
    └── registration.py        # Dataset mapping utilities
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
# Module 1
python module1/src/demo.py

# Module 2
python module2/src/demo.py
```

Module 3 notebooks are designed to run on **Google Colab** (see `CV-Project/` folder).

---

## Skills Demonstrated

- Image preprocessing, filtering, and quality measurement (PSNR, SSIM)
- Classical feature engineering: SIFT, HOG, LBP, Bag of Visual Words
- SVM classification with PCA and hyperparameter tuning
- CNN transfer learning with layer freezing strategy
- Object detection integration (YOLOv8) as a preprocessing step
- Model explainability with Grad-CAM and saliency maps
