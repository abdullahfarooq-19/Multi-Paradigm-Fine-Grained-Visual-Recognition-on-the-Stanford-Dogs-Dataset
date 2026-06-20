# Fine-Grained Dog Breed Recognition Pipeline
> **AI Context File** · Prepared for CV assistance · Author: Student Developer

---

## 1. What This Project Does

This is a **modular computer vision research project** that tackles the fine-grained image classification problem of identifying dog breeds from photographs. The dataset used is the **Stanford Dogs Dataset** — 20,580 images spanning **120 dog breeds** — a standard benchmark for fine-grained visual recognition.

The project is divided into **three progressive modules**, each representing a different paradigm of computer vision: low-level signal processing, classical machine learning, and deep learning. This structured progression makes it academically rigorous and demonstrates a complete understanding of the CV pipeline from raw pixel manipulation to state-of-the-art neural networks.

---

## 2. Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.8+ | Primary language across all modules |
| **Core CV** | OpenCV (`cv2`) | Image I/O, geometric transforms, filtering, edge detection |
| **Numerical** | NumPy | Array operations, noise generation, matrix math |
| **Classical ML** | scikit-learn | SVM classifier, PCA, GridSearchCV, cross-validation |
| **Image Features** | scikit-image | HOG and LBP feature extraction |
| **Deep Learning** | PyTorch + torchvision | ResNet50 transfer learning, model training |
| **Object Detection** | Ultralytics YOLOv8 | Pre-classification dog localization/detection |
| **Visualization** | Matplotlib, seaborn | Plots, confusion matrices, PCA scatter plots |
| **Notebooks** | Jupyter / Google Colab | All three modules run as `.ipynb` notebooks |
| **Data loading** | h5py, scipy (`.mat` files) | Loading Stanford Dogs train/test split lists |

---

## 3. Project Architecture

```
CV-Project/
├── images/                    # Stanford Dogs Dataset images (120 breed folders)
├── lists/                     # Train/test splits in .mat format (MATLAB)
├── requirements.txt           # pip dependencies
│
├── module1/                   # Low-Level Vision & Image Processing
│   └── src/
│       ├── transforms.py      # Geometric (rotate, scale, flip, affine) + Intensity transforms
│       ├── edge_detection.py  # Sobel, Canny, Laplacian, LoG, Auto-Canny detectors
│       ├── noise_restoration.py # Noise models + denoising filters
│       ├── dataset_utils.py   # Stanford Dogs loader, .mat file parsing
│       ├── metrics.py         # PSNR, SSIM image quality metrics
│       ├── comparative_study.py # Automated comparisons across transform/filter configs
│       └── demo.py            # Standalone demo runner
│
├── module2/                   # Classical Feature-Based Recognition (50 breeds)
│   └── src/
│       ├── feature_extraction.py  # SIFT, HOG, LBP, FeatureFusion classes
│       ├── classifier.py          # SVM classifier with CV and GridSearch
│       ├── dimensionality.py      # PCA reduction + 2D/3D scatter visualization
│       ├── breed_subset.py        # Stratified sampling of 50 breed subset
│       └── demo.py                # Full classification pipeline demo
│
└── module3/                   # Deep Learning with Transfer Learning (120 breeds)
    └── src/
        ├── model.py           # ResNet50 custom head definition
        ├── gradcam.py         # Grad-CAM + saliency map implementation
        ├── yolo_detection.py  # YOLOv8 dog detection and crop pipeline
        └── registration.py   # Dataset registration / mapping utilities
```

---

## 4. How Each Module Works

### Module 1 — Low-Level Vision & Image Processing

**Goal:** Study and implement fundamental image processing operations and measure their effect on image quality.

**What is implemented:**

- **Geometric Transforms** (`transforms.py`): Rotation (affine warp matrix), scaling (INTER_CUBIC / INTER_AREA), flipping, translation, and general affine transformation via point correspondences.
- **Intensity Transforms** (`transforms.py`): Histogram equalization (global CLAHE via LAB color space), gamma correction (LUT-based), brightness and contrast adjustment, min-max and z-score normalization.
- **Edge Detection** (`edge_detection.py`): Sobel (Gx, Gy, combined magnitude), Canny (with Gaussian pre-blur + hysteresis), Laplacian, Laplacian of Gaussian (LoG / "Mexican hat"), and adaptive auto-Canny (thresholds set from pixel intensity median).
- **Noise Modeling + Restoration** (`noise_restoration.py`):
  - *Noise types:* Gaussian (additive), Salt-and-Pepper (impulse), Speckle (multiplicative), Poisson (shot noise)
  - *Restoration filters:* Gaussian blur, Median filter, Bilateral filter (edge-preserving), Non-Local Means (NLM) denoising
- **Quality Metrics** (`metrics.py`): PSNR and SSIM to quantitatively measure restoration quality.
- **Comparative Study** (`comparative_study.py`): Automated batch comparisons across multiple noise levels and filter configurations, producing metric tables.

**How it flows:** Images are loaded from the Stanford Dogs dataset → degraded with a chosen noise model → restored with a filter → PSNR/SSIM are computed before and after to evaluate restoration effectiveness.

---

### Module 2 — Classical Feature-Based Dog Breed Classification

**Goal:** Classify 50 dog breeds using hand-crafted visual features and a machine learning classifier — **no neural networks**.

**What is implemented:**

- **SIFT** (`feature_extraction.py`): Scale-Invariant Feature Transform — detects up to 500 keypoints per image and computes 128-dimensional descriptors. Used via Bag of Visual Words (BoVW) for classification.
- **HOG** (`feature_extraction.py`): Histogram of Oriented Gradients — images resized to 128×128, divided into 8×8 cells and 2×2 blocks with 9 orientation bins, producing a fixed-length shape descriptor. Captures body silhouette, ear shape, snout shape.
- **LBP** (`feature_extraction.py`): Local Binary Patterns — images resized to 128×128, LBP computed with 24 sample points at radius 3 using the uniform method. A **spatial pyramid (2×2 grid)** divides the image into 4 regions, each contributing a separate histogram. Captures fur texture.
- **Feature Fusion** (`feature_extraction.py`): HOG + LBP descriptors are L2-normalized and concatenated into a single feature vector for joint classification.
- **PCA** (`dimensionality.py`): `StandardScaler` + `sklearn.decomposition.PCA` used to reduce high-dimensional feature vectors. Explained variance plotted; 2D and 3D scatter plots generated for visual cluster analysis.
- **SVM Classifier** (`classifier.py`): `sklearn.svm.SVC` with RBF kernel, balanced class weights, and probability estimates enabled. Hyperparameter tuning via `GridSearchCV` (C, gamma, kernel). Evaluated with Top-1 and Top-5 accuracy, per-class F1-score, and normalized confusion matrix heatmap.
- **Breed Subset** (`breed_subset.py`): Stratified sampling selects a balanced 50-breed subset from the full 120-breed dataset.

**How it flows:** Dataset loaded → 50 breeds selected → HOG + LBP features extracted for all images → PCA applied → SVM trained with cross-validation → evaluation metrics computed and plotted.

---

### Module 3 — Deep Learning Transfer Learning (120 Breeds)

**Goal:** Achieve high-accuracy breed classification across all 120 breeds using transfer learning on a pretrained CNN, with explainability via Grad-CAM.

**What is implemented:**

- **ResNet50 Transfer Learning** (`model.py`): ImageNet-pretrained ResNet50 loaded from `torchvision.models`. Early layers (conv1 through layer3) are **frozen**; only `layer4` and a new custom `fc` head are fine-tuned. The custom head: `Dropout(0.5) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→120)`. ~23M total parameters, ~6M trainable.
- **YOLOv8 Dog Detection** (`yolo_detection.py`): `ultralytics.YOLO` (`yolov8n.pt`) runs on each image before classification to detect the dog region (COCO class ID 16). The detected bounding box is cropped with padding and fed to ResNet50, improving classification by removing background clutter.
- **Grad-CAM** (`gradcam.py`): Gradient-weighted Class Activation Mapping using PyTorch forward and backward hooks on `model.layer4[-1]`. Produces spatial heatmaps showing which image regions drive the breed prediction. Used to verify the model is attending to the dog's body rather than background.
- **Saliency Maps** (`gradcam.py`): Input gradient-based saliency (`requires_grad=True` on input tensor) as a complementary explainability method.

**How it flows:** Image → YOLOv8 detects and crops dog region → ResNet50 (fine-tuned) classifies 120 breeds → Grad-CAM generates heatmap overlaid on original image to explain decision.

---

## 5. Dataset

| Property | Detail |
|---|---|
| Name | Stanford Dogs Dataset |
| Size | 20,580 images |
| Classes | 120 dog breeds |
| Train/test split | Provided as `.mat` files (parsed with `scipy.io.loadmat`) |
| Image format | JPEG, variable sizes |
| Module 1 usage | Any subset (qualitative + PSNR/SSIM analysis) |
| Module 2 usage | 50-breed stratified subset |
| Module 3 usage | Full 120 breeds |

---

## 6. Key Technical Decisions & Design Choices

| Decision | Rationale |
|---|---|
| Modular pipeline (3 separate modules) | Allows clear comparison: low-level → classical ML → deep learning |
| CLAHE over global histogram equalization | Prevents noise amplification in flat regions; better for textured fur images |
| Bilateral filter as primary denoiser | Preserves edges (body contour, eyes) while smoothing noise |
| Spatial Pyramid LBP (2×2 grid) | Adds spatial context to otherwise orderless texture histogram |
| HOG + LBP fusion over single descriptor | Shape (HOG) and texture (LBP) are complementary; fusion improves SVM accuracy |
| PCA before SVM | Reduces feature dimensionality, removes noise components, speeds up SVM training |
| Freezing ResNet50 layers 1–3 | Avoids overfitting on a dataset smaller than ImageNet; only task-specific layers fine-tuned |
| YOLOv8 pre-processing step in Module 3 | Background clutter is a primary error source in fine-grained recognition; cropping removes it |
| Grad-CAM on `layer4[-1]` | Deepest convolutional layer — highest semantic abstraction, largest receptive field |

---

## 7. Skills Demonstrated (for CV)

- **Computer Vision**: Image preprocessing, geometric/intensity transforms, edge detection operators (Sobel, Canny, LoG), noise modeling, and restoration (Gaussian, Median, Bilateral, NLM)
- **Classical ML**: SIFT keypoint detection, HOG shape descriptors, LBP texture descriptors, Bag of Visual Words, PCA dimensionality reduction, SVM classification with GridSearchCV
- **Deep Learning**: PyTorch transfer learning, layer freezing strategy, custom classifier head design, fine-tuning CNN (ResNet50)
- **Model Explainability**: Grad-CAM and saliency map implementation using PyTorch hooks
- **Object Detection**: YOLOv8 integration for pre-classification localization
- **Evaluation**: PSNR, SSIM, Top-1/Top-5 accuracy, confusion matrix, per-class F1-score
- **Python Engineering**: Object-oriented design with clean class hierarchies, type hints, docstrings, modular source layout

---

*Stack summary for CV bullet points:* **Python 3.8+ · OpenCV · NumPy · scikit-learn · scikit-image · PyTorch · torchvision · ResNet50 · YOLOv8 (Ultralytics) · Grad-CAM · SIFT · HOG · LBP · SVM · PCA · Stanford Dogs Dataset · Jupyter / Google Colab**
