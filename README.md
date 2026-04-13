  **Apziva Project code** - **oGcFw3VgYQq3cvNC**
# MonReader — Page Flip Detection with Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-Image%20Pipeline-5C3EE8)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-0A66C2)
![Jupyter Notebook](https://img.shields.io/badge/Workflow-Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

> A computer vision project that detects whether a page is actively **flipping** or **not flipping** from image data using a lightweight convolutional neural network built in **PyTorch**.

## ✨ At a Glance

- **Problem:** Binary image classification for page-flip detection
- **Approach:** Custom `TinyVGG` CNN with model comparison and sequence-level detection
- **Best Verified Result:** **99.84% test accuracy** and **0.9983 macro F1** on ResNet18 (Model 2)
- **Five Models Tested:** TinyVGG baseline/augmented, ResNet18, MobileNet V2, EfficientNet B0
- **Sequence Signal:** Flip-folder testing shows **>86% flip ratio**, while a non-flip folder is reported at about **0.98%**
- **Tools:** Python, PyTorch, Torchvision, scikit-learn, Matplotlib
- **Portfolio Value:** Demonstrates end-to-end deep learning expertise with transfer learning and training best practices

---

## 📌 Project Summary

MonReader is a **PyTorch-based computer vision project** that classifies page images as `flip` or `notflip` and extends those predictions to **ordered frame sequences**. It demonstrates the full ML workflow: data preparation, CNN development, experiment comparison, metric-driven evaluation, transfer learning, and practical inference.

## 📚 Table of Contents

- [MonReader — Page Flip Detection with Deep Learning](#monreader--page-flip-detection-with-deep-learning)
  - [✨ At a Glance](#-at-a-glance)
  - [📌 Project Summary](#-project-summary)
  - [📚 Table of Contents](#-table-of-contents)
    - [Why this project stands out](#why-this-project-stands-out)
  - [🎯 Business / Use-Case Relevance](#-business--use-case-relevance)
  - [🧰 Tech Stack](#-tech-stack)
  - [🆕 Updated Notebook Highlights](#-updated-notebook-highlights)
  - [🔄 Step-by-Step Project Flow](#-step-by-step-project-flow)
    - [1. Data Acquisition](#1-data-acquisition)
    - [2. Data Exploration](#2-data-exploration)
    - [3. Image Preprocessing](#3-image-preprocessing)
    - [4. Data Augmentation](#4-data-augmentation)
    - [5. Dataset \& DataLoader Creation](#5-dataset--dataloader-creation)
    - [6. Model Development](#6-model-development)
    - [7. Training \& Evaluation](#7-training--evaluation)
    - [8. Model Comparison](#8-model-comparison)
    - [9. Detecting If a Page Is Flipping or Not](#9-detecting-if-a-page-is-flipping-or-not)
  - [🧠 Model Approach](#-model-approach)
  - [📈 Results](#-results)
  - [📁 Project Structure](#-project-structure)
  - [▶️ How to Run](#️-how-to-run)
    - [1. Install dependencies](#1-install-dependencies)
    - [2. Open the notebook](#2-open-the-notebook)
    - [3. Run the workflow](#3-run-the-workflow)
  - [✅ What This Project Demonstrates](#-what-this-project-demonstrates)
  - [💼 Concluding Note](#-concluding-note)

### Why this project stands out

- **Architectural exploration:** comparison of custom TinyVGG against state-of-the-art models (ResNet, MobileNet, EfficientNet)
- **Transfer learning mastery:** demonstrating 4.5% accuracy improvement via pre-trained ImageNet weights
- **Training best practices:** early stopping to optimize convergence and prevent overfitting
- **Comprehensive evaluation:** systematic comparison across five models with consistent metrics (loss, accuracy, F1)
- **Practical extension:** sequence-level flip detection beyond single-image classification

---

## 🎯 Business / Use-Case Relevance

Page-flip detection is useful in:
- **document digitization** and scanning workflows,
- **smart reading** or assistive systems,
- and **automated content-capture** pipelines.

This makes the project a practical example of applying deep learning to a focused real-world vision task.

---

## 🧰 Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Data & Analysis:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib
- **Utilities:** Pillow, Requests, tqdm, torchinfo

---

## 🆕 Updated Notebook Highlights

Key notebook capabilities:

- **Five trained models** for comprehensive comparison:
  - **Model 0:** Custom TinyVGG (no augmentation)
  - **Model 1:** Custom TinyVGG (with data augmentation)
  - **Model 2:** ResNet18 (pre-trained on ImageNet)
  - **Model 3:** MobileNet V2 (lightweight, efficient)
  - **Model 4:** EfficientNet B0 (state-of-the-art efficiency)
- **Early Stopping** function integrated into training loop:
  - Monitors validation loss for improvement
  - Configurable patience parameter (default=3 epochs)
  - Automatically saves the best model state
  - Prevents overfitting and reduces unnecessary training
- **comprehensive model comparison** across loss, accuracy, and **macro F1 score** for all five models
- **data augmentation** using TrivialAugmentWide for robust feature learning
- a dedicated final section for **page-flip detection**
- **sequence utilities** such as `predict_flipping_sequence(...)` and `predict_flipping_from_folder(...)`
- **threshold-based decision logic** using `min_flip_frames` and `min_flip_ratio`

Together, these updates demonstrate both architectural diversity (custom models vs. pre-trained architectures) and training best practices (early stopping, augmentation).

---

## 🔄 Step-by-Step Project Flow

### 1. Data Acquisition
The notebook downloads the image dataset and organizes it into training and testing folders with two classes:
- `flip`
- `notflip`

### 2. Data Exploration
Initial exploration checks:
- folder structure,
- sample images,
- class balance,
- and image characteristics.

This helps validate the dataset before model training begins.

### 3. Image Preprocessing
Images are transformed with `torchvision.transforms`, including:
- resizing to `64x64`,
- tensor conversion,
- and consistent formatting for model input.

### 4. Data Augmentation
The updated workflow applies **random horizontal flipping** during training to improve generalization and reduce overfitting.

### 5. Dataset & DataLoader Creation
The project uses `ImageFolder` and `DataLoader` to create a scalable batching pipeline for both training and evaluation.

### 6. Model Development
The notebook implements and trains **five models** to explore different architectural approaches:

**Custom Architectures:**
- **Model 0:** TinyVGG without augmentation (baseline)
- **Model 1:** TinyVGG with data augmentation (improved baseline)

**Pre-trained Transfer Learning Models:**
- **Model 2:** ResNet18 with ImageNet pre-training (deep residual learning)
- **Model 3:** MobileNet V2 with ImageNet pre-training (lightweight and fast)
- **Model 4:** EfficientNet B0 with ImageNet pre-training (optimal efficiency/accuracy trade-off)

### 7. Training & Evaluation with Early Stopping
Each model is trained using:
- **Loss function:** `CrossEntropyLoss` for binary classification
- **Optimizer:** Adam with learning rate 0.001
- **Evaluation metrics:** loss, accuracy, and macro F1 score
- **Early Stopping:** Integrated callback that:
  - Tracks validation loss across epochs
  - Saves the best-performing model state
  - Stops training when validation loss plateaus (patience=3)
  - Prevents overfitting and optimizes training time

Each model trains for up to **10 epochs** (or until early stopping triggers).

### 8. Model Comparison
Results from all **five models** are compared visually through comprehensive learning curves showing:
- train/test loss across epochs for all models
- train/test accuracy for all models
- train/test macro F1 score for all models

This comparison reveals the performance trade-offs between:
- Custom lightweight models (TinyVGG) vs. heavy pre-trained architectures
- Models with/without data augmentation
- Impact of early stopping on convergence and final accuracy

### 9. Detecting If a Page Is Flipping or Not
The final workflow performs **folder-based sequence analysis**. Ordered frames are scored one by one, a **flip ratio** is computed, and the sequence is classified using `min_flip_frames` and `min_flip_ratio` thresholds.

---

## 🧠 Model Approach

The notebook explores multiple architectural approaches, starting with a custom **TinyVGG** architecture and extending to industry-standard pre-trained models.

Why this is meaningful:
- demonstrates understanding of CNN building blocks (custom TinyVGG),
- shows mastery of transfer learning and fine-tuning with pre-trained models,
- compares lightweight custom models against production-ready architectures,
- and applies professional training practices like early stopping and validation monitoring.

---

## 📈 Results

The updated workflow with five models and early stopping provides comprehensive performance insights across architectural styles:

### Overall Winner — Model 2 (ResNet18)

| Metric | Verified Result |
|---|---:|
| Test Accuracy | **0.9984** |
| Test Macro F1 | **0.9983** |
| Best Epoch | **4-6** |
| Early Stopping | Triggered at epoch 3 |

### Detailed Performance Comparison

| Model | Architecture | Best Accuracy | Best F1 | Best Epoch | Type |
|---|---|---:|---:|---:|---|
| **Model 0** | TinyVGG (baseline) | 0.9539 | 0.9548 | 8 | Custom |
| **Model 1** | TinyVGG (augmented) | 0.8799 | 0.8793 | 8 | Custom |
| **Model 2** | ResNet18 (pre-trained) | **0.9984** | **0.9983** | 4-6 | **Transfer** |
| **Model 3** | MobileNet V2 (pre-trained) | 0.9967 | 0.9966 | 1-3 | Transfer |
| **Model 4** | EfficientNet B0 (pre-trained) | 0.9967 | 0.9966 | 2-3, 5 | Transfer |

### Key Findings

1. **Transfer Learning Dominance:** Pre-trained models vastly outperformed custom architectures
   - ResNet18: 99.84% accuracy (vs. TinyVGG: 95.39%)
   - 4.45% absolute improvement via ImageNet pre-training

2. **ResNet18 Best Performer:** Achieved highest accuracy with deep residual connections
   - Benefited most from early stopping mechanism
   - Converged to peak performance by epoch 4

3. **MobileNet & EfficientNet Match ResNet:** Achieved competitive 99.67% accuracy
   - Lighter parameter footprints for efficient deployment
   - Demonstrated that accuracy-efficiency trade-off is minimal at this scale

4. **Early Stopping Effectiveness:** Reduced training iterations while maintaining/improving accuracy
   - Pre-trained models completed training in 3-4 epochs instead of 10
   - Prevented overfitting and saved computational resources

5. **Data Augmentation Paradox:** TinyVGG with augmentation (Model 1) underperformed baseline
   - Custom model's limited capacity (10 hidden units) couldn't leverage augmentation effectively
   - Suggests need for larger model capacity to benefit from advanced regularization

6. **Sequence Detection Capability:** All models successfully learned flip-vs-notflip patterns
   - Flip folder: **>86% flip ratio** detected
   - Non-flip folder: **~0.98% flip ratio** detected
   - Clear sequence-level signal demonstrates practical effectiveness

The notebook also demonstrates practical sequence behavior:

| Sequence Test | Observed Outcome |
|---|---:|
| `testing/flip` folder | **flip ratio > 86%** |
| `testing/notflip` folder | **flip ratio ≈ 0.98%** |

These results show that the project does more than classify isolated frames — it also produces a clear sequence-level signal that can distinguish flipping from non-flipping behavior in ordered image sets.

---

## 📁 Project Structure

```text
MonReader/
├── data/
│   └── flip_notflip/
├── notebooks/
│   └── MonReader_v3.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open the notebook
Launch:
```text
notebooks/MonReader_v3.ipynb
```

### 3. Run the workflow
Execute the notebook cells in order to:
- download and prepare the dataset,
- train all five models with early stopping,
- compare performance across architectures,
- evaluate sequence-level flip detection,
- and run inference on sample images or folders of frames.

---

## ✅ What This Project Demonstrates

This project highlights the ability to:
- frame a real-world computer vision problem clearly,
- implement custom CNN architectures from scratch (TinyVGG),
- apply transfer learning and fine-tuning with pre-trained models,
- implement professional training practices (early stopping, validation monitoring),
- evaluate performance systematically across multiple metrics and architectures,
- compare custom models against industry-standard approaches,
- and extend a research-style notebook into practical inference workflows.

---

## 💼 Concluding Note 

For hiring managers and recruiters, MonReader highlights:
- **practical expertise** with PyTorch and deep learning fundamentals,
- **transfer learning mastery** — achieving 99.84% accuracy through ImageNet pre-training,
- **architectural understanding** — building custom models AND evaluating pre-trained architectures,
- **professional practices** — implementing early stopping, comprehensive evaluation, and rigorous comparison,
- and **practical thinking** by extending notebook models into usable sequence-level detection logic.

In short, this is a portfolio-ready example of **end-to-end deep learning development** that demonstrates both foundational knowledge and production-level practices.
