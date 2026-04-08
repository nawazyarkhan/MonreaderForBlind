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
- **Approach:** Custom `TinyVGG` CNN with augmentation-based experimentation
- **Best Verified Result:** **91.93% test accuracy** and **0.9196 F1 score**
- **Tools:** Python, PyTorch, Torchvision, scikit-learn, Matplotlib
- **Portfolio Value:** Demonstrates end-to-end computer vision project ownership

---

## 📌 Project Summary

MonReader is an end-to-end image classification project focused on a practical document-analysis problem: **identifying page-flip activity from visual input**. The workflow covers the full machine learning lifecycle — from data ingestion and preprocessing to model experimentation, evaluation, and inference.

## 📚 Table of Contents

- [MonReader — Page Flip Detection with Deep Learning](#monreader--page-flip-detection-with-deep-learning)
  - [✨ At a Glance](#-at-a-glance)
  - [📌 Project Summary](#-project-summary)
  - [📚 Table of Contents](#-table-of-contents)
  - [🎯 Business / Use-Case Relevance](#-business--use-case-relevance)
  - [🧰 Tech Stack](#-tech-stack)
  - [🔄 Step-by-Step Project Flow](#-step-by-step-project-flow)
    - [1. Data Acquisition](#1-data-acquisition)
    - [2. Data Exploration](#2-data-exploration)
    - [3. Image Preprocessing](#3-image-preprocessing)
    - [4. Data Augmentation](#4-data-augmentation)
    - [5. Dataset \& DataLoader Creation](#5-dataset--dataloader-creation)
    - [6. Model Development](#6-model-development)
    - [7. Training \& Evaluation](#7-training--evaluation)
    - [8. Inference on New Images](#8-inference-on-new-images)
    - [9. Sequence-Level Extension](#9-sequence-level-extension)
  - [🧠 Model Approach](#-model-approach)
  - [📈 Results](#-results)
  - [📁 Project Structure](#-project-structure)
  - [▶️ How to Run](#️-how-to-run)
    - [1. Install dependencies](#1-install-dependencies)
    - [2. Open the notebook](#2-open-the-notebook)
    - [3. Run the workflow](#3-run-the-workflow)
  - [✅ What This Project Demonstrates](#-what-this-project-demonstrates)
  - [💼 Concluding Note](#-concluding-note)

This project demonstrates the ability to:
- build a complete deep learning pipeline in Python,
- work with image datasets using `torchvision`,
- design and train a custom CNN architecture,
- compare baseline vs augmented training strategies,
- and extend a frame-level classifier into a **sequence-level page-flip detection workflow**.

---

## 🎯 Business / Use-Case Relevance

Detecting page-flip activity is useful in scenarios such as:
- **digitization and document scanning** workflows,
- **smart reading systems**,
- **automated content capture** pipelines,
- and **human activity understanding** from image sequences.

The project shows how deep learning can be applied to a real classification problem where robust visual recognition is important.

---

## 🧰 Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Data & Analysis:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib
- **Utilities:** Pillow, Requests, tqdm, torchinfo

---

## 🔄 Step-by-Step Project Flow

### 1. Data Acquisition
The notebook downloads the image dataset and organizes it into training and testing folders with two classes:
- `flip`
- `notflip`

### 2. Data Exploration
Initial exploration is performed to inspect:
- folder structure,
- class labels,
- image dimensions,
- and representative samples from the dataset.

This helps validate the quality and distribution of the input data before training.

### 3. Image Preprocessing
Images are prepared using `torchvision.transforms`, including:
- resizing to `64x64`,
- conversion to tensors,
- and normalization-ready formatting for model input.

### 4. Data Augmentation
To improve generalization, the project introduces **random horizontal flipping** during training. This creates more variability in the dataset and helps the model become more robust to visual differences.

### 5. Dataset & DataLoader Creation
The project uses `ImageFolder` and `DataLoader` to build an efficient training pipeline for batch-based model learning and evaluation.

### 6. Model Development
A custom **TinyVGG**-style CNN is implemented in PyTorch. The notebook compares two setups:
- **Model 0:** TinyVGG without augmentation
- **Model 1:** TinyVGG with data augmentation

This allows for a clear experiment-driven comparison of training strategy and generalization performance.

### 7. Training & Evaluation
The model training loop tracks:
- training loss,
- test loss,
- accuracy,
- and F1 score.

This gives a more complete view of performance than accuracy alone.

### 8. Inference on New Images
After training, the model is used to predict whether a new frame represents a flipping or non-flipping page.

### 9. Sequence-Level Extension
A strong practical addition in this project is the sequence inference logic that evaluates **ordered folders of frames** and estimates page-flipping behavior across an image sequence rather than only on isolated images.

---

## 🧠 Model Approach

The final workflow centers on a custom **TinyVGG** architecture — a compact convolutional neural network well-suited for learning visual patterns from relatively small images.

Why this is meaningful:
- it shows understanding of CNN building blocks,
- it keeps the solution lightweight and interpretable,
- and it demonstrates hands-on model development rather than relying only on high-level APIs.

---

## 📈 Results

From the saved notebook run, the augmented model achieved strong performance:

| Metric | Result |
|---|---:|
| Test Accuracy | **0.9193** |
| Test F1 Score | **0.9196** |
| Epoch Reached | **4** |

These results indicate that the model learned to distinguish between `flip` and `notflip` effectively, while the augmentation strategy improved robustness.

---

## 📁 Project Structure

```text
MonReader/
├── data/
│   └── flip_notflip/
├── notebooks/
│   └── MonReader_updated.ipynb
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
notebooks/MonReader_updated.ipynb
```

### 3. Run the workflow
Execute the notebook cells in order to:
- download and prepare the dataset,
- train the model,
- evaluate performance,
- and run prediction on sample images or folders of frames.

---

## ✅ What This Project Demonstrates

This project highlights the ability to:
- frame a real-world computer vision problem clearly,
- build and train a deep learning model from scratch,
- apply data augmentation thoughtfully,
- evaluate performance with relevant metrics,
- and extend a research-style notebook into a more practical inference workflow.

---

## 💼 Concluding Note 

MonReader reflects **end-to-end ownership of a machine learning project** — from understanding the problem and preparing the data to designing the model, evaluating outcomes, and building usable inference logic.

From a portfolio perspective, this project demonstrates:
- solid hands-on experience with **deep learning and computer vision**,
- practical use of **PyTorch and data pipelines**,
- an experimentation mindset through **model comparison and augmentation**,
- and the ability to translate technical work into a clear, outcome-focused solution.

If you are reviewing this project as part of a candidate portfolio, MonReader is a strong example of applied ML work that combines **technical depth, structured experimentation, and business relevance**.
