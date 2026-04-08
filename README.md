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
- **Approach:** Custom `TinyVGG` CNN with model comparison, custom-image inference, and sequence-level detection
- **Best Verified Result:** **95.31% test accuracy** and **0.9530 macro F1** on the saved `model_1` run
- **Sequence Signal:** Flip-folder testing shows **>86% flip ratio**, while a non-flip folder is reported at about **0.98%**
- **Tools:** Python, PyTorch, Torchvision, scikit-learn, Matplotlib
- **Portfolio Value:** Demonstrates end-to-end computer vision project ownership, from training to practical inference

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
    - [9. Custom Image Inference](#9-custom-image-inference)
    - [10. Sequence-Level Page-Flip Detection](#10-sequence-level-page-flip-detection)
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

## 🆕 Updated Notebook Highlights

The latest notebook version now goes beyond basic model training and includes several practical additions:

- **10-epoch training workflow** for `model_1` using `CrossEntropyLoss` and the Adam optimizer
- **side-by-side comparison** of `model_0` and `model_1` across loss, accuracy, and **macro F1**
- **custom image inference** on a downloaded sample image (`flipping_page.jpeg`)
- **sequence-level prediction utilities**:
  - `predict_flipping_sequence(...)`
  - `predict_flipping_from_folder(...)`
  - `show_sequence_predictions(...)`
- **threshold-based decision logic** using both `min_flip_frames` and `min_flip_ratio`

These updates make the notebook more portfolio-ready because they show not only model training, but also how the solution can be applied to realistic, ordered image sequences.

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
A custom **TinyVGG**-style CNN is implemented in PyTorch and tested in two variants:
- **Model 0:** TinyVGG without augmentation
- **Model 1:** TinyVGG with data augmentation

### 7. Training & Evaluation
The notebook trains the model for **10 epochs** using:
- `CrossEntropyLoss`,
- the Adam optimizer,
- and evaluation metrics including **loss, accuracy, and macro F1 score**.

### 8. Model Comparison
Results from `model_0` and `model_1` are compared visually through learning curves for:
- train/test loss,
- train/test accuracy,
- and train/test F1.

### 9. Custom Image Inference
The notebook downloads a standalone sample image, resizes it to `64x64`, runs it through `model_1`, and converts logits to prediction probabilities using `torch.softmax`.

### 10. Sequence-Level Page-Flip Detection
A major update in the notebook is the move from single-image prediction to **folder-based sequence analysis**. Ordered frames are evaluated using helper functions that:
- score each frame,
- compute a **flip ratio**,
- and decide whether a sequence contains flipping based on `min_flip_frames` and `min_flip_ratio`.

---

## 🧠 Model Approach

The final workflow centers on a custom **TinyVGG** architecture — a compact convolutional neural network well-suited for learning visual patterns from relatively small images.

Why this is meaningful:
- it shows understanding of CNN building blocks,
- it keeps the solution lightweight and interpretable,
- and it demonstrates hands-on model development rather than relying only on high-level APIs.

---

## 📈 Results

From the saved notebook outputs, the updated workflow achieved strong performance on the augmented `model_1` run:

| Metric | Verified Result |
|---|---:|
| Best Test Accuracy | **0.9531** |
| Best Test Macro F1 | **0.9530** |
| Best Performing Epoch | **8** |
| Logged Training Time | **485.109 seconds** |

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

- MonReader is a strong example of applied ML work that combines **technical depth, structured experimentation, and business relevance**.
