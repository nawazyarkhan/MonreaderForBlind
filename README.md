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
- **Best Verified Result:** **95.31% test accuracy** and **0.9530 macro F1** on the saved `model_1` run
- **Sequence Signal:** Flip-folder testing shows **>86% flip ratio**, while a non-flip folder is reported at about **0.98%**
- **Tools:** Python, PyTorch, Torchvision, scikit-learn, Matplotlib
- **Portfolio Value:** Demonstrates end-to-end computer vision project ownership, from training to practical inference

---

## 📌 Project Summary

MonReader is a **PyTorch-based computer vision project** that classifies page images as `flip` or `notflip` and extends those predictions to **ordered frame sequences**. It demonstrates the full ML workflow: data preparation, CNN development, experiment comparison, metric-driven evaluation, and practical inference.

## 📚 Table of Contents

- [✨ At a Glance](#-at-a-glance)
- [📌 Project Summary](#-project-summary)
- [🎯 Business / Use-Case Relevance](#-business--use-case-relevance)
- [🧰 Tech Stack](#-tech-stack)
- [🆕 Updated Notebook Highlights](#-updated-notebook-highlights)
- [🔄 Step-by-Step Project Flow](#-step-by-step-project-flow)
- [🧠 Model Approach](#-model-approach)
- [📈 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [▶️ How to Run](#️-how-to-run)
- [✅ What This Project Demonstrates](#-what-this-project-demonstrates)
- [💼 Concluding Note](#-concluding-note)

### Why this project stands out

- **Custom modeling:** a hand-built `TinyVGG` CNN rather than a black-box workflow
- **Clear experimentation:** baseline vs augmentation with tracked accuracy and macro F1
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

- **10-epoch training** for `model_1` with Adam and `CrossEntropyLoss`
- **model comparison** across loss, accuracy, and **macro F1**
- a dedicated final section for **page-flip detection**
- **sequence utilities** such as `predict_flipping_sequence(...)` and `predict_flipping_from_folder(...)`
- **threshold-based decision logic** using `min_flip_frames` and `min_flip_ratio`

Together, these updates show a project that moves beyond training into **usable inference logic**.

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

### 9. Detecting If a Page Is Flipping or Not
The final workflow performs **folder-based sequence analysis**. Ordered frames are scored one by one, a **flip ratio** is computed, and the sequence is classified using `min_flip_frames` and `min_flip_ratio` thresholds.

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

For hiring managers and recruiters, MonReader highlights:
- hands-on experience with **PyTorch and computer vision**,
- the ability to **design experiments and interpret metrics**,
- and practical thinking by extending a notebook model into **usable detection logic**.

In short, this is a concise, portfolio-ready example of building and applying a deep learning solution **end to end**.
