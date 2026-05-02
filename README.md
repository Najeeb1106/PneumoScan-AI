# 🫁 Pneumonia Detection using CNN
### AI Subject Assignment | BS Software Engineering — 6th Semester

---

## 📌 Project Overview

This project trains a **Convolutional Neural Network (CNN)** to detect **Pneumonia** from chest X-ray images using deep learning. The model classifies X-rays into two categories:

- ✅ **NORMAL** — Healthy lungs
- 🔴 **PNEUMONIA** — Infected lungs

---

## 📂 Dataset

- **Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** ~5,800 chest X-ray images
- **Classes:** NORMAL / PNEUMONIA
- **Split:**
  - `train/` — Model learning
  - `val/` — Validation during training
  - `test/` — Final evaluation

---

## 🧠 What is CNN?

A **Convolutional Neural Network (CNN)** is a type of AI model designed for image recognition. It works by:

1. Scanning images with small filters to detect **edges and patterns**
2. Combining patterns to identify **shapes and textures**
3. Making a final **decision** (Normal or Pneumonia)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| TensorFlow / Keras | Building and training the CNN model |
| NumPy | Numerical operations |
| Matplotlib | Plotting accuracy/loss graphs |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Evaluation metrics |
| Google Colab | Cloud-based training environment (GPU) |

---

## 🏗️ Model Architecture

```
Input Image (150x150x3)
        ↓
Conv2D (32 filters) → ReLU → MaxPooling
        ↓
Conv2D (64 filters) → ReLU → MaxPooling
        ↓
Conv2D (128 filters) → ReLU → MaxPooling
        ↓
Flatten
        ↓
Dense (128 neurons) → ReLU
        ↓
Dropout (50%)
        ↓
Dense (1 neuron) → Sigmoid
        ↓
Output: NORMAL (0) or PNEUMONIA (1)
```

---

## ⚙️ How to Run

### Step 1 — Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com)

### Step 2 — Enable GPU
`Runtime → Change Runtime Type → GPU → Save`

### Step 3 — Get Kaggle API Key
`Kaggle → Profile → Settings → API → Create New Token → Download kaggle.json`

### Step 4 — Run the Code
Paste and run each code block in order:
1. Install Kaggle & download dataset
2. Import libraries
3. Load and preprocess data
4. Build CNN model
5. Train the model (10 epochs)
6. Plot accuracy/loss graphs
7. Evaluate on test data
8. Generate confusion matrix

---

## 📊 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Test Accuracy | 85% – 92% |
| Training Time | 10 – 15 minutes (GPU) |
| Epochs | 10 |
| Batch Size | 32 |

---

## 📈 Output Graphs

- **Accuracy Graph** — Shows training vs validation accuracy over epochs
- **Loss Graph** — Shows training vs validation loss over epochs
- **Confusion Matrix** — Shows correct and incorrect predictions

---

## 🔑 Key Concepts Explained

| Term | Simple Meaning |
|------|---------------|
| Epoch | One full pass through all training images |
| Batch Size | Number of images processed at once |
| Dropout | Randomly disabling neurons to prevent memorization |
| Overfitting | Model memorizes training data, fails on new data |
| ReLU | Activation function — keeps positive values only |
| Sigmoid | Converts output to a probability (0 to 1) |
| Augmentation | Creating variations of images to improve learning |
| Normalization | Scaling pixel values from 0–255 to 0–1 |

---

## 👤 Author

**Najeeb Ullah**
BS Software Engineering — 6th Semester
Codrix.dev | Sargodha, Pakistan
