# Project Report: Pneumonia Detection using CNN
**Subject:** Artificial Intelligence | BS Software Engineering (6th Semester)  
**Author:** Najeeb Ullah  

---

## 1. Executive Summary
This project aims to automate the detection of Pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model achieved a high test accuracy of **87.34%**, with a specific focus on high sensitivity (98% recall for Pneumonia), ensuring that very few infected cases go undetected.

---

## 2. Introduction
Pneumonia is a life-threatening lung infection. Timely diagnosis via chest X-rays is crucial. Traditional diagnosis requires expert radiologists, which can be time-consuming or unavailable in remote areas. This AI-based solution provides an automated, rapid screening tool to assist medical professionals.

---

## 3. Dataset Analysis
The dataset was sourced from Kaggle (Chest X-Ray Images - Pneumonia).
- **Total Images:** ~5,800 images.
- **Classes:** Normal (Healthy) and Pneumonia (Infected).
- **Split:** 80% Training, 10% Validation, 10% Testing.
- **Preprocessing:** 
    - Images were resized to **150x150 pixels**.
    - Pixel values were normalized from [0, 255] to **[0, 1]**.
    - Data Augmentation (Rotation, Zoom, Horizontal Flip) was applied to the training set to prevent overfitting.

---

## 4. Methodology (CNN Architecture)
A Deep Learning model was built using a **Sequential CNN architecture**:

1.  **Convolutional Layers:** Three layers (32, 64, and 128 filters) were used to extract spatial features (edges, textures, and lung patterns).
2.  **Activation (ReLU):** Used to introduce non-linearity.
3.  **MaxPooling:** Downsampled the feature maps to reduce computational load and highlight key features.
4.  **Flatten Layer:** Converted the 2D feature maps into a 1D vector.
5.  **Fully Connected (Dense) Layer:** 128 neurons for classification logic.
6.  **Dropout (50%):** Randomly disabled neurons during training to ensure the model generalizes well to new data.
7.  **Output Layer:** Single neuron with **Sigmoid** activation for binary classification (0 = Normal, 1 = Pneumonia).

---

## 5. Training Results
The model was trained for **10 epochs** using the **Adam Optimizer**.

| Metric | Value |
|--------|-------|
| Training Accuracy | 94.29% |
| Validation Accuracy | 87.50% |
| **Final Test Accuracy** | **87.34%** |
| Test Loss | 0.4306 |

---

## 6. Performance Evaluation
### Confusion Matrix Analysis
The model showed exceptional performance in detecting pneumonia:
- **Pneumonia Recall: 98%** — This means the model correctly identified 98% of all actual pneumonia cases. In medical AI, this is the most critical metric as it minimizes "False Negatives."
- **Normal Precision: 95%** — When the model predicts "Normal," it is correct 95% of the time.

### Observations
The Accuracy and Loss graphs showed steady improvement. The slight gap between training (94%) and validation (87%) indicates that while there was some overfitting, the **Dropout layer** and **Data Augmentation** kept the model stable enough to perform well on the unseen test data.

---

## 7. Conclusion
The CNN model successfully identifies pneumonia with high reliability. With an accuracy of **87.34%** and a recall of **98%**, the project demonstrates that deep learning can be a powerful tool in medical diagnostics. Future improvements could include using a larger dataset or transfer learning (e.g., VGG16) to further increase precision.
