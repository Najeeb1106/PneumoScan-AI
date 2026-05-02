# ✅ Pneumonia CNN Assignment — Checklist
### Track your progress step by step

---

## 📋 PHASE 1: Setup

- [x] Created a Google account (for Colab)
- [x] Opened [Google Colab](https://colab.research.google.com)
- [x] Enabled GPU → `Runtime → Change Runtime Type → GPU → Save`
- [x] Created a [Kaggle](https://www.kaggle.com) account
- [x] Downloaded `kaggle.json` API key from Kaggle settings

---

## 📋 PHASE 2: Dataset

- [x] Uploaded `kaggle.json` to Colab using `files.upload()`
- [x] Ran Kaggle install and setup commands
- [x] Downloaded the Pneumonia dataset successfully
- [x] Unzipped the dataset
- [x] Verified folder structure exists:
  - [x] `chest_xray/train/NORMAL/`
  - [x] `chest_xray/train/PNEUMONIA/`
  - [x] `chest_xray/val/`
  - [x] `chest_xray/test/`

---

## 📋 PHASE 3: Code Execution

- [x] Imported all libraries without errors
- [x] Loaded training data with `ImageDataGenerator`
- [x] Loaded validation data
- [x] Loaded test data
- [x] Built CNN model using `Sequential`
- [x] Ran `model.summary()` — verified layers are correct
- [x] Compiled model with Adam optimizer
- [x] Trained model for 10 epochs — no crashes
- [x] Saved training history in `history` variable

---

## 📋 PHASE 4: Results & Evaluation

- [x] Plotted **Accuracy Graph** (train vs validation)
- [x] Plotted **Loss Graph** (train vs validation)
- [x] Evaluated model on test data
- [x] Printed **Test Accuracy** (target: 85%+)
- [x] Generated **Confusion Matrix**
- [x] Printed **Classification Report** (Precision, Recall, F1)

---

## 📋 PHASE 5: Understanding (Viva Ready)

- [x] I can explain what CNN is in simple words
- [x] I can explain what each layer does (Conv2D, MaxPooling, Flatten, Dense, Dropout)
- [x] I can explain what an epoch is
- [x] I can explain what overfitting is and how Dropout prevents it
- [x] I can explain what the confusion matrix shows
- [x] I can explain why we normalize pixel values (divide by 255)
- [x] I can explain what data augmentation is and why we use it
- [x] I can explain the difference between train/val/test sets
- [x] I can explain what sigmoid activation does in the output layer
- [x] I can explain what binary_crossentropy loss means

---

## 📋 PHASE 6: Submission Prep

- [x] Code is clean and has comments
- [x] All graphs are saved/screenshots taken
- [x] README.md is ready
- [x] Can demo the model running live if required
- [ ] Prepared 2-3 line summary of the project for viva:

> *"I trained a CNN model on chest X-ray images from Kaggle to detect Pneumonia. The model achieved over 85% accuracy on test data. It uses three convolutional layers for feature extraction, followed by dense layers for classification."*

---

## 🚨 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `kaggle.json not found` | Re-upload the file using `files.upload()` |
| `No such file or directory: chest_xray/train` | Check unzip worked, verify folder name |
| `CUDA out of memory` | Reduce batch_size from 32 to 16 |
| `Module not found` | Run `!pip install <module_name>` |
| Accuracy stuck at 50% | Check class_mode is `'binary'`, check dataset loaded correctly |
| Training very slow | Make sure GPU is enabled in Runtime settings |

---

## 📊 Results Log (Fill after running)

| Metric | Your Result |
|--------|-------------|
| Train Accuracy (Epoch 10) | 94.29% |
| Val Accuracy (Epoch 10) | 87.50% |
| Test Accuracy | 87.34% |
| Test Loss | 0.4306 |
| Training Time | ~50 mins (CPU/GPU) |

---

**Status:** 🟢 Complete
