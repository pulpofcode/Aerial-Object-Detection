# 🛸 Aerial Object Classification – Bird vs Drone

## 📌 Overview
This project implements a deep learning–based computer vision system to classify aerial images as **Bird** or **Drone**.
The solution targets **aerial surveillance, security monitoring, and wildlife protection**, where accurately distinguishing drones from birds is critical.

The project follows an end-to-end machine learning workflow:
- Data understanding and preprocessing
- Custom CNN baseline modeling
- Performance diagnosis using precision/recall
- Recall-focused optimization
- Transfer learning with MobileNetV2
- Final deployment using Streamlit

---

## 🎯 Problem Statement
In aerial surveillance scenarios (airports, restricted zones, wildlife areas), drones and birds often appear visually similar in the sky.
Misclassifying drones as birds can pose serious security risks.

**Goal:**
Build a robust image classification system that accurately identifies **drones**, prioritizing **high recall** while maintaining strong overall performance.

---

## 🧠 Key Features
- Binary image classification (Bird vs Drone)
- Custom CNN built from scratch as a baseline
- Recall-focused optimization using:
  - Class-weighted loss
  - Decision threshold tuning
- Transfer learning with **MobileNetV2**
- Final model achieves **97% accuracy** and **96% drone recall** on unseen test data
- Interactive Streamlit web app for real-time inference

---

## 🗂️ Dataset
- **Type:** RGB aerial images (.jpg)
- **Classes:** Bird, Drone

### Dataset Split
| Split | Bird | Drone |
|------|------|-------|
| Train | 1414 | 1248 |
| Validation | 217 | 225 |
| Test | 121 | 94 |

Dataset structure follows PyTorch `ImageFolder` format.

---

## 🏗️ Project Structure
```
aerial_object_classification/
│
├── app.py                     # Streamlit deployment
├── train.py                   # Custom CNN training
├── train_transfer.py          # Transfer learning (MobileNetV2)
├── eval.py                    # Test set evaluation
├── mobilenet_transfer.pth     # Final trained model
│
├── dataset/                   #Dataset is not shared in this repo
│   ├── train/
│   ├── val/
│   └── test/
│
├── requirements.txt
└── README.md
```

---

## 🧪 Modeling Approach

### 1️⃣ Custom CNN (Baseline)
- Built from scratch using PyTorch
- Used as a learning and comparison baseline
- Achieved ~79% test accuracy
- Diagnosed limitation: **low drone recall**

### 2️⃣ Recall-Focused Optimization
- Introduced class-weighted loss
- Tuned prediction threshold
- Improved drone recall from **63% → 88%**

### 3️⃣ Transfer Learning (Final Model)
- Model: **MobileNetV2 (ImageNet pretrained)**
- Feature extractor frozen
- Custom binary classification head
- Final performance on test set:
  - **Accuracy:** 97%
  - **Drone Recall:** 96%
  - **Drone Precision:** 97%

---

## 📊 Final Test Results

| Metric | Value |
|-------|-------|
| Accuracy | 97% |
| Drone Precision | 97% |
| Drone Recall | 96% |
| Drone F1-score | 96% |

**Confusion Matrix:**
```
[[118   3]
 [  4  90]]
```

---

## 🚀 Streamlit Deployment

The trained model is deployed using Streamlit for real-time inference.

### Run the App
```bash
streamlit run app.py
```

### Features
- Upload an aerial image
- Get prediction: **Bird 🐦 / Drone 🚁**
- View confidence score
- Security-focused thresholding

---

## 🛠️ Tech Stack
- Python
- PyTorch
- TorchVision
- NumPy
- scikit-learn
- Streamlit
- PIL

---

## 🔍 Key Learnings
- Accuracy alone is insufficient for security applications
- Recall optimization is critical when missing a threat is costly
- Transfer learning dramatically improves performance on limited datasets
- Proper evaluation on unseen data is essential before deployment

---

## 👤 Author
**Harshit Jaiswal**

This project was built as an end-to-end deep learning system with a strong focus on real-world applicability and evaluation-driven improvements.
