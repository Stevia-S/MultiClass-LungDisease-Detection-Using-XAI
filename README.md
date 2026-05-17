
# 🫁 Explainable Deep Learning-Based Lung Disease Detection using ResNet50–VGG16 Fusion

🧠 Python • TensorFlow • Transfer Learning • ResNet50-VGG16 Fusion • Grad-CAM • Medical AI

An end-to-end deep learning system that detects lung diseases from CT scan images using a CNN-based architecture and provides **explainable AI (XAI) visualizations** to improve interpretability in medical diagnosis.

---

## 📊 Project Overview

Medical image diagnosis requires high accuracy and interpretability. Traditional deep learning models often achieve strong performance but lack transparency in decision-making.

This project presents an **Explainable Deep Learning-Based Lung Disease Detection System** using a **ResNet50–VGG16 Fusion Architecture** with **Transfer Learning** and **Grad-CAM Explainability**.

The system is designed to:

- Classify CT scan images
- Detect multiple lung diseases
- Visualize affected regions using Grad-CAM heatmaps
- Improve interpretability in AI-assisted medical diagnosis

It is designed to simulate real-world **AI-assisted radiology systems** for medical image analysis.

---

### 🔍 Key Features

- ResNet50–VGG16 Fusion Model
- Transfer Learning-Based Feature Extraction
- Grad-CAM Explainable AI Visualization
- Streamlit-Based Web Application
- Multi-Class CT Scan Classification

---

### 🩺 Diseases Detected

- COVID
- Normal
- Pneumonia

---

# 🎯 Problem Statement

Manual analysis of CT scans is:

* Time-consuming
* Dependent on expert availability
* Prone to human error in complex cases

This project addresses these challenges by building a deep learning model that automatically:

* Detects lung disease patterns
* Provides interpretable visual explanations
* Assists medical decision-making

---

# 🏗️ Pipeline Architecture

```
CT Scan Image Input
        │
        ▼
┌──────────────────────┐
│ Image Preprocessing  │  ← Resize, normalization
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│   ResNet50 + VGG16   │  ← Feature extraction
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Classification Head  │  ← Softmax output
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Explainability (XAI) │  ← Grad-CAM heatmaps
└────────┬─────────────┘
         │
         ▼
 Prediction + Visualization Output
```

---

# 🧩 System Components

## 🔬 Image Processing

* CT scan resizing and normalization
* Noise reduction for better feature learning

---

## 🧠 Deep Learning Model

* CNN-based architecture
* Automatic feature extraction from CT images
* Multi-class classification

---

## 🔥 Explainable AI (XAI)

* Grad-CAM heatmaps
* Highlights infected lung regions
* Improves model transparency

---

## 🎯 Output System

* Disease prediction
* Confidence score
* Visual explanation map

---

# 🧠 Model Architecture

| Layer        | Description          |
| ------------ | -------------------- |
| Conv2D       | Feature extraction   |
| MaxPooling   | Spatial reduction    |
| Dropout      | Prevent overfitting  |
| Flatten      | Vector conversion    |
| Dense Layers | Classification       |
| Softmax      | Output probabilities |

---

# 📊 Model Performance

| Metric           | Score                |
| ---------------- | -------------------- |
| Accuracy         | ~90% – 97%           |
| Precision        | High                 |
| Recall           | High                 |
| Interpretability | Enabled via Grad-CAM |

> Performance varies based on dataset quality and training configuration.

---

# 🔍 Explainable AI (XAI)

Grad-CAM is used to:

* Highlight infected lung regions
* Show decision-making areas of CNN
* Improve trust in predictions
* Assist radiologists in validation

---

# ⚠️ Key Challenges Addressed

* Similar visual patterns between lung diseases
* Reducing false negatives in medical diagnosis
* Improving interpretability of deep learning models
* Handling limited labeled medical datasets

---

# 🚀 Getting Started

## 📦 Installation

```bash id="inst2"
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

---

## ▶️ Run Project

```bash id="run3"
python train.py
python predict.py
```

---

# 📁 Project Structure

```
CT-Lung-Disease-XAI/
│
├── dataset/
├── models/
├── utils/
│   ├── gradcam.py
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── gradcam_result.png
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 📈 Results

* CNN achieves strong classification performance
* Grad-CAM consistently highlights infected regions
* Reliable multi-class lung disease prediction
* Suitable for AI-assisted diagnosis systems

---

# 🔮 Future Improvements

* Web-based CT scan upload system
* Real-time hospital integration
* Mobile deployment
* Larger dataset training
* Advanced XAI (Grad-CAM++)

---

# 🧠 Key Concepts Demonstrated

* Convolutional Neural Networks (CNN)
* Medical image preprocessing
* Multi-class classification
* Explainable AI (Grad-CAM)
* Deep learning model evaluation

---

# 📜 License

This project is for academic and research purposes only.

---


