🫁 Multi-Class Lung Disease Detection using CNN + Explainable AI (XAI)

An explainable deep learning system that detects lung diseases from CT scan images using a CNN model and provides visual explanations using Grad-CAM.

📌 Project Overview

This project focuses on detecting lung diseases from CT scan images using deep learning. The model classifies images into multiple disease categories and improves trust using Explainable AI (XAI) techniques.

It helps in:

Early detection of lung diseases
Automated medical image analysis
Providing interpretable AI predictions

🎯 Objective
Build a CNN model for lung disease classification
Improve accuracy of CT scan-based diagnosis
Use Grad-CAM for model explainability
Provide risk-aware predictions

🧠 Technologies Used
Python
TensorFlow / Keras
CNN (Deep Learning)
OpenCV
NumPy
Matplotlib
Grad-CAM (XAI)

📂 Dataset
CT scan image dataset
Classes:
COVID-19
Pneumonia
Normal
Images are resized and normalized before training

⚙️ Project Workflow
Load dataset
Preprocess CT scan images
Build CNN model
Train and validate model
Evaluate performance
Predict disease
Generate Grad-CAM heatmaps

🧩 Model Architecture
Conv2D → feature extraction
MaxPooling → downsampling
Flatten → vector conversion
Dense layers → classification
Dropout → reduce overfitting
Softmax → output layer

🔍 Explainable AI (XAI)
Grad-CAM is used to:
        Highlight infected lung regions
        Show model decision areas
        Improve trust in predictions
        Visualize disease-affected zones
        
📊 Results
Accuracy: ~90% – 97%
Strong performance on CT scan classification
Grad-CAM shows clear infection regions
Reliable multi-class predictions

🚀 Installation
git clone https://github.com/your-username/lung-ct-xai.git
cd lung-ct-xai
pip install -r requirements.txt

▶️ Usage
python app/predict.py
or run training:
python train.py

📈 Output
Disease prediction: COVID-19 / Pneumonia / Normal
Grad-CAM heatmap visualization
Model accuracy metrics

🔮 Future Scope
Web app for CT scan upload
Real-time hospital integration
Mobile deployment
Larger dataset training
AI-assisted diagnosis system

📜 License
This project is for educational and research purposes only.
