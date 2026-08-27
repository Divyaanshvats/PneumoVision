# 🫁 PneumoVision

### Chest X-Ray Pneumonia Classification with DenseNet121

🔗 **Live Demo:** https://1389-34-139-222-208.ngrok-free.app/

PneumoVision is a PyTorch-based deep learning project that classifies chest
X-rays into **NORMAL** and **PNEUMONIA** using transfer learning with
**DenseNet121**.

## 🔬 Method

Chest X-Rays
     ↓
Preprocessing (224×224 + ImageNet Normalization)
     ↓
Pretrained DenseNet121
     ↓
Train Classification Head
     ↓
Fine-Tune Final Dense Block
     ↓
Internal Evaluation
     ↓
Independent External Validation
     ↓
Streamlit Deployment

Training: Adam, CrossEntropyLoss, batch size 32, 5 epochs initial training
+ 5 epochs fine-tuning, NVIDIA Tesla T4.

## 📊 Results
Internal Test
Metric	Score
Accuracy	85.42%
Precision	82.29%
Recall	97.69%
Specificity	64.96%
F1	89.33%
ROC-AUC	95.57%
External Validation

The frozen model was evaluated on an independent dataset of 504 X-rays
(204 NORMAL, 300 PNEUMONIA) without retraining.

Metric	Score
Accuracy	61.11%
Recall	99.33%
Specificity	4.90%
ROC-AUC	75.68%

The performance drop demonstrates cross-dataset domain shift. While
pneumonia sensitivity remained high, the model misclassified most normal
external images as pneumonia.

## 🌐 Deployment

A Streamlit application allows users to upload a chest X-ray and receive
a NORMAL/PNEUMONIA prediction with confidence using the trained model.

## 🛠️ Tech Stack

Python · PyTorch · Torchvision · DenseNet121 · Scikit-learn · Streamlit ·
Google Colab · NVIDIA Tesla T4

## 🔭 Future Work

This DenseNet121 implementation serves as the baseline for comparing
Ensemble Learning, Federated Learning, and Quantum Machine Learning.

### ⚠️ For educational and research purposes only. Not a medical diagnostic tool.
