# 🎯 FinSentExplainer

<img width="1916" height="1079" alt="Screenshot 2025-10-24 153120" src="https://github.com/user-attachments/assets/4084b755-b56b-484d-ab77-4ff5f9dced2e" />
<img width="1919" height="1007" alt="Screenshot 2025-10-24 153109" src="https://github.com/user-attachments/assets/388c5757-27ad-4ace-9782-9e3c9680adab" />
<img width="1919" height="1015" alt="Screenshot 2025-10-24 153055" src="https://github.com/user-attachments/assets/7cafee5c-e5bb-4a9a-9c8a-e5ceb4137d57" />
<img width="1919" height="1001" alt="Screenshot 2025-10-24 153047" src="https://github.com/user-attachments/assets/61eed2d2-6c90-4160-9bca-0f605c4a9c84" />



### 🧠 End-to-End Financial Sentiment & Complaint Analysis System

FinSentExplainer is an **NLP-based project** designed to analyze and interpret financial customer reviews using **deep learning** and **transformer-based architectures**.  
It integrates multiple models to perform complaint detection, sentiment analysis, emotion recognition, and domain classification — all within a unified pipeline.

---

## 🔍 Key Highlights

### ✅ Complaint Detection  
Developed a **hybrid RoBERTa + BiGRU + Attention + CentralNet** model to classify reviews as *Complaint* or *Non-Complaint*, achieving **91% accuracy**.

### 😊 Sentiment Analysis  
Built a **BERT + ANN** model to detect *Positive*, *Negative*, and *Neutral* sentiments.

### 😠 Emotion Analysis  
Implemented **BERT + SVC** to capture underlying emotions in customer feedback.

### 🏦 Domain Finder  
Designed a **BERT + XGBoost** model to identify issue domains such as *Transaction*, *Loan*, *Credit Card*, etc.

---

## ⚙️ Common Preprocessing  
Applied consistent NLP preprocessing steps across all models:  
- Word splitting and tokenization  
- Stopword removal  
- Lemmatization  
- Embedding generation using transformer models  
- Sequence padding and truncation  

---

## 💻 Tech Stack

| Layer | Technologies |
|--------|---------------|
| **Frontend** | React, Tailwind CSS |
| **Backend** | Flask (Python) |
| **Machine Learning / NLP** | BERT, RoBERTa, BiGRU, CentralNet, XGBoost, SVC, ANN |
| **Tools** | Jupyter Notebook, VS Code |
| **Version Control** | GitHub |

---

## 📊 Results

- Achieved **91% accuracy** for Complaint vs Non-Complaint classification  
- Created modular architecture for multi-model integration  
- Built RESTful APIs for model inference and frontend visualization  

---

## 🧩 Architecture Overview
Frontend (React + Tailwind)
↓
Flask Backend API
↓
Model Pipeline:
├── Complaint Detector (RoBERTa + BiGRU + Attention + CentralNet)
├── Sentiment Analyzer (BERT + ANN)
├── Emotion Classifier (BERT + SVC)
└── Domain Finder (BERT + XGBoost)


