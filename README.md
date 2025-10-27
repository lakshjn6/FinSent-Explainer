🧠 FinSentExplainer — Financial Sentiment & Complaint Analysis System

🎉 Project Completed: FinSentExplainer 🚀
An end-to-end NLP system that analyzes and interprets financial customer reviews using advanced deep learning and transformer-based models.

🔍 Project Overview

FinSentExplainer is designed to understand customer feedback in the financial domain — from identifying complaints to detecting underlying emotions and domains of issues.
This multi-task system uses state-of-the-art transformer architectures (BERT, RoBERTa) combined with hybrid deep learning and machine learning models.

🌟 Key Highlights
✅ Complaint Detection

Model: RoBERTa + BiGRU + Attention + CentralNet

Goal: Classify customer reviews as Complaint or Non-Complaint

Performance: Achieved 91% accuracy

😊 Sentiment Analysis

Model: BERT + ANN

Goal: Detect Positive, Negative, and Neutral sentiments

😠 Emotion Analysis

Model: BERT + SVC

Goal: Capture nuanced emotions (anger, frustration, happiness, etc.) from customer feedback

🏦 Domain Finder

Model: BERT + XGBoost

Goal: Identify issue domains such as Transaction, Loan, Credit Card, etc.

⚙️ Common Preprocessing

Applied consistent text-cleaning and preprocessing steps across all modules:

Tokenization & word splitting

Stopword removal

Lemmatization

Embedding generation using transformer encoders


System Architecture
Customer Review → Preprocessing → Model Pipeline → Multi-Task Outputs
       │                 │                  │
       ├── Complaint Detection (RoBERTa + BiGRU + Attention + CentralNet)
       ├── Sentiment Analysis (BERT + ANN)
       ├── Emotion Analysis (BERT + SVC)
       └── Domain Finder (BERT + XGBoost)

💻 Tech Stack

| Layer                 | Tools & Technologies                               |
| --------------------- | -------------------------------------------------- |
| **Frontend**          | React, Tailwind CSS                                |
| **Backend**           | Flask (Python)                                     |
| **Model Development** | PyTorch, Scikit-learn, Transformers (Hugging Face) |
| **Tools**             | Jupyter Notebook, VS Code                          |
| **Version Control**   | Git & GitHub                                       |

🎯 Learning & Impact

This project enhanced my understanding of:

Transformer architectures (BERT, RoBERTa)

Hybrid model fusion using CentralNet

Multi-level text classification in financial NLP

Practical integration of deep learning with web applications

🙏 Acknowledgment

A heartfelt thanks to Manish Pandey Sir for his invaluable guidance and support — especially for helping design the architecture inspired by cutting-edge research papers.


FinSentExplainer/
│
├── frontend/           # React + Tailwind frontend
├── backend/            # Flask backend API
├── notebooks/          # Model training notebooks
├── models/             # Saved model weights
├── data/               # Dataset samples (if public)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

