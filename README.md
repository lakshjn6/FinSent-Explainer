🧠 FinSentExplainer — Financial Sentiment & Complaint Analysis System
<img width="1916" height="1079" alt="Screenshot 2025-10-24 153120" src="https://github.com/user-attachments/assets/2e58a45e-c604-45ea-996a-c6615c17e30a" />
<img width="1919" height="1007" alt="Screenshot 2025-10-24 153109" src="https://github.com/user-attachments/assets/4a3e994d-4652-4f8a-b5a0-010347d7e153" />
<img width="1566" height="785" alt="Screenshot 2025-10-24 153101" src="https://github.com/user-attachments/assets/b3bffe5e-8cfc-4da6-8146-33f1a87065bf" />
<img width="1919" height="1001" alt="Screenshot 2025-10-24 153047" src="https://github.com/user-attachments/assets/f44077d3-b82d-4b95-b614-a9d5aa8f7581" />


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

