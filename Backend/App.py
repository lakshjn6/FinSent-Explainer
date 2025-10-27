import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
from tqdm import tqdm
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import date, datetime
import os

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the backend directory path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# ==============================
# 1️⃣ COMPLAINT DETECTION MODEL
# ==============================
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiGRUModel, self).__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        output, hidden = self.bigru(x)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, gru_output):
        scores = self.attention(gru_output)
        attn_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(attn_weights * gru_output, dim=1)
        return context_vector, attn_weights

class CentralBrain(nn.Module):
    def __init__(self, roberta_hidden_size=768, gru_hidden_size=128, num_classes=2):
        super(CentralBrain, self).__init__()
        self.central_fc = nn.Linear(roberta_hidden_size + gru_hidden_size * 2, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, cls_embedding, attn_output):
        combined = torch.cat((cls_embedding, attn_output), dim=1)
        central_output = torch.relu(self.central_fc(combined))
        logits = self.classifier(central_output)
        return logits

class FullModel(nn.Module):
    def __init__(self, roberta, gru_hidden_size=128, num_classes=2):
        super(FullModel, self).__init__()
        self.roberta = roberta
        self.bigru = BiGRUModel(input_size=768, hidden_size=gru_hidden_size)
        self.attention = Attention(gru_hidden_size)
        self.central_brain = CentralBrain(768, gru_hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_embedding = outputs.pooler_output
        gru_output, _ = self.bigru(sequence_output)
        attn_output, attn_weights = self.attention(gru_output)
        logits = self.central_brain(cls_embedding, attn_output)
        return logits

def predict_complaint_from_text(text):
    tokenizer_path = os.path.join(BACKEND_DIR, "saved_model-Copy1")
    model_path = os.path.join(BACKEND_DIR, "saved_model/full_model.pth")
    
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    roberta = RobertaModel.from_pretrained("roberta-base").to(device)
    model = FullModel(roberta, gru_hidden_size=128, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    encoded = tokenizer(text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return str(predicted_class)

# ==============================
# 2️⃣ DOMAIN CLASSIFICATION MODEL
# ==============================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, bert_output):
        attention_scores = self.attention(bert_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended_output = torch.sum(attention_weights * bert_output, dim=1)
        return attended_output, attention_weights

def load_domain_models():
    domain_model_dir = os.path.join(BACKEND_DIR, "Domain_Model")
    
    tokenizer = BertTokenizer.from_pretrained(os.path.join(domain_model_dir, "tokenizer"))
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.load_state_dict(torch.load(os.path.join(domain_model_dir, "bert_model.pth"), map_location=device))
    attention_layer = AttentionLayer()
    attention_layer.load_state_dict(torch.load(os.path.join(domain_model_dir, "attention_layer.pth"), map_location=device))

    with open(os.path.join(domain_model_dir, "xgb_model.pkl"), 'rb') as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(domain_model_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)

    return bert_model.to(device), attention_layer.to(device), tokenizer, xgb_model, label_encoder

def extract_features(texts, bert_model, attention_layer, tokenizer, device, max_len=128):
    bert_model.eval()
    attention_layer.eval()
    features = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            seq_output = outputs.last_hidden_state
            attended_output, _ = attention_layer(seq_output)
            features.append(attended_output.cpu().numpy().flatten())
    return np.array(features)

def predict_domain(text, bert_model, attention_layer, tokenizer, xgb_model, label_encoder):
    feats = extract_features([text], bert_model, attention_layer, tokenizer, device)
    probs = xgb_model.predict_proba(feats)[0]
    pred_idx = np.argmax(probs)
    return label_encoder.inverse_transform([pred_idx])[0]

# ==============================
# 3️⃣ EMOTION DETECTION MODEL
# ==============================
emotion_model_dir = os.path.join(BACKEND_DIR, "Emotion_model")

svm_classifier = joblib.load(os.path.join(emotion_model_dir, 'svm_emotion_classifier.pkl'))
emotion_classes = joblib.load(os.path.join(emotion_model_dir, 'emotion_classes.pkl'))

tokenizer_emotion = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model_emotion = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model_emotion.eval()

def get_bert_embeddings_emotion(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer_emotion.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            outputs = bert_model_emotion(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
    return np.array(embeddings)

def predict_emotion_label(text):
    embeddings = get_bert_embeddings_emotion([text])
    return svm_classifier.predict(embeddings)[0]

# ==============================
# 4️⃣ SENTIMENT ANALYSIS MODEL
# ==============================
sentiment_model_dir = os.path.join(BACKEND_DIR, "Sentiment_Model")

tokenizer_sentiment = BertTokenizer.from_pretrained(os.path.join(sentiment_model_dir, "saved_bert_model"))
bert_model_sentiment = BertModel.from_pretrained(os.path.join(sentiment_model_dir, "saved_bert_model"))
bert_model_sentiment.to(device)
bert_model_sentiment.eval()

def get_bert_embeddings_sentiment(text_list, max_len=128):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer_sentiment(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_len
            ).to(device)
            
            outputs = bert_model_sentiment(**inputs)
            last_hidden_state = outputs.last_hidden_state  
            cls_embedding = last_hidden_state[:, 0, :].cpu().numpy().flatten()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

input_dim = 768  
output_dim = 3   

sentiment_model = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, output_dim)
)
sentiment_model.to(device)
sentiment_model.load_state_dict(torch.load(os.path.join(sentiment_model_dir, "best_model.pth"), map_location=device))
sentiment_model.eval()

def predict_sentiment(text):
    embedding = get_bert_embeddings_sentiment([text])[0]
    X = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = sentiment_model(X)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[predicted_class]

# ==============================
# LOAD DATA AND PROCESS
# ==============================
print("=" * 50)
print("Loading data...")
print("=" * 50)

# Try to find the Reviews.xlsx file
excel_path = os.path.join(PROJECT_ROOT, "Test_Review.xlsx")
if not os.path.exists(excel_path):
    excel_path = os.path.join(BACKEND_DIR, "Test_Review.xlsx")
    
if not os.path.exists(excel_path):
    print(f"ERROR: Reviews.xlsx not found in {PROJECT_ROOT} or {BACKEND_DIR}")
    print("Please place Reviews.xlsx in the FinsentExplainer folder or Backend folder")
    exit(1)

df = pd.read_excel(excel_path)
print(f"Total records in file: {len(df)}")




print("\nProcessing complaints...")
df["Complaint Label"] = df["Domain Complaint/Opinion"].apply(predict_complaint_from_text)

print("Loading domain models...")
bert_model_domain, attention_layer, tokenizer_domain, xgb_model, label_encoder = load_domain_models()

print("Predicting domains...")
df["Domain"] = df["Domain Complaint/Opinion"].apply(
    lambda x: predict_domain(x, bert_model_domain, attention_layer, tokenizer_domain, xgb_model, label_encoder)
)

print("Predicting emotions...")
df["Emotion"] = df["Domain Complaint/Opinion"].apply(predict_emotion_label)

print("Predicting sentiments...")
df["Sentiment"] = df["Domain Complaint/Opinion"].apply(predict_sentiment)

print("\n" + "=" * 50)
print("Data processing complete!")
print("=" * 50)
print("\nSample output:")
print(df[['Domain Complaint/Opinion', 'Complaint Label', 'Domain', 'Emotion', 'Sentiment']].head())

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)
CORS(app)

@app.route('/get-data', methods=['GET'])
def get_data():
    data_json = df.to_dict(orient='records')
    return jsonify(data_json)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running", 
        "records": len(df),
        "backend_dir": BACKEND_DIR
    })

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Starting Flask server on http://127.0.0.1:5000")
    print("=" * 50)
    print(f"Backend directory: {BACKEND_DIR}")
    print(f"Data file: {excel_path}")
    print(f"Total records available: {len(df)}")
    print("\nAPI Endpoints:")
    print("  - GET http://127.0.0.1:5000/get-data")
    print("  - GET http://127.0.0.1:5000/health")
    print("=" * 50 + "\n")
    app.run(port=5000, debug=True)
