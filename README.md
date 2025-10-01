FinSent Expainer
<img width="1883" height="898" alt="image" src="https://github.com/user-attachments/assets/c1784c2d-7a61-4d35-98a6-d120580ffc83" />
<img width="1821" height="843" alt="image" src="https://github.com/user-attachments/assets/b3d0c89e-f706-4739-b4d8-824b92f14fc8" />
<img width="1684" height="840" alt="image" src="https://github.com/user-attachments/assets/c6ac5703-f1b7-4707-967a-9674ae4d169c" />



FinSent Expainer is a financial sentiment explanation and analysis framework. It integrates advanced transformer-based architectures with recurrent and attention mechanisms to extract nuanced insights from financial texts.

🚀 Features

Roberta + BiGRU + Attention + CentralNet + Sigmoid for robust financial sentiment classification.

SpanBERT for cause extraction.

Complaint Labeling for identifying and tagging user complaints.

Versatility Layer for multi-domain adaptability.

Emotion Detector for fine-grained emotional state analysis.

🏗️ Architecture
1. Base Encoder

RoBERTa: Contextual embedding generator for financial text.

BiGRU: Captures sequential dependencies from both directions.

Attention Mechanism: Highlights key tokens driving predictions.

2. CentralNet Fusion

Integrates embeddings from RoBERTa, BiGRU, and Attention.

Provides a unified representation.

Sigmoid Layer: Used for multi-label financial sentiment classification.

3. Auxiliary Modules

SpanBERT: Extracts spans responsible for causes in financial sentiment shifts.

Complaint Labeling Head: Flags complaint-type sentences.

Versatility Head: Adapts model to multiple financial subdomains.

Emotion Detection Head: Identifies emotions such as anxiety, optimism, fear, etc.

🔄 Workflow

Input: Raw financial text (tweets, reports, complaints, etc.).

Text Encoder: RoBERTa encodes → BiGRU → Attention applied.

CentralNet Fusion: Combines signals for sentiment classification.

Downstream Tasks:

SpanBERT extracts causes.

Complaint labeler flags issues.

Versatility module adapts outputs.

Emotion detector refines emotional states.

Output:

Sentiment Score(s)

Cause Spans

Complaint Labels

Emotion Tags

📊 Tasks
Task	Model Component	Output Example
Financial Sentiment	RoBERTa + BiGRU + Attention + CentralNet	Positive / Negative / Neutral
Cause Extraction	SpanBERT	“due to high market volatility”
Complaint Labeling	Custom classifier head	Complaint / Non-Complaint
Versatility	Multi-domain module	Adapted for banking / insurance etc
Emotion Detection	Auxiliary classifier	Anxiety, Optimism, Anger, Fear
📦 Future Extensions

Explainable AI visualization (highlighting causal spans).

Domain-specific fine-tuning (banking, stock markets, insurance).

Integration with financial dashboards.
