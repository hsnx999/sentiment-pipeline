# Sentiment Analysis Pipeline

A production-style NLP pipeline that fine-tunes BERT on the SST-2 dataset
and serves predictions through a FastAPI REST API with a live Streamlit dashboard.

---

## Results

    Training examples:   4,000
    Evaluation examples: 400
    Epochs:              1
    Final accuracy:      87.25%
    Final F1 score:      87.26%

Model published at: huggingface.co/hsnx000/bert-sentiment-sst2

---

## How it works

    1. Fine-tune   BERT-base-uncased trained on SST-2 sentiment data
                   Classification head learns to map [CLS] token → positive/negative
                   Loss dropped from 0.626 → 0.262 over 1 epoch

    2. Serve       FastAPI wraps the model in a REST API
                   Model loads once at startup — not per request
                   /predict for single text, /predict/batch for up to 50 texts

    3. Dashboard   Streamlit UI calls the API in real time
                   Confidence bar chart per prediction
                   Batch mode with sentiment distribution pie chart

---

## Tech stack

    Library          Role
    HuggingFace      BERT model, tokenizer, Trainer, Hub
    PyTorch          Model training and inference
    FastAPI          REST API serving layer
    Streamlit        Interactive dashboard
    Plotly           Confidence and distribution charts
    Scikit-learn     Accuracy and F1 evaluation metrics
    SST-2 (GLUE)     Stanford Sentiment Treebank training data

---

## Run it locally

Prerequisites: Python 3.10+

    git clone https://github.com/hsnx999/sentiment-pipeline.git
    cd sentiment-pipeline
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Start the API (Terminal 1):

    uvicorn api:app --reload

Start the dashboard (Terminal 2):

    streamlit run app.py

Open http://localhost:8501

API docs available at http://localhost:8000/docs

---

## Project structure

    sentiment-pipeline/
    ├── train.py          Fine-tune BERT and push to HuggingFace Hub
    ├── api.py            FastAPI REST API with /predict and /predict/batch
    ├── app.py            Streamlit dashboard with charts
    ├── src/
    │   ├── model.py      Model loading and inference logic
    │   └── data.py       SST-2 loading and tokenization
    └── requirements.txt

---

## What I learned building this

- How transfer learning works — fine-tuning a pre-trained model vs training from scratch
- Why the classification head initialises randomly while BERT weights are pre-trained
- How tokenization works — subword tokens, padding, attention masks
- How to wrap a PyTorch model in a production REST API using FastAPI
- Why models should load at startup not per-request
- How to publish a fine-tuned model to HuggingFace Hub