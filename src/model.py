import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME  = "bert-base-uncased"
NUM_LABELS  = 2   # negative, positive
ID2LABEL    = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID    = {"NEGATIVE": 0, "POSITIVE": 1}


def load_model_for_training():
    """
    Load BERT with a classification head for fine-tuning.
    AutoModelForSequenceClassification adds a linear layer on top
    of BERT's [CLS] token output — that's where the sentiment prediction comes from.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def load_model_for_inference(model_path: str):
    """
    Load a fine-tuned model from disk or HuggingFace Hub.
    Used by the API and dashboard at inference time.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict(text: str, model, tokenizer, device="cpu") -> dict:
    """
    Run inference on a single string.
    Returns a dict with label and confidence score.
    """
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert raw logits to probabilities with softmax
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    predicted_id = probs.argmax().item()

    return {
        "label":      ID2LABEL[predicted_id],
        "confidence": round(probs[predicted_id].item(), 4),
        "scores": {
            "NEGATIVE": round(probs[0].item(), 4),
            "POSITIVE": round(probs[1].item(), 4),
        }
    }