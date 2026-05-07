# Fine-tune BERT on SST-2 and push to HuggingFace Hub.
# Run with: python train.py
# Training takes ~30-60 min on CPU, ~5-10 min on GPU.

import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import login

from src.data import get_tokenizer, load_sst2, tokenize_dataset
from src.model import load_model_for_training

# ── Config ─────────────────────────────────────────────────────────────────
HF_REPO_NAME = "hsnx000/bert-sentiment-sst2"
OUTPUT_DIR   = "model_output"


def compute_metrics(eval_pred):
    """
    Called by Trainer after each evaluation step.
    Returns accuracy and F1 so you can track training progress.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1":       f1_score(labels, predictions, average="weighted"),
    }


def main():
    # ── Login to HuggingFace ───────────────────────────────
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env — add it before training")
    login(token=hf_token)
    print("Logged in to HuggingFace")

    # ── Load data ──────────────────────────────────────────
    print("Loading and tokenizing SST-2...")
    tokenizer  = get_tokenizer()
    dataset    = load_sst2()
    tokenized  = tokenize_dataset(dataset, tokenizer)

    train_dataset = tokenized["train"]
    eval_dataset  = tokenized["validation"]

    # Use a subset for faster training — remove these lines to train on full data
    train_dataset = train_dataset.select(range(4000))
    eval_dataset  = eval_dataset.select(range(400))
    print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)}")

    # ── Load model ─────────────────────────────────────────
    print("Loading BERT...")
    model = load_model_for_training()

    # ── Training config ────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,          # standard BERT fine-tuning rate
        weight_decay=0.01,           # regularisation to avoid overfitting
        eval_strategy="epoch",       # evaluate after every epoch
        save_strategy="epoch",
        load_best_model_at_end=True, # keep the best checkpoint
        metric_for_best_model="accuracy",
        logging_steps=50,
        push_to_hub=True,
        hub_model_id=HF_REPO_NAME,
        hub_token=hf_token,
        report_to="none",            # disable wandb/mlflow for now
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Train ──────────────────────────────────────────────
    print("Starting training...")
    trainer.train()

    # ── Evaluate ───────────────────────────────────────────
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['eval_f1']:.4f}")

    # ── Push to HuggingFace ────────────────────────────────
    print(f"\nPushing model to {HF_REPO_NAME}...")
    trainer.push_to_hub()
    tokenizer.save_pretrained(OUTPUT_DIR)
    tokenizer.push_to_hub(HF_REPO_NAME, token=hf_token)
    print("Done! Model available at huggingface.co/" + HF_REPO_NAME)


if __name__ == "__main__":
    main()