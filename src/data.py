from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128  # max tokens per sentence — 128 is enough for short reviews


def get_tokenizer():
    """Load the BERT tokenizer. Downloads once, cached locally after."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_sst2():
    """
    Load the SST-2 dataset from HuggingFace.
    Returns a DatasetDict with 'train' and 'validation' splits.

    SST-2 labels:
      0 = negative
      1 = positive
    """
    dataset = load_dataset("glue", "sst2")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """
    Apply the tokenizer to every example in the dataset.

    truncation=True  — cut sequences longer than MAX_LENGTH
    padding=True     — pad shorter sequences to MAX_LENGTH
                       so all batches have the same shape
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    # batched=True processes 1000 examples at a time — much faster
    tokenized = dataset.map(tokenize_fn, batched=True)

    # Remove raw text column — BERT only needs token IDs
    tokenized = tokenized.remove_columns(["sentence", "idx"])

    # Rename label → labels (what HuggingFace Trainer expects)
    tokenized = tokenized.rename_column("label", "labels")

    # Tell PyTorch to return tensors
    tokenized.set_format("torch")

    return tokenized