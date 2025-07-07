import os
import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def compute_max_lengths(dataset, tokenizer):
    """
    Computes the optimal input and target lengths using percentiles
    via batched .map, casting batches to plain Python lists of str.
    """
    print("Computing optimal sequence lengths…")
    combined = concatenate_datasets([dataset["train"], dataset["test"]])

    # 1) Source (dialogue) lengths
    tokenized_inputs = combined.map(
        lambda batch: tokenizer(
            [str(d) for d in batch["dialogue"]],  # force list[str]
            truncation=True
        ),
        batched=True,
        remove_columns=["dialogue", "summary"]
    )
    input_lengths = [len(ids) for ids in tokenized_inputs["input_ids"]]
    max_source_length = int(np.percentile(input_lengths, 85))
    print(f"Max source length: {max_source_length}")

    # 2) Target (summary) lengths
    tokenized_targets = combined.map(
        lambda batch: tokenizer(
            [str(s) for s in batch["summary"]],  # force list[str]
            truncation=True
        ),
        batched=True,
        remove_columns=["dialogue", "summary"]
    )
    target_lengths = [len(ids) for ids in tokenized_targets["input_ids"]]
    max_target_length = int(np.percentile(target_lengths, 90))
    print(f"Max target length: {max_target_length}")

    return max_source_length, max_target_length


def preprocess_function(sample, tokenizer, max_source_length, max_target_length, padding="max_length"):
    """
    Tokenizes input dialogues and summaries, handling any None values.
    """
    # Safely build inputs: replace None dialogues with empty string
    inputs = [
        "summarize: " + (item if isinstance(item, str) else "")
        for item in sample["dialogue"]
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=padding,
        truncation=True
    )

    # Safely build target summaries: replace None with empty string
    raw_summaries = [
        item if isinstance(item, str) else ""
        for item in sample["summary"]
    ]
    labels = tokenizer(
        text_target=raw_summaries,
        max_length=max_target_length,
        padding=padding,
        truncation=True
    )

    if padding == "max_length":
        # mask padding token for loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(model_id, output_dir):
    dataset = load_dataset("knkarthick/samsum")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Train size: {len(dataset['train'])} | Test size: {len(dataset['test'])}")
    max_source_length, max_target_length = compute_max_lengths(dataset, tokenizer)
    print(f"Max input length: {max_source_length}, Max target length: {max_target_length}")

    def preprocess(sample):
        return preprocess_function(sample, tokenizer, max_source_length, max_target_length)

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["dialogue", "summary", "id"])

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset["train"].save_to_disk(os.path.join(output_dir, "train"))
    tokenized_dataset["test"].save_to_disk(os.path.join(output_dir, "eval"))

    print("Tokenized datasets saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SAMSum dataset for FLAN-T5 training.")
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xxl", help="Base model ID")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save tokenized datasets")
    args = parser.parse_args()

    main(args.model_id, args.output_dir)
