import os
import argparse
import numpy as np
import evaluate
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig


def load_peft_model(peft_model_dir):
    """
    Load base model and merge with trained PEFT adapter.
    """
    print("Loading PEFT model...")
    config = PeftConfig.from_pretrained(peft_model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_dir, device_map="auto")
    model.eval()
    return model, tokenizer


def decode_labels(labels, tokenizer):
    """
    Convert label token IDs into readable strings, ignoring -100 values.
    """
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    return tokenizer.decode(labels, skip_special_tokens=True)


def evaluate_model(model, tokenizer, dataset, max_target_length=50):
    """
    Generate predictions and compute ROUGE scores.
    """
    metric = evaluate.load("rouge")
    predictions, references = [], []

    for sample in tqdm(dataset, desc="Evaluating"):
        input_ids = sample["input_ids"].unsqueeze(0).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_target_length,
                do_sample=True,
                top_p=0.9
            )
        pred = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        label = decode_labels(sample["labels"], tokenizer)
        predictions.append(pred)
        references.append(label)

    print("Computing ROUGE scores...")
    rouge = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    print(f"\n📊 ROUGE Scores:")
    print(f"ROUGE-1: {rouge['rouge1'] * 100:.2f}%")
    print(f"ROUGE-2: {rouge['rouge2'] * 100:.2f}%")
    print(f"ROUGE-L: {rouge['rougeL'] * 100:.2f}%")
    print(f"ROUGE-Lsum: {rouge['rougeLsum'] * 100:.2f}%")


def main(peft_model_dir, eval_data_dir):
    print("Loading test dataset...")
    dataset = load_from_disk(eval_data_dir).with_format("torch")

    model, tokenizer = load_peft_model(peft_model_dir)
    evaluate_model(model, tokenizer, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PEFT-finetuned FLAN-T5 model.")
    parser.add_argument("--peft_model_dir", type=str, default="results/model", help="Path to trained LoRA adapter")
    parser.add_argument("--eval_data_dir", type=str, default="data/eval", help="Path to saved tokenized test data")
    args = parser.parse_args()

    main(args.peft_model_dir, args.eval_data_dir)
