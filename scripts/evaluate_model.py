import argparse
import numpy as np
import evaluate
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

def set_seed(seed: int):
    """Set deterministic seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_peft_model(peft_model_dir, device):
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
    model.to(device)
    return model, tokenizer

def decode_labels(labels_batch, tokenizer):
    """
    Convert a batch of label token IDs into readable strings, ignoring -100.
    """
    texts = []
    for labels in labels_batch:
        cleaned = [l if l != -100 else tokenizer.pad_token_id for l in labels]
        texts.append(tokenizer.decode(cleaned, skip_special_tokens=True))
    return texts

def evaluate_model(model, tokenizer, dataset, device, batch_size, max_new_tokens, top_p):
    """
    Generate predictions in batches and compute ROUGE scores.
    """
    metric = evaluate.load("rouge")
    predictions, references = [], []
    
    total = len(dataset)
    for start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        end = min(start + batch_size, total)
        batch = dataset[start:end]
        
        # Prepare inputs
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p
            )
        
        # Decode predictions
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = decode_labels(batch["labels"], tokenizer)
        
        predictions.extend(preds)
        references.extend(refs)

    print("Computing ROUGE scores...")
    rouge = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    print(f"\nROUGE Scores:")
    print(f"ROUGE-1: {rouge['rouge1'] * 100:.2f}%")
    print(f"ROUGE-2: {rouge['rouge2'] * 100:.2f}%")
    print(f"ROUGE-L: {rouge['rougeL'] * 100:.2f}%")
    print(f"ROUGE-Lsum: {rouge['rougeLsum'] * 100:.2f}%")

def main(peft_model_dir, eval_data_dir, batch_size, max_new_tokens, top_p, seed):
    # Set deterministic seed
    set_seed(seed)

    # Device agnostic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading test dataset...")
    dataset = load_from_disk(eval_data_dir).with_format("numpy")

    # Load model and tokenizer
    model, tokenizer = load_peft_model(peft_model_dir, device)

    # Evaluate
    evaluate_model(
        model, tokenizer, dataset, device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        top_p=top_p
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PEFT-finetuned FLAN-T5 model with batching and deterministic sampling.")
    parser.add_argument("--peft_model_dir", type=str, default="results/model", help="Path to trained LoRA adapter")
    parser.add_argument("--eval_data_dir", type=str, default="data/eval", help="Path to saved tokenized test data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(
        args.peft_model_dir,
        args.eval_data_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        seed=args.seed
    )