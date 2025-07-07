import argparse
from random import randrange
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig


def load_peft_model(peft_model_dir):
    """
    Load PEFT adapter with base model and tokenizer.
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


def infer_sample(model, tokenizer, max_new_tokens=50):
    """
    Generate summary from a random test dialogue.
    """
    dataset = load_dataset("samsum")
    sample = dataset["test"][randrange(len(dataset["test"]))]
    input_text = sample["dialogue"]

    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nDialogue:")
    print(input_text)
    print("\nGenerated Summary:")
    print(summary)
    print("\nReference Summary:")
    print(sample["summary"])


def main(peft_model_dir):
    model, tokenizer = load_peft_model(peft_model_dir)
    infer_sample(model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a random SAMSum test sample.")
    parser.add_argument("--peft_model_dir", type=str, default="results/model", help="Path to trained PEFT adapter")
    args = parser.parse_args()

    main(args.peft_model_dir)