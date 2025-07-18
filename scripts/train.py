import os
import argparse
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig, 
)

def load_model_with_lora(model_id):
    """
    Load the base FLAN-T5 model in 8-bit and prepare it with LoRA.
    """
    print("Loading base model...")
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype="bfloat16",
    )

    # Prepare for LoRA training on k-bit (8-bit) weights
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main(model_id, data_dir, output_dir, epochs, lr):
    print("Loading tokenizer and tokenized dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))

    model = load_model_with_lora(model_id)
    model.config.use_cache = False  # avoid warning during training

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard",
        fp16=False,
        bf16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 with PEFT LoRA.")
    parser.add_argument("--model_id", type=str, default="philschmid/flan-t5-xxl-sharded-fp16", help="LoRA-compatible base model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with preprocessed tokenized data")
    parser.add_argument("--output_dir", type=str, default="results/model", help="Where to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    main(args.model_id, args.data_dir, args.output_dir, args.epochs, args.lr)
