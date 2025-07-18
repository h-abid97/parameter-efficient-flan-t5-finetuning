# 🦙 PEFT Fine-Tuning FLAN-T5-XXL on SAMSum with LoRA

This project demonstrates how to apply **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** on the **FLAN-T5-XXL** model, specifically trained on the [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) dataset. The entire training pipeline is optimized for 8-bit with `bitsandbytes`, reducing memory footprint and cost.


## 💡 Project Highlights

- 💾 Fine-tunes FLAN-T5-XXL with LoRA in 8-bit using `bitsandbytes`
- ⚡ Efficient training via `peft`, `transformers`, and `accelerate`
- 📉 Evaluation using ROUGE metrics
- 📁 Modular scripts for preprocessing, training, and evaluation


## 🛠️ Setup

### 1. Create and activate environment

```bash
conda create -n peft-flan-t5 python=3.9 -y
conda activate peft-flan-t5
```

### 2. Install dependencies

```bash
git clone https://github.com/h-abid97/parameter-efficient-flan-t5-finetuning.git
cd parameter-efficient-flan-t5-finetuning
pip install -r requirements.txt
```

## 📊 Dataset
The [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) dataset contains dialogue-summary pairs and is used for training a summarization model. It is collection of about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English.

```
{
  "id": "13818513",
  "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
  "dialogue": "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"
}
```

## 🚀 How to Run
### 1. Preprocess and tokenize dataset

```bash
python scripts/preprocess.py
```

#### This will download and tokenize the dataset to the data/ directory.

### 2. Fine-tune the model

```bash
python scripts/train.py \
  --model_id philschmid/flan-t5-xxl-sharded-fp16 \
  --data_dir data \
  --output_dir results/model \
  --epochs 5 \
  --lr 1e-4
```

### 3. Evaluate the model

```bash
python scripts/evaluate.py \
  --peft_model_dir results/model \
  --eval_data_dir data/eval
```

#### This will output ROUGE scores on the test set.

## 📁 Project Structure

```
peft-FLAN-t5/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   └── [processed datasets]
├── results/
│   └── model/
├── requirements.txt
└── README.md
```
