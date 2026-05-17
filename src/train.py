import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)

from config import *

df = pd.read_csv("data/train.csv")
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(example):
    model_inputs = tokenizer(
        str(example["input"]),
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        str(example["target"]),
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    label_ids = labels["input_ids"]

    # Ignorar padding durante el entrenamiento
    label_ids = [
        token if token != tokenizer.pad_token_id else -100
        for token in label_ids
    ]

    model_inputs["labels"] = label_ids

    return model_inputs

dataset_tokenizado = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=3e-4,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenizado
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Modelo entrenado correctamente")