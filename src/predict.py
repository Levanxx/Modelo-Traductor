from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import *

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_LENGTH
    )

    outputs = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result