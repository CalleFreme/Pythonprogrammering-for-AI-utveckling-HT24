from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

def load_data(file_path):
    """Load preprocessed data."""
    import json
    with open(file_path, "r") as f:
        return json.load(f)

def tokenize_data(data, tokenizer, max_length=512):
    """Tokenize data for model training."""
    inputs = tokenizer(data["train_texts"], max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(data["train_summaries"], max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    return Dataset.from_dict({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]})

def fine_tune_model(train_dataset, val_dataset, model_name="t5-small"):
    """Fine-tune T5 model on custom dataset."""
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir="../models/fine_tuned_model",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    model.save_pretrained("../models/fine_tuned_model")
    tokenizer.save_pretrained("../models/fine_tuned_model")

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_data = load_data("../data/processed/train.json")
    val_data = load_data("../data/processed/val.json")
    
    train_dataset = tokenize_data(train_data, tokenizer)
    val_dataset = tokenize_data(val_data, tokenizer)
    
    fine_tune_model(train_dataset, val_dataset)
