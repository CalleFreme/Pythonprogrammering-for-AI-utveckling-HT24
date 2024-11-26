import json
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load JSON data."""
    with open(file_path, "r") as f:
        return json.load(f)

def preprocess_data(data, test_size=0.2):
    """Preprocess and split data into training and validation sets."""
    texts = [item["text"] for item in data]
    summaries = [item["summary"] for item in data]
    return train_test_split(texts, summaries, test_size=test_size, random_state=42)

def save_data(data, output_path):
    """Save preprocessed data to file."""
    with open(output_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    raw_data = load_data("../data/legal_texts.json")
    train_texts, val_texts, train_summaries, val_summaries = preprocess_data(raw_data)
    save_data({"train_texts": train_texts, "train_summaries": train_summaries}, "../data/processed/train.json")
    save_data({"val_texts": val_texts, "val_summaries": val_summaries}, "../data/processed/val.json")
