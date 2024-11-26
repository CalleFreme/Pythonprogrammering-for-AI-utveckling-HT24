from sklearn.metrics import rouge_score
import json

def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def evaluate_model(model, data):
    predictions = [model.summarize(text) for text in data["val_texts"]]
    scores = rouge_score.rouge_l(predictions, data["val_summaries"])
    return scores

if __name__ == "__main__":
    from inference import LegalSummarizer
    summarizer = LegalSummarizer("../models/fine_tuned_model")
    val_data = load_data("../data/processed/val.json")
    scores = evaluate_model(summarizer, val_data)
    print("ROUGE Scores:", scores)
