from transformers import T5Tokenizer, T5ForConditionalGeneration

class LegalSummarizer:
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    summarizer = LegalSummarizer("../models/fine_tuned_model")
    test_text = "Your lengthy legal text here."
    print("Summary:", summarizer.summarize(test_text))
