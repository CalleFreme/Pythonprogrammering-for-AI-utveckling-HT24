# python app.py
# curl -x -H "Content-type: application/jsoin" -d '{"text": "Your legal text here"}'
# Eller liknande

from flask import Flask, request, jsonify
from src.inference import LegalSummarizer

app = Flask(__name__)
summarizer = LegalSummarizer("../models/fine_tuned_model")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    summary = summarizer.summarize(data["text"])
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
