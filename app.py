from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import requests


app = Flask(__name__)
CORS(app)


# Home route to check server
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running ✅"})


# ---------------- IMAGE CLASSIFICATION ----------------
model_name = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)


@app.route("/classify-image", methods=["POST"])
def classify_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400


        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")


        inputs = processor(images=image, return_tensors="pt")


        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()


        labels = ["Battery", "Biological", "Cardboard", "Clothes", "Glass",
                  "Metal", "Paper", "Plastic", "Shoes", "Trash"]


        predicted_index = torch.argmax(probs)
        predicted_label = labels[predicted_index.item()]
        confidence = probs[predicted_index].item() * 100


        recyclable = ["Paper", "Cardboard", "Plastic", "Metal", "Glass"]
        compostable = ["Biological"]
        trash = ["Battery", "Shoes", "Clothes"]


        if predicted_label in recyclable:
            general_label = "Recyclable"
        elif predicted_label in compostable:
            general_label = "Compostable"
        else:
            general_label = "Trash"


        return jsonify({"label": general_label, "confidence": round(confidence, 2)})


    except Exception as e:
        return jsonify({"error": str(e)}), 500


        


# ---------------- TEXT CLASSIFICATION ----------------


# ----------------------------
# Hugging Face Query Function (for text classification)
# ----------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_HEADERS = {"Authorization": "Bearer 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"}


def query_hf(payload):
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None




# ----------------------------
# Text Classification Route
# ----------------------------
@app.route("/classify-text", methods=["POST"])
def classify_text():
    try:
        data = request.json
        user_text = data.get("text", "")


        if not user_text:
            return jsonify({"error": "No text provided"}), 400


        # Call Hugging Face
        response = query_hf({"inputs": user_text})


        # Keyword-based mapping
        user_text_lower = user_text.lower()


        recyclable_keywords = ["paper", "plastic", "metal", "glass", "cardboard", "bottle", "can", "tin"]
        compostable_keywords = ["food", "banana", "apple", "vegetable", "fruit", "biological", "organic", "leaves"]
        trash_keywords = ["shoes", "clothes", "battery", "trash", "wrapper", "styrofoam"]


        label = "Trash"  # default
        if any(word in user_text_lower for word in recyclable_keywords):
            label = "Recyclable"
        elif any(word in user_text_lower for word in compostable_keywords):
            label = "Compostable"
        elif any(word in user_text_lower for word in trash_keywords):
            label = "Trash"


        # fallback if HF returns sentiment
        sentiment, score = "Unknown", 0.0
        if response and isinstance(response, list) and "label" in response[0]:
            sentiment = response[0]["label"]
            score = response[0]["score"]


        return jsonify({
            "label": label,
            "sentiment": sentiment,
            "confidence": round(score * 100, 2)
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500




# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
