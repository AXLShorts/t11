from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

app = Flask(__name__)

# Load model and label encoder
@app.before_first_request
def load_model_and_label_encoder():
    global fine_tuned_model, label_encoder, tokenizer
    fine_tuned_model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('./label_encoder_classes.npy', allow_pickle=True)
    tokenizer = BertTokenizer.from_pretrained("./tokenizer")  # Load tokenizer from local file

@app.route('/predict', methods=['POST'])
def predict():
    user_input_symptom = request.json.get('symptom')
    if not user_input_symptom:
        return jsonify({'error': 'Please provide a symptom.'}), 400

    user_input_encoding = tokenizer(user_input_symptom, padding=True, truncation=True, return_tensors='pt', max_length=512, return_attention_mask=True, return_token_type_ids=True)

    with torch.no_grad():
        logits = fine_tuned_model(**user_input_encoding)
        probabilities = torch.nn.functional.softmax(logits.logits, dim=1).numpy()[0]
        predicted_labels = np.argsort(-probabilities)[:5]
        predicted_diseases = label_encoder.inverse_transform(predicted_labels)
        predicted_probabilities = probabilities[predicted_labels]

    predictions = [{'disease': disease, 'probability': probability * 100} for disease, probability in zip(predicted_diseases, predicted_probabilities)]

    return jsonify({'predictions': predictions})

# Render expects the app to be served using Gunicorn
if __name__ == '__main__':
    app.run(debug=True)
