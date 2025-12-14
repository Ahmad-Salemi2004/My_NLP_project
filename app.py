from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
USE_GPU = os.getenv('USE_GPU', 'True').lower() == 'true'
MODEL_PATH = os.getenv('MODEL_PATH', './models/fine_tuned_bart')
MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '1024'))
MAX_SUMMARY_LENGTH = int(os.getenv('MAX_SUMMARY_LENGTH', '150'))
MIN_SUMMARY_LENGTH = int(os.getenv('MIN_SUMMARY_LENGTH', '40'))

app = Flask(__name__)

# Load the model
def load_model():
    try:
        device = 0 if torch.cuda.is_available() else -1
        model_path = "./models/fine_tuned_bart"
        
        # Check if model exists
        if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
            print("Loading fine-tuned model...")
            summarizer = pipeline(
                "summarization", 
                model=model_path, 
                tokenizer=model_path, 
                device=device
            )
        else:
            print("Fine-tuned model not found. Loading base model...")
            summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn", 
                device=device
            )
        
        print("Model loaded successfully!")
        return summarizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

summarizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text.split()) < 50:
            return jsonify({'error': 'Text too short. Please provide at least 50 words.'}), 400
        
        if summarizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        summary = summarizer(
            text, 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        return jsonify({'summary': summary[0]['summary_text']})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': summarizer is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
