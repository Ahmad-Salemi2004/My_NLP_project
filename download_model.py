import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_pretrained_model():
    """Download base model if fine-tuned model doesn't exist."""
    model_path = "./models/fine_tuned_bart"
    
    if not os.path.exists(model_path):
        print("Downloading base BART model...")
        os.makedirs(model_path, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"Base model saved to {model_path}")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_pretrained_model()
