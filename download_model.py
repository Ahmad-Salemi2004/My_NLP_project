from transformers import BartForConditionalGeneration, BartTokenizer
import os

def download_and_setup_model():
    print("Downloading BART-large-cnn model and tokenizer...")
    
    # Load model and tokenizer directly from Hugging Face
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Save them to your local 'models' directory
    save_path = "./models/fine_tuned_bart"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Model and tokenizer saved to '{save_path}'")

if __name__ == "__main__":
    download_and_setup_model()
