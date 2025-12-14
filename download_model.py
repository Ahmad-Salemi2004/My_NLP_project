from transformers import BartForConditionalGeneration, BartTokenizer
import os

def download_and_save_model():
    # Create models directory if it doesn't exist
    os.makedirs("models/fine_tuned_bart", exist_ok=True)
    
    # Load pre-trained model and tokenizer
    print("Downloading BART model and tokenizer...")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Save to models directory
    print("Saving model to models/fine_tuned_bart/...")
    model.save_pretrained("models/fine_tuned_bart")
    tokenizer.save_pretrained("models/fine_tuned_bart")
    
    print("Model download complete!")

if __name__ == "__main__":
    download_and_save_model()
