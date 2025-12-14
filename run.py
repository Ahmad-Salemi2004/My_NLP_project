import os
import sys
import subprocess

def main():
    """Run the Flask application."""
    print("Starting Text Summarization Application...")
    
    # Check if requirements are installed
    try:
        import flask
        import torch
        import transformers
    except ImportError:
        print("Missing dependencies. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check if model exists
    model_path = "./models/fine_tuned_bart"
    if not os.path.exists(model_path):
        print("Model not found. Running download script...")
        subprocess.check_call([sys.executable, "download_model.py"])
    
    # Run the Flask app
    print("\nStarting web server...")
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    subprocess.check_call([sys.executable, "app.py"])

if __name__ == "__main__":
    main()
