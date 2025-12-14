#!/bin/bash

echo "Setting up Text Summarization Project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

# Create necessary directories
echo "Creating project directories..."
mkdir -p models/fine_tuned_bart
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
mkdir -p notebooks
mkdir -p logs
mkdir -p results

# Check if model needs to be downloaded
if [ ! -f "models/fine_tuned_bart/config.json" ]; then
    echo "Downloading base model..."
    python download_model.py
fi

echo "Setup completed!"
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: python app.py"
echo "3. Open browser to: http://localhost:5000"
