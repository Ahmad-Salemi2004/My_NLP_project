# Text Summarization with Fine-Tuned BART

## Project Overview
This project implements a text summarization system using a fine-tuned BART model. The model is trained on the DialogSum dataset to generate concise summaries of dialogue text. The project includes both the training pipeline and a web interface for real-time summarization.

## Features
- Fine-tuned BART-large-cnn model on dialog summarization
- Web interface for text summarization
- REST API endpoint for programmatic access
- Model evaluation using ROUGE metrics
- Support for both CPU and GPU inference

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/text-summarization-project.git
cd text-summarization-project
```
## Install dependencies:
```bash
pip install -r requirements.txt
```
## Download NLTK data:
```python
import nltk
nltk.download('punkt')
```
# Usage

## Running the Web Application
```bash
python app.py
```

## Using the API
```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here..."}'
```

# Training the Model
## The training code is available in the Jupyter notebook:
```bash
jupyter notebook notebooks/text_summarization_finetuning.ipynb
```

## Project Structure
```text
.
├── app.py              # Flask web application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks
├── static/           # Static web files
├── templates/        # HTML templates
└── tests/           # Unit tests
```

## Model Details
. Base Model: facebook/bart-large-cnn

. Dataset: DialogSum (knkarthick/dialogsum)

. Training Epochs: 2

. Batch Size: 8

. Sequence Length: 128 tokens

## Performance
