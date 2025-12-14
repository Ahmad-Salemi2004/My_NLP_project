# Models Directory

This directory contains the fine-tuned BART model for text summarization.

## Model Information
- **Base Model**: facebook/bart-large-cnn
- **Fine-tuned on**: DialogSum dataset
- **Training Steps**: 3,116
- **Training Loss**: 1.26
- **Evaluation Loss**: 1.68

## File Structure
models/
├── fine_tuned_bart/ # Fine-tuned model files

│ ├── config.json # Model configuration

│ ├── pytorch_model.bin # Model weights

│ ├── tokenizer.json # Tokenizer configuration

│ └── ...

└── README.md # This file


## Usage
To use the fine-tuned model:
```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="./models/fine_tuned_bart",
    tokenizer="./models/fine_tuned_bart"
)
```

## Note
If the fine-tuned model is not available in this directory, the application will automatically use the base BART-large-cnn model from Hugging Face.
