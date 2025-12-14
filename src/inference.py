import torch
from transformers import pipeline

class TextSummarizer:
    def __init__(self, model_path=None, device=None):
        """Initialize the summarizer."""
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        if model_path:
            self.summarizer = pipeline(
                "summarization",
                model=model_path,
                tokenizer=model_path,
                device=device
            )
        else:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
    
    def summarize(self, text, max_length=150, min_length=40, **kwargs):
        """Generate summary for given text."""
        output = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            **kwargs
        )
        return output[0]['summary_text']
