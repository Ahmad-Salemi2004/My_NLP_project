from datasets import load_dataset
from transformers import AutoTokenizer

def load_dialogsum_dataset():
    """Load the DialogSum dataset from Hugging Face."""
    ds = load_dataset("knkarthick/dialogsum")
    return ds

def preprocess_function(batch, tokenizer, max_length=128):
    """Preprocess the dataset for summarization task."""
    source = batch['dialogue']
    target = batch['summary']
    
    # Tokenize source and target
    source_ids = tokenizer(
        source, 
        truncation=True, 
        padding='max_length', 
        max_length=max_length
    )
    target_ids = tokenizer(
        target, 
        truncation=True, 
        padding='max_length', 
        max_length=max_length
    )
    
    # Prepare labels (replace padding token id with -100 for loss calculation)
    labels = target_ids['input_ids']
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in labels
    ]
    
    return {
        'input_ids': source_ids['input_ids'],
        'attention_mask': source_ids['attention_mask'],
        'labels': labels
    }

def prepare_dataset(tokenizer_name="facebook/bart-large-cnn"):
    """Prepare the dataset for training."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load dataset
    dataset = load_dialogsum_dataset()
    
    # Preprocess dataset
    tokenized_dataset = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True
    )
    
    return tokenized_dataset, tokenizer
