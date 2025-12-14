from transformers import (
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

def initialize_model(model_name="facebook/bart-large-cnn"):
    """Initialize the model for sequence-to-sequence learning."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def get_training_args(output_dir="./results"):
    """Define training arguments."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=True
    )
    return training_args

def train_model(model, tokenized_dataset, training_args):
    """Train the model."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
    )
    
    # Train the model
    trainer.train()
    
    return trainer

def save_model(model, tokenizer, save_path="./models/fine_tuned_bart"):
    """Save the trained model and tokenizer."""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
