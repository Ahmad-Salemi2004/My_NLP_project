#!/usr/bin/env python3
import os
import sys
import argparse
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print a nice banner"""
    print("\n" + "="*60)
    print("BART-LARGE-CNN MODEL DOWNLOADER")
    print("="*60)
    print("Model: facebook/bart-large-cnn")
    print("Size:  ~1.6 GB")
    print("Task:  Text Summarization")
    print("="*60 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['transformers', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"ERROR: Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install transformers torch")
        return False
    return True

def check_disk_space(required_mb=2000):
    """Check if there's enough disk space (2GB recommended)"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_mb = free // (1024 * 1024)
        
        if free_mb < required_mb:
            logger.warning(f"WARNING: Low disk space: {free_mb} MB available")
            logger.warning(f"   Recommended: {required_mb} MB for BART-large-cnn")
            
            if free_mb < 500:
                logger.error("ERROR: Not enough space (less than 500 MB)")
                return False
            else:
                response = input(f"   Continue anyway? (y/N): ").strip().lower()
                return response == 'y'
        
        logger.info(f"SUCCESS: Disk space available: {free_mb} MB")
        return True
    except Exception:
        logger.warning("WARNING: Could not check disk space, continuing anyway...")
        return True

def download_model_with_progress(model_name, save_path, force=False):
    """Download model with progress tracking"""
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    # Check if model already exists
    if os.path.exists(save_path) and not force:
        # Check for essential model files
        model_files = ["config.json", "pytorch_model.bin"]
        tokenizer_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
        
        essential_files_exist = any(os.path.exists(os.path.join(save_path, f)) for f in model_files)
        
        if essential_files_exist:
            logger.info(f"SUCCESS: Model already exists at: {save_path}")
            logger.info("  Use --force to re-download")
            
            response = input("  Load existing model? (Y/n): ").strip().lower()
            if response not in ['', 'y', 'yes']:
                logger.info("Download cancelled.")
                return None, None
            
            try:
                logger.info("Loading existing model...")
                model = BartForConditionalGeneration.from_pretrained(save_path)
                tokenizer = BartTokenizer.from_pretrained(save_path)
                logger.info("SUCCESS: Existing model loaded successfully")
                return model, tokenizer
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                logger.info("Will download fresh copy...")
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Downloading to: {save_path}")
    
    try:
        logger.info("Starting download... (this may take 5-15 minutes)")
        
        # Download with progress indication
        print("Downloading model files...")
        
        # Method 1: Try with resume_download
        try:
            model = BartForConditionalGeneration.from_pretrained(
                model_name,
                resume_download=True,
                force_download=force
            )
            
            tokenizer = BartTokenizer.from_pretrained(
                model_name,
                resume_download=True,
                force_download=force
            )
            
            logger.info("SUCCESS: Model downloaded from Hugging Face Hub")
            
        except Exception as e:
            logger.warning(f"Standard download failed: {e}")
            logger.info("Trying alternative method...")
            
            # Method 2: Try with huggingface_hub
            try:
                from huggingface_hub import snapshot_download
                
                print("Downloading model files (this may take a while)...")
                snapshot_download(
                    repo_id=model_name,
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
                )
                
                # Load from local directory
                model = BartForConditionalGeneration.from_pretrained(save_path)
                tokenizer = BartTokenizer.from_pretrained(save_path)
                
                logger.info("SUCCESS: Model downloaded using huggingface_hub")
                
            except Exception as e2:
                logger.error(f"Alternative download failed: {e2}")
                raise
        
        # Save locally
        logger.info("Saving model locally for future use...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"SUCCESS: Model saved to {save_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        
        # Clean up partial downloads
        if os.path.exists(save_path):
            try:
                import shutil
                shutil.rmtree(save_path)
                logger.info("Cleaned up partial download")
            except:
                pass
        
        return None, None

def verify_model(model, tokenizer):
    """Verify the downloaded model works correctly"""
    logger.info("Verifying model...")
    
    try:
        # Import torch for verification
        import torch
        
        # Simple test
        test_text = "The quick brown fox jumps over the lazy dog. This is a simple test to verify the model works correctly."
        
        inputs = tokenizer(
            test_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        logger.info("SUCCESS: Model verification successful!")
        logger.info(f"  Test input:  '{test_text[:50]}...'")
        logger.info(f"  Test output: '{summary}'")
        
        return True
        
    except ImportError:
        logger.error("ERROR: Torch not available for verification")
        return False
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def create_readme(save_path, model_name):
    """Create a README file in the model directory"""
    readme_path = os.path.join(save_path, "README.md")
    
    readme_content = f"""# BART-Large-CNN Model for Text Summarization

## Model Information
- **Model**: {model_name}
- **Task**: Text summarization
- **Download Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Size**: ~1.6 GB

## Usage

### Python Code
```python
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the model
model = BartForConditionalGeneration.from_pretrained('{save_path}')
tokenizer = BartTokenizer.from_pretrained('{save_path}')

# Generate summary
text = "Your long text here..."
inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
summary_ids = model.generate(inputs['input_ids'], max_length=150)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
