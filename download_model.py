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

def create_progress_callback():
    """Create a simple progress tracker"""
    start_time = time.time()
    last_update = start_time
    
    def update_progress(current, total):
        nonlocal last_update
        current_time = time.time()
        
        # Update every 2 seconds or when done
        if current_time - last_update >= 2 or current == total:
            progress = (current / total) * 100
            elapsed = current_time - start_time
            
            # Estimate remaining time
            if current > 0 and elapsed > 0:
                speed = current / elapsed / (1024 * 1024)  # MB/s
                remaining = (total - current) / (current / elapsed) if current > 0 else 0
                
                print(f"\rDownloading: {progress:.1f}% | "
                      f"{current/(1024*1024):.1f}/{total/(1024*1024):.1f} MB | "
                      f"Speed: {speed:.1f} MB/s | "
                      f"ETA: {remaining:.0f}s", end='', flush=True)
            
            last_update = current_time
        
        if current == total:
            print()  # New line when complete
    
    return update_progress

def download_model_with_progress(model_name, save_path, force=False):
    """Download model with progress tracking"""
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    # Check if model already exists
    if os.path.exists(save_path) and not force:
        required_files = ["config.json", "pytorch_model.bin"]
        existing_files = all(os.path.exists(os.path.join(save_path, f)) 
                           for f in required_files)
        
        if existing_files:
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
                return model, tokenizer
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                logger.info("Will download fresh copy...")
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Downloading to: {save_path}")
    
    try:
        # Try to use the built-in progress tracking if available
        logger.info("Starting download... (this may take 5-15 minutes)")
        
        # Method 1: Try with resume_download and progress tracking
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
        
        # Save locally
        logger.info("Saving model locally...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        
        # Try alternative approach
        logger.info("Trying alternative download method...")
        try:
            # Simpler approach without progress tracking
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=model_name,
                local_dir=save_path,
                resume_download=True
            )
            
            # Load from local
            model = BartForConditionalGeneration.from_pretrained(save_path)
            tokenizer = BartTokenizer.from_pretrained(save_path)
            
            return model, tokenizer
            
        except Exception as e2:
            logger.error(f"Alternative download also failed: {e2}")
            return None, None

def verify_model(model, tokenizer):
    """Verify the downloaded model works correctly"""
    logger.info("Verifying model...")
    
    try:
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
