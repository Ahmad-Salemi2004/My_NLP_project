import os
import sys
import torch
import logging
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Application configuration
class Config:
    """Configuration class for the application."""
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', './models/fine_tuned_bart')
    BASE_MODEL = os.getenv('BASE_MODEL', 'facebook/bart-large-cnn')
    USE_GPU = os.getenv('USE_GPU', 'True').lower() == 'true'
    
    # Summarization parameters
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '1024'))
    MAX_SUMMARY_LENGTH = int(os.getenv('MAX_SUMMARY_LENGTH', '150'))
    MIN_SUMMARY_LENGTH = int(os.getenv('MIN_SUMMARY_LENGTH', '40'))
    LENGTH_PENALTY = float(os.getenv('LENGTH_PENALTY', '2.0'))
    NUM_BEAMS = int(os.getenv('NUM_BEAMS', '4'))
    
    # Input validation
    MIN_WORDS = int(os.getenv('MIN_WORDS', '30'))
    MAX_WORDS = int(os.getenv('MAX_WORDS', '2000'))

# Initialize config
config = Config()

def setup_device():
    """Configure and return the device for model inference."""
    if config.USE_GPU and torch.cuda.is_available():
        device = 0  # Use first GPU
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        return device
    else:
        device = -1  # Use CPU
        logger.info("Using CPU for inference")
        return device

def load_model():
    """
    Load the text summarization model.
    Attempts to load fine-tuned model first, falls back to base model.
    """
    try:
        device = setup_device()
        
        # Check if fine-tuned model exists
        model_exists = False
        if os.path.exists(config.MODEL_PATH):
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            existing_files = os.listdir(config.MODEL_PATH)
            model_exists = all(f in existing_files for f in required_files)
        
        if model_exists:
            logger.info(f"Loading fine-tuned model from: {config.MODEL_PATH}")
            model = pipeline(
                "summarization",
                model=config.MODEL_PATH,
                tokenizer=config.MODEL_PATH,
                device=device,
                framework="pt"
            )
            model_type = "fine-tuned"
        else:
            logger.info(f"Loading base model: {config.BASE_MODEL}")
            model = pipeline(
                "summarization",
                model=config.BASE_MODEL,
                device=device,
                framework="pt"
            )
            model_type = "base"
        
        logger.info(f"Model loaded successfully ({model_type})")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        
        # Try loading with CPU as fallback
        try:
            logger.info("Attempting to load model with CPU...")
            model = pipeline(
                "summarization",
                model=config.BASE_MODEL,
                device=-1,
                framework="pt"
            )
            logger.info("Model loaded successfully with CPU")
            return model
        except Exception as e2:
            logger.error(f"Failed to load model even with CPU: {str(e2)}")
            return None

# Global model instance
logger.info("Initializing text summarization application...")
summarizer = load_model()

def validate_input(text):
    """
    Validate input text.
    Returns (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "No text provided"
    
    # Clean and count words
    cleaned_text = ' '.join(text.split())
    words = cleaned_text.split()
    word_count = len(words)
    
    if word_count < config.MIN_WORDS:
        return False, f"Text too short. Please provide at least {config.MIN_WORDS} words."
    
    if word_count > config.MAX_WORDS:
        return False, f"Text too long. Please provide less than {config.MAX_WORDS} words."
    
    # Check character length
    char_count = len(cleaned_text)
    if char_count > config.MAX_INPUT_LENGTH * 5:  # Approximate token count
        return False, f"Text exceeds maximum length. Please shorten your text."
    
    return True, ""

@app.route('/')
def home():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    API endpoint for text summarization.
    Expects JSON: {'text': 'your text here'}
    Returns JSON: {'summary': 'generated summary', 'stats': {...}}
    """
    try:
        # Parse request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        
        # Validate input
        is_valid, error_msg = validate_input(text)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Check model availability
        if summarizer is None:
            return jsonify({'error': 'Model not available. Please try again later.'}), 503
        
        # Generate summary
        logger.info(f"Generating summary for {len(text.split())} words...")
        
        try:
            summary_result = summarizer(
                text,
                max_length=config.MAX_SUMMARY_LENGTH,
                min_length=config.MIN_SUMMARY_LENGTH,
                length_penalty=config.LENGTH_PENALTY,
                num_beams=config.NUM_BEAMS,
                early_stopping=True,
                truncation=True,
                no_repeat_ngram_size=3
            )
            
            summary = summary_result[0]['summary_text']
            
            # Calculate statistics
            original_words = len(text.split())
            summary_words = len(summary.split())
            compression_ratio = original_words / summary_words if summary_words > 0 else 0
            
            response = {
                'summary': summary,
                'stats': {
                    'original_length': original_words,
                    'summary_length': summary_words,
                    'compression_ratio': round(compression_ratio, 2),
                    'reduction_percentage': round((1 - (summary_words / original_words)) * 100, 1)
                },
                'success': True
            }
            
            logger.info(f"Summary generated: {summary_words} words ({compression_ratio:.1f}x compression)")
            return jsonify(response)
            
        except Exception as gen_error:
            logger.error(f"Error during generation: {str(gen_error)}")
            return jsonify({'error': 'Failed to generate summary. Please try again.'}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in summarize endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    health_status = {
        'status': 'healthy',
        'service': 'text-summarization',
        'version': '1.0.0',
        'model': {
            'loaded': summarizer is not None,
            'type': 'fine-tuned' if summarizer and hasattr(summarizer, 'model') else 'base',
            'device': 'cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu'
        },
        'limits': {
            'min_words': config.MIN_WORDS,
            'max_words': config.MAX_WORDS,
            'max_summary_length': config.MAX_SUMMARY_LENGTH
        }
    }
    return jsonify(health_status)

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint."""
    info = {
        'name': 'Text Summarization API',
        'version': '1.0.0',
        'description': 'AI-powered text summarization using BART model',
        'endpoints': {
            'GET /': 'Web interface',
            'POST /summarize': 'Generate summary from text',
            'GET /health': 'Health check',
            'GET /api/info': 'API information'
        },
        'parameters': {
            'max_input_length': config.MAX_INPUT_LENGTH,
            'max_summary_length': config.MAX_SUMMARY_LENGTH,
            'min_summary_length': config.MIN_SUMMARY_LENGTH,
            'length_penalty': config.LENGTH_PENALTY,
            'num_beams': config.NUM_BEAMS
        },
        'model': {
            'base': config.BASE_MODEL,
            'fine_tuned_path': config.MODEL_PATH
        }
    }
    return jsonify(info)

@app.route('/api/example', methods=['GET'])
def example():
    """Provide example usage."""
    example_text = """Text summarization is the process of creating a shorter version of a longer text while preserving the key information and meaning. This is useful for quickly understanding long documents, articles, or reports. Natural language processing techniques like transformer models have greatly improved the quality of automated text summarization. These models can understand context, identify important information, and generate coherent summaries that capture the essence of the original text."""
    
    example_request = {
        'text': example_text,
        'expected_response': {
            'summary': 'A generated summary will appear here...',
            'stats': {
                'original_length': len(example_text.split()),
                'summary_length': 'varies',
                'compression_ratio': 'varies'
            }
        }
    }
    
    return jsonify({
        'example': example_request,
        'curl_example': f'curl -X POST http://localhost:5000/summarize -H "Content-Type: application/json" -d \'{{"text": "{example_text[:100]}..."}}\''
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the application."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))
    
    # Print startup banner
    banner = f"""
    {'='*60}
    Text Summarization Application
    {'='*60}
    Version: 1.0.0
    Port: {port}
    Model: {'Loaded' if summarizer else 'Failed to load'}
    Device: {'GPU' if torch.cuda.is_available() and config.USE_GPU else 'CPU'}
    Max Input: {config.MAX_INPUT_LENGTH} tokens
    Max Summary: {config.MAX_SUMMARY_LENGTH} tokens
    {'='*60}
    """
    
    print(banner)
    logger.info("Application starting up...")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('FLASK_ENV', 'development') == 'development',
        threaded=True
    )

if __name__ == '__main__':
    main()
