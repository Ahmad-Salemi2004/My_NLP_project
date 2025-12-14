#!/usr/bin/env python3
import os
import sys
import torch
import time
import logging
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
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
    
    # API settings
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_PERIOD = int(os.getenv('RATE_LIMIT_PERIOD', '3600'))  # seconds
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))  # seconds

# Initialize config
config = Config()

# Request tracking for rate limiting (simple in-memory implementation)
request_tracker = {}

def handle_errors(f):
    """Decorator for comprehensive error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        request_id = f"{time.time()}-{id(request)}"
        
        try:
            logger.info(f"Request {request_id}: {request.method} {request.path}")
            result = f(*args, **kwargs)
            processing_time = time.time() - start_time
            logger.info(f"Request {request_id}: Completed in {processing_time:.2f}s")
            return result
            
        except torch.cuda.OutOfMemoryError:
            error_msg = "GPU memory exhausted. Model requires more GPU memory."
            logger.error(f"Request {request_id}: {error_msg}")
            return jsonify({
                'error': 'Resource limit exceeded',
                'message': error_msg,
                'suggestion': 'Try using CPU or reducing input length',
                'request_id': request_id,
                'status': 'gpu_out_of_memory'
            }), 507  # Insufficient Storage
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                error_msg = "System memory exhausted during processing."
                logger.error(f"Request {request_id}: {error_msg}")
                return jsonify({
                    'error': 'Memory limit exceeded',
                    'message': error_msg,
                    'suggestion': 'Reduce input length or use smaller model',
                    'request_id': request_id,
                    'status': 'memory_exhausted'
                }), 507
            elif 'CUDA' in str(e):
                error_msg = "CUDA error occurred during processing."
                logger.error(f"Request {request_id}: {error_msg} - {str(e)}")
                return jsonify({
                    'error': 'GPU processing error',
                    'message': error_msg,
                    'suggestion': 'Try using CPU mode',
                    'request_id': request_id,
                    'status': 'gpu_error'
                }), 500
            else:
                error_msg = f"Runtime error: {str(e)}"
                logger.error(f"Request {request_id}: {error_msg}")
                return jsonify({
                    'error': 'Processing error',
                    'message': error_msg,
                    'request_id': request_id,
                    'status': 'runtime_error'
                }), 500
                
        except ValueError as e:
            if 'too long' in str(e).lower():
                error_msg = "Input text exceeds maximum length limit."
                logger.error(f"Request {request_id}: {error_msg}")
                return jsonify({
                    'error': 'Input too long',
                    'message': error_msg,
                    'max_length': config.MAX_INPUT_LENGTH,
                    'request_id': request_id,
                    'status': 'input_too_long'
                }), 400
            else:
                error_msg = f"Invalid input: {str(e)}"
                logger.error(f"Request {request_id}: {error_msg}")
                return jsonify({
                    'error': 'Invalid request',
                    'message': error_msg,
                    'request_id': request_id,
                    'status': 'validation_error'
                }), 400
                
        except TimeoutError:
            error_msg = "Request processing timed out."
            logger.error(f"Request {request_id}: {error_msg}")
            return jsonify({
                'error': 'Processing timeout',
                'message': error_msg,
                'timeout_seconds': config.REQUEST_TIMEOUT,
                'request_id': request_id,
                'status': 'timeout'
            }), 408  # Request Timeout
            
        except Exception as e:
            error_msg = f"Unexpected server error: {str(e)}"
            logger.error(f"Request {request_id}: {error_msg}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred',
                'request_id': request_id,
                'status': 'internal_error'
            }), 500
            
    return decorated_function

def check_rate_limit(client_ip):
    """Simple rate limiting implementation."""
    current_time = time.time()
    
    if client_ip not in request_tracker:
        request_tracker[client_ip] = []
    
    # Clean old requests
    request_tracker[client_ip] = [
        req_time for req_time in request_tracker[client_ip]
        if current_time - req_time < config.RATE_LIMIT_PERIOD
    ]
    
    # Check if limit exceeded
    if len(request_tracker[client_ip]) >= config.RATE_LIMIT_REQUESTS:
        return False, "Rate limit exceeded. Please try again later."
    
    # Add current request
    request_tracker[client_ip].append(current_time)
    return True, ""

def setup_device():
    """Configure and return the device for model inference."""
    if config.USE_GPU and torch.cuda.is_available():
        try:
            device = 0  # Use first GPU
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"Using GPU: {device_name} ({gpu_memory:.1f} GB)")
            return device
        except Exception as e:
            logger.warning(f"GPU setup failed, falling back to CPU: {e}")
            return -1
    else:
        logger.info("Using CPU for inference")
        return -1  # Use CPU

def model_exists(model_path):
    """Check if model files exist."""
    if not os.path.exists(model_path):
        return False
    
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    existing_files = set(os.listdir(model_path))
    
    # Check for at least the essential model files
    essential_files = ['config.json']
    if not any(f in existing_files for f in essential_files):
        return False
    
    return True

def load_model_with_fallback():
    """
    Load the text summarization model with multiple fallback strategies.
    """
    device = setup_device()
    model_type = "unknown"
    
    # Strategy 1: Try fine-tuned model
    if model_exists(config.MODEL_PATH):
        try:
            logger.info(f"Attempting to load fine-tuned model from: {config.MODEL_PATH}")
            model = pipeline(
                "summarization",
                model=config.MODEL_PATH,
                tokenizer=config.MODEL_PATH,
                device=device,
                framework="pt"
            )
            model_type = "fine-tuned"
            logger.info(f"SUCCESS: Fine-tuned model loaded from {config.MODEL_PATH}")
            return model, model_type
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model: {e}")
    
    # Strategy 2: Try base model from Hugging Face
    try:
        logger.info(f"Loading base model from Hugging Face: {config.BASE_MODEL}")
        model = pipeline(
            "summarization",
            model=config.BASE_MODEL,
            device=device,
            framework="pt"
        )
        model_type = "base"
        logger.info(f"SUCCESS: Base model loaded: {config.BASE_MODEL}")
        return model, model_type
    except Exception as e:
        logger.warning(f"Failed to load base model from Hugging Face: {e}")
    
    # Strategy 3: Try CPU fallback for base model
    try:
        logger.info("Attempting to load base model with CPU...")
        model = pipeline(
            "summarization",
            model=config.BASE_MODEL,
            device=-1,  # Force CPU
            framework="pt"
        )
        model_type = "base-cpu"
        logger.info("SUCCESS: Base model loaded with CPU")
        return model, model_type
    except Exception as e:
        logger.error(f"Failed to load model with CPU: {e}")
    
    # Strategy 4: Try alternative model
    try:
        logger.info("Trying alternative model (facebook/bart-base)...")
        model = pipeline(
            "summarization",
            model="facebook/bart-base",
            device=device,
            framework="pt"
        )
        model_type = "alternative"
        logger.info("SUCCESS: Alternative model loaded")
        return model, model_type
    except Exception as e:
        logger.error(f"All model loading attempts failed: {e}")
    
    return None, "failed"

def check_model_availability():
    """Check if model is available and working."""
    if summarizer is None:
        return False, "Model not loaded"
    
    try:
        # Quick test inference
        test_text = "This is a test to verify the model is working correctly."
        test_result = summarizer(
            test_text,
            max_length=50,
            min_length=10,
            length_penalty=1.0,
            num_beams=2
        )
        if test_result and len(test_result) > 0:
            return True, "Model is working"
        else:
            return False, "Model returned empty result"
    except Exception as e:
        return False, f"Model test failed: {str(e)}"

def validate_input(text):
    """
    Validate input text comprehensively.
    Returns (is_valid, error_message, validation_details)
    """
    if not text or not isinstance(text, str):
        return False, "Input must be a non-empty string", {}
    
    # Remove extra whitespace and clean
    cleaned_text = ' '.join(text.strip().split())
    
    if len(cleaned_text) == 0:
        return False, "Input text cannot be empty or only whitespace", {}
    
    # Check minimum length
    if len(cleaned_text) < 10:
        return False, "Input text is too short (minimum 10 characters)", {}
    
    # Word-based validation
    words = cleaned_text.split()
    word_count = len(words)
    char_count = len(cleaned_text)
    
    validation_details = {
        'word_count': word_count,
        'char_count': char_count,
        'cleaned_text': cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
    }
    
    if word_count < config.MIN_WORDS:
        return False, f"Text too short. Please provide at least {config.MIN_WORDS} words.", validation_details
    
    if word_count > config.MAX_WORDS:
        return False, f"Text too long. Maximum allowed is {config.MAX_WORDS} words.", validation_details
    
    # Check for unusual content
    if cleaned_text.isdigit():
        return False, "Input appears to be only numbers. Please provide meaningful text.", validation_details
    
    # Check character diversity (basic spam detection)
    unique_chars = len(set(cleaned_text.lower()))
    if unique_chars < 5 and len(cleaned_text) > 20:
        return False, "Input has low character diversity.", validation_details
    
    return True, "Input is valid", validation_details

# Global model instance
logger.info("Initializing text summarization application...")
summarizer, model_info = load_model_with_fallback()

@app.before_request
def before_request():
    """Log each request and check rate limits."""
    client_ip = request.remote_addr
    is_allowed, message = check_rate_limit(client_ip)
    
    if not is_allowed:
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': message,
            'limits': {
                'requests_per_hour': config.RATE_LIMIT_REQUESTS,
                'client_ip': client_ip
            }
        }), 429  # Too Many Requests

@app.route('/')
def home():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
@handle_errors
def summarize_text():
    """
    API endpoint for text summarization.
    Expects JSON: {'text': 'your text here'}
    Returns JSON with summary and statistics.
    """
    # Parse and validate request format
    if not request.is_json:
        return jsonify({
            'error': 'Invalid content type',
            'message': 'Content-Type must be application/json',
            'received_content_type': request.content_type
        }), 400
    
    data = request.get_json()
    if not data:
        return jsonify({
            'error': 'Empty request body',
            'message': 'Request body must contain JSON data'
        }), 400
    
    # Extract text
    text = data.get('text', '').strip()
    
    # Comprehensive input validation
    is_valid, error_msg, validation_details = validate_input(text)
    if not is_valid:
        return jsonify({
            'error': 'Invalid input',
            'message': error_msg,
            'validation_details': validation_details,
            'limits': {
                'min_words': config.MIN_WORDS,
                'max_words': config.MAX_WORDS
            }
        }), 400
    
    # Check model availability
    if summarizer is None:
        return jsonify({
            'error': 'Service unavailable',
            'message': 'Text summarization model is not available',
            'suggestion': 'Please try again later or contact support'
        }), 503
    
    # Get optional parameters with validation
    try:
        max_length = min(int(data.get('max_length', config.MAX_SUMMARY_LENGTH)), 200)
        min_length = max(int(data.get('min_length', config.MIN_SUMMARY_LENGTH)), 10)
        num_beams = max(min(int(data.get('num_beams', config.NUM_BEAMS)), 8), 1)
        
        if min_length >= max_length:
            return jsonify({
                'error': 'Invalid parameters',
                'message': 'min_length must be less than max_length'
            }), 400
    except ValueError:
        return jsonify({
            'error': 'Invalid parameters',
            'message': 'Parameters must be valid integers'
        }), 400
    
    # Generate summary
    logger.info(f"Generating summary for {validation_details['word_count']} words...")
    
    try:
        start_time = time.time()
        
        summary_result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            length_penalty=config.LENGTH_PENALTY,
            num_beams=num_beams,
            early_stopping=True,
            truncation=True,
            no_repeat_ngram_size=3
        )
        
        processing_time = time.time() - start_time
        
        if not summary_result or len(summary_result) == 0:
            return jsonify({
                'error': 'Generation failed',
                'message': 'Model returned empty result',
                'processing_time': round(processing_time, 2)
            }), 500
        
        summary = summary_result[0]['summary_text']
        
        # Calculate statistics
        original_words = validation_details['word_count']
        summary_words = len(summary.split())
        
        stats = {
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': round(original_words / summary_words, 2) if summary_words > 0 else 0,
            'reduction_percentage': round((1 - (summary_words / original_words)) * 100, 1) if original_words > 0 else 0,
            'processing_time_seconds': round(processing_time, 2),
            'processing_speed': round(original_words / processing_time, 1) if processing_time > 0 else 0
        }
        
        # Prepare response
        response = {
            'summary': summary,
            'stats': stats,
            'model_info': {
                'type': model_info,
                'device': 'gpu' if torch.cuda.is_available() and config.USE_GPU else 'cpu'
            },
            'parameters_used': {
                'max_length': max_length,
                'min_length': min_length,
                'num_beams': num_beams,
                'length_penalty': config.LENGTH_PENALTY
            },
            'timestamp': datetime.now().isoformat(),
            'request_id': f"{time.time()}-{id(request)}"
        }
        
        logger.info(f"Summary generated: {summary_words} words in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}\n{traceback.format_exc()}")
        raise  # Let the error handler deal with it

@app.route('/health', methods=['GET'])
@handle_errors
def health_check():
    """Comprehensive health check endpoint."""
    model_available, model_message = check_model_availability()
    
    health_data = {
        'status': 'healthy' if model_available else 'degraded',
        'service': 'text-summarization-api',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'uptime': round(time.time() - app_start_time, 2),
        'model': {
            'available': model_available,
            'status': model_message,
            'type': model_info,
            'device': 'gpu' if torch.cuda.is_available() and config.USE_GPU else 'cpu'
        },
        'system': {
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'memory_allocated_gb': round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0
        },
        'limits': {
            'min_words': config.MIN_WORDS,
            'max_words': config.MAX_WORDS,
            'max_summary_length': config.MAX_SUMMARY_LENGTH,
            'rate_limit': f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_PERIOD}s"
        }
    }
    
    status_code = 200 if model_available else 503
    return jsonify(health_data), status_code

@app.route('/api/info', methods=['GET'])
@handle_errors
def api_info():
    """API information endpoint."""
    info = {
        'name': 'Text Summarization API',
        'version': '1.0.0',
        'description': 'AI-powered text summarization using BART model',
        'documentation': {
            'endpoints': {
                'GET /': 'Web interface for interactive summarization',
                'POST /summarize': 'Generate summary from text (JSON required)',
                'GET /health': 'Comprehensive health check',
                'GET /api/info': 'API information (this endpoint)',
                'GET /api/example': 'Example usage and curl command'
            },
            'authentication': 'None required for basic endpoints',
            'rate_limits': f"{config.RATE_LIMIT_REQUESTS} requests per {config.RATE_LIMIT_PERIOD} seconds"
        },
        'parameters': {
            'input_validation': {
                'min_words': config.MIN_WORDS,
                'max_words': config.MAX_WORDS,
                'max_input_length': config.MAX_INPUT_LENGTH
            },
            'generation_parameters': {
                'max_summary_length': config.MAX_SUMMARY_LENGTH,
                'min_summary_length': config.MIN_SUMMARY_LENGTH,
                'length_penalty': config.LENGTH_PENALTY,
                'num_beams': config.NUM_BEAMS
            }
        },
        'model': {
            'base_model': config.BASE_MODEL,
            'fine_tuned_path': config.MODEL_PATH,
            'current_model': model_info
        },
        'support': {
            'issues': 'https://github.com/Ahmad-Salemi2004/My_NLP_project/issues',
            'source': 'https://github.com/Ahmad-Salemi2004/My_NLP_project'
        }
    }
    return jsonify(info)

@app.route('/api/example', methods=['GET'])
@handle_errors
def example():
    """Provide example usage with curl command."""
    example_text = """Text summarization is the process of creating a shorter version of a longer text while preserving the key information and meaning. This is useful for quickly understanding long documents, articles, or reports. Natural language processing techniques like transformer models have greatly improved the quality of automated text summarization. These models can understand context, identify important information, and generate coherent summaries that capture the essence of the original text."""
    
    example_request = {
        'method': 'POST',
        'endpoint': '/summarize',
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': {
            'text': example_text,
            'max_length': 100,
            'min_length': 30,
            'num_beams': 4
        },
        'expected_response_structure': {
            'summary': 'Generated summary text',
            'stats': {
                'original_words': 'Number of words in input',
                'summary_words': 'Number of words in summary',
                'compression_ratio': 'Compression ratio',
                'processing_time_seconds': 'Time taken to generate'
            },
            'model_info': {
                'type': 'Type of model used',
                'device': 'Device used for inference'
            }
        }
    }
    
    curl_command = f"""curl -X POST http://localhost:5000/summarize \\
  -H "Content-Type: application/json" \\
  -d '{{
    "text": "{example_text[:100]}...",
    "max_length": 100,
    "min_length": 30,
    "num_beams": 4
  }}'"""
    
    return jsonify({
        'example': example_request,
        'curl_command': curl_command,
        'note': 'Replace localhost:5000 with your actual server address'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server',
        'available_endpoints': [
            'GET /',
            'POST /summarize',
            'GET /health',
            'GET /api/info',
            'GET /api/example'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The method {request.method} is not allowed for this endpoint',
        'allowed_methods': error.description.get('allowed', [])
    }), 405

def main():
    """Main function to run the application."""
    global app_start_time
    app_start_time = time.time()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))
    
    # Print startup banner
    banner = f"""
    {'='*60}
    Text Summarization Application
    {'='*60}
    Version: 1.0.0
    Port: {port}
    Model Status: {'LOADED' if summarizer else 'FAILED TO LOAD'}
    Model Type: {model_info}
    Device: {'GPU' if torch.cuda.is_available() and config.USE_GPU else 'CPU'}
    Max Input Words: {config.MAX_WORDS}
    Max Summary Length: {config.MAX_SUMMARY_LENGTH}
    {'='*60}
    Application starting...
    {'='*60}
    """
    
    print(banner)
    logger.info("Application starting up...")
    
    # Log configuration
    logger.info(f"Configuration: MODEL_PATH={config.MODEL_PATH}")
    logger.info(f"Configuration: BASE_MODEL={config.BASE_MODEL}")
    logger.info(f"Configuration: USE_GPU={config.USE_GPU}")
    logger.info(f"Configuration: Rate limit: {config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_PERIOD}s")
    
    # Check model
    if summarizer is None:
        logger.error("CRITICAL: Model failed to load. Some endpoints will return errors.")
        logger.error("Run 'python download_model.py' to download the model.")
    else:
        logger.info(f"SUCCESS: Model loaded successfully: {model_info}")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('FLASK_ENV', 'development') == 'development',
        threaded=True
    )

if __name__ == '__main__':
    main()
