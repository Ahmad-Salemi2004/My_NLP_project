"""
config.py
Configuration settings for the application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration."""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    
    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', './models/fine_tuned_bart')
    USE_GPU = os.getenv('USE_GPU', 'True').lower() == 'true'
    
    # Summarization parameters
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '1024'))
    MAX_SUMMARY_LENGTH = int(os.getenv('MAX_SUMMARY_LENGTH', '150'))
    MIN_SUMMARY_LENGTH = int(os.getenv('MIN_SUMMARY_LENGTH', '40'))
    
    # Validation
    MIN_WORDS = int(os.getenv('MIN_WORDS', '30'))
    MAX_WORDS = int(os.getenv('MAX_WORDS', '1000'))
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not os.path.exists(cls.MODEL_PATH):
            print(f"WARNING: Model path '{cls.MODEL_PATH}' does not exist")
        
        if cls.MAX_INPUT_LENGTH < cls.MAX_SUMMARY_LENGTH:
            print(f"WARNING: MAX_INPUT_LENGTH ({cls.MAX_INPUT_LENGTH}) should be greater than MAX_SUMMARY_LENGTH ({cls.MAX_SUMMARY_LENGTH})")
        
        return True

# Validate configuration on import
Config.validate()
