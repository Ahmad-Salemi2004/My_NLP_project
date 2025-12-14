#!/bin/bash

echo "=========================================="
echo "Text Summarization Project Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if [[ $python_version < "3.8" ]]; then
    echo "ERROR: Python 3.8 or higher required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version detected"

# Create virtual environment
echo -e "\nCreating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\nActivating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\nInstalling dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "ERROR: requirements.txt not found"
    exit 1
fi

# Download NLTK data
echo -e "\nDownloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt')" 2>/dev/null || echo "NLTK download failed (non-critical)"

# Create necessary directories
echo -e "\nCreating project directories..."
mkdir -p models/fine_tuned_bart
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
mkdir -p notebooks
mkdir -p logs
mkdir -p results
echo "✓ Directories created"

# Check and download model
echo -e "\nChecking for model..."
if [ -f "download_model.py" ]; then
    if [ ! -f "models/fine_tuned_bart/config.json" ]; then
        echo "Model not found. Downloading..."
        python3 download_model.py
    else
        echo "✓ Model already exists"
    fi
else
    echo "WARNING: download_model.py not found"
fi

# Create default template if not exists
if [ ! -f "templates/index.html" ]; then
    echo -e "\nCreating default template..."
    cat > templates/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Text Summarization</title>
</head>
<body>
    <h1>Text Summarization Project</h1>
    <p>Please add your HTML template file.</p>
</body>
</html>
EOF
    echo "✓ Default template created"
fi

# Set execute permissions
chmod +x setup.sh 2>/dev/null || true

echo -e "\n=========================================="
echo "SETUP COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo -e "\nTo run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: python app.py"
echo "3. Open browser: http://localhost:5000"
echo -e "\nAlternative ways to run:"
echo "- python run.py"
echo "./setup.sh (to re-run setup)"
echo -e "\nFor Docker:"
echo "docker-compose up --build"
