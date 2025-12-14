.PHONY: setup run test clean docker-build docker-run help

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup     - Set up the project"
	@echo "  make run       - Run the application"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up temporary files"
	@echo "  make docker    - Build and run with Docker"
	@echo "  make notebook  - Start Jupyter notebook"

setup:
	@chmod +x setup.sh
	@./setup.sh

run:
	@source venv/bin/activate && python app.py

test:
	@source venv/bin/activate && python test_app.py

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cleaned!"

docker-build:
	@docker build -t text-summarization .

docker-run:
	@docker run -p 5000:5000 text-summarization

docker: docker-build docker-run

notebook:
	@source venv/bin/activate && jupyter notebook notebooks/

update-model:
	@source venv/bin/activate && python download_model.py
