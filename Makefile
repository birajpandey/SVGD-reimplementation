.PHONY: all set-permissions setup-environment test-environment clean

ENV_NAME="svgd_env"

# Default target
all: set-permissions setup-environment test-environment clean

## Make setup.sh executable
set-permissions:
	@echo "Making setup.sh file executable..."
	chmod +x setup.sh

## Setup Python environment
setup-environment:
	@echo "Setting up Conda environment..."
	./setup.sh


## Check Python environment
test-environment:
	@echo "Activating the Conda environment $(ENV_NAME) and running tests..."
	@bash -c "source activate $(ENV_NAME); python -m pytest tests/test_environment.py"


## Delete all compiled Python files
 clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Cleaning compiled Python files"
