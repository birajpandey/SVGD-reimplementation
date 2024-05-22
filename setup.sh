#!/bin/bash

# Define global variables
ENV_NAME="svgd_env"
OS_NAME="$(uname -s)"
HARDWARE_NAME="$(uname -m)"

echo "Detected OS: $OS_NAME, Hardware: $HARDWARE_NAME."
echo

# Function to check if Conda is installed
check_conda(){
  if ! command -V conda &> /dev/null; then
    echo "Conda could not be found. Please install Anaconda before
    continuing. https://www.anaconda.com/download"
    exit 1

  fi
}

# Creating a new Conda environment and activating it
create_conda_env(){
  echo "Creating a new Conda environment..."
  conda create --name "$ENV_NAME" python=3.10 -y
  echo

}


# Global variable for conda
ENV_BIN_PATH="$(conda info --base)/envs/$ENV_NAME/bin"


# Installing Python packages from requirements.txt
install_requirements(){
  echo "Installing requirements from requirements.txt in the conda
  environment $ENV_NAME..."
  "$ENV_BIN_PATH/pip" install -r requirements.txt
  echo
}


# Main script execution
main(){
  check_conda
  create_conda_env
  install_requirements
  echo "Setup complete."
}

main