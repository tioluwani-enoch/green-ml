#!/bin/bash

# Abort on any error
set -e

# Variables
REPO_URL="https://github.com/tioluwani-enoch/green-ml"
PROJECT_DIR="green-ml"
VENV_DIR="venv"

echo "=== Updating System ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Python3 Virtualenv Support ==="
sudo apt install -y python3-venv python3-pip git

# Clone/Update repository
if [ -d "$PROJECT_DIR" ]; then
    echo "Repository exists — pulling latest changes"
    cd "$PROJECT_DIR"
    git pull
else
    echo "Cloning repository"
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
fi

# Create virtual environment
echo "=== Creating Python Virtual Environment ==="
python3 -m venv "$VENV_DIR"

# Activate the virtual environment
echo "=== Activating Virtual Environment ==="
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "=== Upgrading pip ==="
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "=== Installing Dependencies ==="
    pip install -r requirements.txt
else
    echo "No requirements.txt found — skipping dependency installation"
fi

# Run the main script
echo "=== Running Project ==="
python3 live_stram_waste_identification.py

# Keep environment active if needed
echo "=== Script Finished ==="