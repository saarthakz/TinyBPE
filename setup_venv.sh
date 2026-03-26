#!/bin/bash

# Exit on error
set -e

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install it first (https://github.com/astral-sh/uv)."
    exit 1
fi

echo "Creating virtual environment using uv..."
uv venv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "source .venv/bin/activate"
