#!/bin/bash

set -e  # Exit on first error

# Set HOME directory
HOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$HOME_DIR/weights"
ENV_NAME="grounding_env"


# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3,10,16))')

if [[ "$PYTHON_VERSION" != "True" ]]; then
    echo "❌ Error: Python 3.10 or higher is required. Current version: $(python3 --version)"
    exit 1
else
    echo "✅ Python version check passed: $(python3 --version)"
fi

# Create Conda environment if it doesn't exist
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating Conda environment '$ENV_NAME' with Python 3.10.16..."
    conda create -y -n "$ENV_NAME" python=3.10.16
else
    echo "✅ Conda environment '$ENV_NAME' already exists. Skipping creation."
fi

# Activate Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel

# Install PyTorch
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone repositories (if not already cloned)
echo "Cloning repositories..."
[[ -d "$HOME_DIR/GroundingDINO" ]] || git clone https://github.com/IDEA-Research/GroundingDINO.git "$HOME_DIR/GroundingDINO"
cd "$HOME_DIR/GroundingDINO"
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -q -e .

cd "$HOME_DIR"
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

[[ -d "$HOME_DIR/samv2" ]] || git clone https://github.com/SauravMaheshkar/samv2.git "$HOME_DIR/samv2"

# Download weights
echo "Downloading weights..."
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

[[ -f "groundingdino_swint_ogc.pth" ]] || wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
[[ -f "sam_vit_b_01ec64.pth" ]] || wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Download SAM2 Hiera Tiny model
MODEL_URL="https://huggingface.co/facebook/sam2-hiera-tiny/resolve/f245b47be73d8858fb7543a8b9c1c720d9f98779/sam2_hiera_tiny.pt"
wget -O "$WEIGHTS_DIR/sam2_hiera_tiny.pt" "$MODEL_URL" --progress=bar:force
echo "✅ SAM2 Hiera Tiny model downloaded to: $WEIGHTS_DIR/sam2_hiera_tiny.pt"

# Install dependencies
echo "Installing dependencies..."
cd "$HOME_DIR"
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found! Skipping dependency installation."
fi

echo "✅ Setup complete! Activate the environment using: conda activate $ENV_NAME"
