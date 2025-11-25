#!/bin/bash

##############################################################################
# ONNX Model Re-Export Environment Setup
##############################################################################
#
# This script creates and manages a Python virtual environment for ONNX model
# re-export with Optimum CLI. The environment is shared across all re-export
# scripts to avoid redundant dependency installations.
#
# USAGE:
#   source ./setup_optimum_env.sh
#
# DEPENDENCIES INSTALLED:
#   - optimum[exporters,onnxruntime]: ONNX export with graph optimization
#   - transformers: HuggingFace model loading
#   - torch: PyTorch (CPU-only for export, no CUDA needed)
#   - onnx: ONNX graph manipulation
#   - onnxruntime: ONNX Runtime for validation
#
# ENVIRONMENT LOCATION:
#   models/.venv_optimum/
#
# NOTES:
#   - Virtual environment is reused across all re-export scripts
#   - Use 'source' instead of executing directly to activate venv in current shell
#   - CPU-only PyTorch is sufficient for model export (no GPU needed)
#
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_optimum"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}ONNX Model Re-Export Environment Setup${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists at: $VENV_DIR${NC}"
else
    echo -e "${YELLOW}→ Creating virtual environment at: $VENV_DIR${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}→ Activating virtual environment${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}→ Upgrading pip${NC}"
pip install --upgrade pip > /dev/null 2>&1

# Check if dependencies are already installed
if python -c "import optimum" 2>/dev/null && \
   python -c "import transformers" 2>/dev/null && \
   python -c "import torch" 2>/dev/null && \
   python -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
else
    echo -e "${YELLOW}→ Installing dependencies (this may take a few minutes)...${NC}"

    # Install PyTorch CPU-only (faster, smaller, sufficient for export)
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # Install Optimum with ONNX exporters and ONNX Runtime
    pip install optimum[exporters,onnxruntime]

    # Install Transformers
    pip install transformers

    # Install ONNX tools
    pip install onnx onnxruntime

    # Install HuggingFace Hub for model downloads
    pip install huggingface_hub

    pip install --upgrade optimum[onnxruntime]

    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Environment ready for ONNX model re-export${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Virtual environment: ${BLUE}$VENV_DIR${NC}"
echo -e "Python: ${BLUE}$(which python)${NC}"
echo -e "Optimum version: ${BLUE}$(python -c 'import importlib.metadata; print(importlib.metadata.version("optimum"))')${NC}"
echo -e "Transformers version: ${BLUE}$(python -c 'import transformers; print(transformers.__version__)')${NC}"
echo ""
