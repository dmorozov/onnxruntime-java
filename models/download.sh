#!/bin/bash
##############################################################################
# Model Download Wrapper Script
##############################################################################
#
# This script wraps download_model.py and ensures the Python virtual
# environment is set up before downloading models.
#
# USAGE:
#   ./download.sh --model <model_name> [--variants <variants>]
#   ./download.sh --list
#
# EXAMPLES:
#   ./download.sh --list
#   ./download.sh --model qwen3
#   ./download.sh --model flan-t5-small --variants int8,q4
#
##############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_optimum"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}→ Virtual environment not found. Setting up...${NC}"
    source "$SCRIPT_DIR/setup.sh"
else
    # Activate existing virtual environment
    source "$VENV_DIR/bin/activate"
fi

# Run Python download script with all arguments
echo -e "${BLUE}→ Running model downloader...${NC}"
echo ""

python3 "$SCRIPT_DIR/download_model.py" "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Download script completed successfully${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Download script exited with code $exit_code${NC}"
fi

exit $exit_code
