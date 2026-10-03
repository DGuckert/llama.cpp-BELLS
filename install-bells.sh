#!/bin/bash
# BELLS one-line installer for Linux/macOS
# Usage: curl -sSL https://raw.githubusercontent.com/DGuckert/llama.cpp-BELLS/bells-next/install-bells.sh | bash

set -e

echo ""
echo "  ____  _____ _     _     ____"
echo " | __ )| ____| |   | |   / ___|"
echo " |  _ \|  _| | |   | |   \___ \\"
echo " | |_) | |___| |___| |___ ___) |"
echo " |____/|_____|_____|_____|____/"
echo ""
echo " Per-layer VRAM expert cache for MoE models"
echo ""

# Check Python
PY=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY="$cmd"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "[BELLS] ERROR: Python 3.10+ required"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  Fedora:        sudo dnf install python3 python3-pip"
    echo "  macOS:         brew install python@3.12"
    exit 1
fi

echo "[BELLS] Found $($PY --version)"

# Determine repo location
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
REPO_DIR="$SCRIPT_DIR"

if [ ! -f "$REPO_DIR/tools/bells-manager/install.py" ]; then
    REPO_DIR="$HOME/llama.cpp-BELLS"
    if [ ! -d "$REPO_DIR" ]; then
        echo "[BELLS] Cloning repository..."
        git clone https://github.com/DGuckert/llama.cpp-BELLS.git "$REPO_DIR"
    else
        echo "[BELLS] Repository already exists at $REPO_DIR"
    fi
fi

echo "[BELLS] Running installer..."
"$PY" "$REPO_DIR/tools/bells-manager/install.py"

echo ""
echo "[BELLS] Done! Run 'bells' to start."
