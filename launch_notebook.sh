#!/bin/bash
# Launch script for eFAST Interactive Notebook

echo "🚀 Launching eFAST Interactive Notebook..."
echo ""

# Check if we're in the efast directory
if [ ! -f "selected_sites.geojson" ]; then
    echo "❌ Error: Run this script from the efast directory"
    exit 1
fi

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "   Create one with: python -m venv .venv"
    exit 1
fi

# Ensure pip is available
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "📦 Installing pip..."
    .venv/bin/python -m ensurepip --upgrade || {
        echo "❌ Failed to install pip!"
        exit 1
    }
fi

# Check if package is installed
if ! .venv/bin/python -c "import efast" 2>/dev/null; then
    echo "⚠️  efast package not installed in venv!"
    echo "   Installing: .venv/bin/python -m pip install -e ."
    .venv/bin/python -m pip install -e . || {
        echo "❌ Installation failed!"
        exit 1
    }
fi

# Check if run_efast can be imported (needs dependencies)
if ! .venv/bin/python -c "import run_efast" 2>/dev/null; then
    echo "⚠️  run_efast dependencies missing, installing from requirements.txt..."
    .venv/bin/python -m pip install -r requirements.txt || {
        echo "⚠️  Some dependencies may be missing - download may be disabled"
    }
fi

# Launch directly with venv jupyter
if [ -f ".venv/bin/jupyter" ]; then
    echo "📓 Opening efast_interactive.ipynb..."
    echo ""
    echo "Jupyter will open in your browser shortly..."
    echo "Press Ctrl+C to stop the server when done."
    echo ""
    .venv/bin/jupyter notebook efast_interactive.ipynb --ip=127.0.0.1
else
    echo "❌ Jupyter not found!"
    echo "   Install: .venv/bin/python -m pip install -e '.[notebook]'"
    exit 1
fi
