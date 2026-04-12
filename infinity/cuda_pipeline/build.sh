#!/bin/bash

# Build script for GPU pipeline extension

set -e

echo "Building GPU pipeline extension..."

cd "$(dirname "$0")"

# Check PyTorch installation
python -c "import torch; assert torch.cuda.is_available(), 'GPU backend not available in PyTorch'" || {
    echo "ERROR: PyTorch with GPU support is required."
    exit 1
}

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info *.so

# Determine build mode
BUILD_MODE="${1:-simple}"

if [ "$BUILD_MODE" = "simple" ]; then
    echo ""
    echo "Building SIMPLE version (C++ only, no custom CUDA kernels)..."
    echo "This version is more compatible but may be slightly slower."
    echo ""
    python setup.py build_ext --inplace --simple
elif [ "$BUILD_MODE" = "cuda" ] || [ "$BUILD_MODE" = "rocm" ] || [ "$BUILD_MODE" = "gpu" ]; then
    BACKEND=$(python - <<'PY'
import torch
if getattr(torch.version, "hip", None):
    print("rocm")
elif getattr(torch.version, "cuda", None) and torch.cuda.is_available():
    print("cuda")
else:
    print("none")
PY
)

    if [ "$BACKEND" = "cuda" ] && ! command -v nvcc &> /dev/null; then
        echo "ERROR: nvcc not found. Please install the CUDA toolkit or use 'simple' mode."
        exit 1
    fi

    if [ "$BACKEND" = "rocm" ] && ! command -v hipcc &> /dev/null; then
        echo "ERROR: hipcc not found. Please install ROCm or use 'simple' mode."
        exit 1
    fi

    if [ "$BACKEND" = "none" ]; then
        echo "ERROR: No CUDA or ROCm backend detected in PyTorch."
        exit 1
    fi

    echo ""
    echo "Building full ${BACKEND^^} version (with custom GPU kernels)..."
    echo "This version has custom kernels for maximum performance."
    echo ""
    python setup.py build_ext --inplace
else
    echo "Usage: $0 [simple|gpu|cuda|rocm]"
    echo "  simple - Build C++ version (default, more compatible)"
    echo "  gpu    - Auto-detect CUDA or ROCm and build the GPU version"
    echo "  cuda   - Alias for gpu mode when CUDA is installed"
    echo "  rocm   - Alias for gpu mode when ROCm is installed"
    exit 1
fi

# Install extension
echo ""
echo "Installing extension..."
pip install -e .

echo ""
echo "Build complete!"
echo ""
echo "To verify installation, run:"
echo "  python test_extension.py"
