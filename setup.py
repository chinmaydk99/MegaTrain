"""Setup script for MegaTrain: Single-GPU Large Model Training Toolkit."""

from setuptools import setup, find_packages
from pathlib import Path
import sys
import os

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Try to import torch and CUDA extension builder
ext_modules = []
cmdclass = {}


def detect_gpu_backend(torch_module) -> str | None:
    """Detect which GPU backend this PyTorch build targets."""
    if getattr(torch_module.version, "hip", None):
        return "rocm"
    if getattr(torch_module.version, "cuda", None) and torch_module.cuda.is_available():
        return "cuda"
    return None


def extension_compile_args(backend: str) -> dict[str, list[str]]:
    """Return compile args for CUDA/ROCm extensions."""
    cxx_args = ["-O3", "-std=c++17"]
    gpu_args = ["-O3"]

    if backend == "cuda":
        gpu_args.append("--use_fast_math")
    elif backend == "rocm":
        cxx_args.append("-DUSE_ROCM")
        gpu_args.extend(["-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__"])

    return {"cxx": cxx_args, "nvcc": gpu_args}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, ROCM_HOME

    backend = detect_gpu_backend(torch)
    toolchain_home = ROCM_HOME if backend == "rocm" else CUDA_HOME

    if backend and toolchain_home:
        print(f"{backend.upper()} detected! Building GPU extensions...")
        compile_args = extension_compile_args(backend)

        ext_modules.extend(
            [
                CUDAExtension(
                    name="cuda_pipeline",
                    sources=["infinity/cuda_pipeline/batched_copy.cu"],
                    extra_compile_args=compile_args,
                ),
                CUDAExtension(
                    name="infinity_memory_ops",
                    sources=["csrc/memory_ops.cpp"],
                    extra_compile_args=compile_args,
                ),
            ]
        )
        cmdclass['build_ext'] = BuildExtension
        print("✓ GPU extensions will be built")
    else:
        print("GPU toolchain not available, skipping native GPU extensions")
        
except ImportError:
    print("PyTorch not installed yet, skipping native GPU extensions")
    print("You can install GPU extensions later by running: pip install -e .")

setup(
    name="megatrain",
    version="0.2.0",
    author="MegaTrain Team",
    description="MegaTrain: Single-GPU Large Model Training with CPU-backed parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DLYuanGod/MegaTrain",
    packages=find_packages(include=["infinity", "infinity.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "flash-attn": [
            "flash-attn>=2.0.0",
        ],
        "deepspeed": [
            "deepspeed>=0.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="deep-learning, large-language-models, training, gpu, cpu-offloading, megatrain",
)
