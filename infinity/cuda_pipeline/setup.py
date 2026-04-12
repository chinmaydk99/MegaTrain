from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, ROCM_HOME
import torch
import sys

# Determine which version to build
use_simple = '--simple' in sys.argv
if use_simple:
    sys.argv.remove('--simple')

if use_simple:
    # Simple C++ version (no custom CUDA kernels)
    print("Building simple C++ version (no custom CUDA kernels)")
    ext_modules = [
        CppExtension(
            name='cuda_pipeline',
            sources=['simple_pipeline.cpp'],
            extra_compile_args={'cxx': ['-O3', '-std=c++17']}
        )
    ]
else:
    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    toolchain_home = ROCM_HOME if backend == "rocm" else CUDA_HOME
    if not toolchain_home:
        raise RuntimeError(f"{backend.upper()} toolchain not found; cannot build gpu pipeline extension")

    cxx_args = ['-O3', '-std=c++17']
    gpu_args = ['-O3']
    if backend == "cuda":
        gpu_args.extend([
            '--use_fast_math',
            '-gencode=arch=compute_80,code=sm_80',  # A100
            '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
            '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
            '-gencode=arch=compute_90,code=sm_90',  # H100
        ])
    else:
        cxx_args.append('-DUSE_ROCM')
        gpu_args.extend(['-DUSE_ROCM', '-D__HIP_PLATFORM_AMD__'])

    print(f"Building full {backend.upper()}-compatible GPU version")
    ext_modules = [
        CUDAExtension(
            name='cuda_pipeline',
            sources=['batched_copy.cu'],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': gpu_args,
            }
        )
    ]

setup(
    name='cuda_pipeline',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
