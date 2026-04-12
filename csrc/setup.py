from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, ROCM_HOME


def detect_gpu_backend(torch_module):
    if getattr(torch_module.version, "hip", None):
        return "rocm"
    if getattr(torch_module.version, "cuda", None) and torch_module.cuda.is_available():
        return "cuda"
    return None


def extension_compile_args(backend):
    cxx_args = ["-O3", "-std=c++17"]
    gpu_args = ["-O3"]

    if backend == "cuda":
        gpu_args.append("--use_fast_math")
    elif backend == "rocm":
        cxx_args.append("-DUSE_ROCM")
        gpu_args.extend(["-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__"])

    return {"cxx": cxx_args, "nvcc": gpu_args}


backend = detect_gpu_backend(torch)
if not backend:
    raise RuntimeError("PyTorch GPU backend not detected; cannot build infinity_memory_ops")

toolchain_home = ROCM_HOME if backend == "rocm" else CUDA_HOME
if not toolchain_home:
    raise RuntimeError(f"{backend.upper()} toolchain not found; cannot build infinity_memory_ops")

setup(
    name='infinity_memory_ops',
    ext_modules=[
        CUDAExtension(
            'infinity_memory_ops',
            ['memory_ops.cpp'],
            extra_compile_args=extension_compile_args(backend)
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
