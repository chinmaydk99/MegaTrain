#pragma once

#include <torch/extension.h>
#include <c10/core/StreamGuard.h>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;

namespace infinity_gpu_compat {
using Stream = decltype(c10::hip::getCurrentHIPStream());
using StreamGuard = c10::StreamGuard;

inline Stream get_current_stream() {
    return c10::hip::getCurrentHIPStream();
}

inline Stream get_stream_from_pool(bool is_high_priority, int device_index) {
    return c10::hip::getStreamFromPool(is_high_priority, device_index);
}

inline gpuStream_t raw_stream(Stream stream) {
    return stream.stream();
}
}  // namespace infinity_gpu_compat

#define GPU_BACKEND_NAME "rocm"
#define GPU_SUCCESS hipSuccess
#define GPU_ERROR_NOT_READY hipErrorNotReady
#define GPU_CHECK(status) TORCH_CHECK((status) == hipSuccess, "HIP error: ", hipGetErrorString(status))
#define GPU_GET_ERROR_STRING(status) hipGetErrorString(status)
#define GPU_MALLOC_HOST(ptr, size) hipHostMalloc(ptr, size)
#define GPU_FREE_HOST(ptr) hipHostFree(ptr)
#define GPU_MALLOC(ptr, size) hipMalloc(ptr, size)
#define GPU_FREE(ptr) hipFree(ptr)
#define GPU_MEMCPY_ASYNC hipMemcpyAsync
#define GPU_MEMCPY_H2D hipMemcpyHostToDevice
#define GPU_MEMCPY_D2H hipMemcpyDeviceToHost
#define GPU_EVENT_CREATE hipEventCreate
#define GPU_EVENT_DESTROY hipEventDestroy
#define GPU_EVENT_RECORD hipEventRecord
#define GPU_EVENT_QUERY hipEventQuery
#define GPU_EVENT_SYNCHRONIZE hipEventSynchronize
#define GPU_STREAM_WAIT_EVENT hipStreamWaitEvent
#define GPU_EVENT_ELAPSED_TIME hipEventElapsedTime

#else

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuError_t = cudaError_t;

namespace infinity_gpu_compat {
using Stream = c10::cuda::CUDAStream;
using StreamGuard = c10::StreamGuard;

inline Stream get_current_stream() {
    return c10::cuda::getCurrentCUDAStream();
}

inline Stream get_stream_from_pool(bool is_high_priority, int device_index) {
    return c10::cuda::getStreamFromPool(is_high_priority, device_index);
}

inline gpuStream_t raw_stream(Stream stream) {
    return stream.stream();
}
}  // namespace infinity_gpu_compat

#define GPU_BACKEND_NAME "cuda"
#define GPU_SUCCESS cudaSuccess
#define GPU_ERROR_NOT_READY cudaErrorNotReady
#define GPU_CHECK(status) TORCH_CHECK((status) == cudaSuccess, "CUDA error: ", cudaGetErrorString(status))
#define GPU_GET_ERROR_STRING(status) cudaGetErrorString(status)
#define GPU_MALLOC_HOST(ptr, size) cudaMallocHost(ptr, size)
#define GPU_FREE_HOST(ptr) cudaFreeHost(ptr)
#define GPU_MALLOC(ptr, size) cudaMalloc(ptr, size)
#define GPU_FREE(ptr) cudaFree(ptr)
#define GPU_MEMCPY_ASYNC cudaMemcpyAsync
#define GPU_MEMCPY_H2D cudaMemcpyHostToDevice
#define GPU_MEMCPY_D2H cudaMemcpyDeviceToHost
#define GPU_EVENT_CREATE cudaEventCreate
#define GPU_EVENT_DESTROY cudaEventDestroy
#define GPU_EVENT_RECORD cudaEventRecord
#define GPU_EVENT_QUERY cudaEventQuery
#define GPU_EVENT_SYNCHRONIZE cudaEventSynchronize
#define GPU_STREAM_WAIT_EVENT cudaStreamWaitEvent
#define GPU_EVENT_ELAPSED_TIME cudaEventElapsedTime

#endif
