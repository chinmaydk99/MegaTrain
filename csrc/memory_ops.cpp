// PyTorch C++/CUDA extension for memory management primitives
#include <torch/extension.h>
#include "gpu_compat.h"
#include <vector>
#include <mutex>

// ============================================================================
// Pinned Memory Buffer Pool
// ============================================================================

class PinnedBufferPool {
public:
    PinnedBufferPool(size_t buffer_size, size_t num_buffers)
        : buffer_size_(buffer_size), num_buffers_(num_buffers) {
        buffers_.resize(num_buffers);
        free_list_.reserve(num_buffers);

        for (size_t i = 0; i < num_buffers; i++) {
            GPU_CHECK(GPU_MALLOC_HOST(&buffers_[i], buffer_size));
            free_list_.push_back(i);
        }
    }

    ~PinnedBufferPool() {
        for (auto ptr : buffers_) {
            if (ptr) {
                auto status = GPU_FREE_HOST(ptr);
                (void)status;
            }
        }
    }

    int64_t acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_.empty()) return -1;
        int64_t idx = free_list_.back();
        free_list_.pop_back();
        return idx;
    }

    void release(int64_t idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (idx >= 0 && idx < static_cast<int64_t>(num_buffers_)) {
            free_list_.push_back(idx);
        }
    }

    void* get_ptr(int64_t idx) {
        if (idx < 0 || idx >= static_cast<int64_t>(num_buffers_)) return nullptr;
        return buffers_[idx];
    }

    size_t buffer_size() const { return buffer_size_; }
    size_t num_free() const { return free_list_.size(); }

private:
    size_t buffer_size_;
    size_t num_buffers_;
    std::vector<void*> buffers_;
    std::vector<int64_t> free_list_;
    std::mutex mutex_;
};

// Global pool instance
static std::unique_ptr<PinnedBufferPool> g_pool;

void init_pool(int64_t buffer_size, int64_t num_buffers) {
    g_pool = std::make_unique<PinnedBufferPool>(buffer_size, num_buffers);
}

void destroy_pool() {
    g_pool.reset();
}

int64_t pool_acquire() {
    TORCH_CHECK(g_pool, "Pool not initialized");
    return g_pool->acquire();
}

void pool_release(int64_t idx) {
    TORCH_CHECK(g_pool, "Pool not initialized");
    g_pool->release(idx);
}

int64_t pool_num_free() {
    TORCH_CHECK(g_pool, "Pool not initialized");
    return g_pool->num_free();
}

// ============================================================================
// Async Memory Copy
// ============================================================================

void memcpy_h2d_async(
    torch::Tensor dst,
    int64_t pool_idx,
    int64_t num_bytes,
    int64_t stream_ptr
) {
    TORCH_CHECK(g_pool, "Pool not initialized");
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA tensor");

    void* src = g_pool->get_ptr(pool_idx);
    TORCH_CHECK(src, "Invalid pool index");

    gpuStream_t stream = reinterpret_cast<gpuStream_t>(stream_ptr);
    GPU_CHECK(GPU_MEMCPY_ASYNC(
        dst.data_ptr(),
        src,
        num_bytes,
        GPU_MEMCPY_H2D,
        stream
    ));
}

void memcpy_d2h_async(
    int64_t pool_idx,
    torch::Tensor src,
    int64_t num_bytes,
    int64_t stream_ptr
) {
    TORCH_CHECK(g_pool, "Pool not initialized");
    TORCH_CHECK(src.is_cuda(), "src must be CUDA tensor");

    void* dst = g_pool->get_ptr(pool_idx);
    TORCH_CHECK(dst, "Invalid pool index");

    gpuStream_t stream = reinterpret_cast<gpuStream_t>(stream_ptr);
    GPU_CHECK(GPU_MEMCPY_ASYNC(
        dst,
        src.data_ptr(),
        num_bytes,
        GPU_MEMCPY_D2H,
        stream
    ));
}

// View pinned buffer as tensor
torch::Tensor pool_to_tensor(int64_t pool_idx, std::vector<int64_t> shape, torch::ScalarType dtype) {
    TORCH_CHECK(g_pool, "Pool not initialized");
    void* ptr = g_pool->get_ptr(pool_idx);
    TORCH_CHECK(ptr, "Invalid pool index");

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(ptr, shape, options);
}

// ============================================================================
// CUDA Events
// ============================================================================

int64_t event_create() {
    gpuEvent_t event;
    GPU_CHECK(GPU_EVENT_CREATE(&event));
    return reinterpret_cast<int64_t>(event);
}

void event_destroy(int64_t event_ptr) {
    gpuEvent_t event = reinterpret_cast<gpuEvent_t>(event_ptr);
    GPU_CHECK(GPU_EVENT_DESTROY(event));
}

void event_record(int64_t event_ptr, int64_t stream_ptr) {
    gpuEvent_t event = reinterpret_cast<gpuEvent_t>(event_ptr);
    gpuStream_t stream = reinterpret_cast<gpuStream_t>(stream_ptr);
    GPU_CHECK(GPU_EVENT_RECORD(event, stream));
}

bool event_query(int64_t event_ptr) {
    gpuEvent_t event = reinterpret_cast<gpuEvent_t>(event_ptr);
    gpuError_t status = GPU_EVENT_QUERY(event);
    if (status == GPU_SUCCESS) return true;
    if (status == GPU_ERROR_NOT_READY) return false;
    GPU_CHECK(status);
    return false;
}

void event_synchronize(int64_t event_ptr) {
    gpuEvent_t event = reinterpret_cast<gpuEvent_t>(event_ptr);
    GPU_CHECK(GPU_EVENT_SYNCHRONIZE(event));
}

void stream_wait_event(int64_t stream_ptr, int64_t event_ptr) {
    gpuStream_t stream = reinterpret_cast<gpuStream_t>(stream_ptr);
    gpuEvent_t event = reinterpret_cast<gpuEvent_t>(event_ptr);
    GPU_CHECK(GPU_STREAM_WAIT_EVENT(stream, event, 0));
}

float event_elapsed_time(int64_t start_ptr, int64_t end_ptr) {
    gpuEvent_t start = reinterpret_cast<gpuEvent_t>(start_ptr);
    gpuEvent_t end = reinterpret_cast<gpuEvent_t>(end_ptr);
    float ms;
    GPU_CHECK(GPU_EVENT_ELAPSED_TIME(&ms, start, end));
    return ms;
}

// ============================================================================
// Stream utilities
// ============================================================================

int64_t get_current_stream_ptr() {
    auto stream = infinity_gpu_compat::get_current_stream();
    return reinterpret_cast<int64_t>(infinity_gpu_compat::raw_stream(stream));
}

// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Pool management
    m.def("init_pool", &init_pool, "Initialize pinned buffer pool");
    m.def("destroy_pool", &destroy_pool, "Destroy pinned buffer pool");
    m.def("pool_acquire", &pool_acquire, "Acquire buffer from pool");
    m.def("pool_release", &pool_release, "Release buffer to pool");
    m.def("pool_num_free", &pool_num_free, "Get number of free buffers");
    m.def("pool_to_tensor", &pool_to_tensor, "View pool buffer as tensor");

    // Async memcpy
    m.def("memcpy_h2d_async", &memcpy_h2d_async, "Async copy from pinned pool to GPU");
    m.def("memcpy_d2h_async", &memcpy_d2h_async, "Async copy from GPU to pinned pool");

    // Events
    m.def("event_create", &event_create, "Create CUDA event");
    m.def("event_destroy", &event_destroy, "Destroy CUDA event");
    m.def("event_record", &event_record, "Record event on stream");
    m.def("event_query", &event_query, "Query if event completed");
    m.def("event_synchronize", &event_synchronize, "Wait for event");
    m.def("stream_wait_event", &stream_wait_event, "Make stream wait for event");
    m.def("event_elapsed_time", &event_elapsed_time, "Get elapsed time between events");

    // Stream
    m.def("get_current_stream_ptr", &get_current_stream_ptr, "Get current stream pointer");
}
