import gc
import importlib

import pytest
import torch

import infinity.csrc as memory_ext


def _require_memory_extension():
    if not memory_ext.HAS_GPU_EXT:
        pytest.skip("infinity_memory_ops is not built")


def _require_cuda_pipeline():
    try:
        return importlib.import_module("cuda_pipeline")
    except ImportError:
        pytest.skip("cuda_pipeline is not built")


@pytest.fixture
def pinned_pool(skip_no_gpu):
    _require_memory_extension()
    pool = memory_ext.PinnedPool(4096, 2)
    try:
        yield pool
    finally:
        del pool
        gc.collect()


@pytest.mark.gpu
def test_memory_ops_extension_imports(skip_no_gpu):
    _require_memory_extension()
    assert memory_ext._C is not None
    stream_ptr = memory_ext._C.get_current_stream_ptr()
    assert isinstance(stream_ptr, int)
    assert stream_ptr >= 0


@pytest.mark.gpu
def test_pinned_pool_acquire_release(skip_no_gpu, pinned_pool):
    assert pinned_pool.num_free() == 2
    idx = pinned_pool.acquire()
    assert idx >= 0
    assert pinned_pool.num_free() == 1

    host_tensor = pinned_pool.as_tensor(idx, [256], torch.float32)
    host_tensor.fill_(3.5)
    assert host_tensor.is_pinned()
    assert torch.allclose(host_tensor, torch.full((256,), 3.5))

    pinned_pool.release(idx)
    assert pinned_pool.num_free() == 2


@pytest.mark.gpu
def test_copy_h2d_and_d2h_async(skip_no_gpu, pinned_pool):
    idx = pinned_pool.acquire()
    host_tensor = pinned_pool.as_tensor(idx, [256], torch.float32)
    source = torch.arange(256, dtype=torch.float32)
    host_tensor.copy_(source)

    device_tensor = torch.empty(256, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        memory_ext.copy_h2d_async(
            device_tensor,
            idx,
            host_tensor.numel() * host_tensor.element_size(),
            stream,
        )
    stream.synchronize()
    assert torch.allclose(device_tensor.cpu(), source)

    device_tensor.add_(1.0)
    with torch.cuda.stream(stream):
        memory_ext.copy_d2h_async(
            idx,
            device_tensor,
            host_tensor.numel() * host_tensor.element_size(),
            stream,
        )
    stream.synchronize()
    roundtrip = pinned_pool.as_tensor(idx, [256], torch.float32).clone()
    assert torch.allclose(roundtrip, source + 1.0)

    pinned_pool.release(idx)


@pytest.mark.gpu
def test_event_record_wait_and_elapsed_time(skip_no_gpu):
    _require_memory_extension()
    start = memory_ext.Event()
    end = memory_ext.Event()
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        start.record(stream)
        tensor = torch.randn(4096, device="cuda")
        tensor.mul_(tensor)
        end.record(stream)

    memory_ext.stream_wait_event(torch.cuda.current_stream(), end)
    end.synchronize()

    assert end.query()
    assert start.elapsed_time(end) >= 0.0


@pytest.mark.gpu
def test_cuda_pipeline_batched_copy_params(skip_no_gpu):
    cuda_pipeline = _require_cuda_pipeline()
    cpu_tensors = [torch.randn(128, dtype=torch.float32).pin_memory() for _ in range(4)]
    gpu_tensors = [torch.empty(128, dtype=torch.float32, device="cuda") for _ in range(4)]

    cuda_pipeline.batched_copy_params(cpu_tensors, gpu_tensors)
    torch.cuda.synchronize()

    for cpu_tensor, gpu_tensor in zip(cpu_tensors, gpu_tensors):
        assert torch.allclose(gpu_tensor.cpu(), cpu_tensor)


@pytest.mark.gpu
def test_cuda_pipeline_async_accumulate_grads(skip_no_gpu):
    cuda_pipeline = _require_cuda_pipeline()
    gpu_params = [torch.nn.Parameter(torch.randn(128, device="cuda")) for _ in range(4)]
    cpu_params = [torch.nn.Parameter(torch.zeros(128, dtype=torch.float32)) for _ in range(4)]

    for gpu_param in gpu_params:
        gpu_param.grad = torch.randn_like(gpu_param)

    cuda_pipeline.async_accumulate_grads(gpu_params, cpu_params)
    torch.cuda.synchronize()

    for gpu_param, cpu_param in zip(gpu_params, cpu_params):
        assert cpu_param.grad is not None
        assert torch.allclose(cpu_param.grad, gpu_param.grad.cpu(), atol=1e-6, rtol=1e-6)
