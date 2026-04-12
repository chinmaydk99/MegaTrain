import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU backend required"
)


def test_multi_stream_creation():
    streams = [torch.cuda.Stream() for _ in range(3)]
    assert all(isinstance(stream, torch.cuda.Stream) for stream in streams)


def test_event_record_and_wait_across_streams():
    src = torch.ones(8, device="cuda")
    dst = torch.zeros_like(src)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    event = torch.cuda.Event()

    with torch.cuda.stream(stream_a):
        src.add_(1.0)
        event.record(stream_a)

    with torch.cuda.stream(stream_b):
        stream_b.wait_event(event)
        dst.copy_(src)

    torch.cuda.synchronize()
    assert torch.equal(dst, torch.full_like(dst, 2.0))


def test_stream_wait_event_ordering():
    src = torch.arange(16, device="cuda", dtype=torch.float32)
    dst = torch.zeros_like(src)
    stream = torch.cuda.Stream()
    event = torch.cuda.Event()

    with torch.cuda.stream(stream):
        src.mul_(2.0)
        event.record(stream)

    torch.cuda.current_stream().wait_event(event)
    dst.copy_(src)
    torch.cuda.synchronize()

    assert torch.equal(dst, torch.arange(16, device="cuda", dtype=torch.float32) * 2.0)


def test_pinned_memory_allocation():
    cpu_tensor = torch.empty(1024, dtype=torch.float32).pin_memory()
    assert cpu_tensor.is_pinned()


def test_async_h2d_copy():
    cpu_tensor = torch.arange(64, dtype=torch.float32).pin_memory()
    gpu_tensor = torch.empty_like(cpu_tensor, device="cuda")
    gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(gpu_tensor.cpu(), cpu_tensor.cpu())


def test_async_d2h_copy():
    gpu_tensor = torch.arange(64, dtype=torch.float32, device="cuda")
    cpu_tensor = torch.empty(64, dtype=torch.float32).pin_memory()
    cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(cpu_tensor.cpu(), gpu_tensor.cpu())


def test_double_buffer_ping_pong():
    cpu_buffers = [
        torch.full((32,), float(i + 1), dtype=torch.float32).pin_memory()
        for i in range(4)
    ]
    gpu_buffers = [
        torch.empty(32, dtype=torch.float32, device="cuda"),
        torch.empty(32, dtype=torch.float32, device="cuda"),
    ]
    copy_stream = torch.cuda.Stream()

    for i, cpu_tensor in enumerate(cpu_buffers):
        buf_idx = i % 2
        with torch.cuda.stream(copy_stream):
            gpu_buffers[buf_idx].copy_(cpu_tensor, non_blocking=True)

    torch.cuda.synchronize()
    assert torch.all(gpu_buffers[0] == 3.0)
    assert torch.all(gpu_buffers[1] == 4.0)


def test_flat_buffer_unflatten_view():
    flat = torch.arange(24, dtype=torch.float32, device="cuda")
    first = flat[:8].view(2, 4)
    second = flat[8:].view(4, 4)
    torch.cuda.synchronize()
    assert first.shape == (2, 4)
    assert second.shape == (4, 4)
    assert torch.equal(first.flatten(), torch.arange(8, dtype=torch.float32, device="cuda"))
    assert torch.equal(second.flatten(), torch.arange(8, 24, dtype=torch.float32, device="cuda"))


def test_gpu_memory_tracking():
    torch.cuda.reset_peak_memory_stats()
    tensor = torch.empty(1024, 1024, device="cuda")
    current = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    del tensor
    torch.cuda.synchronize()
    assert current > 0
    assert peak >= current


def test_autograd_grad_per_layer():
    layer = nn.Linear(8, 4, bias=False, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    y = layer(x).sum()

    grads = torch.autograd.grad(
        outputs=y,
        inputs=(x, *layer.parameters()),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    assert grads[0] is not None
    assert grads[1] is not None


def test_autograd_grad_returns_correct_shapes():
    layer = nn.Linear(8, 4, bias=True, device="cuda", dtype=torch.float32)
    x = torch.randn(3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    y = layer(x).sum()

    grad_x, grad_w, grad_b = torch.autograd.grad(
        outputs=y,
        inputs=(x, layer.weight, layer.bias),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    assert grad_x.shape == x.shape
    assert grad_w.shape == layer.weight.shape
    assert grad_b.shape == layer.bias.shape
