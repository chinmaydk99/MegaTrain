import torch

from infinity.device import detect_backend, get_backend


def test_detect_backend_returns_known_type():
    backend = detect_backend()
    assert backend.name in {"cpu", "cuda", "rocm"}


def test_backend_has_required_attributes():
    backend = get_backend()
    for attr in (
        "name",
        "is_available",
        "device_count",
        "device_name",
        "memory_total",
        "pcie_gen",
        "stream_class",
        "event_class",
    ):
        assert hasattr(backend, attr)


def test_backend_stream_creation_matches_torch():
    backend = get_backend()
    if not backend.is_available:
        assert backend.create_stream() is None
        return

    stream = backend.create_stream()
    assert isinstance(stream, torch.cuda.Stream)


def test_backend_event_creation_matches_torch():
    backend = get_backend()
    if not backend.is_available:
        assert backend.create_event() is None
        return

    event = backend.create_event()
    assert isinstance(event, torch.cuda.Event)


def test_pin_memory_works_on_both_backends():
    tensor = torch.empty(16, dtype=torch.float32).pin_memory()
    assert tensor.is_pinned()


def test_backend_properties_reflect_hardware():
    backend = get_backend()
    if not backend.is_available:
        assert backend.device_count == 0
        assert backend.device_name is None
        assert backend.memory_total == 0
        return

    assert backend.device_count >= 1
    assert backend.device_name
    assert backend.memory_total > 0
