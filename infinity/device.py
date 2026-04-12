"""Backend detection helpers for CUDA and ROCm runtimes."""

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Backend:
    """Normalized view of the active runtime backend."""

    name: str
    is_available: bool
    device_count: int
    device_name: Optional[str]
    memory_total: int
    pcie_gen: Optional[str]
    stream_class: Optional[type]
    event_class: Optional[type]

    def create_stream(self, device: int = 0):
        """Create a GPU stream for the active backend."""
        if not self.is_available or self.stream_class is None:
            return None
        return torch.cuda.Stream(device=device)

    def create_event(self, enable_timing: bool = False):
        """Create a GPU event for the active backend."""
        if not self.is_available or self.event_class is None:
            return None
        return torch.cuda.Event(enable_timing=enable_timing)

    @property
    def is_rocm(self) -> bool:
        return self.name == "rocm"

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"


def _detect_backend_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    return "cuda"


def detect_backend() -> Backend:
    """Detect the active execution backend."""
    name = _detect_backend_name()
    is_available = name != "cpu"

    if not is_available:
        return Backend(
            name="cpu",
            is_available=False,
            device_count=0,
            device_name=None,
            memory_total=0,
            pcie_gen=os.environ.get("MEGATRAIN_PCIE_GEN"),
            stream_class=None,
            event_class=None,
        )

    device_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    return Backend(
        name=name,
        is_available=True,
        device_count=device_count,
        device_name=torch.cuda.get_device_name(0),
        memory_total=int(props.total_memory),
        pcie_gen=os.environ.get("MEGATRAIN_PCIE_GEN"),
        stream_class=torch.cuda.Stream,
        event_class=torch.cuda.Event,
    )


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    """Return the cached backend description."""
    backend = detect_backend()
    if backend.is_available:
        logger.info(
            "MegaTrain backend: %s (%s, %d device%s, %.1f GiB on device 0)",
            backend.name,
            backend.device_name,
            backend.device_count,
            "" if backend.device_count == 1 else "s",
            backend.memory_total / 1024**3,
        )
    else:
        logger.info("MegaTrain backend: cpu")
    return backend


BACKEND = get_backend()
