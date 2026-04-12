"""Training configuration for CPU Master model."""

from dataclasses import dataclass, field
import importlib.util

import torch

from infinity.device import get_backend


def _module_exists(module_name: str) -> bool:
    """Return True when an importable module is available."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def flash_attn_available() -> bool:
    """Whether the flash-attn package is installed."""
    return _module_exists("flash_attn")


def deepspeed_cpu_adam_available() -> bool:
    """Whether DeepSpeed CPUAdam is importable."""
    return _module_exists("deepspeed.ops.adam")


def flash_linear_attention_available() -> bool:
    """Whether flash-linear-attention is importable."""
    return _module_exists("fla") or _module_exists("flash_linear_attention")


def default_attn_implementation() -> str:
    """Choose the safest default attention implementation for this backend."""
    backend = get_backend()
    if backend.is_rocm and not flash_attn_available():
        return "sdpa"
    return "flash_attention_2"


@dataclass
class CPUMasterConfig:
    """Configuration for CPUMasterModel training.

    Supports any HuggingFace decoder-only model (Llama, Qwen, Mistral, Phi, etc.).

    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        max_seq_len: Maximum sequence length
        batch_size: Batch size per training step
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_steps: Total number of training steps
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        eps: Adam epsilon parameter
        max_grad_norm: Maximum gradient norm for clipping
        device: CUDA device index
        dtype: Data type for GPU computations (bfloat16 or float16)
        seed: Random seed for reproducibility
        log_interval: Steps between logging
        checkpoint_interval: Layers between checkpoints (for gradient checkpointing)
        dataset_path: Path to training dataset
        enable_timing: Enable CUDA timing (adds sync overhead)
        num_grad_slabs: Number of gradient slab buffers (>= 2 * checkpoint_interval recommended)
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
        trust_remote_code: Trust remote code when loading HF models
        system_prompt: Optional system prompt for chat dataset (empty = no system message)
        query_field: Field name for user query in dataset
        response_field: Field name for assistant response in dataset
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    device: int = 0
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = field(default_factory=default_attn_implementation)
    trust_remote_code: bool = True

    # Dataset
    dataset_path: str = ""
    dataset_name: str = ""
    dataset_dir: str = "data"
    max_seq_len: int = 1024
    split_seed: int = 42
    train_ratio: float = 1.0
    eval_ratio: float = 0.0
    system_prompt: str = ""
    query_field: str = "query"
    response_field: str = "response"
    train_on_prompt: bool = False

    # VLM
    freeze_vision_encoder: bool = True
    freeze_projector: bool = False

    # Training
    batch_size: int = 96
    gradient_accumulation_steps: int = 1
    num_steps: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # Optimizer
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Memory
    checkpoint_interval: int = 4
    num_grad_slabs: int = 12

    # Logging
    log_interval: int = 1
    enable_timing: bool = True

    # Evaluation
    eval_enabled: bool = False
    eval_batch_size: int = 4
    eval_max_new_tokens: int = 128
    eval_num_samples: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_grad_slabs < 2 * self.checkpoint_interval:
            import warnings
            warnings.warn(
                f"num_grad_slabs ({self.num_grad_slabs}) < 2 * checkpoint_interval "
                f"({2 * self.checkpoint_interval}). This may cause gradient slab starvation."
            )

        if not self.dataset_path and not self.dataset_name:
            raise ValueError("Must specify either dataset_path or dataset_name")

        valid_attn = ("flash_attention_2", "sdpa", "eager")
        if self.attn_implementation not in valid_attn:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn}, "
                f"got '{self.attn_implementation}'"
            )

        for name, ratio in (("train_ratio", self.train_ratio), ("eval_ratio", self.eval_ratio)):
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {ratio}")
        if self.train_ratio + self.eval_ratio > 1.0 + 1e-8:
            raise ValueError(
                f"train_ratio + eval_ratio must be <= 1, got {self.train_ratio + self.eval_ratio}"
            )
        if self.eval_enabled and self.eval_ratio <= 0.0:
            raise ValueError("eval_enabled=True requires eval_ratio > 0")
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1")
        if self.eval_max_new_tokens < 1:
            raise ValueError("eval_max_new_tokens must be >= 1")
        if self.eval_num_samples < 0:
            raise ValueError("eval_num_samples must be >= 0")
