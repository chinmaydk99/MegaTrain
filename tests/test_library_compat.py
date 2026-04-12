import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from infinity.config.training import (
    default_attn_implementation,
    deepspeed_cpu_adam_available,
    flash_attn_available,
    flash_linear_attention_available,
)
from infinity.config.yaml_loader import yaml_to_training_config
from infinity.device import get_backend
from infinity.model import cpu_master as cpu_master_module


def _load_train_module():
    train_path = Path(__file__).resolve().parents[1] / "examples" / "train.py"
    spec = importlib.util.spec_from_file_location("megatrain_examples_train", train_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flash_attention_available_or_fallback():
    attn_impl = default_attn_implementation()
    assert attn_impl in {"flash_attention_2", "sdpa"}

    backend = get_backend()
    if backend.is_rocm and not flash_attn_available():
        assert attn_impl == "sdpa"


def test_yaml_loader_uses_backend_default_attention():
    config = yaml_to_training_config(
        {
            "model": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "dtype": "bfloat16",
                "device": 0,
            },
            "dataset": {
                "name": "alpaca_en_demo",
                "dataset_dir": "data",
            },
        }
    )

    assert config.attn_implementation == default_attn_implementation()


@pytest.mark.gpu
def test_sdpa_attention_forward_backward(skip_no_gpu):
    q = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)

    out = F.scaled_dot_product_attention(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.gpu
def test_flash_attention_forward_backward(skip_no_gpu):
    pytest.importorskip("flash_attn")
    flash_attn = importlib.import_module("flash_attn")
    flash_attn_func = getattr(flash_attn, "flash_attn_func", None)
    if flash_attn_func is None:
        pytest.skip("flash_attn_func is unavailable in this build")

    q = torch.randn(2, 8, 4, 16, device="cuda", dtype=torch.float16, requires_grad=True)
    out = flash_attn_func(q, q, q, 0.0, causal=False)
    out.sum().backward()
    assert q.grad is not None


def test_flash_ce_or_pytorch_fallback(cpu_master_model):
    if cpu_master_module.FLASH_CE_AVAILABLE:
        assert cpu_master_model.ce_loss is not None
    else:
        assert cpu_master_model.ce_loss is None


@pytest.mark.gpu
def test_cross_entropy_numerical_parity(skip_no_gpu, cpu_master_model):
    logits = torch.randn(6, 11, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 1, 5, 3, 7, 2], device="cuda", dtype=torch.long)

    torch_loss = F.cross_entropy(logits, labels, reduction="none")
    manual_loss = -F.log_softmax(logits, dim=-1)[torch.arange(labels.numel(), device="cuda"), labels]
    assert torch.allclose(torch_loss, manual_loss, atol=1e-5)

    if cpu_master_model.ce_loss is not None:
        flash_loss = cpu_master_model.ce_loss(logits, labels)
        assert torch.allclose(flash_loss, torch_loss, atol=1e-4, rtol=1e-4)


def test_deepspeed_cpuadam_or_pytorch_fallback():
    available = deepspeed_cpu_adam_available()
    assert available is True or available is False


def test_optimizer_step_updates_params():
    param = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
    grad = torch.randn_like(param)
    before = param.detach().clone()

    if deepspeed_cpu_adam_available():
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        optimizer = DeepSpeedCPUAdam([param], lr=1e-3, adamw_mode=True)
    else:
        optimizer = torch.optim.AdamW([param], lr=1e-3)

    param.grad = grad.clone()
    optimizer.step()
    assert not torch.allclose(param.detach(), before)


def test_optimizer_numerical_parity():
    if not deepspeed_cpu_adam_available():
        pytest.skip("DeepSpeed CPUAdam is not installed")

    from deepspeed.ops.adam import DeepSpeedCPUAdam

    base = torch.randn(8, 8, dtype=torch.float32)
    grad = torch.randn_like(base)

    deepspeed_param = torch.nn.Parameter(base.clone())
    torch_param = torch.nn.Parameter(base.clone())

    deepspeed_opt = DeepSpeedCPUAdam([deepspeed_param], lr=1e-3, adamw_mode=True)
    torch_opt = torch.optim.AdamW([torch_param], lr=1e-3)

    deepspeed_param.grad = grad.clone()
    torch_param.grad = grad.clone()

    deepspeed_opt.step()
    torch_opt.step()

    assert torch.allclose(
        deepspeed_param.detach(),
        torch_param.detach(),
        atol=5e-5,
        rtol=5e-4,
    )


def test_flash_linear_attention_import():
    if not flash_linear_attention_available():
        pytest.skip("flash-linear-attention is not installed")

    try:
        module = importlib.import_module("fla")
    except ModuleNotFoundError:
        module = importlib.import_module("flash_linear_attention")
    assert module is not None


def test_optional_processor_falls_back_without_vision_deps(monkeypatch, caplog):
    train_module = _load_train_module()

    def _raise_import_error(*args, **kwargs):
        raise ImportError("torchvision missing")

    monkeypatch.setattr(train_module.AutoProcessor, "from_pretrained", _raise_import_error)

    with caplog.at_level("WARNING"):
        processor = train_module._load_optional_processor(
            "Qwen/Qwen3.5-27B", trust_remote_code=True
        )

    assert processor is None
    assert "Continuing without processor" in caplog.text
    assert "image/video datasets still require" in caplog.text


def test_optional_processor_returns_loaded_processor(monkeypatch):
    train_module = _load_train_module()
    sentinel = object()

    monkeypatch.setattr(
        train_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: sentinel,
    )

    processor = train_module._load_optional_processor(
        "Qwen/Qwen3.5-27B", trust_remote_code=True
    )

    assert processor is sentinel
