import pytest
import torch
from transformers import AutoModelForCausalLM, GPT2Config

from infinity import CPUMasterConfig, CPUMasterModel


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a GPU backend")
    config.addinivalue_line("markers", "slow: runs a slower integration-style check")
    config.addinivalue_line("markers", "benchmark: records benchmark-oriented checks")
    config.addinivalue_line("markers", "mi355x: targets the MI355X validation path")


@pytest.fixture
def skip_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU backend required")


@pytest.fixture
def tiny_model_config():
    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=64,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def tiny_hf_model(tiny_model_config):
    return AutoModelForCausalLM.from_config(tiny_model_config)


@pytest.fixture
def cpu_master_model(skip_no_gpu, tiny_hf_model):
    config = CPUMasterConfig(
        model_name="tiny-gpt2",
        dataset_path="dummy",
        batch_size=2,
        max_seq_len=16,
        num_steps=1,
        checkpoint_interval=1,
        num_grad_slabs=4,
        device=0,
        dtype=torch.float32,
        attn_implementation="eager",
        enable_timing=False,
    )
    model = CPUMasterModel(tiny_hf_model, config)
    try:
        yield model
    finally:
        model.cleanup()


@pytest.fixture
def dummy_batch():
    batch_size = 2
    seq_len = 16
    vocab_size = 128

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
