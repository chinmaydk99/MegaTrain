import os
import subprocess
import sys
import math

import psutil
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from infinity import CPUMasterConfig, CPUMasterModel
from infinity.model import cpu_master as cpu_master_module


def _make_streaming_model(tiny_model_config, state_dict=None):
    hf_model = AutoModelForCausalLM.from_config(tiny_model_config)
    if state_dict is not None:
        hf_model.load_state_dict(state_dict)
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
    return CPUMasterModel(hf_model, config)


def _make_tiny_qwen2_streaming_model():
    qwen_config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
    )
    qwen_config._attn_implementation = "sdpa"
    hf_model = AutoModelForCausalLM.from_config(qwen_config)
    config = CPUMasterConfig(
        model_name="tiny-qwen2",
        dataset_path="dummy",
        batch_size=2,
        max_seq_len=16,
        num_steps=1,
        checkpoint_interval=1,
        num_grad_slabs=4,
        device=0,
        dtype=torch.float32,
        attn_implementation="sdpa",
        enable_timing=False,
    )
    return CPUMasterModel(hf_model, config)


def _build_layer_kwargs(model, attention_mask):
    batch_size, seq_len = attention_mask.shape
    mask = attention_mask.to(model.device)
    cache_position = torch.arange(seq_len, device=model.device)
    position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = None

    if model.rotary_gpu and model.layer_accepts_position_embeddings:
        dummy = torch.empty((1, 1, seq_len, model.head_dim), device=model.device, dtype=torch.float32)
        cos, sin = model.rotary_gpu(dummy, position_ids[:1])
        position_embeddings = (cos.to(model.config.dtype), sin.to(model.config.dtype))

    return model._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)


def _run_streamed_layer(model, layer_idx, buffer_idx, hidden, layer_kwargs):
    model._load_layer_to_buffer_async(layer_idx, buffer_idx)
    model.weight_stream.synchronize()
    with torch.no_grad():
        model._unflatten_to_layer(layer_idx, buffer_idx)
        gpu_layer = model._get_gpu_layer(layer_idx, buffer_idx)
        out = gpu_layer(hidden, **layer_kwargs)
    torch.cuda.synchronize()
    return (out[0] if isinstance(out, tuple) else out).detach()


@pytest.mark.gpu
def test_tiny_model_loss_decreases(skip_no_gpu, cpu_master_model):
    optimizer = torch.optim.AdamW(cpu_master_model.get_parameters(), lr=5e-3)
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()

    losses = []
    for _ in range(20):
        loss_val, _, _ = cpu_master_model.forward_and_backward(input_ids, attention_mask, labels)
        optimizer.step()
        cpu_master_model._sync_params_to_gpu()
        cpu_master_model.zero_grad()
        optimizer.zero_grad()
        losses.append(loss_val)

    assert losses[-1] < losses[0] * 0.5


@pytest.mark.gpu
def test_forward_backward_numerical_parity(skip_no_gpu, tiny_model_config, dummy_batch):
    reference_model = AutoModelForCausalLM.from_config(tiny_model_config).to("cuda", dtype=torch.float32)
    state_dict = {k: v.detach().cpu().clone() for k, v in reference_model.state_dict().items()}
    streaming_model = _make_streaming_model(tiny_model_config, state_dict)

    try:
        ref_out = reference_model(
            input_ids=dummy_batch["input_ids"].to("cuda"),
            attention_mask=dummy_batch["attention_mask"].to("cuda"),
            labels=dummy_batch["labels"].to("cuda"),
        )
        ref_loss = ref_out.loss
        ref_loss.backward()

        loss_val, _, _ = streaming_model.forward_and_backward(
            dummy_batch["input_ids"],
            dummy_batch["attention_mask"],
            dummy_batch["labels"],
        )

        assert abs(loss_val - ref_loss.item()) < 5e-2
    finally:
        streaming_model.cleanup()


@pytest.mark.gpu
def test_gradient_accumulation_correctness(skip_no_gpu, cpu_master_model, dummy_batch):
    cpu_master_model.zero_grad()

    input_ids_gpu = dummy_batch["input_ids"].to(cpu_master_model.device)
    hidden = cpu_master_model.emb_gpu(input_ids_gpu).detach().requires_grad_(True)
    layer_kwargs = _build_layer_kwargs(cpu_master_model, dummy_batch["attention_mask"])

    cpu_master_model._load_layer_to_buffer_async(0, 0)
    cpu_master_model.weight_stream.synchronize()
    cpu_master_model._unflatten_to_layer(0, 0)
    gpu_layer = cpu_master_model._get_gpu_layer(0, 0)

    for param in gpu_layer.parameters():
        param.requires_grad_(True)

    out = gpu_layer(hidden, **layer_kwargs)
    layer_output = out[0] if isinstance(out, tuple) else out
    grad_output = torch.randn_like(layer_output)
    grads = torch.autograd.grad(
        outputs=layer_output,
        inputs=(hidden, *gpu_layer.parameters()),
        grad_outputs=grad_output,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    reference_param_grads = [grad.detach().cpu().clone() for grad in grads[1:]]

    for param, grad in zip(gpu_layer.parameters(), grads[1:]):
        param.grad = grad

    cpu_master_model.backward_done_events[0].record(torch.cuda.current_stream(cpu_master_model.device))
    cpu_master_model._collect_layer_grads_async(0, 0)
    cpu_master_model._accumulate_grads_batch()

    for param in gpu_layer.parameters():
        param.requires_grad_(False)

    for cpu_param, reference_grad in zip(cpu_master_model.layer_cpu_params[0], reference_param_grads):
        assert cpu_param.grad is not None
        assert torch.allclose(cpu_param.grad, reference_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_checkpoint_recompute_correctness(skip_no_gpu, cpu_master_model, dummy_batch):
    input_ids_gpu = dummy_batch["input_ids"].to(cpu_master_model.device)
    hidden = cpu_master_model.emb_gpu(input_ids_gpu)
    layer_kwargs = _build_layer_kwargs(cpu_master_model, dummy_batch["attention_mask"])

    checkpoint = _run_streamed_layer(cpu_master_model, 0, 0, hidden, layer_kwargs)
    original = _run_streamed_layer(cpu_master_model, 1, 1, checkpoint, layer_kwargs)
    recomputed = _run_streamed_layer(cpu_master_model, 1, 0, checkpoint.clone(), layer_kwargs)

    assert torch.allclose(recomputed, original, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_double_buffer_produces_correct_output(skip_no_gpu, cpu_master_model, dummy_batch):
    input_ids_gpu = dummy_batch["input_ids"].to(cpu_master_model.device)
    hidden = cpu_master_model.emb_gpu(input_ids_gpu)
    layer_kwargs = _build_layer_kwargs(cpu_master_model, dummy_batch["attention_mask"])

    out_buf0 = _run_streamed_layer(cpu_master_model, 0, 0, hidden, layer_kwargs)
    out_buf1 = _run_streamed_layer(cpu_master_model, 0, 1, hidden, layer_kwargs)

    assert torch.allclose(out_buf0, out_buf1, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_optimizer_sync_to_gpu(skip_no_gpu, cpu_master_model, dummy_batch):
    optimizer = torch.optim.AdamW(cpu_master_model.get_parameters(), lr=1e-3)
    cpu_master_model.forward_and_backward(
        dummy_batch["input_ids"],
        dummy_batch["attention_mask"],
        dummy_batch["labels"],
    )
    optimizer.step()
    cpu_master_model._sync_params_to_gpu()

    emb_cpu = next(cpu_master_model.embedding.parameters()).detach().cpu()
    emb_gpu = next(cpu_master_model.emb_gpu.parameters()).detach().cpu()
    assert torch.allclose(emb_cpu, emb_gpu, atol=1e-6, rtol=1e-6)

    head_cpu = next(cpu_master_model.lm_head.parameters()).detach().cpu()
    head_gpu = next(cpu_master_model.lm_head_gpu.parameters()).detach().cpu()
    assert torch.allclose(head_cpu, head_gpu, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
def test_materialize_hf_model_matches_reference_update(skip_no_gpu, tiny_model_config, dummy_batch):
    base_model = AutoModelForCausalLM.from_config(tiny_model_config)
    state_dict = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    streaming_model = _make_streaming_model(tiny_model_config, state_dict)
    export_model = AutoModelForCausalLM.from_config(tiny_model_config)

    try:
        streaming_optimizer = torch.optim.AdamW(streaming_model.get_parameters(), lr=1e-3)

        streaming_model.forward_and_backward(
            dummy_batch["input_ids"],
            dummy_batch["attention_mask"],
            dummy_batch["labels"],
        )
        streaming_optimizer.step()
        streaming_model._sync_params_to_gpu()

        streaming_model.materialize_hf_model(export_model)
        exported = cpu_master_module._discover_model_components(export_model)

        for exported_param, streaming_param in zip(
            exported["embedding"].parameters(), streaming_model.embedding.parameters()
        ):
            assert torch.allclose(exported_param, streaming_param, atol=1e-6, rtol=1e-6)
        for exported_layer, streaming_layer in zip(exported["layers"], streaming_model.cpu_layers):
            for exported_param, streaming_param in zip(
                exported_layer.parameters(), streaming_layer.parameters()
            ):
                assert torch.allclose(exported_param, streaming_param, atol=1e-6, rtol=1e-6)
        for exported_param, streaming_param in zip(
            exported["lm_head"].parameters(), streaming_model.lm_head.parameters()
        ):
            assert torch.allclose(exported_param, streaming_param, atol=1e-6, rtol=1e-6)
    finally:
        streaming_model.cleanup()


@pytest.mark.gpu
def test_qwen2_sdpa_mask_normalization(skip_no_gpu):
    model = _make_tiny_qwen2_streaming_model()
    try:
        input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = input_ids.clone()

        prepared_mask = model._prepare_attention_mask(attention_mask.to(model.device))
        assert prepared_mask.shape == (2, 1, 16, 16)
        assert prepared_mask.dtype == torch.float32

        loss_val, _, _ = model.forward_and_backward(input_ids, attention_mask, labels)
        assert math.isfinite(loss_val)
    finally:
        model.cleanup()


@pytest.mark.gpu
def test_bf16_runtime_uses_fp32_cpu_master_weights(skip_no_gpu, tiny_model_config):
    hf_model = AutoModelForCausalLM.from_config(tiny_model_config).to(dtype=torch.bfloat16)
    config = CPUMasterConfig(
        model_name="tiny-gpt2-bf16",
        dataset_path="dummy",
        batch_size=2,
        max_seq_len=16,
        num_steps=1,
        checkpoint_interval=1,
        num_grad_slabs=4,
        device=0,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        enable_timing=False,
    )
    model = CPUMasterModel(hf_model, config)
    try:
        assert next(model.embedding.parameters()).dtype == torch.float32
        assert next(model.cpu_layers[0].parameters()).dtype == torch.float32
        assert next(model.emb_gpu.parameters()).dtype == torch.bfloat16
        assert next(model._get_gpu_layer(0, 0).parameters()).dtype == torch.bfloat16
    finally:
        model.cleanup()


@pytest.mark.gpu
@pytest.mark.slow
def test_7b_model_trains_if_memory_sufficient(skip_no_gpu):
    if os.environ.get("MEGATRAIN_RUN_7B_TEST") != "1":
        pytest.skip("Set MEGATRAIN_RUN_7B_TEST=1 to run the 7B integration test")

    if psutil.virtual_memory().total < 100 * 1024**3:
        pytest.skip("Insufficient host memory for the 7B integration test")

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "examples/train.py",
            "--config",
            "examples/configs/qwen_7b_rocm.yaml",
            "--num-steps",
            "5",
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    combined_output = result.stdout + result.stderr
    assert "Step 5/5" in combined_output
