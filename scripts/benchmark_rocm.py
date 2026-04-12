#!/usr/bin/env python3
"""ROCm/CUDA benchmark helpers for MegaTrain.

This module is importable from tests and executable as a standalone script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, GPT2Config

from infinity import CPUMasterConfig, CPUMasterModel
from infinity.device import get_backend


PCIe_GEN5_REFERENCE_GBPS = 20.0


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU backend required for benchmarks")


def _benchmark_host_timed(op, warmup: int = 2, repeats: int = 5) -> dict[str, Any]:
    _require_gpu()

    for _ in range(warmup):
        op()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        op()
        times.append(time.perf_counter() - start)

    return {
        "seconds_min": min(times),
        "seconds_avg": sum(times) / len(times),
        "samples": times,
    }


def _dtype_numel_for_bytes(num_bytes: int, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return max(1, num_bytes // element_size)


def expected_pcie_bandwidth_floor_gbps() -> float:
    backend = get_backend()
    device_name = backend.device_name.lower()
    if any(token in device_name for token in ("mi300", "mi325", "mi350", "mi355", "h100", "h200")):
        return PCIe_GEN5_REFERENCE_GBPS
    return 5.0


def measure_h2d_bandwidth(
    num_bytes: int = 256 * 1024**2,
    repeats: int = 5,
    warmup: int = 2,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    _require_gpu()

    numel = _dtype_numel_for_bytes(num_bytes, dtype)
    host = torch.randn(numel, dtype=dtype).pin_memory()
    device = torch.empty(numel, dtype=dtype, device="cuda")
    stream = torch.cuda.Stream()

    def op() -> None:
        with torch.cuda.stream(stream):
            device.copy_(host, non_blocking=True)
        stream.synchronize()

    stats = _benchmark_host_timed(op, warmup=warmup, repeats=repeats)
    total_bytes = numel * host.element_size()
    gbps = total_bytes / stats["seconds_min"] / 1024**3
    return {
        "direction": "h2d",
        "num_bytes": total_bytes,
        "gbps": gbps,
        **stats,
    }


def measure_d2h_bandwidth(
    num_bytes: int = 256 * 1024**2,
    repeats: int = 5,
    warmup: int = 2,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    _require_gpu()

    numel = _dtype_numel_for_bytes(num_bytes, dtype)
    device = torch.randn(numel, dtype=dtype, device="cuda")
    host = torch.empty(numel, dtype=dtype).pin_memory()
    stream = torch.cuda.Stream()

    def op() -> None:
        with torch.cuda.stream(stream):
            host.copy_(device, non_blocking=True)
        stream.synchronize()

    stats = _benchmark_host_timed(op, warmup=warmup, repeats=repeats)
    total_bytes = numel * host.element_size()
    gbps = total_bytes / stats["seconds_min"] / 1024**3
    return {
        "direction": "d2h",
        "num_bytes": total_bytes,
        "gbps": gbps,
        **stats,
    }


def measure_overlap_efficiency(
    copy_num_bytes: int = 256 * 1024**2,
    matmul_dim: int = 4096,
    repeats: int = 5,
    warmup: int = 2,
    copy_dtype: torch.dtype = torch.float32,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    _require_gpu()

    copy_numel = _dtype_numel_for_bytes(copy_num_bytes, copy_dtype)
    host = torch.randn(copy_numel, dtype=copy_dtype).pin_memory()
    device_copy = torch.empty(copy_numel, dtype=copy_dtype, device="cuda")

    lhs = torch.randn((matmul_dim, matmul_dim), device="cuda", dtype=compute_dtype)
    rhs = torch.randn((matmul_dim, matmul_dim), device="cuda", dtype=compute_dtype)
    out = torch.empty((matmul_dim, matmul_dim), device="cuda", dtype=compute_dtype)

    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def copy_only() -> None:
        with torch.cuda.stream(copy_stream):
            device_copy.copy_(host, non_blocking=True)
        copy_stream.synchronize()

    def compute_only() -> None:
        with torch.cuda.stream(compute_stream):
            torch.mm(lhs, rhs, out=out)
        compute_stream.synchronize()

    def overlapped() -> None:
        with torch.cuda.stream(copy_stream):
            device_copy.copy_(host, non_blocking=True)
        with torch.cuda.stream(compute_stream):
            torch.mm(lhs, rhs, out=out)
        copy_stream.synchronize()
        compute_stream.synchronize()

    copy_stats = _benchmark_host_timed(copy_only, warmup=warmup, repeats=repeats)
    compute_stats = _benchmark_host_timed(compute_only, warmup=warmup, repeats=repeats)
    overlap_stats = _benchmark_host_timed(overlapped, warmup=warmup, repeats=repeats)

    copy_time = copy_stats["seconds_min"]
    compute_time = compute_stats["seconds_min"]
    overlap_time = overlap_stats["seconds_min"]
    max_overlap = max(min(copy_time, compute_time), 1e-9)
    efficiency = (copy_time + compute_time - overlap_time) / max_overlap
    efficiency = max(0.0, min(1.0, efficiency))

    return {
        "copy_seconds": copy_time,
        "compute_seconds": compute_time,
        "overlap_seconds": overlap_time,
        "overlap_efficiency": efficiency,
    }


def tiny_model_config() -> GPT2Config:
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


def make_dummy_batch(
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 128,
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_tiny_cpu_master_model() -> CPUMasterModel:
    hf_model = AutoModelForCausalLM.from_config(tiny_model_config())
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


def measure_single_layer_forward_latency(
    hf_model=None,
    batch_size: int = 2,
    seq_len: int = 16,
    repeats: int = 10,
    warmup: int = 3,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    _require_gpu()

    owns_model = hf_model is None
    if hf_model is None:
        hf_model = AutoModelForCausalLM.from_config(tiny_model_config())

    hf_model = hf_model.to("cuda", dtype=dtype).eval()
    hidden = torch.randn(batch_size, seq_len, hf_model.config.n_embd, device="cuda", dtype=dtype)
    layer = hf_model.transformer.h[0]
    layer_param_count = sum(p.numel() for p in layer.parameters())

    def op() -> None:
        with torch.no_grad():
            layer(hidden, use_cache=False, output_attentions=False)
        torch.cuda.synchronize()

    stats = _benchmark_host_timed(op, warmup=warmup, repeats=repeats)
    latency_ms = stats["seconds_avg"] * 1000.0
    approx_tflops = (2 * layer_param_count * batch_size * seq_len) / stats["seconds_min"] / 1e12

    if owns_model:
        del hf_model
        torch.cuda.empty_cache()

    return {
        "latency_ms": latency_ms,
        "approx_tflops": approx_tflops,
        **stats,
    }


def measure_full_step_throughput(
    model: CPUMasterModel | None = None,
    batch: dict[str, torch.Tensor] | None = None,
    repeats: int = 3,
    warmup: int = 1,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    _require_gpu()

    owns_model = model is None
    if model is None:
        model = build_tiny_cpu_master_model()
    if batch is None:
        batch = make_dummy_batch()

    optimizer = torch.optim.AdamW(model.get_parameters(), lr=learning_rate)
    num_params = sum(p.numel() for p in model.get_parameters())

    def run_step() -> tuple[float, int]:
        loss_val, n_tokens, _ = model.forward_and_backward(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        optimizer.step()
        model._sync_params_to_gpu()
        model.zero_grad()
        optimizer.zero_grad()
        return loss_val, n_tokens

    for _ in range(warmup):
        run_step()

    step_times = []
    token_rates = []
    tflops = []
    losses = []
    for _ in range(repeats):
        start = time.perf_counter()
        loss_val, n_tokens = run_step()
        elapsed = time.perf_counter() - start

        step_times.append(elapsed)
        token_rates.append(n_tokens / elapsed)
        tflops.append((6 * num_params * n_tokens) / elapsed / 1e12)
        losses.append(loss_val)

    if owns_model:
        model.cleanup()

    return {
        "avg_step_time_s": sum(step_times) / len(step_times),
        "min_step_time_s": min(step_times),
        "tokens_per_sec": sum(token_rates) / len(token_rates),
        "tflops": sum(tflops) / len(tflops),
        "losses": losses,
        "num_params": num_params,
    }


def run_benchmark_suite(
    copy_sizes_mib: list[int],
    repeats: int,
    warmup: int,
    matmul_dim: int,
) -> dict[str, Any]:
    h2d = [
        measure_h2d_bandwidth(num_bytes=size * 1024**2, repeats=repeats, warmup=warmup)
        for size in copy_sizes_mib
    ]
    d2h = [
        measure_d2h_bandwidth(num_bytes=size * 1024**2, repeats=repeats, warmup=warmup)
        for size in copy_sizes_mib
    ]
    layer = measure_single_layer_forward_latency(repeats=repeats, warmup=warmup)
    step = measure_full_step_throughput(repeats=max(2, repeats // 2), warmup=1)
    overlap = measure_overlap_efficiency(
        copy_num_bytes=max(copy_sizes_mib) * 1024**2,
        matmul_dim=matmul_dim,
        repeats=repeats,
        warmup=warmup,
    )
    return {
        "backend": get_backend().name,
        "device_name": get_backend().device_name,
        "pcie_reference_gbps": PCIe_GEN5_REFERENCE_GBPS,
        "h2d": h2d,
        "d2h": d2h,
        "single_layer": layer,
        "full_step": step,
        "overlap": overlap,
    }


def _print_summary(results: dict[str, Any]) -> None:
    print(f"Backend: {results['backend']} ({results['device_name']})")
    print(f"PCIe Gen5 reference floor: {results['pcie_reference_gbps']:.1f} GB/s")
    print()

    print("H2D bandwidth:")
    for item in results["h2d"]:
        print(f"  {item['num_bytes'] / 1024**2:6.0f} MiB -> {item['gbps']:.2f} GB/s")

    print("D2H bandwidth:")
    for item in results["d2h"]:
        print(f"  {item['num_bytes'] / 1024**2:6.0f} MiB -> {item['gbps']:.2f} GB/s")

    overlap = results["overlap"]
    print("Overlap:")
    print(
        "  copy {:.4f}s, compute {:.4f}s, overlap {:.4f}s, efficiency {:.2f}".format(
            overlap["copy_seconds"],
            overlap["compute_seconds"],
            overlap["overlap_seconds"],
            overlap["overlap_efficiency"],
        )
    )

    layer = results["single_layer"]
    print("Single layer:")
    print(f"  latency {layer['latency_ms']:.3f} ms, approx {layer['approx_tflops']:.4f} TFLOPS")

    step = results["full_step"]
    print("Full step:")
    print(
        "  avg step {:.4f}s, {:.1f} tokens/s, {:.4f} TFLOPS".format(
            step["avg_step_time_s"],
            step["tokens_per_sec"],
            step["tflops"],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MegaTrain on ROCm/CUDA")
    parser.add_argument("--copy-sizes-mib", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--matmul-dim", type=int, default=4096)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a readable summary")
    args = parser.parse_args()

    results = run_benchmark_suite(
        copy_sizes_mib=args.copy_sizes_mib,
        repeats=args.repeats,
        warmup=args.warmup,
        matmul_dim=args.matmul_dim,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_summary(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
