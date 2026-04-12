import math

import pytest

from scripts.benchmark_rocm import (
    expected_pcie_bandwidth_floor_gbps,
    measure_d2h_bandwidth,
    measure_full_step_throughput,
    measure_h2d_bandwidth,
    measure_overlap_efficiency,
    measure_single_layer_forward_latency,
)


@pytest.mark.gpu
@pytest.mark.benchmark
def test_h2d_bandwidth(skip_no_gpu, record_property):
    metrics = measure_h2d_bandwidth(num_bytes=256 * 1024**2, repeats=5, warmup=2)
    record_property("h2d_gbps", metrics["gbps"])
    assert metrics["gbps"] > expected_pcie_bandwidth_floor_gbps()


@pytest.mark.gpu
@pytest.mark.benchmark
def test_d2h_bandwidth(skip_no_gpu, record_property):
    metrics = measure_d2h_bandwidth(num_bytes=256 * 1024**2, repeats=5, warmup=2)
    record_property("d2h_gbps", metrics["gbps"])
    assert metrics["gbps"] > expected_pcie_bandwidth_floor_gbps()


@pytest.mark.gpu
@pytest.mark.benchmark
def test_double_buffer_overlap_efficiency(skip_no_gpu, record_property):
    metrics = measure_overlap_efficiency(
        copy_num_bytes=256 * 1024**2,
        matmul_dim=4096,
        repeats=5,
        warmup=2,
    )
    record_property("overlap_efficiency", metrics["overlap_efficiency"])
    assert metrics["overlap_efficiency"] > 0.8


@pytest.mark.gpu
@pytest.mark.benchmark
def test_single_layer_forward_latency(skip_no_gpu, tiny_hf_model, record_property):
    metrics = measure_single_layer_forward_latency(
        hf_model=tiny_hf_model,
        batch_size=2,
        seq_len=16,
        repeats=10,
        warmup=3,
    )
    record_property("single_layer_latency_ms", metrics["latency_ms"])
    record_property("single_layer_approx_tflops", metrics["approx_tflops"])
    assert math.isfinite(metrics["latency_ms"])
    assert metrics["latency_ms"] > 0.0
    assert math.isfinite(metrics["approx_tflops"])
    assert metrics["approx_tflops"] > 0.0


@pytest.mark.gpu
@pytest.mark.benchmark
def test_full_step_throughput(skip_no_gpu, cpu_master_model, dummy_batch, record_property):
    metrics = measure_full_step_throughput(
        model=cpu_master_model,
        batch=dummy_batch,
        repeats=3,
        warmup=1,
    )
    record_property("full_step_tokens_per_sec", metrics["tokens_per_sec"])
    record_property("full_step_tflops", metrics["tflops"])
    assert math.isfinite(metrics["avg_step_time_s"])
    assert metrics["avg_step_time_s"] > 0.0
    assert math.isfinite(metrics["tokens_per_sec"])
    assert metrics["tokens_per_sec"] > 0.0
    assert math.isfinite(metrics["tflops"])
    assert metrics["tflops"] > 0.0
    assert all(math.isfinite(loss) for loss in metrics["losses"])
