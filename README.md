# MegaTrain on AMD ROCm / MI355X

Full-parameter LLM training past the HBM fit boundary, validated on a single AMD MI355X accelerator.

[![Paper](https://img.shields.io/badge/Paper-arXiv%202604.05091-red)](https://arxiv.org/abs/2604.05091)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch ROCm](https://img.shields.io/badge/PyTorch-ROCm-orange)](https://pytorch.org/)

This fork carries a ROCm-focused MegaTrain validation branch for AMD MI355X. MegaTrain stores persistent training state in host memory and treats GPU HBM as a transient compute cache. That makes it useful in the regime where native PyTorch is fast but no longer fits, and where conventional CPU offload paths such as ZeRO-3 or FSDP pay high synchronization and memory overhead.

The headline result in this branch is a paper-aligned single-accelerator study on one MI355X using `MetaMathQA`, `max_seq_len=1024`, BF16, and steady-state throughput after dropping step 1 warmup.

## What This Branch Adds

- ROCm/HIP compatibility for MegaTrain native extension paths.
- A repo-owned ROCm bootstrap: [`scripts/install_rocm.sh`](scripts/install_rocm.sh).
- AMD-specific requirements in [`requirements-rocm.txt`](requirements-rocm.txt).
- ROCm configs for Qwen 7B, 14B, 32B, 72B, and related stress runs.
- Baseline harnesses for:
  - `DeepSpeed ZeRO-3 + CPU offload`
  - `PyTorch Native`
  - `FSDP + CPU offload` probe
- Reproducible benchmark scripts:
  - [`scripts/run_batch_sweep.py`](scripts/run_batch_sweep.py)
  - [`scripts/benchmark_offload_baseline.py`](scripts/benchmark_offload_baseline.py)
  - [`scripts/parse_benchmark_log.py`](scripts/parse_benchmark_log.py)
  - [`scripts/assemble_mi355x_paper_demo.py`](scripts/assemble_mi355x_paper_demo.py)
- A committed MI355X artifact bundle under [`artifacts/mi355x_paper_demo/`](artifacts/mi355x_paper_demo/).

## Why This Matters

Native PyTorch remains the right answer when the full training state fits in HBM. The interesting question is what happens after that boundary.

At 14B on one MI355X:

- Native PyTorch is strong up to `BS=12`, then OOMs at `BS=16`.
- ZeRO-3 CPU offload reaches `BS=20`, then OOMs at `BS=24`.
- MegaTrain reaches `BS=256` while keeping GPU memory at `137.1 GB`.

So the honest claim is not "MegaTrain is fastest everywhere." The supported claim is:

> On one MI355X, MegaTrain is the strongest validated path in this branch for full-parameter training once HBM residency becomes the limiter.

## MI355X Results

All numbers below are from [`artifacts/mi355x_paper_demo/`](artifacts/mi355x_paper_demo/). The primary sweep is `Qwen2.5-14B-Instruct` on `MetaMathQA`, `max_seq_len=1024`, BF16, one MI355X.

### Same Workload: 14B, Batch Size 16

| Method | Status | Batch | Steady TFLOPS | Peak GPU | Peak CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| MegaTrain | success | 16 | 109.4 | 16.8 GB | 190.5 GB |
| ZeRO-3 + CPU offload | success | 16 | 37.4 | 214.1 GB | 344.1 GB |
| PyTorch Native | OOM | 16 | - | 214.3 GB | 5.9 GB |

At the same `14B, BS=16` workload, MegaTrain is `2.93x` faster than ZeRO-3 CPU offload and uses `12.8x` less GPU memory.

### Validated 14B Batch Ceiling

| Method | Largest Validated Batch | Steady TFLOPS | Peak GPU | Peak CPU |
| --- | ---: | ---: | ---: | ---: |
| MegaTrain | 256 | 526.7 | 137.1 GB | 190.5 GB |
| ZeRO-3 + CPU offload | 20 | 42.2 | 260.9 GB | 344.1 GB |
| PyTorch Native | 12 | 134.3 | 223.3 GB | 5.8 GB |
| FSDP + CPU offload | 16 probe | 23.9 | 242.0 GB | 126.7 GB |

The maximum-batch comparison is not an apples-to-apples throughput claim. It shows each method's practical operating point on this node.

### Full 14B Batch Sweep

| Method | Batch | Status | Steady TFLOPS | Tokens/s | Peak GPU | Peak CPU |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| MegaTrain | 16 | success | 109.4 | 1234.3 | 16.8 GB | 190.5 GB |
| MegaTrain | 32 | success | 217.3 | 2451.9 | 24.1 GB | 190.5 GB |
| MegaTrain | 64 | success | 367.8 | 4150.2 | 39.7 GB | 190.5 GB |
| MegaTrain | 96 | success | 424.1 | 4785.4 | 55.5 GB | 190.5 GB |
| MegaTrain | 128 | success | 466.5 | 5264.2 | 71.5 GB | 190.5 GB |
| MegaTrain | 192 | success | 507.4 | 5725.8 | 104.3 GB | 190.5 GB |
| MegaTrain | 256 | success | 526.7 | 5943.4 | 137.1 GB | 190.5 GB |
| ZeRO-3 + CPU offload | 8 | success | 20.5 | 231.9 | 121.2 GB | 344.1 GB |
| ZeRO-3 + CPU offload | 12 | success | 28.8 | 325.0 | 167.8 GB | 344.0 GB |
| ZeRO-3 + CPU offload | 16 | success | 37.4 | 421.6 | 214.1 GB | 344.1 GB |
| ZeRO-3 + CPU offload | 20 | success | 42.2 | 475.8 | 260.9 GB | 344.1 GB |
| ZeRO-3 + CPU offload | 24 | OOM | - | - | - | - |
| ZeRO-3 + CPU offload | 32 | OOM | - | - | - | - |
| PyTorch Native | 4 | success | 111.1 | 1253.8 | 139.5 GB | 5.9 GB |
| PyTorch Native | 8 | success | 125.8 | 1419.6 | 176.8 GB | 5.9 GB |
| PyTorch Native | 12 | success | 134.3 | 1515.9 | 223.3 GB | 5.8 GB |
| PyTorch Native | 16 | OOM | - | - | 214.3 GB | 5.9 GB |

### MegaTrain Capability Ladder

| Model | Batch | Throughput | GPU Mem | CPU Mem | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-7B-Instruct | 96 | 406.5 TFLOPS | 46.9 GB | 102.4 GB | average |
| Qwen2.5-14B-Instruct | 256 | 507.2 TFLOPS | 137.1 GB | 190.6 GB | average |
| Qwen2.5-32B-Instruct | 300 | 524.6 TFLOPS | 221.0 GB | 393.2 GB | average |
| Qwen2.5-72B-Instruct | 200 | 556.1 TFLOPS | 215.3 GB | 855.8 GB | proof-of-life step 2 |

The 72B result is intentionally labeled proof-of-life. It is not a long-run convergence or production-throughput claim.

## Artifact Bundle

Start here:

- Summary: [`artifacts/mi355x_paper_demo/README.md`](artifacts/mi355x_paper_demo/README.md)
- Machine-readable summary: [`artifacts/mi355x_paper_demo/a_first_summary.json`](artifacts/mi355x_paper_demo/a_first_summary.json)
- Batch sweep CSV: [`artifacts/mi355x_paper_demo/14b_batch_sweep_points.csv`](artifacts/mi355x_paper_demo/14b_batch_sweep_points.csv)
- Capability ladder CSV: [`artifacts/mi355x_paper_demo/megatrain_capability_ladder.csv`](artifacts/mi355x_paper_demo/megatrain_capability_ladder.csv)
- Raw logs and per-run summaries: [`artifacts/mi355x_paper_demo/batch_sweep/`](artifacts/mi355x_paper_demo/batch_sweep/)

Generated charts include:

- `14b_same_workload_tflops.svg`
- `14b_validated_batch_ceiling.svg`
- `14b_batch_scaling.svg`
- `14b_gpu_memory_vs_batch.svg`
- `14b_cpu_memory_vs_batch.svg`
- `megatrain_throughput_vs_model_size.svg`

## Installation On ROCm

This branch expects a ROCm-enabled PyTorch build to already be installed in the environment. The install script verifies that first; it does not install a generic CPU/CUDA PyTorch wheel.

```bash
git clone https://github.com/chinmaydk99/MegaTrain.git
cd MegaTrain
git checkout ck-megatrain-rocm

bash scripts/install_rocm.sh
```

The ROCm bootstrap:

- verifies `torch.version.hip`
- installs MegaTrain editable
- installs [`requirements-rocm.txt`](requirements-rocm.txt)
- builds `causal-conv1d` from source with ROCm-safe flags
- checks imports for Flash Attention, Flash CE, DeepSpeed CPUAdam, flash-linear-attention, and `causal_conv1d`

See [`ROCM_ENVIRONMENT.md`](ROCM_ENVIRONMENT.md) for package notes and caveats.

## Quick Smoke Test

```bash
python examples/train.py \
  --config examples/configs/qwen_7b_rocm_paper.yaml \
  --num-steps 5 \
  --eval-num-samples 8
```

## Reproduce The 14B Sweep

MegaTrain:

```bash
python scripts/run_batch_sweep.py \
  --config examples/configs/qwen_14b_rocm_batch_sweep.yaml \
  --method megatrain \
  --batch-sizes 16 32 64 96 128 192 256 \
  --num-steps 5 \
  --output-dir artifacts/mi355x_paper_demo/batch_sweep \
  --stop-after-first-failure
```

ZeRO-3 CPU offload:

```bash
python scripts/run_batch_sweep.py \
  --config examples/configs/qwen_14b_rocm_batch_sweep.yaml \
  --method zero3_cpu_offload \
  --batch-sizes 8 12 16 20 24 32 \
  --num-steps 5 \
  --output-dir artifacts/mi355x_paper_demo/batch_sweep \
  --stop-after-first-failure
```

Native PyTorch:

```bash
python scripts/run_batch_sweep.py \
  --config examples/configs/qwen_14b_rocm_batch_sweep.yaml \
  --method native \
  --batch-sizes 4 8 12 16 \
  --num-steps 5 \
  --output-dir artifacts/mi355x_paper_demo/batch_sweep \
  --stop-after-first-failure
```

FSDP CPU-offload probe:

```bash
python scripts/run_batch_sweep.py \
  --config examples/configs/qwen_14b_rocm_batch_sweep.yaml \
  --method fsdp_cpu_offload \
  --batch-sizes 8 16 \
  --num-steps 5 \
  --output-dir artifacts/mi355x_paper_demo/batch_sweep \
  --stop-after-first-failure
```

Regenerate the artifact summary, CSVs, and SVGs:

```bash
python scripts/assemble_mi355x_paper_demo.py
```

## How MegaTrain Works

MegaTrain inverts the usual GPU-resident training model:

1. Host memory is the authoritative store for weights, gradients, and optimizer states.
2. GPU memory holds only the active layer, activation checkpoints, staging buffers, and transient compute state.
3. Weights stream H2D layer-by-layer.
4. Gradients stream D2H asynchronously.
5. CPU-side optimizer state avoids keeping Adam moments resident in HBM.
6. Double buffering overlaps parameter movement with compute.
7. Block-wise recomputation bounds activation memory.

The MI355X results are a good fit for this design because the accelerator has enough HBM to spend memory on large activation batches while host RAM carries persistent model state.

## Supported Model And Data Paths

MegaTrain uses HuggingFace model loading and local YAML configs. This branch has been exercised most heavily with Qwen2.5 models on MetaMathQA:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`

Relevant configs:

- [`examples/configs/qwen_7b_rocm_paper.yaml`](examples/configs/qwen_7b_rocm_paper.yaml)
- [`examples/configs/qwen_14b_rocm_batch_sweep.yaml`](examples/configs/qwen_14b_rocm_batch_sweep.yaml)
- [`examples/configs/qwen_14b_rocm_paper.yaml`](examples/configs/qwen_14b_rocm_paper.yaml)
- [`examples/configs/qwen_32b_mi355x.yaml`](examples/configs/qwen_32b_mi355x.yaml)
- [`examples/configs/qwen_72b_mi355x.yaml`](examples/configs/qwen_72b_mi355x.yaml)

The data path supports `MetaMathQA` through the repo dataset loader. The paper-aligned split policy used for the benchmark is:

- `train_ratio=0.7`
- `eval_ratio=0.3`
- `split_seed=42`
- `max_seq_len=1024`

## Caveats

- This branch is an AMD ROCm / MI355X validation branch, not the upstream CUDA-default README.
- The primary benchmark is single-accelerator. It does not claim to beat multi-GPU distributed training in that regime.
- Native PyTorch is still the right baseline when the model and batch fit comfortably in HBM.
- The short batch sweeps are throughput and capacity measurements, not final accuracy reproductions.
- The 72B result is proof-of-life unless extended by a longer sustained run.
- The ZeRO-3 baseline uses the DeepSpeed-managed optimizer config path required by this ROCm stack rather than a client-provided optimizer object.
- `torchvision` is intentionally not part of the text-only ROCm bootstrap path.

## Original Paper

This work builds on MegaTrain:

```bibtex
@misc{yuan2026megatrainprecisiontraining100b,
      title={MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU},
      author={Zhengqing Yuan and Hanchi Sun and Lichao Sun and Yanfang Ye},
      year={2026},
      eprint={2604.05091},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05091},
}
```

## Acknowledgements

This branch builds on the original MegaTrain implementation and the PyTorch ecosystem, including HuggingFace Transformers, DeepSpeed, Flash Attention, Flash Linear Attention, and LLaMA-Factory-style dataset conventions.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).
