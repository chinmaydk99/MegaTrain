#!/usr/bin/env python
"""Benchmark single-GPU baseline methods against MegaTrain."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import deepspeed
from infinity import ChatDataset, collate_fn
from infinity.config import get_num_workers, load_training_config, load_yaml_config
from infinity.evaluation import build_dataset_splits


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark single-GPU CPU-offload baselines")
    parser.add_argument("--config", required=True, help="Path to an existing MegaTrain YAML config")
    parser.add_argument(
        "--mode",
        choices=("zero3_cpu_offload", "native", "fsdp_cpu_offload"),
        default="zero3_cpu_offload",
        help="Baseline mode to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--num-steps", type=int, default=None, help="Override number of steps from config")
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to write a machine-readable run summary",
    )
    return parser.parse_args()


def _select_train_indices(dataset_length: int, train_ratio: float, eval_ratio: float, seed: int):
    train_indices, eval_indices = build_dataset_splits(
        dataset_length,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
    )
    if train_ratio >= 1.0 and eval_ratio <= 0.0:
        train_indices = list(range(dataset_length))
    return train_indices, eval_indices


def _build_dataset_and_loader(config, tokenizer, num_workers: int):
    dataset = ChatDataset(
        tokenizer,
        config.max_seq_len,
        dataset_name=config.dataset_name,
        dataset_dir=config.dataset_dir,
        system_prompt=config.system_prompt if config.system_prompt else None,
        train_on_prompt=config.train_on_prompt,
    )
    train_indices, eval_indices = _select_train_indices(
        len(dataset),
        train_ratio=config.train_ratio,
        eval_ratio=config.eval_ratio,
        seed=config.split_seed,
    )
    train_dataset = Subset(dataset, train_indices)
    if len(train_indices) != len(dataset):
        logger.info(
            f"Using deterministic split: {len(train_indices)} train / {len(eval_indices)} eval "
            f"(seed={config.split_seed})"
        )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataset, dataloader


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device):
    prepared = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            prepared[key] = value.to(device, non_blocking=True)
    return prepared


def _make_deepspeed_config(config) -> dict:
    return {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        "bf16": {"enabled": config.dtype == torch.bfloat16},
        "fp16": {"enabled": config.dtype == torch.float16},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.beta1, config.beta2],
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "contiguous_gradients": True,
            "overlap_comm": False,
        },
        "steps_per_print": config.log_interval,
        "wall_clock_breakdown": False,
    }


def _write_summary(summary_json: str | None, payload: dict):
    if not summary_json:
        return
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _set_single_rank_dist_env(config):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", str(config.device))


def _ensure_single_rank_distributed(config):
    if dist.is_initialized():
        return
    _set_single_rank_dist_env(config)
    dist.init_process_group(backend="nccl")


def _wrap_fsdp_cpu_offload(model, config):
    _ensure_single_rank_distributed(config)
    return FSDP(
        model.to(device=f"cuda:{config.device}", dtype=config.dtype),
        cpu_offload=CPUOffload(offload_params=True),
        device_id=config.device,
        use_orig_params=True,
    )


def _cleanup(*objects):
    for obj in objects:
        del obj
    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run():
    args = parse_args()

    yaml_config = load_yaml_config(args.config)
    config = load_training_config(args.config)
    num_workers = get_num_workers(yaml_config)

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_steps:
        config.num_steps = args.num_steps

    torch.manual_seed(config.seed)
    device = torch.device(f"cuda:{config.device}")
    torch.cuda.set_device(device)

    logger.info("=" * 70)
    logger.info("OFFLOAD BASELINE BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Attention: {config.attn_implementation}")
    logger.info(f"Dataset: {config.dataset_name or config.dataset_path}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Training steps: {config.num_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if hasattr(model_config, "vision_config"):
        raise ValueError("benchmark_offload_baseline.py only supports text-only CausalLM models")

    logger.info(f"Loading model with attn_implementation='{config.attn_implementation}'...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    _, dataloader = _build_dataset_and_loader(config, tokenizer, num_workers)
    data_iter = iter(dataloader)

    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()

    engine = None
    optimizer = None

    if args.mode == "zero3_cpu_offload":
        ds_config = _make_deepspeed_config(config)
        logger.info("Initializing DeepSpeed ZeRO-3 with CPU param+optimizer offload")
        _set_single_rank_dist_env(config)
        deepspeed.init_distributed(dist_backend="nccl")
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_config,
        )
        train_model = engine
    elif args.mode == "fsdp_cpu_offload":
        logger.info("Initializing FSDP with CPU offload")
        model = _wrap_fsdp_cpu_offload(model, config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        train_model = model
    else:
        logger.info("Initializing native single-GPU training baseline")
        model = model.to(device=device, dtype=config.dtype)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        train_model = model

    step_records = []
    losses = []
    step_times = []
    throughputs = []
    gpu_mems = []
    cpu_mems = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    clip_times = []

    logger.info("=" * 70)
    logger.info("Starting baseline training...")
    logger.info("=" * 70)

    try:
        for step in range(config.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = _prepare_batch(batch, device)
            n_tokens = int((batch["labels"] != -100).sum().item())

            torch.cuda.synchronize(device)
            step_start = time.perf_counter()

            torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            outputs = train_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            torch.cuda.synchronize(device)
            forward_time = time.perf_counter() - forward_start

            clip_time = 0.0
            optimizer_time = 0.0

            if args.mode == "zero3_cpu_offload":
                torch.cuda.synchronize(device)
                backward_start = time.perf_counter()
                engine.backward(loss)
                torch.cuda.synchronize(device)
                backward_time = time.perf_counter() - backward_start

                torch.cuda.synchronize(device)
                optimizer_start = time.perf_counter()
                engine.step()
                torch.cuda.synchronize(device)
                optimizer_time = time.perf_counter() - optimizer_start
                grad_norm = None
            else:
                torch.cuda.synchronize(device)
                backward_start = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize(device)
                backward_time = time.perf_counter() - backward_start

                clip_start = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                clip_time = time.perf_counter() - clip_start

                torch.cuda.synchronize(device)
                optimizer_start = time.perf_counter()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                optimizer_time = time.perf_counter() - optimizer_start

            step_time = time.perf_counter() - step_start
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            cpu_mem = process.memory_info().rss / 1024**3
            flops = 6 * num_params * n_tokens
            gflops = (flops / 1e9) / step_time
            tps = n_tokens / step_time
            loss_val = float(loss.detach().item())

            losses.append(loss_val)
            step_times.append(step_time)
            throughputs.append(gflops)
            gpu_mems.append(gpu_mem)
            cpu_mems.append(cpu_mem)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            optimizer_times.append(optimizer_time)
            clip_times.append(clip_time)

            step_records.append(
                {
                    "step": step + 1,
                    "loss": loss_val,
                    "step_time_s": step_time,
                    "tokens_per_s": tps,
                    "gflops": gflops,
                    "forward_s": forward_time,
                    "backward_s": backward_time,
                    "clip_s": clip_time,
                    "optimizer_s": optimizer_time,
                    "gpu_gb": gpu_mem,
                    "cpu_gb": cpu_mem,
                }
            )

            logger.info(f"Step {step+1}/{config.num_steps} | Loss {loss_val:.4f} | Avg {sum(losses)/len(losses):.4f}")
            logger.info(f"  Time: {step_time:.2f}s | Tokens/s {tps:.1f} | GFLOPS {gflops:.1f}")
            logger.info(f"  FWD: {forward_time:.2f}s | BWD: {backward_time:.2f}s")
            logger.info(
                f"  Breakdown: fwd_bwd {forward_time + backward_time:.2f}s | "
                f"clip {clip_time:.2f}s | optim {optimizer_time:.2f}s"
            )
            logger.info(
                f"  GPU: {gpu_mem:.2f}GB (alloc {torch.cuda.memory_allocated() / 1024**3:.2f}GB / "
                f"reserved {torch.cuda.memory_reserved() / 1024**3:.2f}GB)"
            )
            logger.info(f"  CPU: {cpu_mem:.2f}GB")
            if grad_norm is not None:
                logger.info(f"  Grad norm: {float(grad_norm):.4f}")

    except Exception as exc:
        failure_summary = {
            "success": False,
            "mode": args.mode,
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "num_steps": config.num_steps,
            "error": repr(exc),
            "steps": step_records,
        }
        _write_summary(args.summary_json, failure_summary)
        raise

    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100.0) if initial_loss > 0 else 0.0

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Loss: {initial_loss:.4f} -> {final_loss:.4f} ({loss_reduction:.1f}% reduction)")
    logger.info(f"Peak GPU: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"Peak CPU: {max(cpu_mems):.2f} GB")
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  Avg Latency: {sum(step_times) / len(step_times):.3f}s per step")
    logger.info(f"  Avg Throughput: {sum(throughputs) / len(throughputs):.1f} GFLOPS")
    logger.info("Timing Breakdown:")
    logger.info(f"  Avg forward: {sum(forward_times) / len(forward_times):.3f}s")
    logger.info(f"  Avg backward: {sum(backward_times) / len(backward_times):.3f}s")
    logger.info(f"  Avg clip_grad_norm: {sum(clip_times) / len(clip_times):.3f}s")
    logger.info(f"  Avg optimizer.step: {sum(optimizer_times) / len(optimizer_times):.3f}s")

    summary = {
        "success": True,
        "mode": args.mode,
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "num_steps": config.num_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_cpu_gb": max(cpu_mems) if cpu_mems else 0.0,
        "avg_latency_s": sum(step_times) / len(step_times),
        "avg_gflops": sum(throughputs) / len(throughputs),
        "avg_forward_s": sum(forward_times) / len(forward_times),
        "avg_backward_s": sum(backward_times) / len(backward_times),
        "avg_clip_s": sum(clip_times) / len(clip_times),
        "avg_optimizer_s": sum(optimizer_times) / len(optimizer_times),
        "steps": step_records,
    }
    _write_summary(args.summary_json, summary)

    _cleanup(train_model, model, optimizer, tokenizer)


if __name__ == "__main__":
    run()
