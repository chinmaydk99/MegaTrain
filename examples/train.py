"""
MegaTrain Training Script — Universal Single-GPU Large Model Training

Supports any HuggingFace decoder-only model: Llama 2/3/4, Qwen 2/3/3.5,
Mistral, DeepSeek, Phi, Gemma, and more.

Usage:
    # Train with YAML config
    python examples/train.py --config examples/configs/qwen_7b.yaml

    # Train with config + overrides
    python examples/train.py --config examples/configs/llama3_8b.yaml --batch-size 64 --num-steps 500
"""

import argparse
import gc
import importlib.util
import logging
import os
import time
import psutil
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
    HAS_VLM_CLASS = True
except ImportError:
    HAS_VLM_CLASS = False

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import load_training_config, load_yaml_config, get_optimizer_type, get_num_workers, CPUMasterConfig
from infinity.evaluation import (
    build_dataset_splits,
    exact_match_score,
    split_messages_for_evaluation,
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HAS_ACCELERATE = importlib.util.find_spec("accelerate") is not None

# Try to import DeepSpeed CPUAdam
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    CPU_ADAM_AVAILABLE = True
    logger.info("DeepSpeed CPUAdam available (5-7x faster than PyTorch AdamW)!")
except ImportError:
    CPU_ADAM_AVAILABLE = False
    logger.info("DeepSpeed CPUAdam not available, using PyTorch AdamW")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train large language models with CPU-backed parameters")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset path from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override number of training steps from config")
    parser.add_argument("--eval-num-samples", type=int, default=None,
                        help="Override evaluation sample count (0 = full eval split)")
    parser.add_argument("--debug-numerics", action="store_true",
                        help="Enable expensive per-step gradient/parameter diagnostics")
    return parser.parse_args()


def _load_optional_processor(model_name: str, trust_remote_code: bool):
    """Load AutoProcessor when available, but allow text-only VLM training without vision deps."""
    try:
        return AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except ImportError as exc:
        logger.warning(
            "AutoProcessor unavailable for %s: %s. Continuing without processor; "
            "text-only datasets can still train, but image/video datasets still "
            "require the missing vision dependencies.",
            model_name,
            exc,
        )
        return None


def _count_nonfinite_params(params, check_grad=False):
    """Count parameters with non-finite values."""
    count = 0
    for param in params:
        tensor = param.grad if check_grad else param.data
        if tensor is not None and not torch.isfinite(tensor).all():
            count += 1
    return count


def _max_abs_grad(params):
    """Return the max absolute gradient across all parameters."""
    max_abs = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad_max = param.grad.detach().abs().max().item()
        if grad_max > max_abs:
            max_abs = grad_max
    return max_abs


def _iter_named_model_parameters(model):
    """Yield unique parameter names from the CPU master model."""
    seen = set()

    def _yield_named(prefix, module):
        if module is None:
            return
        for name, param in module.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            yield f"{prefix}.{name}", param

    yield from _yield_named("embedding", model.embedding)
    for idx, layer in enumerate(model.cpu_layers):
        yield from _yield_named(f"layers.{idx}", layer)
    yield from _yield_named("norm", model.norm)
    yield from _yield_named("lm_head", model.lm_head)


def _summarize_nonfinite_tensors(named_params, check_grad=False, limit=5):
    """Return a short summary of tensors containing NaN/Inf."""
    summaries = []
    for name, param in named_params:
        tensor = param.grad if check_grad else param.data
        if tensor is None:
            continue
        finite_mask = torch.isfinite(tensor)
        if finite_mask.all():
            continue
        nonfinite_count = (~finite_mask).sum().item()
        summaries.append(f"{name}({nonfinite_count})")
        if len(summaries) >= limit:
            break
    return summaries


def _summarize_largest_grads(named_params, limit=5):
    """Return the parameters with the largest finite/infinite grad magnitudes."""
    summaries = []
    for name, param in named_params:
        if param.grad is None:
            continue
        grad_max = param.grad.detach().abs().max().item()
        summaries.append((grad_max, name))
    summaries.sort(reverse=True, key=lambda item: item[0])
    return [f"{name}({value:.4e})" for value, name in summaries[:limit]]


def _select_train_and_eval_indices(dataset, config):
    """Return deterministic train/eval splits for a dataset."""
    train_indices, eval_indices = build_dataset_splits(
        len(dataset),
        train_ratio=config.train_ratio,
        eval_ratio=config.eval_ratio,
        seed=config.split_seed,
    )
    if config.train_ratio >= 1.0 and config.eval_ratio <= 0.0:
        train_indices = list(range(len(dataset)))
    return train_indices, eval_indices


def _evaluate_exact_match(eval_model, tokenizer, dataset, eval_indices, batch_size, max_new_tokens):
    """Run greedy generation and compute exact-match accuracy."""
    if not eval_indices:
        return None, 0

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    matches = 0
    total = 0
    device = next(eval_model.parameters()).device

    try:
        for start in range(0, len(eval_indices), batch_size):
            batch_indices = eval_indices[start:start + batch_size]
            prompts = []
            references = []

            for idx in batch_indices:
                messages, _ = dataset._get_messages(idx)
                prompt_messages, reference = split_messages_for_evaluation(messages)
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_text)
                references.append(reference)

            encoded = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = eval_model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = encoded["input_ids"].shape[1]
            predictions = tokenizer.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
            )

            for prediction, reference in zip(predictions, references):
                matches += int(exact_match_score(prediction, reference))
                total += 1
    finally:
        tokenizer.padding_side = old_padding_side

    return (matches / total) if total else None, total


def main():
    args = parse_args()

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        yaml_config = load_yaml_config(args.config)
        config = load_training_config(args.config)
        optimizer_type = get_optimizer_type(yaml_config)
        num_workers = get_num_workers(yaml_config)
    else:
        logger.info("Using default configuration")
        config = CPUMasterConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_seq_len=1024,
            batch_size=148,
            gradient_accumulation_steps=1,
            num_steps=100,
            learning_rate=1e-5,
            weight_decay=0.01,
            checkpoint_interval=4,
            dataset_path="dataset/Math/train",
            num_grad_slabs=12,
            device=0,
            dtype=torch.bfloat16,
            seed=42,
            log_interval=1,
        )
        optimizer_type = "deepspeed_adam"
        num_workers = 2

    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.eval_num_samples is not None:
        config.eval_num_samples = args.eval_num_samples

    logger.info("=" * 70)
    logger.info("MEGATRAIN: RAM-CENTRIC SINGLE-GPU TRAINING")
    logger.info("=" * 70)
    dataset_display = config.dataset_name or config.dataset_path
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Attention: {config.attn_implementation}")
    logger.info(f"Dataset: {dataset_display}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Training steps: {config.num_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if args.debug_numerics:
        logger.info("Numerics diagnostics: enabled")

    torch.manual_seed(config.seed)

    # Load tokenizer and detect VLM
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect VLM: check if model config has vision_config
    model_config = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    is_vlm = hasattr(model_config, 'vision_config') or (
        HAS_VLM_CLASS and type(model_config) in AutoModelForImageTextToText._model_mapping.keys()
    )

    processor = None
    if is_vlm:
        logger.info("VLM detected — loading with AutoModelForImageTextToText")
        processor = _load_optional_processor(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        if processor is None:
            logger.info("AutoProcessor unavailable; continuing in text-only VLM mode")
        load_class = AutoModelForImageTextToText if HAS_VLM_CLASS else AutoModelForCausalLM
    else:
        load_class = AutoModelForCausalLM

    # Load model with specified attention implementation
    logger.info(f"Loading model with attn_implementation='{config.attn_implementation}'...")
    model_load_kwargs = {
        "dtype": config.dtype,
        "trust_remote_code": config.trust_remote_code,
        "attn_implementation": config.attn_implementation,
    }
    if HAS_ACCELERATE:
        model_load_kwargs["device_map"] = "cpu"
    else:
        logger.info("Accelerate not available; loading model on CPU without device_map")

    hf_model = load_class.from_pretrained(
        config.model_name,
        **model_load_kwargs,
    )

    # Verify attention implementation was applied
    attn_impl = getattr(hf_model.config, '_attn_implementation', 'unknown')
    logger.info(f"Model loaded. Attention implementation: {attn_impl}")
    if is_vlm:
        logger.info(f"VLM mode: vision encoder will be CPU-offloaded")

    # Create CPU Master model
    model = CPUMasterModel(hf_model, config)
    del hf_model

    # Setup optimizer
    if optimizer_type == "deepspeed_adam" and CPU_ADAM_AVAILABLE:
        logger.info("Using DeepSpeed CPUAdam optimizer (SIMD-accelerated)")
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            adamw_mode=True
        )
    else:
        logger.info("Using PyTorch AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    # Setup dataset (universal: uses tokenizer's native chat template)
    dataset_kwargs = dict(
        system_prompt=config.system_prompt if config.system_prompt else None,
        train_on_prompt=config.train_on_prompt,
        processor=processor,
    )
    if config.dataset_name:
        dataset = ChatDataset(
            tokenizer, config.max_seq_len,
            dataset_name=config.dataset_name,
            dataset_dir=config.dataset_dir,
            **dataset_kwargs,
        )
    else:
        dataset = ChatDataset(
            tokenizer, config.max_seq_len,
            dataset_path=config.dataset_path,
            query_field=config.query_field,
            response_field=config.response_field,
            **dataset_kwargs,
        )
    train_indices, eval_indices = _select_train_and_eval_indices(dataset, config)
    train_dataset = Subset(dataset, train_indices)
    if len(train_indices) != len(dataset):
        logger.info(
            f"Using deterministic split: {len(train_indices)} train / {len(eval_indices)} eval "
            f"(seed={config.split_seed})"
        )
    elif eval_indices:
        logger.info(
            f"Evaluation enabled with {len(eval_indices)} held-out samples "
            f"(seed={config.split_seed})"
        )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    data_iter = iter(dataloader)

    # Training metrics
    total_loss = 0.0
    losses = []
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    throughputs = []
    gpu_mems = []
    cpu_mems = []
    forward_backward_times = []
    grad_accum_wait_times = []
    clip_times = []
    optimizer_step_times = []
    sync_params_times = []
    process = psutil.Process()

    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    # Training loop
    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        start_time = time.perf_counter()

        # Forward and backward
        fwd_kwargs = {}
        if is_vlm and "pixel_values" in batch:
            fwd_kwargs["pixel_values"] = batch["pixel_values"]
            # Pass any additional vision kwargs (e.g., image_grid_thw for Qwen-VL)
            for k in batch:
                if k not in ("input_ids", "attention_mask", "labels", "pixel_values", "prompt_length"):
                    fwd_kwargs[k] = batch[k]

        forward_backward_start = time.perf_counter()
        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"], **fwd_kwargs
        )
        forward_backward_time = time.perf_counter() - forward_backward_start
        grad_accum_wait_time = timing.get("grad_accum_wait", 0.0)

        # Optimizer step
        clip_time = 0.0
        optimizer_step_time = 0.0
        sync_params_time = 0.0
        if (step + 1) % config.gradient_accumulation_steps == 0:
            params = model.get_parameters()
            if args.debug_numerics:
                named_params = list(_iter_named_model_parameters(model))
                bad_grads_before_clip = _count_nonfinite_params(params, check_grad=True)
                max_grad_before_clip = _max_abs_grad(params)
                bad_grad_names_before_clip = _summarize_nonfinite_tensors(named_params, check_grad=True)
                largest_grad_names_before_clip = (
                    _summarize_largest_grads(named_params)
                    if max_grad_before_clip > 1.0e10
                    else []
                )
            else:
                named_params = []
                bad_grads_before_clip = 0
                max_grad_before_clip = 0.0
                bad_grad_names_before_clip = []
                largest_grad_names_before_clip = []

            clip_start = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            clip_time = time.perf_counter() - clip_start

            if args.debug_numerics:
                bad_grads_after_clip = _count_nonfinite_params(params, check_grad=True)
                bad_grad_names_after_clip = _summarize_nonfinite_tensors(named_params, check_grad=True)
            else:
                bad_grads_after_clip = 0
                bad_grad_names_after_clip = []

            optimizer_step_start = time.perf_counter()
            optimizer.step()
            optimizer_step_time = time.perf_counter() - optimizer_step_start

            sync_params_start = time.perf_counter()
            model._sync_params_to_gpu()
            sync_params_time = time.perf_counter() - sync_params_start

            if args.debug_numerics:
                bad_params_after_step = _count_nonfinite_params(params, check_grad=False)
                bad_param_names_after_step = _summarize_nonfinite_tensors(named_params, check_grad=False)
            else:
                bad_params_after_step = 0
                bad_param_names_after_step = []
            model.zero_grad()
            optimizer.zero_grad()
        else:
            grad_norm = None
            bad_grads_before_clip = 0
            bad_grads_after_clip = 0
            bad_params_after_step = 0
            max_grad_before_clip = 0.0
            bad_grad_names_before_clip = []
            bad_grad_names_after_clip = []
            largest_grad_names_before_clip = []
            bad_param_names_after_step = []

        step_time = time.perf_counter() - start_time

        # Calculate metrics
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        cpu_mem = process.memory_info().rss / 1024**3

        num_params = sum(p.numel() for p in model.get_parameters())
        flops = 6 * num_params * n_tokens
        gflops = (flops / 1e9) / step_time

        fwd_time = timing['forward']
        bwd_time = timing['backward']

        step_times.append(step_time)
        throughputs.append(gflops)
        gpu_mems.append(gpu_mem)
        cpu_mems.append(cpu_mem)
        forward_backward_times.append(forward_backward_time)
        grad_accum_wait_times.append(grad_accum_wait_time)
        clip_times.append(clip_time)
        optimizer_step_times.append(optimizer_step_time)
        sync_params_times.append(sync_params_time)

        total_loss += loss_val
        losses.append(loss_val)

        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            tps = n_tokens / step_time
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3

            logger.info(f"Step {step+1}/{config.num_steps} | Loss {loss_val:.4f} | Avg {avg_loss:.4f}")
            logger.info(f"  Time: {step_time:.2f}s | Tokens/s {tps:.1f} | GFLOPS {gflops:.1f}")
            logger.info(f"  FWD: {fwd_time:.2f}s | BWD: {bwd_time:.2f}s")
            logger.info(
                f"  Breakdown: fwd_bwd {forward_backward_time:.2f}s | "
                f"grad_accum_wait {grad_accum_wait_time:.2f}s | "
                f"clip {clip_time:.2f}s | optim {optimizer_step_time:.2f}s | sync {sync_params_time:.2f}s"
            )
            logger.info(f"  GPU: {gpu_mem:.2f}GB (alloc {mem_alloc:.2f}GB / reserved {mem_reserved:.2f}GB)")
            logger.info(f"  CPU: {cpu_mem:.2f}GB")
            if grad_norm is not None:
                if args.debug_numerics:
                    logger.info(
                        f"  Grad stats: norm {float(grad_norm):.4f} | max_abs {max_grad_before_clip:.4e} | "
                        f"nonfinite grads before/after clip {bad_grads_before_clip}/{bad_grads_after_clip} | "
                        f"nonfinite params after step {bad_params_after_step}"
                    )
                    if bad_grad_names_before_clip:
                        logger.info(f"  Nonfinite grad tensors before clip: {', '.join(bad_grad_names_before_clip)}")
                    if largest_grad_names_before_clip:
                        logger.info(f"  Largest grad tensors before clip: {', '.join(largest_grad_names_before_clip)}")
                    if bad_grad_names_after_clip:
                        logger.info(f"  Nonfinite grad tensors after clip: {', '.join(bad_grad_names_after_clip)}")
                    if bad_param_names_after_step:
                        logger.info(f"  Nonfinite param tensors after step: {', '.join(bad_param_names_after_step)}")
                else:
                    logger.info(f"  Grad norm: {float(grad_norm):.4f}")

    # Training summary
    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0.0

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Loss: {initial_loss:.4f} -> {final_loss:.4f} ({loss_reduction:.1f}% reduction)")
    logger.info(f"Peak GPU: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"Peak CPU: {max(cpu_mems):.2f} GB")
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  Avg Latency: {sum(step_times)/len(step_times):.3f}s per step")
    logger.info(f"  Avg Throughput: {sum(throughputs)/len(throughputs):.1f} GFLOPS")
    logger.info("Timing Breakdown:")
    logger.info(f"  Avg fwd_bwd: {sum(forward_backward_times)/len(forward_backward_times):.3f}s")
    logger.info(f"  Avg grad_accum_wait: {sum(grad_accum_wait_times)/len(grad_accum_wait_times):.3f}s")
    logger.info(f"  Avg clip_grad_norm: {sum(clip_times)/len(clip_times):.3f}s")
    logger.info(f"  Avg optimizer.step: {sum(optimizer_step_times)/len(optimizer_step_times):.3f}s")
    logger.info(f"  Avg _sync_params_to_gpu: {sum(sync_params_times)/len(sync_params_times):.3f}s")

    if config.eval_enabled:
        if not eval_indices:
            logger.warning("Evaluation requested, but eval_ratio produced an empty eval split")
        else:
            selected_eval_indices = eval_indices
            if config.eval_num_samples:
                selected_eval_indices = eval_indices[:config.eval_num_samples]

            logger.info("=" * 70)
            logger.info("Starting exact-match evaluation...")
            logger.info("=" * 70)
            logger.info(
                f"Eval samples: {len(selected_eval_indices)} / {len(eval_indices)} "
                f"| Max new tokens: {config.eval_max_new_tokens} | Batch size: {config.eval_batch_size}"
            )

            eval_load_kwargs = {
                "dtype": config.dtype,
                "trust_remote_code": config.trust_remote_code,
                "attn_implementation": config.attn_implementation,
            }
            eval_model = load_class.from_pretrained(
                config.model_name,
                **eval_load_kwargs,
            )
            model.materialize_hf_model(eval_model)

            model.cleanup()
            del optimizer
            del model
            gc.collect()
            torch.cuda.empty_cache()

            eval_model = eval_model.to(device=f"cuda:{config.device}", dtype=config.dtype).eval()
            exact_match, eval_total = _evaluate_exact_match(
                eval_model,
                tokenizer,
                dataset,
                selected_eval_indices,
                batch_size=config.eval_batch_size,
                max_new_tokens=config.eval_max_new_tokens,
            )
            logger.info(
                f"Exact-match accuracy: {exact_match * 100:.2f}% "
                f"({int(round(exact_match * eval_total))}/{eval_total})"
            )

            del eval_model
            torch.cuda.empty_cache()
            return

    # Cleanup
    model.cleanup()


if __name__ == "__main__":
    main()
