#!/usr/bin/env python3
"""Diagnose the first backward pass on a full Hugging Face Qwen model."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from infinity import ChatDataset, collate_fn
from infinity.config import load_training_config, load_yaml_config


def _build_dataset(config, yaml_config, tokenizer):
    dataset_kwargs = dict(
        system_prompt=config.system_prompt if config.system_prompt else None,
        train_on_prompt=config.train_on_prompt,
        processor=None,
    )
    if config.dataset_name:
        return ChatDataset(
            tokenizer,
            config.max_seq_len,
            dataset_name=config.dataset_name,
            dataset_dir=config.dataset_dir,
            **dataset_kwargs,
        )
    return ChatDataset(
        tokenizer,
        config.max_seq_len,
        dataset_path=config.dataset_path,
        query_field=config.query_field,
        response_field=config.response_field,
        **dataset_kwargs,
    )


def _nonfinite_grad_summaries(model, limit=20):
    summaries = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        finite_mask = torch.isfinite(param.grad)
        if finite_mask.all():
            continue
        summaries.append(f"{name}({(~finite_mask).sum().item()})")
        if len(summaries) >= limit:
            break
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    yaml_config = load_yaml_config(args.config)
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    torch.manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _build_dataset(config, yaml_config, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    batch = next(iter(dataloader))

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    ).to("cuda")
    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=batch["input_ids"].to("cuda"),
        attention_mask=batch["attention_mask"].to("cuda"),
        labels=batch["labels"].to("cuda"),
    )
    loss = outputs.loss
    print(f"loss={loss.item()}")
    loss.backward()

    nonfinite = _nonfinite_grad_summaries(model)
    print(f"nonfinite_grad_tensors={nonfinite}")

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm_grad = model.model.norm.weight.grad
        if norm_grad is not None:
            print(f"norm_weight_all_finite={torch.isfinite(norm_grad).all().item()}")
            print(f"norm_weight_nonfinite_count={(~torch.isfinite(norm_grad)).sum().item()}")


if __name__ == "__main__":
    main()
