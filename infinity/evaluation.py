"""Evaluation helpers for paper-aligned MegaTrain runs."""

from __future__ import annotations

import random
import re
from typing import Sequence


def build_dataset_splits(
    num_samples: int,
    train_ratio: float = 1.0,
    eval_ratio: float = 0.0,
    seed: int = 42,
):
    """Build deterministic train/eval splits from a single dataset."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    for name, ratio in (("train_ratio", train_ratio), ("eval_ratio", eval_ratio)):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {ratio}")
    if train_ratio + eval_ratio > 1.0 + 1e-8:
        raise ValueError(
            f"train_ratio + eval_ratio must be <= 1, got {train_ratio + eval_ratio}"
        )

    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)

    train_count = int(num_samples * train_ratio)
    eval_count = int(num_samples * eval_ratio)

    train_indices = indices[:train_count]
    eval_indices = indices[train_count:train_count + eval_count]
    return train_indices, eval_indices


def split_messages_for_evaluation(messages: Sequence[dict]):
    """Split a chat transcript into prompt messages and reference answer."""
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if not assistant_indices:
        raise ValueError("messages must contain at least one assistant turn")

    target_idx = assistant_indices[-1]
    return list(messages[:target_idx]), messages[target_idx]["content"]


def _extract_last_boxed(text: str):
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    depth = 1
    chars = []
    i = start + len(marker)
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
        i += 1
    return None


def _extract_last_prefixed_line(text: str, pattern: str):
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_final_answer(text: str) -> str:
    """Extract the final answer span from a math-style response."""
    if not text:
        return ""

    boxed_pos = text.rfind("\\boxed{")
    hash_pos = text.rfind("####")
    answer_pos = text.rfind("The answer is")

    latest_pos = max(boxed_pos, hash_pos, answer_pos)
    if latest_pos == boxed_pos:
        answer = _extract_last_boxed(text)
        if answer:
            return answer
    if latest_pos == hash_pos:
        answer = _extract_last_prefixed_line(text, r"####\s*(.+)")
        if answer:
            return answer
    if latest_pos == answer_pos:
        answer = _extract_last_prefixed_line(text, r"The answer is:?\s*(.+)")
        if answer:
            return answer

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def normalize_math_answer(text: str) -> str:
    """Normalize simple formatting differences before exact-match comparison."""
    answer = extract_final_answer(text).strip()
    answer = answer.strip("$")
    answer = answer.rstrip(".")
    answer = re.sub(r"\s+", " ", answer)
    if answer.startswith("{") and answer.endswith("}") and len(answer) > 2:
        answer = answer[1:-1].strip()
    return answer.strip()


def exact_match_score(prediction: str, reference: str) -> bool:
    """Return True when normalized answers match exactly."""
    return normalize_math_answer(prediction) == normalize_math_answer(reference)
