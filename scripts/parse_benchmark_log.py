#!/usr/bin/env python3
"""Parse MegaTrain and baseline benchmark logs into JSON summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"Step (\d+)/(\d+) \| Loss ([0-9.]+) \| Avg ([0-9.]+)")
TIME_RE = re.compile(r"Time: ([0-9.]+)s \| Tokens/s ([0-9.]+) \| GFLOPS ([0-9.]+)")
GPU_RE = re.compile(r"GPU: ([0-9.]+)GB")
CPU_RE = re.compile(r"CPU: ([0-9.]+)GB")
SUMMARY_LOSS_RE = re.compile(r"Loss: ([0-9.]+) -> ([0-9.]+) \(([0-9.]+)% reduction\)")
PEAK_GPU_RE = re.compile(r"Peak GPU: ([0-9.]+) GB")
PEAK_CPU_RE = re.compile(r"Peak CPU: ([0-9.]+) GB")
AVG_LATENCY_RE = re.compile(r"Avg Latency: ([0-9.]+)s per step")
AVG_THROUGHPUT_RE = re.compile(r"Avg Throughput: ([0-9.]+) GFLOPS")
EXACT_MATCH_RE = re.compile(r"Exact-match accuracy: ([0-9.]+)% \((\d+)/(\d+)\)")
HEADER_KEY_RE = re.compile(r"(Mode|Model|Attention|Dataset|Batch size|Training steps|Learning rate): (.+)")


def parse_log(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = {
        "log_path": str(path),
        "header": {},
        "steps": [],
        "summary": {},
    }

    pending_step = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        header_match = HEADER_KEY_RE.search(line)
        if header_match:
            key = header_match.group(1).lower().replace(" ", "_")
            result["header"].setdefault(key, header_match.group(2))

        step_match = STEP_RE.search(line)
        if step_match:
            pending_step = {
                "step": int(step_match.group(1)),
                "total_steps": int(step_match.group(2)),
                "loss": float(step_match.group(3)),
                "avg_loss": float(step_match.group(4)),
            }
            result["steps"].append(pending_step)
            continue

        if pending_step is not None:
            time_match = TIME_RE.search(line)
            if time_match:
                pending_step["step_time_s"] = float(time_match.group(1))
                pending_step["tokens_per_s"] = float(time_match.group(2))
                pending_step["gflops"] = float(time_match.group(3))
                continue

            gpu_match = GPU_RE.search(line)
            if gpu_match:
                pending_step["gpu_gb"] = float(gpu_match.group(1))
                continue

            cpu_match = CPU_RE.search(line)
            if cpu_match:
                pending_step["cpu_gb"] = float(cpu_match.group(1))
                pending_step = None
                continue

        if (match := SUMMARY_LOSS_RE.search(line)):
            result["summary"]["initial_loss"] = float(match.group(1))
            result["summary"]["final_loss"] = float(match.group(2))
            result["summary"]["loss_reduction_pct"] = float(match.group(3))
            continue

        if (match := PEAK_GPU_RE.search(line)):
            result["summary"]["peak_gpu_gb"] = float(match.group(1))
            continue

        if (match := PEAK_CPU_RE.search(line)):
            result["summary"]["peak_cpu_gb"] = float(match.group(1))
            continue

        if (match := AVG_LATENCY_RE.search(line)):
            result["summary"]["avg_latency_s"] = float(match.group(1))
            continue

        if (match := AVG_THROUGHPUT_RE.search(line)):
            result["summary"]["avg_gflops"] = float(match.group(1))
            continue

        if (match := EXACT_MATCH_RE.search(line)):
            result["summary"]["exact_match_pct"] = float(match.group(1))
            result["summary"]["exact_match_correct"] = int(match.group(2))
            result["summary"]["exact_match_total"] = int(match.group(3))
            continue

    if result["steps"]:
        steady_steps = result["steps"][1:] if len(result["steps"]) > 1 else result["steps"]
        result["summary"]["steady_state_avg_gflops"] = sum(s["gflops"] for s in steady_steps) / len(steady_steps)
        result["summary"]["steady_state_avg_tokens_per_s"] = (
            sum(s["tokens_per_s"] for s in steady_steps) / len(steady_steps)
        )
        result["summary"]["steady_state_avg_step_time_s"] = (
            sum(s["step_time_s"] for s in steady_steps) / len(steady_steps)
        )
        result["summary"]["steady_state_peak_gpu_gb"] = max(s.get("gpu_gb", 0.0) for s in steady_steps)
        result["summary"]["steady_state_peak_cpu_gb"] = max(s.get("cpu_gb", 0.0) for s in steady_steps)

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse MegaTrain benchmark logs into JSON")
    parser.add_argument("log_path", help="Path to a benchmark log")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    payload = parse_log(Path(args.log_path))
    text = json.dumps(payload, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
