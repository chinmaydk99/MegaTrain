#!/usr/bin/env python3
"""Run a batch-size sweep for one single-accelerator training method."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run a batch-size sweep for one method")
    parser.add_argument("--config", required=True, help="Sweep config path")
    parser.add_argument(
        "--method",
        required=True,
        choices=("megatrain", "zero3_cpu_offload", "native", "fsdp_cpu_offload"),
        help="Method to sweep",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True, help="Ordered batch sizes to try")
    parser.add_argument("--num-steps", type=int, default=5, help="Training steps per point")
    parser.add_argument(
        "--output-dir",
        default="artifacts/mi355x_paper_demo/batch_sweep",
        help="Directory for logs, summaries, and manifest",
    )
    parser.add_argument("--label", default=None, help="Optional filename prefix")
    parser.add_argument(
        "--stop-after-first-failure",
        action="store_true",
        help="Stop the sweep after the first non-zero exit code",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip points that already have a summary JSON",
    )
    return parser.parse_args()


def make_basename(label: str, method: str, batch_size: int) -> str:
    return f"{label}_{method}_bs{batch_size}"


def build_command(config: str, method: str, batch_size: int, num_steps: int, summary_path: Path) -> list[str]:
    if method == "megatrain":
        return [
            sys.executable,
            str(REPO_ROOT / "examples" / "train.py"),
            "--config",
            config,
            "--batch-size",
            str(batch_size),
            "--num-steps",
            str(num_steps),
        ]

    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_offload_baseline.py"),
        "--config",
        config,
        "--mode",
        method,
        "--batch-size",
        str(batch_size),
        "--num-steps",
        str(num_steps),
        "--summary-json",
        str(summary_path),
    ]


def write_failure_summary(summary_path: Path, method: str, batch_size: int, num_steps: int, log_path: Path):
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    tail_lines = log_text.splitlines()[-40:]
    error_line = next(
        (
            line.strip()
            for line in reversed(tail_lines)
            if "OutOfMemoryError" in line or "RuntimeError" in line or "Traceback" in line or "ERROR" in line
        ),
        "Run failed; inspect log tail for details.",
    )
    payload = {
        "success": False,
        "mode": method,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "error": error_line,
        "log_path": str(log_path),
        "steps": [],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def ensure_megatrain_summary(log_path: Path, summary_path: Path):
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "parse_benchmark_log.py"),
            str(log_path),
            "--output",
            str(summary_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def read_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def summary_success(summary: dict) -> bool:
    return summary.get("success", True)


def main():
    args = parse_args()

    config_path = str((REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    label = args.label or Path(config_path).stem

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    logs_dir = output_dir / "logs"
    summaries_dir = output_dir / "summaries"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": config_path,
        "method": args.method,
        "batch_sizes": args.batch_sizes,
        "num_steps": args.num_steps,
        "runs": [],
    }

    for batch_size in args.batch_sizes:
        base = make_basename(label, args.method, batch_size)
        log_path = logs_dir / f"{base}.log"
        summary_path = summaries_dir / f"{base}.json"

        if args.skip_existing and summary_path.exists():
            summary = read_summary(summary_path)
            manifest["runs"].append(
                {
                    "batch_size": batch_size,
                    "status": "skipped_existing",
                    "success": summary_success(summary),
                    "log_path": str(log_path),
                    "summary_path": str(summary_path),
                }
            )
            continue

        command = build_command(config_path, args.method, batch_size, args.num_steps, summary_path)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if args.method == "megatrain":
            if process.returncode == 0:
                ensure_megatrain_summary(log_path, summary_path)
            else:
                write_failure_summary(summary_path, args.method, batch_size, args.num_steps, log_path)
        elif process.returncode != 0 and not summary_path.exists():
            write_failure_summary(summary_path, args.method, batch_size, args.num_steps, log_path)

        summary = read_summary(summary_path)
        manifest["runs"].append(
            {
                "batch_size": batch_size,
                "status": "completed" if process.returncode == 0 else "failed",
                "success": summary_success(summary),
                "return_code": process.returncode,
                "log_path": str(log_path),
                "summary_path": str(summary_path),
                "error": summary.get("error"),
            }
        )

        if process.returncode != 0 and args.stop_after_first_failure:
            break

    manifest_path = output_dir / f"{label}_{args.method}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
