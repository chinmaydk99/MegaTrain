#!/usr/bin/env python3
"""Assemble A-first MI355X benchmark artifacts from measured runs."""

from __future__ import annotations

import csv
import json
import math
import re
from html import escape
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "mi355x_paper_demo"
ANCHORS_DIR = ARTIFACTS_DIR / "anchors"
BASELINES_DIR = ARTIFACTS_DIR / "baselines"
SWEEPS_DIR = ARTIFACTS_DIR / "sweeps"
BATCH_SWEEP_DIR = ARTIFACTS_DIR / "batch_sweep"
SWEEP_LABEL = "qwen_14b_rocm_batch_sweep"


COLORS = {
    "MegaTrain": "#1f77b4",
    "ZeRO-3 + CPU offload": "#d62728",
    "PyTorch Native": "#2ca02c",
    "FSDP + CPU offload": "#ff7f0e",
    "GPU memory": "#9467bd",
    "CPU memory": "#8c564b",
}

METHOD_DISPLAY = {
    "megatrain": "MegaTrain",
    "zero3_cpu_offload": "ZeRO-3 + CPU offload",
    "native": "PyTorch Native",
    "fsdp_cpu_offload": "FSDP + CPU offload",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_none(values):
    return mean(values) if values else None


def steady_value(steps: list[dict], key: str) -> float | None:
    if not steps:
        return None
    steady_steps = steps[1:] if len(steps) > 1 else steps
    return mean_or_none([step[key] for step in steady_steps])


def peak_value(steps: list[dict], key: str) -> float | None:
    if not steps:
        return None
    return max(step[key] for step in steps)


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "OOM"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def parse_showcase_results() -> list[dict]:
    showcase_path = REPO_ROOT / "MI355X_SHOWCASE_RESULTS.md"
    results = []
    for line in showcase_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `Qwen2.5-"):
            continue

        model_match = re.search(r"`Qwen2\.5-(\d+)B-Instruct`", line)
        batch_match = re.search(r"`BS=(\d+)`", line)
        if not model_match or not batch_match:
            continue

        model_size_b = int(model_match.group(1))
        if "average throughput about" in line:
            throughput_match = re.search(r"average throughput about `([0-9.]+) TFLOPS`", line)
            throughput_note = "average"
        else:
            throughput_match = re.search(r"step 2: `([0-9.]+) TFLOPS`", line, flags=re.IGNORECASE)
            throughput_note = "proof-of-life step 2"
        if not throughput_match:
            continue

        gpu_match = (
            re.search(r"peak GPU `([0-9.]+) GB`", line)
            or re.search(r"trainer-reported GPU `([0-9.]+) GB`", line)
        )
        cpu_match = re.search(r"peak CPU `([0-9.]+) GB`", line) or re.search(r"CPU about `([0-9.]+) GB`", line)
        if not gpu_match or not cpu_match:
            continue

        results.append(
            {
                "model_name": f"Qwen2.5-{model_size_b}B-Instruct",
                "model_size_b": model_size_b,
                "batch_size": int(batch_match.group(1)),
                "throughput_tflops": float(throughput_match.group(1)),
                "throughput_note": throughput_note,
                "gpu_gb": float(gpu_match.group(1)),
                "cpu_gb": float(cpu_match.group(1)),
            }
        )

    return sorted(results, key=lambda item: item["model_size_b"])


def build_legacy_summary() -> dict:
    megatrain_14b_bs16 = read_json(SWEEPS_DIR / "qwen_14b_megatrain_bs16.json")
    megatrain_14b_bs96 = read_json(SWEEPS_DIR / "qwen_14b_megatrain_bs96.json")
    megatrain_14b_bs256 = read_json(ANCHORS_DIR / "qwen_14b_anchor.json")

    zero3_14b_bs16 = read_json(BASELINES_DIR / "qwen_14b_zero3_bs16.json")
    zero3_14b_bs24 = read_json(BASELINES_DIR / "qwen_14b_zero3_bs24.json")
    zero3_14b_bs32 = read_json(BASELINES_DIR / "qwen_14b_zero3_bs32.json")

    native_14b_bs8 = read_json(BASELINES_DIR / "qwen_14b_native_bs8.json")
    native_14b_bs16 = read_json(BASELINES_DIR / "qwen_14b_native_bs16.json")

    def baseline_success_row(name: str, batch_size: int, payload: dict) -> dict:
        return {
            "method": name,
            "status": "success",
            "batch_size": batch_size,
            "steady_tflops": steady_value(payload["steps"], "gflops") / 1000.0,
            "steady_tokens_per_s": steady_value(payload["steps"], "tokens_per_s"),
            "peak_gpu_gb": payload.get("peak_gpu_gb", peak_value(payload["steps"], "gpu_gb")),
            "peak_cpu_gb": payload.get("peak_cpu_gb", peak_value(payload["steps"], "cpu_gb")),
        }

    same_workload_14b = [
        {
            "method": "MegaTrain",
            "status": "success",
            "batch_size": 16,
            "steady_tflops": megatrain_14b_bs16["summary"]["steady_state_avg_gflops"] / 1000.0,
            "steady_tokens_per_s": megatrain_14b_bs16["summary"]["steady_state_avg_tokens_per_s"],
            "peak_gpu_gb": megatrain_14b_bs16["summary"]["steady_state_peak_gpu_gb"],
            "peak_cpu_gb": megatrain_14b_bs16["summary"]["steady_state_peak_cpu_gb"],
        },
        baseline_success_row("ZeRO-3 + CPU offload", 16, zero3_14b_bs16),
        {
            "method": "PyTorch Native",
            "status": "oom",
            "batch_size": 16,
            "steady_tflops": None,
            "steady_tokens_per_s": None,
            "peak_gpu_gb": peak_value(native_14b_bs16.get("steps", []), "gpu_gb"),
            "peak_cpu_gb": peak_value(native_14b_bs16.get("steps", []), "cpu_gb"),
            "error": native_14b_bs16.get("error"),
        },
    ]

    megatrain_vs_zero3_speedup = same_workload_14b[0]["steady_tflops"] / same_workload_14b[1]["steady_tflops"]

    validated_batch_ceiling_14b = [
        baseline_success_row("PyTorch Native", 8, native_14b_bs8),
        baseline_success_row("ZeRO-3 + CPU offload", 16, zero3_14b_bs16),
        {
            "method": "MegaTrain",
            "status": "success",
            "batch_size": 256,
            "steady_tflops": megatrain_14b_bs256["summary"]["steady_state_avg_gflops"] / 1000.0,
            "steady_tokens_per_s": megatrain_14b_bs256["summary"]["steady_state_avg_tokens_per_s"],
            "peak_gpu_gb": megatrain_14b_bs256["summary"]["steady_state_peak_gpu_gb"],
            "peak_cpu_gb": megatrain_14b_bs256["summary"]["steady_state_peak_cpu_gb"],
        },
    ]

    batch_scaling_14b = {
        "MegaTrain": {
            "points": [
                {"batch_size": 16, "steady_tflops": megatrain_14b_bs16["summary"]["steady_state_avg_gflops"] / 1000.0},
                {"batch_size": 96, "steady_tflops": megatrain_14b_bs96["summary"]["steady_state_avg_gflops"] / 1000.0},
                {"batch_size": 256, "steady_tflops": megatrain_14b_bs256["summary"]["steady_state_avg_gflops"] / 1000.0},
            ],
            "oom_batch_sizes": [],
        },
        "ZeRO-3 + CPU offload": {
            "points": [
                {"batch_size": 16, "steady_tflops": steady_value(zero3_14b_bs16["steps"], "gflops") / 1000.0},
            ],
            "oom_batch_sizes": [24, 32],
        },
        "PyTorch Native": {
            "points": [
                {"batch_size": 8, "steady_tflops": steady_value(native_14b_bs8["steps"], "gflops") / 1000.0},
            ],
            "oom_batch_sizes": [16],
        },
    }

    capability_rows = parse_showcase_results()

    return {
        "methodology": {
            "scope": "single MI355X, single-node, single-accelerator benchmark",
            "headline_comparison": "MegaTrain vs DeepSpeed ZeRO-3 + CPU offload",
            "native_role": "calibration baseline only where full GPU residency still fits",
            "dataset": "MetaMathQA",
            "max_seq_len": 1024,
            "split": "70/30 deterministic split (seed 42)",
        },
        "same_workload_14b": {
            "batch_size": 16,
            "rows": same_workload_14b,
            "megatrain_vs_zero3_speedup": megatrain_vs_zero3_speedup,
        },
        "validated_batch_ceiling_14b": {
            "rows": validated_batch_ceiling_14b,
            "zero3_first_oom_batch": 24,
            "native_first_oom_batch": 16,
            "megatrain_validated_batch": 256,
        },
        "batch_scaling_14b": batch_scaling_14b,
        "capability_ladder": capability_rows,
        "baseline_failures": {
            "zero3_14b_bs24": zero3_14b_bs24.get("error"),
            "zero3_14b_bs32": zero3_14b_bs32.get("error"),
            "native_14b_bs16": native_14b_bs16.get("error"),
        },
    }


def _summary_status(payload: dict) -> str:
    if payload.get("success", True):
        return "success"
    error = str(payload.get("error", "")).lower()
    if "outofmemory" in error or "oom" in error:
        return "oom"
    return "failed"


def _summary_to_row(method_display: str, batch_size: int, payload: dict) -> dict:
    if "summary" in payload:
        summary = payload["summary"]
        return {
            "method": method_display,
            "status": "success",
            "batch_size": batch_size,
            "steady_tflops": summary["steady_state_avg_gflops"] / 1000.0,
            "steady_tokens_per_s": summary["steady_state_avg_tokens_per_s"],
            "peak_gpu_gb": summary["steady_state_peak_gpu_gb"],
            "peak_cpu_gb": summary["steady_state_peak_cpu_gb"],
            "error": None,
        }

    steps = payload.get("steps", [])
    success = payload.get("success", False)
    return {
        "method": method_display,
        "status": _summary_status(payload),
        "batch_size": batch_size,
        "steady_tflops": steady_value(steps, "gflops") / 1000.0 if success and steps else None,
        "steady_tokens_per_s": steady_value(steps, "tokens_per_s") if success and steps else None,
        "peak_gpu_gb": payload.get("peak_gpu_gb", peak_value(steps, "gpu_gb")),
        "peak_cpu_gb": payload.get("peak_cpu_gb", peak_value(steps, "cpu_gb")),
        "error": payload.get("error"),
    }


def _load_batch_sweep_rows(internal_method: str) -> list[dict]:
    manifest_path = BATCH_SWEEP_DIR / f"{SWEEP_LABEL}_{internal_method}_manifest.json"
    if not manifest_path.exists():
        return []

    manifest = read_json(manifest_path)
    rows = []
    for run in manifest.get("runs", []):
        summary_name = Path(run["summary_path"]).name
        summary_path = BATCH_SWEEP_DIR / "summaries" / summary_name
        if not summary_path.exists():
            continue
        payload = read_json(summary_path)
        rows.append(
            _summary_to_row(
                METHOD_DISPLAY[internal_method],
                batch_size=run["batch_size"],
                payload=payload,
            )
        )

    return sorted(rows, key=lambda row: row["batch_size"])


def _success_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["status"] == "success"]


def _failed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["status"] != "success"]


def build_summary() -> dict:
    sweep_rows = {
        internal_method: _load_batch_sweep_rows(internal_method)
        for internal_method in ("megatrain", "zero3_cpu_offload", "native", "fsdp_cpu_offload")
    }

    if not sweep_rows["megatrain"] or not sweep_rows["zero3_cpu_offload"] or not sweep_rows["native"]:
        return build_legacy_summary()

    primary_methods = ("megatrain", "zero3_cpu_offload", "native")
    primary_success_rows = {method: _success_rows(sweep_rows[method]) for method in primary_methods}
    primary_failed_rows = {method: _failed_rows(sweep_rows[method]) for method in primary_methods}

    same_workload_batch = 16
    same_workload_rows = []
    for internal_method in primary_methods:
        matching = next((row for row in sweep_rows[internal_method] if row["batch_size"] == same_workload_batch), None)
        if matching:
            same_workload_rows.append(matching)

    megatrain_same = next(row for row in same_workload_rows if row["method"] == "MegaTrain")
    zero3_same = next(row for row in same_workload_rows if row["method"] == "ZeRO-3 + CPU offload")
    megatrain_vs_zero3_speedup = megatrain_same["steady_tflops"] / zero3_same["steady_tflops"]

    validated_batch_rows = []
    for internal_method in primary_methods:
        validated_batch_rows.append(primary_success_rows[internal_method][-1])

    batch_scaling = {}
    for internal_method in primary_methods:
        method_name = METHOD_DISPLAY[internal_method]
        batch_scaling[method_name] = {
            "points": [
                {"batch_size": row["batch_size"], "steady_tflops": row["steady_tflops"]}
                for row in primary_success_rows[internal_method]
            ],
            "memory_points": [
                {
                    "batch_size": row["batch_size"],
                    "peak_gpu_gb": row["peak_gpu_gb"],
                    "peak_cpu_gb": row["peak_cpu_gb"],
                }
                for row in primary_success_rows[internal_method]
            ],
            "oom_batch_sizes": [row["batch_size"] for row in primary_failed_rows[internal_method]],
        }

    fsdp_probe = {
        "rows": sweep_rows["fsdp_cpu_offload"],
        "success_rows": _success_rows(sweep_rows["fsdp_cpu_offload"]),
        "failed_rows": _failed_rows(sweep_rows["fsdp_cpu_offload"]),
    }

    capability_rows = parse_showcase_results()

    return {
        "methodology": {
            "scope": "single MI355X, single-node, single-accelerator benchmark",
            "headline_comparison": "MegaTrain vs DeepSpeed ZeRO-3 + CPU offload",
            "native_role": "calibration baseline only where full GPU residency still fits",
            "dataset": "MetaMathQA",
            "max_seq_len": 1024,
            "split": "70/30 deterministic split (seed 42)",
            "throughput_definition": "steady-state after dropping step 1 warmup",
            "sweep_config": "examples/configs/qwen_14b_rocm_batch_sweep.yaml",
        },
        "same_workload_14b": {
            "batch_size": same_workload_batch,
            "rows": same_workload_rows,
            "megatrain_vs_zero3_speedup": megatrain_vs_zero3_speedup,
        },
        "validated_batch_ceiling_14b": {
            "rows": validated_batch_rows,
            "zero3_first_oom_batch": primary_failed_rows["zero3_cpu_offload"][0]["batch_size"] if primary_failed_rows["zero3_cpu_offload"] else None,
            "native_first_oom_batch": primary_failed_rows["native"][0]["batch_size"] if primary_failed_rows["native"] else None,
            "megatrain_validated_batch": primary_success_rows["megatrain"][-1]["batch_size"],
        },
        "batch_scaling_14b": batch_scaling,
        "sweep_rows_14b": {
            METHOD_DISPLAY[internal_method]: sweep_rows[internal_method]
            for internal_method in primary_methods
        },
        "fsdp_probe_14b": fsdp_probe,
        "capability_ladder": capability_rows,
        "baseline_failures": {
            internal_method: [
                {"batch_size": row["batch_size"], "error": row["error"]}
                for row in primary_failed_rows[internal_method]
            ]
            for internal_method in primary_methods
        },
    }


def svg_text(x, y, text, size=14, weight="normal", anchor="middle", fill="#222222"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(text)}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke="#444444", width=1.0, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr} />'


def svg_rect(x, y, width, height, fill, stroke="none", stroke_width=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def svg_circle(cx, cy, r, fill, stroke="none", stroke_width=1.0):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'


def svg_polyline(points, stroke, width=2.0, fill="none"):
    points_attr = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{points_attr}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'


def svg_cross(cx, cy, size, stroke, width=2.0):
    half = size / 2.0
    return "\n".join(
        [
            svg_line(cx - half, cy - half, cx + half, cy + half, stroke=stroke, width=width),
            svg_line(cx - half, cy + half, cx + half, cy - half, stroke=stroke, width=width),
        ]
    )


def build_svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'{svg_rect(0, 0, width, height, fill="#ffffff")}\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def render_bar_chart(path: Path, title: str, ylabel: str, rows: list[dict], note: str | None = None):
    width = 940
    height = 560
    left = 90
    right = 40
    top = 80
    bottom = 125
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = [row["value"] for row in rows if row["value"] is not None]
    y_max = nice_max(max(values) * 1.15 if values else 1.0)
    tick_count = 5
    bar_slot = plot_width / max(len(rows), 1)
    bar_width = bar_slot * 0.55

    body = [
        svg_text(width / 2, 34, title, size=24, weight="bold"),
        svg_text(18, top + plot_height / 2, ylabel, size=14, anchor="middle"),
        svg_line(left, top, left, top + plot_height, stroke="#333333", width=1.5),
        svg_line(left, top + plot_height, left + plot_width, top + plot_height, stroke="#333333", width=1.5),
    ]

    for idx in range(tick_count + 1):
        tick_value = (y_max / tick_count) * idx
        y = top + plot_height - (tick_value / y_max) * plot_height
        body.append(svg_line(left - 5, y, left + plot_width, y, stroke="#dddddd", width=1.0))
        body.append(svg_text(left - 10, y + 4, fmt(tick_value), size=12, anchor="end"))

    for idx, row in enumerate(rows):
        x_center = left + bar_slot * idx + bar_slot / 2
        x = x_center - bar_width / 2
        if row["value"] is not None:
            bar_height = (row["value"] / y_max) * plot_height
            y = top + plot_height - bar_height
            body.append(svg_rect(x, y, bar_width, bar_height, fill=row["color"]))
            body.append(svg_text(x_center, y - 10, fmt(row["value"]), size=13))
        else:
            body.append(svg_cross(x_center, top + plot_height - 24, 18, stroke="#d62728", width=2.0))
            body.append(svg_text(x_center, top + plot_height - 36, "OOM", size=12, fill="#d62728"))

        body.append(svg_text(x_center, top + plot_height + 24, row["label"], size=13))
        if row.get("annotation"):
            body.append(svg_text(x_center, top + plot_height + 44, row["annotation"], size=11, fill="#555555"))

    if note:
        body.append(svg_text(width / 2, height - 18, note, size=12, fill="#555555"))

    path.write_text(build_svg(width, height, body), encoding="utf-8")


def render_line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
    oom_markers: list[dict],
    note: str | None = None,
):
    width = 980
    height = 600
    left = 95
    right = 50
    top = 80
    bottom = 140
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [point["x"] for item in series for point in item["points"]] + [item["x"] for item in oom_markers]
    y_values = [point["y"] for item in series for point in item["points"]]
    x_min = min(x_values)
    x_max = max(x_values)
    y_max = nice_max(max(y_values) * 1.15 if y_values else 1.0)
    y_min = 0.0

    def x_to_px(value):
        if x_max == x_min:
            return left + plot_width / 2
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value):
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    body = [
        svg_text(width / 2, 34, title, size=24, weight="bold"),
        svg_text(width / 2, height - 28, x_label, size=14),
        svg_text(24, top + plot_height / 2, y_label, size=14, anchor="middle"),
        svg_line(left, top, left, top + plot_height, stroke="#333333", width=1.5),
        svg_line(left, top + plot_height, left + plot_width, top + plot_height, stroke="#333333", width=1.5),
    ]

    tick_count = 5
    for idx in range(tick_count + 1):
        tick_value = (y_max / tick_count) * idx
        y = y_to_px(tick_value)
        body.append(svg_line(left - 5, y, left + plot_width, y, stroke="#e3e3e3", width=1.0))
        body.append(svg_text(left - 10, y + 4, fmt(tick_value), size=12, anchor="end"))

    for value in sorted(set(x_values)):
        x = x_to_px(value)
        body.append(svg_line(x, top + plot_height, x, top + plot_height + 6, stroke="#333333", width=1.0))
        body.append(svg_text(x, top + plot_height + 24, str(value), size=12))

    legend_x = left + 10
    legend_y = 58
    for idx, item in enumerate(series):
        ly = legend_y + idx * 20
        body.append(svg_line(legend_x, ly, legend_x + 18, ly, stroke=item["color"], width=3.0))
        body.append(svg_text(legend_x + 26, ly + 4, item["name"], size=12, anchor="start"))

    for item in series:
        points = [(x_to_px(point["x"]), y_to_px(point["y"])) for point in item["points"]]
        if len(points) > 1:
            body.append(svg_polyline(points, stroke=item["color"], width=2.5))
        for point, (px, py) in zip(item["points"], points):
            body.append(svg_circle(px, py, 4.5, fill=item["color"]))
            body.append(svg_text(px, py - 10, fmt(point["y"]), size=11))

    for marker in oom_markers:
        px = x_to_px(marker["x"])
        py = y_to_px(y_max * 0.05)
        body.append(svg_cross(px, py, 16, stroke=marker["color"], width=2.0))
        body.append(svg_text(px, py - 12, "OOM", size=11, fill=marker["color"]))

    if note:
        body.append(svg_text(width / 2, height - 10, note, size=12, fill="#555555"))

    path.write_text(build_svg(width, height, body), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_charts(summary: dict):
    same_rows = []
    for row in summary["same_workload_14b"]["rows"]:
        same_rows.append(
            {
                "label": row["method"],
                "value": row["steady_tflops"],
                "annotation": "BS=16",
                "color": COLORS[row["method"]],
            }
        )
    render_bar_chart(
        ARTIFACTS_DIR / "14b_same_workload_tflops.svg",
        title="14B Same-Workload Throughput on One MI355X",
        ylabel="Steady-State TFLOPS",
        rows=same_rows,
        note="PyTorch Native is shown only as a calibration baseline and OOMs at 14B, BS=16.",
    )

    ceiling_rows = []
    for row in summary["validated_batch_ceiling_14b"]["rows"]:
        ceiling_rows.append(
            {
                "label": row["method"],
                "value": row["batch_size"],
                "annotation": f"{fmt(row['steady_tflops'])} TFLOPS",
                "color": COLORS[row["method"]],
            }
        )
    render_bar_chart(
        ARTIFACTS_DIR / "14b_validated_batch_ceiling.svg",
        title="14B Validated Single-GPU Batch Size",
        ylabel="Validated Batch Size",
        rows=ceiling_rows,
        note="Validated ceiling means the largest batch confirmed in this pass, not a proof of the global optimum.",
    )

    series = []
    oom_markers = []
    for method, payload in summary["batch_scaling_14b"].items():
        series.append(
            {
                "name": method,
                "color": COLORS[method],
                "points": [
                    {"x": point["batch_size"], "y": point["steady_tflops"]}
                    for point in payload["points"]
                ],
            }
        )
        for batch_size in payload["oom_batch_sizes"]:
            oom_markers.append({"x": batch_size, "color": COLORS[method]})

    render_line_chart(
        ARTIFACTS_DIR / "14b_batch_scaling.svg",
        title="14B Batch-Size Scaling on One MI355X",
        x_label="Batch Size",
        y_label="Steady-State TFLOPS",
        series=series,
        oom_markers=oom_markers,
        note="MegaTrain scales to much larger validated batches before the conventional baselines hit OOM.",
    )

    gpu_memory_series = []
    cpu_memory_series = []
    for method, payload in summary["batch_scaling_14b"].items():
        gpu_memory_series.append(
            {
                "name": method,
                "color": COLORS[method],
                "points": [
                    {"x": point["batch_size"], "y": point["peak_gpu_gb"]}
                    for point in payload["memory_points"]
                ],
            }
        )
        cpu_memory_series.append(
            {
                "name": method,
                "color": COLORS[method],
                "points": [
                    {"x": point["batch_size"], "y": point["peak_cpu_gb"]}
                    for point in payload["memory_points"]
                ],
            }
        )

    render_line_chart(
        ARTIFACTS_DIR / "14b_gpu_memory_vs_batch.svg",
        title="14B GPU Memory vs Batch Size on One MI355X",
        x_label="Batch Size",
        y_label="GPU Memory (GB)",
        series=gpu_memory_series,
        oom_markers=oom_markers,
        note="MegaTrain keeps GPU memory dramatically lower while scaling to larger validated batches.",
    )

    render_line_chart(
        ARTIFACTS_DIR / "14b_cpu_memory_vs_batch.svg",
        title="14B Host Memory vs Batch Size on One MI355X",
        x_label="Batch Size",
        y_label="CPU Memory (GB)",
        series=cpu_memory_series,
        oom_markers=oom_markers,
        note="Conventional offload baselines consume substantially more host memory even before they hit OOM.",
    )

    capability = summary["capability_ladder"]
    capability_series = [
        {
            "name": "MegaTrain",
            "color": COLORS["MegaTrain"],
            "points": [{"x": row["model_size_b"], "y": row["throughput_tflops"]} for row in capability],
        }
    ]
    render_line_chart(
        ARTIFACTS_DIR / "megatrain_throughput_vs_model_size.svg",
        title="MegaTrain Throughput vs Model Size on One MI355X",
        x_label="Model Size (B parameters)",
        y_label="TFLOPS",
        series=capability_series,
        oom_markers=[],
        note="The 72B point is a proof-of-life step-2 throughput, while 7B/14B/32B are average run summaries.",
    )

    gpu_series = [
        {
            "name": "GPU memory",
            "color": COLORS["GPU memory"],
            "points": [{"x": row["model_size_b"], "y": row["gpu_gb"]} for row in capability],
        }
    ]
    render_line_chart(
        ARTIFACTS_DIR / "megatrain_gpu_memory_vs_model_size.svg",
        title="MegaTrain GPU Memory vs Model Size",
        x_label="Model Size (B parameters)",
        y_label="GPU Memory (GB)",
        series=gpu_series,
        oom_markers=[],
        note="MegaTrain keeps GPU residency bounded enough to sustain single-accelerator training into the 72B regime.",
    )

    cpu_series = [
        {
            "name": "CPU memory",
            "color": COLORS["CPU memory"],
            "points": [{"x": row["model_size_b"], "y": row["cpu_gb"]} for row in capability],
        }
    ]
    render_line_chart(
        ARTIFACTS_DIR / "megatrain_cpu_memory_vs_model_size.svg",
        title="MegaTrain Host Memory vs Model Size",
        x_label="Model Size (B parameters)",
        y_label="CPU Memory (GB)",
        series=cpu_series,
        oom_markers=[],
        note="Host memory becomes the scaling boundary once persistent state is moved out of HBM.",
    )


def write_markdown_summary(summary: dict):
    same_rows = summary["same_workload_14b"]["rows"]
    ceiling_rows = summary["validated_batch_ceiling_14b"]["rows"]
    capability_rows = summary["capability_ladder"]
    sweep_rows = summary["sweep_rows_14b"]
    fsdp_rows = summary["fsdp_probe_14b"]["rows"]

    native_ceiling = next(row for row in ceiling_rows if row["method"] == "PyTorch Native")
    zero3_ceiling = next(row for row in ceiling_rows if row["method"] == "ZeRO-3 + CPU offload")
    megatrain_ceiling = next(row for row in ceiling_rows if row["method"] == "MegaTrain")

    lines = [
        "# MI355X A-First Benchmark Summary",
        "",
        "This artifact keeps the story locked to the paper-faithful single-accelerator question: one MI355X, one node, `MetaMathQA`, `max_seq_len=1024`, and a normalized `14B` batch sweep built from `examples/configs/qwen_14b_rocm_batch_sweep.yaml`.",
        "",
        "All new sweep charts use the same throughput definition: steady-state throughput after dropping step 1 warmup.",
        "",
        "## Headline Findings",
        "",
        f"- At `14B, BS=16`, `MegaTrain` reaches `{same_rows[0]['steady_tflops']:.1f} TFLOPS` versus `{same_rows[1]['steady_tflops']:.1f} TFLOPS` for `ZeRO-3 + CPU offload`, a `{summary['same_workload_14b']['megatrain_vs_zero3_speedup']:.2f}x` speedup on the same single-GPU workload.",
        f"- The validated `14B` batch ceilings in this pass are `BS={megatrain_ceiling['batch_size']}` for `MegaTrain`, `BS={zero3_ceiling['batch_size']}` for `ZeRO-3 + CPU offload`, and `BS={native_ceiling['batch_size']}` for `PyTorch Native`.",
        "- The AMD-specific mechanism story is now explicit: `MegaTrain` keeps gaining throughput as batch size rises, while the conventional baselines hit fit or throughput walls much earlier.",
        "",
        "## 14B Same-Workload View",
        "",
        "| Method | Status | Batch | Steady TFLOPS | Peak GPU (GB) | Peak CPU (GB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in same_rows:
        lines.append(
            f"| `{row['method']}` | `{row['status']}` | `{row['batch_size']}` | "
            f"`{fmt(row['steady_tflops'])}` | `{fmt(row['peak_gpu_gb'])}` | `{fmt(row['peak_cpu_gb'])}` |"
        )

    lines.extend(
        [
            "",
            "## 14B Validated Batch Ceiling",
            "",
            "| Method | Largest Validated Batch | Steady TFLOPS At That Batch | Peak GPU (GB) | Peak CPU (GB) |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in ceiling_rows:
        lines.append(
            f"| `{row['method']}` | `{row['batch_size']}` | `{fmt(row['steady_tflops'])}` | "
            f"`{fmt(row['peak_gpu_gb'])}` | `{fmt(row['peak_cpu_gb'])}` |"
        )

    lines.extend(
        [
            "",
            "## 14B Batch Sweep",
            "",
            "| Method | Batch | Status | Steady TFLOPS | Peak GPU (GB) | Peak CPU (GB) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for method_name, rows in sweep_rows.items():
        for row in rows:
            lines.append(
                f"| `{method_name}` | `{row['batch_size']}` | `{row['status']}` | "
                f"`{fmt(row['steady_tflops'])}` | `{fmt(row['peak_gpu_gb'])}` | `{fmt(row['peak_cpu_gb'])}` |"
            )

    lines.extend(
        [
            "",
            "## MegaTrain Capability Ladder",
            "",
            "| Model | Batch | Throughput | GPU Mem (GB) | CPU Mem (GB) | Note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in capability_rows:
        lines.append(
            f"| `{row['model_name']}` | `{row['batch_size']}` | `{fmt(row['throughput_tflops'])}` | "
            f"`{fmt(row['gpu_gb'])}` | `{fmt(row['cpu_gb'])}` | `{row['throughput_note']}` |"
        )

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![14B same-workload throughput](14b_same_workload_tflops.svg)",
            "",
            "![14B validated batch ceiling](14b_validated_batch_ceiling.svg)",
            "",
            "![14B batch scaling](14b_batch_scaling.svg)",
            "",
            "![14B GPU memory vs batch](14b_gpu_memory_vs_batch.svg)",
            "",
            "![14B CPU memory vs batch](14b_cpu_memory_vs_batch.svg)",
            "",
            "![MegaTrain throughput vs model size](megatrain_throughput_vs_model_size.svg)",
            "",
            "![MegaTrain GPU memory vs model size](megatrain_gpu_memory_vs_model_size.svg)",
            "",
            "![MegaTrain CPU memory vs model size](megatrain_cpu_memory_vs_model_size.svg)",
        ]
    )

    if fsdp_rows:
        lines.extend(
            [
                "",
                "## Appendix: FSDP Probe",
                "",
                "| Batch | Status | Steady TFLOPS | Peak GPU (GB) | Peak CPU (GB) |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in fsdp_rows:
            lines.append(
                f"| `{row['batch_size']}` | `{row['status']}` | `{fmt(row['steady_tflops'])}` | "
                f"`{fmt(row['peak_gpu_gb'])}` | `{fmt(row['peak_cpu_gb'])}` |"
            )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Validated install path:",
            "",
            "```bash",
            "bash scripts/install_rocm.sh",
            "```",
            "",
            "Primary batch-scaling sweeps:",
            "",
            "```bash",
            "python scripts/run_batch_sweep.py --config examples/configs/qwen_14b_rocm_batch_sweep.yaml --method megatrain --batch-sizes 16 32 64 96 128 192 256 --num-steps 5 --output-dir artifacts/mi355x_paper_demo/batch_sweep --stop-after-first-failure",
            "python scripts/run_batch_sweep.py --config examples/configs/qwen_14b_rocm_batch_sweep.yaml --method zero3_cpu_offload --batch-sizes 8 12 16 20 24 32 --num-steps 5 --output-dir artifacts/mi355x_paper_demo/batch_sweep --stop-after-first-failure",
            "python scripts/run_batch_sweep.py --config examples/configs/qwen_14b_rocm_batch_sweep.yaml --method native --batch-sizes 4 8 12 16 --num-steps 5 --output-dir artifacts/mi355x_paper_demo/batch_sweep --stop-after-first-failure",
            "```",
            "",
            "Cheap FSDP probe:",
            "",
            "```bash",
            "python scripts/run_batch_sweep.py --config examples/configs/qwen_14b_rocm_batch_sweep.yaml --method fsdp_cpu_offload --batch-sizes 8 16 --num-steps 5 --output-dir artifacts/mi355x_paper_demo/batch_sweep --stop-after-first-failure",
            "```",
            "",
            "Regenerate the repo-owned artifact bundle:",
            "",
            "```bash",
            "python scripts/assemble_mi355x_paper_demo.py",
            "```",
            "",
            "## Caveats",
            "",
            "- This is intentionally a single-accelerator benchmark. It does not claim to beat multi-GPU distributed methods in their natural regime.",
            "- `PyTorch Native` is a calibration baseline only; it exits the comparison once the model no longer fits.",
            "- On this ROCm stack, the ZeRO-3 benchmark uses the supported DeepSpeed-managed `AdamW` config path for CPU offload rather than a client-provided optimizer object.",
            "- The capability ladder remains separate from the sweep charts so average showcase numbers are not conflated with steady-state sweep results.",
            "- The `72B` throughput point is still a proof-of-life step result, not a long-run average.",
        ]
    )

    (ARTIFACTS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_artifacts(summary: dict):
    same_rows = [
        {
            "method": row["method"],
            "status": row["status"],
            "batch_size": row["batch_size"],
            "steady_tflops": row["steady_tflops"],
            "steady_tokens_per_s": row["steady_tokens_per_s"],
            "peak_gpu_gb": row["peak_gpu_gb"],
            "peak_cpu_gb": row["peak_cpu_gb"],
        }
        for row in summary["same_workload_14b"]["rows"]
    ]
    write_csv(ARTIFACTS_DIR / "14b_same_workload.csv", same_rows)

    ceiling_rows = [
        {
            "method": row["method"],
            "validated_batch_size": row["batch_size"],
            "steady_tflops": row["steady_tflops"],
            "peak_gpu_gb": row["peak_gpu_gb"],
            "peak_cpu_gb": row["peak_cpu_gb"],
        }
        for row in summary["validated_batch_ceiling_14b"]["rows"]
    ]
    write_csv(ARTIFACTS_DIR / "14b_validated_batch_ceiling.csv", ceiling_rows)

    batch_rows = []
    for method_name, rows in summary["sweep_rows_14b"].items():
        for row in rows:
            batch_rows.append(
                {
                    "method": method_name,
                    "batch_size": row["batch_size"],
                    "status": row["status"],
                    "steady_tflops": row["steady_tflops"],
                    "steady_tokens_per_s": row["steady_tokens_per_s"],
                    "peak_gpu_gb": row["peak_gpu_gb"],
                    "peak_cpu_gb": row["peak_cpu_gb"],
                    "error": row.get("error"),
                }
            )
    write_csv(ARTIFACTS_DIR / "14b_batch_sweep_points.csv", batch_rows)

    if summary["fsdp_probe_14b"]["rows"]:
        fsdp_rows = []
        for row in summary["fsdp_probe_14b"]["rows"]:
            fsdp_rows.append(
                {
                    "method": row["method"],
                    "batch_size": row["batch_size"],
                    "status": row["status"],
                    "steady_tflops": row["steady_tflops"],
                    "steady_tokens_per_s": row["steady_tokens_per_s"],
                    "peak_gpu_gb": row["peak_gpu_gb"],
                    "peak_cpu_gb": row["peak_cpu_gb"],
                    "error": row.get("error"),
                }
            )
        write_csv(ARTIFACTS_DIR / "14b_fsdp_probe.csv", fsdp_rows)

    write_csv(ARTIFACTS_DIR / "megatrain_capability_ladder.csv", summary["capability_ladder"])


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (ARTIFACTS_DIR / "a_first_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv_artifacts(summary)
    render_charts(summary)
    write_markdown_summary(summary)
    print(f"Wrote A-first MI355X artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
