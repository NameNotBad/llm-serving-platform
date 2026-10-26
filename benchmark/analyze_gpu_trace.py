
"""
Analyze GPU kernel events from PyTorch Profiler traces.

Input:
    results/gpu_profiles/*rank0*.pt.trace.json.gz

Outputs:
    results/gpu_profiles/gpu_kernel_summary.csv
    results/gpu_profiles/gpu_trace_analysis.json

Usage:
    python benchmark/analyze_gpu_trace.py \
        --results results/gpu_profiles

No GPU inference or external packages required.
"""

import argparse
import csv
import gzip
import json

from collections import Counter, defaultdict
from pathlib import Path


def load_trace(path: Path) -> list[dict]:
    """Load a compressed Chrome Trace JSON file."""

    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError(f"Unsupported trace format: {path}")

    if not isinstance(events, list):
        raise ValueError("Invalid traceEvents field")

    return events


def classify_kernel(name: str) -> str:
    """Assign a descriptive family to a GPU kernel."""

    text = name.lower()

    if "cutlass" in text or "gemm" in text:
        return "GEMM / CUTLASS"

    if "flash" in text and "cache" not in text:
        return "FlashAttention"

    if "reshape_and_cache" in text or "kv_cache" in text:
        return "KV Cache"

    if "triton" in text:
        return "Triton"

    if "memcpy" in text or "memset" in text:
        return "Memory Transfer"

    if "gemv" in text:
        return "GEMV"

    return "Other"


def extract_gpu_events(events: list[dict]) -> list[dict]:
    """
    Select timed GPU activities.

    PyTorch's Chrome Trace normally labels CUDA kernel
    events with category 'kernel'. Copy and memset events
    may use separate categories.
    """

    selected = []

    gpu_categories = {
        "kernel",
        "gpu_memcpy",
        "gpu_memset",
    }

    for event in events:

        if event.get("ph") != "X":
            continue

        category = str(event.get("cat", "")).lower()

        if category not in gpu_categories:
            continue

        duration = float(event.get("dur", 0))

        if duration <= 0:
            continue

        selected.append({
            "name": str(event.get("name", "unknown")),
            "category": category,
            "start_us": float(event.get("ts", 0)),
            "duration_us": duration,
            "end_us": (
                float(event.get("ts", 0)) + duration
            ),
        })

    return selected


def interval_union_us(events: list[dict]) -> float:
    """
    Calculate the union of event time intervals.

    This avoids double-counting overlapping activities.
    It is not a hardware SM utilization measurement.
    """

    if not events:
        return 0.0

    intervals = sorted(
        (e["start_us"], e["end_us"])
        for e in events
    )

    total = 0.0

    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:

        if start <= current_end:
            current_end = max(current_end, end)

        else:
            total += current_end - current_start
            current_start, current_end = start, end

    total += current_end - current_start

    return total


def analyze_trace(path: Path) -> tuple[dict, list[dict]]:
    """Summarize GPU activities by kernel family and name."""

    events = load_trace(path)
    gpu_events = extract_gpu_events(events)

    if not gpu_events:
        categories = Counter(
            str(event.get("cat", "unknown"))
            for event in events
        )

        raise ValueError(
            f"No GPU kernel events found in {path}.\n"
            f"Available event categories: {categories.most_common(20)}"
        )

    start = min(e["start_us"] for e in gpu_events)
    end = max(e["end_us"] for e in gpu_events)

    span_us = end - start
    busy_us = interval_union_us(gpu_events)

    by_name = defaultdict(list)
    by_family = defaultdict(list)

    for event in gpu_events:
        by_name[event["name"]].append(event)
        by_family[
            classify_kernel(event["name"])
        ].append(event)

    kernel_rows = []

    for name, group in by_name.items():
        total_us = sum(
            e["duration_us"]
            for e in group
        )

        kernel_rows.append({
            "trace": path.name,
            "family": classify_kernel(name),
            "kernel_name": name,
            "calls": len(group),
            "total_ms": total_us / 1000,
            "mean_us": total_us / len(group),
        })

    kernel_rows.sort(
        key=lambda row: row["total_ms"],
        reverse=True,
    )

    families = []

    for family, group in by_family.items():
        families.append({
            "family": family,
            "calls": len(group),
            "summed_duration_ms": (
                sum(e["duration_us"] for e in group) / 1000
            ),
        })

    families.sort(
        key=lambda row: row["summed_duration_ms"],
        reverse=True,
    )

    summary = {
        "trace": path.name,
        "total_trace_events": len(events),
        "gpu_activity_events": len(gpu_events),
        "gpu_activity_span_ms": span_us / 1000,
        "gpu_activity_union_ms": busy_us / 1000,
        "gpu_activity_coverage_pct": (
            100 * busy_us / span_us
            if span_us > 0
            else None
        ),
        "kernel_families": families,
        "top_kernels": kernel_rows[:20],
    }

    return summary, kernel_rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    results_dir = args.results

    # Select the EngineCore traces, not the small
    # asynchronous API server traces.
    traces = sorted(
        results_dir.glob("*rank0*.pt.trace.json.gz")
    )

    if not traces:
        raise FileNotFoundError(
            f"No rank0 traces found in {results_dir}"
        )

    analyses = []
    all_rows = []

    for path in traces:
        print(f"\n=== {path.name} ===")

        summary, rows = analyze_trace(path)

        analyses.append(summary)
        all_rows.extend(rows)

        print(
            f"GPU activity events: "
            f"{summary['gpu_activity_events']}"
        )

        print(
            f"GPU activity span: "
            f"{summary['gpu_activity_span_ms']:.2f} ms"
        )

        print(
            f"GPU activity coverage: "
            f"{summary['gpu_activity_coverage_pct']:.2f}%"
        )

        print("\nKernel families:")

        for family in summary["kernel_families"]:
            print(
                f"  {family['family']:20s} "
                f"Calls={family['calls']:7d} | "
                f"Summed={family['summed_duration_ms']:.2f} ms"
            )

        print("\nTop 10 kernels:")

        for row in rows[:10]:
            print(
                f"  {row['total_ms']:9.2f} ms | "
                f"{row['calls']:7d} calls | "
                f"{row['kernel_name'][:75]}"
            )

    output_json = (
        results_dir / "gpu_trace_analysis.json"
    )

    output_json.write_text(
        json.dumps(
            {
                "analyses": analyses,
                "methodology": {
                    "duration_unit": "microseconds",
                    "selected_categories": [
                        "kernel",
                        "gpu_memcpy",
                        "gpu_memset",
                    ],
                    "coverage_definition": (
                        "Union of recorded GPU activity "
                        "intervals divided by the time span "
                        "from first to last recorded activity"
                    ),
                    "limitations": [
                        "Not an SM utilization metric",
                        "Not a memory bandwidth measurement",
                        "Kernel durations may overlap",
                        "Profiles contain instrumentation overhead",
                        "Kernel family classification is heuristic",
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_csv = (
        results_dir / "gpu_kernel_summary.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trace",
                "family",
                "kernel_name",
                "calls",
                "total_ms",
                "mean_us",
            ],
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\nAnalysis completed.")
    print(f"JSON: {output_json}")
    print(f"CSV:  {output_csv}")


if __name__ == "__main__":
    main()
