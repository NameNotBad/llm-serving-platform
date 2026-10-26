
"""
GPU Trace Analysis V2 for vLLM.

Features:
1. Parse PyTorch Profiler Chrome Trace files.
2. Classify GPU kernels with ordered matching rules.
3. Calculate kernel counts and duration statistics.
4. Calculate non-overlapping GPU activity intervals.
5. Normalize GPU activity by actual generated tokens.
6. Generate CSV, JSON, and PNG reports.

Input:
    results/gpu_profiles/*rank0*.pt.trace.json.gz
    results/gpu_profiles/<workload>_*.json

Outputs:
    results/gpu_profiles/analysis_v2/
        summary.csv
        kernel_families.csv
        top_kernels.csv
        analysis.json
        kernel_families.png
        normalized_gpu_time.png
        gpu_activity_coverage.png

Usage:
    python benchmark/analyze_gpu_trace_v2.py \
        --results results/gpu_profiles

No GPU inference or server restart required.
"""

import argparse
import csv
import gzip
import json
import re
import statistics

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

GPU_CATEGORIES = {
    "kernel",
    "gpu_memcpy",
    "gpu_memset",
}

WORKLOAD_LABELS = {
    "prefill_heavy": "Prefill-heavy",
    "decode_heavy": "Decode-heavy",
}

FAMILY_ORDER = [
    "GEMM / CUTLASS",
    "FlashAttention",
    "KV Cache",
    "Triton",
    "GEMV",
    "Memory Transfer",
    "Other",
]


# ============================================================
# 1. File utilities
# ============================================================

def save_csv(path: Path, rows: list[dict]) -> None:
    """Save dictionaries as CSV."""

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict) -> None:
    """Save a JSON report."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_trace(path: Path) -> list[dict]:
    """Read a compressed PyTorch Profiler trace."""

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        events = data.get("traceEvents", [])

    elif isinstance(data, list):
        events = data

    else:
        raise ValueError(
            f"Unsupported trace format: {path}"
        )

    if not isinstance(events, list):
        raise ValueError(
            "traceEvents must be a list"
        )

    return events


# ============================================================
# 2. Kernel classification
# ============================================================

def classify_kernel(
    name: str,
    category: str,
) -> str:
    """
    Classify kernels using ordered matching rules.

    Important:
    FlashAttention must be recognized before generic
    GEMM matching because CUDA names may contain
    additional template parameters.
    """

    text = name.lower()

    # CUDA memory operations are separate from kernels.
    if category in {"gpu_memcpy", "gpu_memset"}:
        return "Memory Transfer"

    # Attention kernels are checked first.
    if (
        "flash_fwd" in text
        or "flash_attn" in text
        or "flashattention" in text
        or "flashinfer" in text
        or "paged_attention" in text
        or "varlen_fwd" in text
    ):
        return "FlashAttention"

    # KV cache writes, updates and copies.
    if (
        "reshape_and_cache" in text
        or "kv_cache" in text
        or "cache_update" in text
        or "copy_blocks" in text
        or "swap_blocks" in text
    ):
        return "KV Cache"

    # Matrix-vector operations.
    if (
        "gemv" in text
        or "gemvx" in text
    ):
        return "GEMV"

    # General matrix multiplication kernels.
    if (
        "cutlass" in text
        or "gemm" in text
        or "cublas" in text
        or "wmma" in text
    ):
        return "GEMM / CUTLASS"

    # Generic Triton-generated kernels.
    if (
        "triton" in text
        or "fused_" in text
    ):
        return "Triton"

    return "Other"


# ============================================================
# 3. Extract GPU activities
# ============================================================

def extract_gpu_events(
    trace_events: list[dict],
) -> list[dict]:
    """
    Extract timed CUDA activity events.

    Duration and timestamps are in microseconds.
    """

    gpu_events = []

    for event in trace_events:

        if event.get("ph") != "X":
            continue

        category = str(
            event.get("cat", "")
        ).lower()

        if category not in GPU_CATEGORIES:
            continue

        duration = float(
            event.get("dur", 0)
        )

        if duration <= 0:
            continue

        start = float(
            event.get("ts", 0)
        )

        name = str(
            event.get("name", "unknown")
        )

        gpu_events.append({
            "name": name,
            "category": category,
            "family": classify_kernel(
                name,
                category,
            ),
            "start_us": start,
            "end_us": start + duration,
            "duration_us": duration,
        })

    return gpu_events


# ============================================================
# 4. Non-overlapping GPU activity duration
# ============================================================

def interval_union_us(
    events: list[dict],
) -> float:
    """
    Calculate the union of GPU event intervals.

    Overlapping events are counted only once.

    This represents recorded activity coverage,
    not hardware compute utilization.
    """

    if not events:
        return 0.0

    intervals = sorted(
        (
            event["start_us"],
            event["end_us"],
        )
        for event in events
    )

    current_start, current_end = intervals[0]
    total = 0.0

    for start, end in intervals[1:]:

        if start <= current_end:
            current_end = max(
                current_end,
                end,
            )

        else:
            total += (
                current_end - current_start
            )

            current_start = start
            current_end = end

    total += (
        current_end - current_start
    )

    return total


# ============================================================
# 5. Find matching workload metadata
# ============================================================

def workload_from_filename(
    path: Path,
) -> str:
    """Identify the workload from the trace filename."""

    for workload in WORKLOAD_LABELS:
        if path.name.startswith(workload + "_"):
            return workload

    raise ValueError(
        f"Unknown workload: {path.name}"
    )


def load_workload_metadata(
    trace_path: Path,
    results_dir: Path,
) -> dict:
    """
    Find the exact experiment JSON corresponding
    to the selected GPU trace.

    Example:
        prefill_heavy_20261025_235707_rank0...gz

    Matches:
        prefill_heavy_20261025_235707.json
    """

    marker = "_rank0"

    if marker not in trace_path.name:
        raise ValueError(
            "Trace filename has no rank0 marker"
        )

    prefix = trace_path.name.split(
        marker,
        1,
    )[0]

    metadata_path = (
        results_dir / f"{prefix}.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing workload metadata: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    requests = metadata.get(
        "requests", []
    )

    if not requests:
        raise ValueError(
            f"No request records in {metadata_path}"
        )

    total_output = sum(
        int(request["output_tokens"])
        for request in requests
    )

    total_input = sum(
        int(request["input_tokens"])
        for request in requests
    )

    if total_output <= 0:
        raise ValueError(
            "No generated output tokens"
        )

    return {
        "metadata_path": str(metadata_path),
        "request_count": len(requests),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }


# ============================================================
# 6. Analyze one GPU trace
# ============================================================

def analyze_trace(
    trace_path: Path,
    results_dir: Path,
) -> tuple[dict, list[dict], list[dict]]:
    """Analyze kernel families and individual kernels."""

    trace = load_trace(trace_path)

    gpu_events = extract_gpu_events(trace)

    if not gpu_events:
        categories = defaultdict(int)

        for event in trace:
            categories[
                str(event.get("cat", "unknown"))
            ] += 1

        raise RuntimeError(
            f"No GPU activities in {trace_path}\n"
            f"Available categories: {dict(categories)}"
        )

    workload = workload_from_filename(
        trace_path
    )

    metadata = load_workload_metadata(
        trace_path,
        results_dir,
    )

    total_output = metadata[
        "total_output_tokens"
    ]

    # --------------------------------------------------------
    # Overall GPU activity statistics
    # --------------------------------------------------------

    first_event = min(
        event["start_us"]
        for event in gpu_events
    )

    last_event = max(
        event["end_us"]
        for event in gpu_events
    )

    span_us = last_event - first_event

    union_us = interval_union_us(
        gpu_events
    )

    summed_us = sum(
        event["duration_us"]
        for event in gpu_events
    )

    # --------------------------------------------------------
    # Group by family and kernel name
    # --------------------------------------------------------

    by_family = defaultdict(list)
    by_kernel = defaultdict(list)

    for event in gpu_events:

        by_family[
            event["family"]
        ].append(event)

        key = (
            event["family"],
            event["name"],
        )

        by_kernel[key].append(event)

    family_rows = []

    for family in FAMILY_ORDER:

        events = by_family.get(
            family, []
        )

        total_us = sum(
            event["duration_us"]
            for event in events
        )

        family_rows.append({
            "workload": workload,
            "family": family,
            "calls": len(events),

            "summed_duration_ms": (
                total_us / 1000
            ),

            "normalized_ms_per_1k_output_tokens": (
                total_us / total_output
            ),
        })

    kernel_rows = []

    for (family, name), events in by_kernel.items():

        durations = [
            event["duration_us"]
            for event in events
        ]

        total_us = sum(durations)

        kernel_rows.append({
            "workload": workload,
            "family": family,
            "kernel_name": name,
            "calls": len(events),
            "total_ms": total_us / 1000,
            "mean_us": statistics.mean(durations),
        })

    kernel_rows.sort(
        key=lambda row: row["total_ms"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "workload": workload,
        "trace_file": trace_path.name,
        "request_count": metadata["request_count"],
        "total_input_tokens": metadata["total_input_tokens"],
        "total_output_tokens": total_output,

        "gpu_activity_events": len(gpu_events),

        "gpu_activity_span_ms": span_us / 1000,
        "gpu_activity_union_ms": union_us / 1000,
        "gpu_activity_summed_ms": summed_us / 1000,

        "gpu_activity_coverage_pct": (
            100 * union_us / span_us
            if span_us > 0
            else None
        ),

        "union_ms_per_1k_output_tokens": (
            union_us / total_output
        ),

        "summed_ms_per_1k_output_tokens": (
            summed_us / total_output
        ),

        "events_per_output_token": (
            len(gpu_events) / total_output
        ),
    }

    return summary, family_rows, kernel_rows


# ============================================================
# 7. Plot kernel families
# ============================================================

def plot_kernel_families(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Compare normalized kernel family durations."""

    workloads = list(
        WORKLOAD_LABELS.keys()
    )

    families = [
        family
        for family in FAMILY_ORDER
        if any(
            row["family"] == family
            and row["calls"] > 0
            for row in rows
        )
    ]

    values = {
        workload: {
            row["family"]:
                row["normalized_ms_per_1k_output_tokens"]
            for row in rows
            if row["workload"] == workload
        }
        for workload in workloads
    }

    y_positions = list(range(len(families)))

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bar_height = 0.36

    for index, workload in enumerate(workloads):

        offsets = [
            y + (index - 0.5) * bar_height
            for y in y_positions
        ]

        widths = [
            values[workload].get(family, 0)
            for family in families
        ]

        ax.barh(
            offsets,
            widths,
            height=bar_height,
            label=WORKLOAD_LABELS[workload],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(families)
    ax.invert_yaxis()

    ax.set_xlabel(
        "Summed GPU kernel duration "
        "(ms / 1,000 output tokens)"
    )

    ax.set_title(
        "Normalized GPU Kernel Time by Family"
    )

    ax.legend()
    ax.grid(
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


# ============================================================
# 8. Plot normalized GPU activity
# ============================================================

def plot_normalized_activity(
    summaries: list[dict],
    output_path: Path,
) -> None:
    """Compare union and summed GPU times."""

    ordered = sorted(
        summaries,
        key=lambda row: list(
            WORKLOAD_LABELS
        ).index(row["workload"]),
    )

    labels = [
        WORKLOAD_LABELS[
            row["workload"]
        ]
        for row in ordered
    ]

    union = [
        row["union_ms_per_1k_output_tokens"]
        for row in ordered
    ]

    summed = [
        row["summed_ms_per_1k_output_tokens"]
        for row in ordered
    ]

    x = list(range(len(labels)))

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    width = 0.35

    ax.bar(
        [value - width / 2 for value in x],
        union,
        width=width,
        label="GPU activity union",
    )

    ax.bar(
        [value + width / 2 for value in x],
        summed,
        width=width,
        label="Summed GPU activity",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "ms per 1,000 generated tokens"
    )

    ax.set_title(
        "Normalized GPU Activity"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


# ============================================================
# 9. Plot GPU activity coverage
# ============================================================

def plot_coverage(
    summaries: list[dict],
    output_path: Path,
) -> None:
    """Plot recorded GPU activity coverage."""

    labels = [
        WORKLOAD_LABELS[row["workload"]]
        for row in summaries
    ]

    values = [
        row["gpu_activity_coverage_pct"]
        for row in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(labels, values)

    ax.set_ylim(0, 100)

    ax.set_ylabel(
        "Recorded GPU activity coverage (%)"
    )

    ax.set_title(
        "GPU Activity Coverage"
    )

    for index, value in enumerate(values):
        ax.text(
            index,
            value + 1,
            f"{value:.2f}%",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


# ============================================================
# 10. Main analysis
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    results_dir = args.results

    output_dir = (
        results_dir / "analysis_v2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    traces = sorted(
        results_dir.glob(
            "*rank0*.pt.trace.json.gz"
        )
    )

    if not traces:
        raise FileNotFoundError(
            f"No GPU rank0 traces in {results_dir}"
        )

    summaries = []
    family_rows = []
    kernel_rows = []

    for trace_path in traces:

        print(
            f"\nAnalyzing {trace_path.name}",
            flush=True,
        )

        summary, families, kernels = analyze_trace(
            trace_path,
            results_dir,
        )

        summaries.append(summary)
        family_rows.extend(families)
        kernel_rows.extend(kernels)

        print(
            f"  Output tokens: "
            f"{summary['total_output_tokens']}"
        )

        print(
            f"  GPU activity coverage: "
            f"{summary['gpu_activity_coverage_pct']:.2f}%"
        )

        print(
            f"  Union time / 1k output tokens: "
            f"{summary['union_ms_per_1k_output_tokens']:.2f} ms"
        )

        print(
            "  Kernel families:"
        )

        for row in families:
            if row["calls"] > 0:
                print(
                    f"    {row['family']:20s} | "
                    f"Calls={row['calls']:7d} | "
                    f"Summed={row['summed_duration_ms']:.2f} ms"
                )

    save_csv(
        output_dir / "summary.csv",
        summaries,
    )

    save_csv(
        output_dir / "kernel_families.csv",
        family_rows,
    )

    save_csv(
        output_dir / "top_kernels.csv",
        kernel_rows,
    )

    save_json(
        output_dir / "analysis.json",
        {
            "summaries": summaries,
            "kernel_families": family_rows,
            "top_kernels": kernel_rows[:30],
            "methodology": {
                "normalization": (
                    "GPU event duration divided by "
                    "actual completion token count"
                ),
                "kernel_classification": (
                    "Heuristic name matching with "
                    "ordered precedence"
                ),
                "limitations": [
                    "Not SM occupancy",
                    "Not GPU utilization",
                    "Not measured DRAM bandwidth",
                    "Summed kernel times can overlap",
                    "Prefill costs are included",
                    "Profiler instrumentation adds overhead",
                ],
            },
        },
    )

    plot_kernel_families(
        family_rows,
        output_dir / "kernel_families.png",
    )

    plot_normalized_activity(
        summaries,
        output_dir / "normalized_gpu_time.png",
    )

    plot_coverage(
        summaries,
        output_dir / "gpu_activity_coverage.png",
    )

    print(
        f"\nAnalysis completed: {output_dir}"
    )


if __name__ == "__main__":
    main()
