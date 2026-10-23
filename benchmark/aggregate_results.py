
"""
Aggregate repeated LLM serving benchmark runs.

Reads summary.csv from a completed benchmark suite and
calculates mean and standard deviation across repeated runs.

Usage:
    python benchmark/aggregate_results.py \
        --results results/benchmark_YYYYMMDD_HHMMSS
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = [
    "throughput_tps",
    "requests_per_sec",
    "ttft_mean_ms",
    "ttft_p95_ms",
    "tpot_mean_ms",
    "tpot_p95_ms",
    "e2e_mean_ms",
    "e2e_p95_ms",
]


def aggregate_results(results_dir: Path):
    summary_path = results_dir / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    # Group successful runs by input length and concurrency.
    groups = defaultdict(list)

    with summary_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] != "success":
                continue

            key = (
                int(row["input_tokens"]),
                int(row["concurrency"]),
            )

            groups[key].append(row)

    aggregated = []

    for (input_tokens, concurrency), runs in sorted(groups.items()):
        result = {
            "input_tokens": input_tokens,
            "concurrency": concurrency,
            "successful_repeats": len(runs),
        }

        for metric in METRICS:
            values = [
                float(run[metric])
                for run in runs
                if run.get(metric)
            ]

            if not values:
                result[f"{metric}_mean"] = None
                result[f"{metric}_std"] = None
                continue

            # Mean performance across repeated benchmark runs.
            result[f"{metric}_mean"] = statistics.mean(values)

            # Sample standard deviation measures run-to-run variation.
            # At least two observations are required.
            result[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else None
            )

        aggregated.append(result)

    return aggregated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory containing benchmark summary.csv",
    )

    args = parser.parse_args()

    rows = aggregate_results(args.results)

    if not rows:
        raise RuntimeError("No successful benchmark runs found")

    output_path = args.results / "aggregated.csv"

    with output_path.open(
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

    print("\n=== Aggregated Benchmark Results ===")

    for row in rows:
        print(
            f"Input={row['input_tokens']} | "
            f"C={row['concurrency']} | "
            f"Runs={row['successful_repeats']} | "
            f"Throughput="
            f"{row['throughput_tps_mean']:.2f} tok/s | "
            f"Mean TTFT="
            f"{row['ttft_mean_ms_mean']:.2f} ms | "
            f"Mean TPOT="
            f"{row['tpot_mean_ms_mean']:.2f} ms"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
