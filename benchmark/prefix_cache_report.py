
"""
Generate a complete 2x2 Prefix Caching report.

Reads existing benchmark CSV files. No GPU inference
or server restart required.

Usage:
    python benchmark/prefix_cache_report.py
"""

import csv
import json
import statistics
from pathlib import Path


RESULTS_DIR = Path("results/prefix_cache_v1")

SCENARIOS = [
    "off_unique",
    "on_unique",
    "off_shared",
    "on_shared",
]

METRICS = [
    "throughput_tps",
    "ttft_mean_ms",
    "tpot_mean_ms",
    "e2e_p95_ms",
]


def load_scenario(scenario):
    path = RESULTS_DIR / f"{scenario}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def aggregate(rows):
    """Aggregate independent benchmark repetitions."""

    results = {}

    for concurrency in sorted({
        int(row["concurrency"]) for row in rows
    }):
        subset = [
            row for row in rows
            if int(row["concurrency"]) == concurrency
        ]

        values = {}

        for metric in METRICS:
            samples = [
                float(row[metric]) for row in subset
            ]

            values[metric] = {
                "mean": statistics.mean(samples),
                "std": (
                    statistics.stdev(samples)
                    if len(samples) > 1
                    else None
                ),
            }

        values["repeats"] = len(subset)
        results[concurrency] = values

    return results


def percentage_change(before, after):
    return (after / before - 1) * 100


def main():
    # Aggregate each scenario independently.
    results = {
        scenario: aggregate(load_scenario(scenario))
        for scenario in SCENARIOS
    }

    comparisons = []

    for concurrency in sorted(results["off_shared"]):
        print(f"\n=== CONCURRENCY {concurrency} ===")

        for workload in ["unique", "shared"]:
            off = results[f"off_{workload}"][concurrency]
            on = results[f"on_{workload}"][concurrency]

            throughput_off = off["throughput_tps"]["mean"]
            throughput_on = on["throughput_tps"]["mean"]

            ttft_off = off["ttft_mean_ms"]["mean"]
            ttft_on = on["ttft_mean_ms"]["mean"]

            throughput_gain = percentage_change(
                throughput_off,
                throughput_on,
            )

            ttft_reduction = (
                1 - ttft_on / ttft_off
            ) * 100

            comparison = {
                "concurrency": concurrency,
                "workload": workload,
                "throughput_off": throughput_off,
                "throughput_on": throughput_on,
                "throughput_gain_pct": throughput_gain,
                "ttft_off_ms": ttft_off,
                "ttft_on_ms": ttft_on,
                "ttft_reduction_pct": ttft_reduction,
            }

            comparisons.append(comparison)

            print(
                f"\n{workload.upper()}:"
                f"\n  Throughput: "
                f"{throughput_off:.2f} -> "
                f"{throughput_on:.2f} tok/s"
                f"\n  Gain: {throughput_gain:+.2f}%"
                f"\n  TTFT: "
                f"{ttft_off:.2f} -> "
                f"{ttft_on:.2f} ms"
                f"\n  Reduction: {ttft_reduction:.2f}%"
            )

    # Machine-readable analysis for future visualization.
    output_json = RESULTS_DIR / "final_report.json"

    output_json.write_text(
        json.dumps(
            {
                "scenarios": results,
                "comparisons": comparisons,
                "notes": [
                    "Metrics are means across benchmark runs.",
                    "Runs use warmup and closed-loop load.",
                    "Shared prompts use approximately 513 input tokens.",
                    "Cache hit rate was not directly measured.",
                    "ON and OFF runs were not interleaved.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_csv = RESULTS_DIR / "final_comparison.csv"

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparisons[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparisons)

    print("\nReports saved:")
    print(output_json)
    print(output_csv)


if __name__ == "__main__":
    main()
