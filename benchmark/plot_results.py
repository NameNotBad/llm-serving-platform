
"""
Generate performance plots from aggregated benchmark results.

Input:
    results/benchmark_.../aggregated.csv

Outputs:
    results/benchmark_.../plots/
        throughput.png
        ttft.png
        tpot.png
        e2e_p95.png

Usage:
    python benchmark/plot_results.py \
        --results results/benchmark_XXXXXXXX
"""

import argparse
import csv
from pathlib import Path

import matplotlib

# Render images without requiring a graphical desktop.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load aggregated benchmark data
# ------------------------------------------------------------

def load_results(csv_path: Path) -> list[dict]:
    """Read aggregated.csv and convert values to numbers."""

    rows = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for raw in reader:
            row = {}

            for key, value in raw.items():
                if key is None:
                    continue

                if key in (
                    "input_tokens",
                    "concurrency",
                    "successful_repeats",
                ):
                    row[key] = int(value)

                elif value is None or value == "":
                    row[key] = None

                else:
                    row[key] = float(value)

            rows.append(row)

    if not rows:
        raise ValueError(
            f"No benchmark records in {csv_path}"
        )

    return rows


# ------------------------------------------------------------
# Plot one metric
# ------------------------------------------------------------

def plot_metric(
    rows: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot a performance metric against concurrency.

    Each input token length gets a separate line.

    Error bars use +/- one sample standard deviation
    across successful repetitions.
    """

    input_lengths = sorted({
        row["input_tokens"]
        for row in rows
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for input_tokens in input_lengths:

        subset = sorted(
            [
                row
                for row in rows
                if row["input_tokens"] == input_tokens
            ],
            key=lambda row: row["concurrency"],
        )

        x = [
            row["concurrency"]
            for row in subset
        ]

        y = [
            row[f"{metric}_mean"]
            for row in subset
        ]

        std_values = [
            row.get(f"{metric}_std")
            for row in subset
        ]

        # A single repetition has no sample standard
        # deviation. Show its point without an error bar.
        yerr = [
            value if value is not None else 0.0
            for value in std_values
        ]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"Input {input_tokens} tokens",
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks(sorted({
        row["concurrency"]
        for row in rows
    }))

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# Generate all plots
# ------------------------------------------------------------

def generate_plots(results_dir: Path) -> None:
    """Generate the benchmark visualization suite."""

    csv_path = results_dir / "aggregated.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Aggregated results not found: {csv_path}"
        )

    rows = load_results(csv_path)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    charts = [
        {
            "metric": "throughput_tps",
            "title": "Output Throughput vs. Concurrency",
            "ylabel": "Output Tokens / Second",
            "filename": "throughput.png",
        },
        {
            "metric": "ttft_mean_ms",
            "title": "Mean TTFT vs. Concurrency",
            "ylabel": "TTFT (ms)",
            "filename": "ttft.png",
        },
        {
            "metric": "tpot_mean_ms",
            "title": "Mean TPOT vs. Concurrency",
            "ylabel": "TPOT (ms)",
            "filename": "tpot.png",
        },
        {
            "metric": "e2e_p95_ms",
            "title": "P95 E2E Latency vs. Concurrency",
            "ylabel": "P95 E2E Latency (ms)",
            "filename": "e2e_p95.png",
        },
    ]

    for chart in charts:
        plot_metric(
            rows=rows,
            metric=chart["metric"],
            title=chart["title"],
            ylabel=chart["ylabel"],
            output_path=plots_dir / chart["filename"],
        )

    print(
        f"\nGenerated {len(charts)} plots "
        f"in {plots_dir}"
    )


# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregated LLM benchmark results"
    )

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory containing aggregated.csv",
    )

    args = parser.parse_args()

    generate_plots(args.results)


if __name__ == "__main__":
    main()
