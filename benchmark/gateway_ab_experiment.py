
import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = (
    PROJECT_ROOT / "benchmark" / "streaming_load_test.py"
)

DIRECT_URL = (
    "http://127.0.0.1:8000/v1/chat/completions"
)
GATEWAY_URL = (
    "http://127.0.0.1:8080/v1/chat/completions"
)

METRICS = [
    "ttft_mean_ms",
    "ttft_p95_ms",
    "tpot_mean_ms",
    "tpot_p95_ms",
    "e2e_mean_ms",
    "e2e_p95_ms",
    "throughput_tps",
    "requests_per_sec",
]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def parse_concurrencies(value):
    try:
        numbers = [
            int(item.strip())
            for item in value.split(",")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Concurrency must be comma-separated integers"
        ) from exc

    if not numbers or any(n <= 0 for n in numbers):
        raise argparse.ArgumentTypeError(
            "Concurrency values must be positive"
        )

    if len(numbers) != len(set(numbers)):
        raise argparse.ArgumentTypeError(
            "Concurrency values must be unique"
        )

    return numbers


def run_benchmark(
    endpoint,
    url,
    concurrency,
    pair_index,
    args,
    output_dir,
):
    name = (
        f"c{concurrency:02d}_"
        f"pair{pair_index:02d}_"
        f"{endpoint}"
    )

    json_path = output_dir / "runs" / f"{name}.json"
    log_path = output_dir / "logs" / f"{name}.log"

    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--url", url,
        "--concurrency", str(concurrency),
        "--requests", str(args.requests),
        "--input-tokens", str(args.input_tokens),
        "--max-tokens", str(args.max_tokens),
        "--output-json", str(json_path),
    ]

    print(
        f"\n[RUN] {name}",
        flush=True,
    )

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            f"Benchmark timed out after {args.timeout}s\n"
            f"{exc}\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Benchmark timeout: {name}"
        ) from exc

    log_path.write_text(
        result.stdout + "\n" + result.stderr,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed: {name}\n"
            f"See: {log_path}\n"
            f"{result.stderr[-1500:]}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    summary = report["summary"]

    print(
        f"  TTFT={summary['ttft_mean_ms']:.2f} ms | "
        f"TPOT={summary['tpot_mean_ms']:.2f} ms | "
        f"E2E={summary['e2e_mean_ms']:.2f} ms | "
        f"Throughput={summary['throughput_tps']:.2f} tok/s",
        flush=True,
    )

    return {
        "endpoint": endpoint,
        "url": url,
        "concurrency": concurrency,
        "pair_index": pair_index,
        "report_path": str(
            json_path.relative_to(output_dir)
        ),
        "summary": summary,
    }


def create_pair_row(
    concurrency,
    pair_index,
    order,
    direct,
    gateway,
):
    row = {
        "concurrency": concurrency,
        "pair_index": pair_index,
        "execution_order": "->".join(order),
    }

    for metric in METRICS:
        direct_value = direct["summary"].get(metric)
        gateway_value = gateway["summary"].get(metric)

        row[f"direct_{metric}"] = direct_value
        row[f"gateway_{metric}"] = gateway_value

        if direct_value is None or gateway_value is None:
            row[f"delta_{metric}"] = None
            row[f"delta_pct_{metric}"] = None
            continue

        delta = gateway_value - direct_value

        row[f"delta_{metric}"] = delta

        row[f"delta_pct_{metric}"] = (
            delta / direct_value * 100
            if direct_value != 0
            else None
        )

    return row


def aggregate_pairs(pair_rows):
    output = []

    concurrencies = sorted({
        row["concurrency"]
        for row in pair_rows
    })

    for concurrency in concurrencies:
        group = [
            row for row in pair_rows
            if row["concurrency"] == concurrency
        ]

        summary = {
            "concurrency": concurrency,
            "pairs": len(group),
        }

        for metric in METRICS:
            direct_values = [
                row[f"direct_{metric}"]
                for row in group
                if row[f"direct_{metric}"] is not None
            ]

            gateway_values = [
                row[f"gateway_{metric}"]
                for row in group
                if row[f"gateway_{metric}"] is not None
            ]

            delta_values = [
                row[f"delta_{metric}"]
                for row in group
                if row[f"delta_{metric}"] is not None
            ]

            if not delta_values:
                continue

            summary[f"direct_mean_{metric}"] = (
                statistics.mean(direct_values)
            )
            summary[f"gateway_mean_{metric}"] = (
                statistics.mean(gateway_values)
            )
            summary[f"mean_delta_{metric}"] = (
                statistics.mean(delta_values)
            )
            summary[f"median_delta_{metric}"] = (
                statistics.median(delta_values)
            )

            if len(delta_values) > 1:
                summary[f"stdev_delta_{metric}"] = (
                    statistics.stdev(delta_values)
                )

            direct_mean = statistics.mean(direct_values)

            summary[f"mean_delta_pct_{metric}"] = (
                statistics.mean(delta_values)
                / direct_mean * 100
                if direct_mean != 0
                else None
            )

        output.append(summary)

    return output


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(aggregated):
    print("\n=== Gateway A/B Benchmark Summary ===")
    print(
        "Delta = Gateway - Direct; "
        "positive latency delta means slower Gateway."
    )

    for row in aggregated:
        print(
            f"\nConcurrency: {row['concurrency']}"
        )

        for metric in [
            "ttft_mean_ms",
            "tpot_mean_ms",
            "e2e_mean_ms",
            "throughput_tps",
        ]:
            direct = row.get(f"direct_mean_{metric}")
            gateway = row.get(f"gateway_mean_{metric}")
            delta = row.get(f"mean_delta_{metric}")
            pct = row.get(f"mean_delta_pct_{metric}")

            if direct is None:
                continue

            print(
                f"  {metric}: "
                f"Direct={direct:.2f}, "
                f"Gateway={gateway:.2f}, "
                f"Delta={delta:+.2f} "
                f"({pct:+.2f}%)"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Paired Direct vs Gateway benchmark"
    )

    parser.add_argument(
        "--concurrencies",
        type=parse_concurrencies,
        default=[1, 4, 8, 16],
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/gateway"),
    )

    args = parser.parse_args()

    if any(
        value <= 0 for value in [
            args.requests,
            args.repeats,
            args.input_tokens,
            args.max_tokens,
            args.timeout,
        ]
    ):
        parser.error(
            "Numeric parameters must be positive"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        PROJECT_ROOT / args.output_root / timestamp
    )

    (output_dir / "runs").mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_dir / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "concurrencies": args.concurrencies,
        "requests_per_run": args.requests,
        "repeats": args.repeats,
        "input_tokens": args.input_tokens,
        "max_tokens": args.max_tokens,
        "direct_url": DIRECT_URL,
        "gateway_url": GATEWAY_URL,
        "timestamp": timestamp,
    }

    save_json(
        output_dir / "config.json",
        config,
    )

    all_runs = []
    pair_rows = []

    endpoints = {
        "direct": DIRECT_URL,
        "gateway": GATEWAY_URL,
    }

    for concurrency_index, concurrency in enumerate(
        args.concurrencies
    ):
        for pair_index in range(1, args.repeats + 1):

            # Alternate order between pairs and
            # concurrency levels to reduce order bias.
            direct_first = (
                (concurrency_index + pair_index) % 2 == 1
            )

            order = (
                ["direct", "gateway"]
                if direct_first
                else ["gateway", "direct"]
            )

            pair_results = {}

            for endpoint in order:
                result = run_benchmark(
                    endpoint=endpoint,
                    url=endpoints[endpoint],
                    concurrency=concurrency,
                    pair_index=pair_index,
                    args=args,
                    output_dir=output_dir,
                )

                all_runs.append(result)
                pair_results[endpoint] = result

                # Save partial progress after each run.
                save_json(
                    output_dir / "all_runs.json",
                    all_runs,
                )

            pair_row = create_pair_row(
                concurrency=concurrency,
                pair_index=pair_index,
                order=order,
                direct=pair_results["direct"],
                gateway=pair_results["gateway"],
            )

            pair_rows.append(pair_row)

            save_json(
                output_dir / "pairs.json",
                pair_rows,
            )

            write_csv(
                output_dir / "paired_results.csv",
                pair_rows,
            )

    aggregated = aggregate_pairs(pair_rows)

    save_json(
        output_dir / "summary.json",
        aggregated,
    )

    write_csv(
        output_dir / "summary.csv",
        aggregated,
    )

    print_summary(aggregated)

    print(
        f"\nAll results saved to: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
