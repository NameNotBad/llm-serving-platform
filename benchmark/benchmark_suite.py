
"""
Automated LLM Serving Benchmark Suite.

This script orchestrates multiple benchmark runs using different:
- Input token lengths
- Concurrency levels
- Repeated trials

Each run executes streaming_load_test.py as a subprocess.

The streaming benchmark writes structured JSON results.
This suite validates the results and saves summary reports.

Example:
    python benchmark/benchmark_suite.py \
        --config benchmark/config_smoke.json
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys

from datetime import datetime
from pathlib import Path

from aggregate_results import aggregate_results

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

# Directory containing this script:
# llm-serving-platform/benchmark/
ROOT = Path(__file__).resolve().parent

# Project root:
# llm-serving-platform/
PROJECT_ROOT = ROOT.parent


# ------------------------------------------------------------
# Run a single benchmark configuration
# ------------------------------------------------------------

def run_benchmark(
    input_tokens: int,
    concurrency: int,
    requests: int,
    max_tokens: int,
    output_path: Path,
) -> dict:
    """
    Execute one streaming benchmark run.

    Parameters
    ----------
    input_tokens:
        Target number of prompt tokens, including chat template.

    concurrency:
        Maximum number of simultaneous HTTP requests.

    requests:
        Total number of requests in this benchmark run.

    max_tokens:
        Maximum number of generated tokens per request.

    output_path:
        Location where streaming_load_test.py saves its JSON report.

    Returns
    -------
    dict:
        Structured benchmark report containing:
        - config
        - summary
        - requests
    """

    # Use the same Python interpreter that runs this suite.
    # This ensures both scripts use the same virtual environment.
    command = [
        sys.executable,
        str(ROOT / "streaming_load_test.py"),
        "--input-tokens",
        str(input_tokens),
        "--concurrency",
        str(concurrency),
        "--requests",
        str(requests),
        "--max-tokens",
        str(max_tokens),
        "--output-json",
        str(output_path),
    ]

    # Execute the child benchmark process.
    #
    # capture_output=True:
    #   Collect stdout and stderr instead of printing them directly.
    #
    # text=True:
    #   Return stdout/stderr as strings.
    #
    # timeout=1800:
    #   Stop the child process if it exceeds 30 minutes.
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    # A nonzero exit code indicates the benchmark failed.
    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark process failed "
            f"(exit code {process.returncode}).\n"
            f"STDERR:\n{process.stderr[-3000:]}\n"
            f"STDOUT:\n{process.stdout[-3000:]}"
        )

    # The child benchmark must produce the expected JSON file.
    if not output_path.exists():
        raise RuntimeError(
            f"Benchmark did not produce JSON report: {output_path}"
        )

    # Load structured metrics directly from JSON.
    # No regular-expression parsing is required.
    with output_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    # Basic schema validation.
    required_sections = {"config", "summary", "requests"}

    missing = required_sections - report.keys()

    if missing:
        raise ValueError(
            f"Missing report sections: {sorted(missing)}"
        )

    return report


# ------------------------------------------------------------
# Validate benchmark results
# ------------------------------------------------------------

def validate_report(
    report: dict,
    expected_input_tokens: int,
    expected_requests: int,
    max_output_tokens: int,
    expected_concurrency: int,
) -> dict:
    """
    Validate a benchmark report before marking it successful.

    Checks:
    1. Configuration matches the requested experiment.
    2. Number of completed requests matches the expected count.
    3. Every prompt contains exactly the requested token count.
    4. Output token counts are within the allowed range.
    5. Reported total output tokens match per-request values.

    Returns the validated summary dictionary.
    """

    config = report["config"]
    summary = report["summary"]
    request_results = report["requests"]

    # Verify the child process used the requested configuration.
    expected_config = {
        "input_tokens": expected_input_tokens,
        "concurrency": expected_concurrency,
        "total_requests": expected_requests,
        "max_output_tokens": max_output_tokens,
    }

    for key, expected_value in expected_config.items():
        actual_value = config.get(key)

        if actual_value != expected_value:
            raise ValueError(
                f"Configuration mismatch for {key}: "
                f"expected={expected_value}, actual={actual_value}"
            )

    # Each request should produce one result entry.
    if len(request_results) != expected_requests:
        raise ValueError(
            f"Request count mismatch: "
            f"expected={expected_requests}, "
            f"actual={len(request_results)}"
        )

    total_output_tokens = 0
    seen_request_ids = set()

    for request in request_results:
        request_id = request["request_id"]

        # Prevent duplicate or missing request IDs.
        if request_id in seen_request_ids:
            raise ValueError(
                f"Duplicate request ID: {request_id}"
            )

        seen_request_ids.add(request_id)

        # Validate actual input length returned by the server.
        actual_input = request["input_tokens"]

        if actual_input != expected_input_tokens:
            raise ValueError(
                f"Request {request_id}: "
                f"expected {expected_input_tokens} input tokens, "
                f"got {actual_input}"
            )

        # Output can be shorter than max_tokens if the model
        # generates an EOS token and stops early.
        actual_output = request["output_tokens"]

        if not (1 <= actual_output <= max_output_tokens):
            raise ValueError(
                f"Request {request_id}: "
                f"invalid output token count {actual_output}"
            )

        total_output_tokens += actual_output

        # Latency metrics should be positive.
        for metric in ("ttft", "e2e"):
            value = request[metric]

            if value is None or value <= 0:
                raise ValueError(
                    f"Request {request_id}: "
                    f"invalid {metric}={value}"
                )

        # TPOT is only defined when more than one token exists.
        tpot = request["tpot"]

        if actual_output > 1:
            if tpot is None or tpot <= 0:
                raise ValueError(
                    f"Request {request_id}: invalid TPOT={tpot}"
                )

    expected_ids = set(range(expected_requests))

    if seen_request_ids != expected_ids:
        raise ValueError(
            "Missing or unexpected request IDs"
        )

    # Cross-check the aggregate token count.
    reported_total = summary["total_output_tokens"]

    if reported_total != total_output_tokens:
        raise ValueError(
            f"Output token total mismatch: "
            f"reported={reported_total}, "
            f"calculated={total_output_tokens}"
        )

    # Ensure the aggregate statistics are sensible.
    for metric in (
        "duration_s",
        "throughput_tps",
        "requests_per_sec",
        "ttft_mean_ms",
        "ttft_p95_ms",
        "e2e_mean_ms",
        "e2e_p95_ms",
    ):
        value = summary.get(metric)

        if value is None or value <= 0:
            raise ValueError(
                f"Invalid summary metric: {metric}={value}"
            )

    return summary


# ------------------------------------------------------------
# Save CSV results
# ------------------------------------------------------------

def save_csv(path: Path, rows: list[dict]) -> None:
    """
    Write one summary row per benchmark run.

    A fixed column order is used because successful and
    failed runs can contain different fields.
    """

    if not rows:
        return

    fieldnames = [
        "input_tokens",
        "concurrency",
        "repeat",
        "status",
        "total_output_tokens",
        "duration_s",
        "throughput_tps",
        "requests_per_sec",
        "ttft_mean_ms",
        "ttft_p95_ms",
        "tpot_mean_ms",
        "tpot_p95_ms",
        "e2e_mean_ms",
        "e2e_p95_ms",
        "report_path",
        "error",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Save JSON results
# ------------------------------------------------------------

def save_json(path: Path, data) -> None:
    """
    Save Python objects as readable JSON.
    """

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ------------------------------------------------------------
# Main benchmark orchestration
# ------------------------------------------------------------

def main() -> None:
    """
    Load the experiment matrix, execute every configuration,
    validate the results, and save reports.
    """

    parser = argparse.ArgumentParser(
        description="Automated LLM serving benchmark suite"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.json",
        help="Path to the benchmark configuration JSON",
    )

    args = parser.parse_args()

    # Load the benchmark matrix.
    with args.config.open("r", encoding="utf-8") as file:
        config = json.load(file)

    input_lengths = config["input_tokens"]
    concurrency_levels = config["concurrency"]
    requests_per_run = config["requests_per_run"]
    max_output_tokens = config["max_output_tokens"]
    repeats = config["repeats"]

    # Validate the experiment configuration.
    if (
        not input_lengths
        or not concurrency_levels
        or requests_per_run < 1
        or max_output_tokens < 1
        or repeats < 1
    ):
        parser.error("Invalid benchmark configuration")

    if any(value < 1 for value in input_lengths):
        parser.error("Input token lengths must be positive")

    if any(value < 1 for value in concurrency_levels):
        parser.error("Concurrency levels must be positive")

    # Create a unique result directory for this execution.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        PROJECT_ROOT
        / "results"
        / f"benchmark_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Save the exact experiment configuration.
    save_json(
        output_dir / "config.json",
        config,
    )

    # Build every combination of:
    # input length × concurrency × repeat.
    combinations = list(
        itertools.product(
            input_lengths,
            concurrency_levels,
            range(1, repeats + 1),
        )
    )

    total_runs = len(combinations)

    print(
        f"\nStarting {total_runs} benchmark runs",
        flush=True,
    )

    print(
        f"Results directory: {output_dir}\n",
        flush=True,
    )

    rows = []
    raw_runs = []

    # Run configurations sequentially.
    #
    # Running multiple benchmark configurations concurrently
    # would cause them to compete for the same GPU, making
    # performance comparisons harder to interpret.
    for index, (
        input_tokens,
        concurrency,
        repeat,
    ) in enumerate(combinations, start=1):

        print(
            f"[{index}/{total_runs}] "
            f"Input={input_tokens}, "
            f"Concurrency={concurrency}, "
            f"Repeat={repeat}",
            flush=True,
        )

        # Construct a unique report path for this run.
        report_path = output_dir / (
            f"input{input_tokens}_"
            f"c{concurrency}_"
            f"r{repeat}.json"
        )

        row = {
            "input_tokens": input_tokens,
            "concurrency": concurrency,
            "repeat": repeat,
            "report_path": str(report_path),
        }

        try:
            # Execute one benchmark and read its JSON report.
            report = run_benchmark(
                input_tokens=input_tokens,
                concurrency=concurrency,
                requests=requests_per_run,
                max_tokens=max_output_tokens,
                output_path=report_path,
            )

            # Reject invalid or incomplete benchmark results.
            metrics = validate_report(
                report=report,
                expected_input_tokens=input_tokens,
                expected_requests=requests_per_run,
                max_output_tokens=max_output_tokens,
                expected_concurrency=concurrency,
            )

            # Add metrics to the CSV summary row.
            row.update({
                "status": "success",
                **metrics,
            })

            print(
                f"  SUCCESS | "
                f"Throughput={metrics['throughput_tps']:.2f} tok/s | "
                f"TTFT={metrics['ttft_mean_ms']:.2f} ms | "
                f"TPOT={metrics['tpot_mean_ms']:.2f} ms",
                flush=True,
            )

        except Exception as exc:
            # A failed run should not terminate the entire suite.
            #
            # The failure is recorded separately instead of
            # silently treating it as valid performance data.
            row["status"] = "failed"
            row["error"] = str(exc)

            print(
                f"  FAILED | {exc}",
                flush=True,
            )

        rows.append(row)
        raw_runs.append(dict(row))

        # Save progress after every configuration.
        # If a later run fails, completed results remain on disk.
        save_csv(
            output_dir / "summary.csv",
            rows,
        )

        save_json(
            output_dir / "runs.json",
            raw_runs,
        )

    # Count successful and failed runs.
    succeeded = sum(
        row["status"] == "success"
        for row in rows
    )

    failed = total_runs - succeeded

    print("\n=== Benchmark Suite Complete ===")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {succeeded}")
    print(f"Failed runs: {failed}")
    print(f"Results: {output_dir}")


    # Aggregate successful benchmark runs automatically.
    if succeeded > 0:
        aggregated_rows = aggregate_results(output_dir)

        aggregated_path = output_dir / "aggregated.csv"

        if aggregated_rows:
            with aggregated_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(aggregated_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(aggregated_rows)

            print("\n=== Aggregated Results ===")

            for row in aggregated_rows:
                print(
                    f"Input={row['input_tokens']} | "
                    f"C={row['concurrency']} | "
                    f"Runs={row['successful_repeats']} | "
                    f"Throughput="
                    f"{row['throughput_tps_mean']:.2f} tok/s | "
                    f"TTFT="
                    f"{row['ttft_mean_ms_mean']:.2f} ms | "
                    f"TPOT="
                    f"{row['tpot_mean_ms_mean']:.2f} ms"
                )

            print(f"\nAggregated CSV: {aggregated_path}")

    # Return a nonzero exit code if any benchmark failed.
    # Useful for CI/CD automation.
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
