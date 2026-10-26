
"""
Prefill vs Decode Benchmark for vLLM.

Workloads:
    short_short: 128 input, 64 max output
    long_short:  1024 input, 64 max output
    short_long:  128 input, 512 max output
    long_long:   1024 input, 512 max output

Runs:
    4 workloads x 4 concurrency levels x 3 repeats
    = 48 benchmark runs

Outputs:
    environment.json
    config.json
    summary.csv
    aggregated.csv
    analysis.json
    runs/*.json

Usage:
    python benchmark/prefill_decode_experiment.py
    python benchmark/prefill_decode_experiment.py --analyze DIR

Run from the project root.
"""

import argparse
import asyncio
import csv
import json
import math
import statistics
import time

from datetime import datetime, timezone
from pathlib import Path

import httpx

from environment import collect_environment
from prompt_generator import generate_token_length_prompts


# ============================================================
# 1. Configuration
# ============================================================

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

WORKLOADS = {
    "short_short": (128, 64),
    "long_short": (1024, 64),
    "short_long": (128, 512),
    "long_long": (1024, 512),
}

CONCURRENCY_LEVELS = [1, 4, 8, 16]

REQUESTS_PER_RUN = 128
REPEATS = 3

EXPECTED_PREFIX_CACHING = False
EXPECTED_MAX_MODEL_LEN = 2048

RESULTS_ROOT = Path("results")


# ============================================================
# 2. General utilities
# ============================================================

def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

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


def percentile(values: list[float], q: float) -> float:
    """Calculate an interpolated percentile."""

    if not values:
        raise ValueError("Cannot calculate empty percentile")

    ordered = sorted(values)

    position = (len(ordered) - 1) * q

    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return ordered[low]

    fraction = position - low

    return (
        ordered[low] * (1 - fraction)
        + ordered[high] * fraction
    )


def safe_mean(values: list[float]):
    return statistics.mean(values) if values else None


# ============================================================
# 3. Server configuration validation
# ============================================================

def validate_server() -> dict:
    """
    Verify that the running vLLM server matches
    the experimental baseline.
    """

    environment = collect_environment()

    server = environment["server"]["running_process"]

    if not server.get("verified"):
        raise RuntimeError("Running vLLM server not verified")

    if not server.get("port_process_verified"):
        raise RuntimeError("vLLM port ownership not verified")

    if server.get("model") != MODEL:
        raise RuntimeError("Unexpected model")

    if server.get("prefix_caching") is not False:
        raise RuntimeError(
            "Prefill/Decode baseline requires "
            "Prefix Caching OFF"
        )

    if server.get("max_model_len") != EXPECTED_MAX_MODEL_LEN:
        raise RuntimeError("Unexpected maximum context length")

    if server.get("dtype") != "float16":
        raise RuntimeError("Unexpected model dtype")

    if (
        server.get("host") != "127.0.0.1"
        or server.get("port") != 8000
    ):
        raise RuntimeError("Unexpected server endpoint")

    return environment


# ============================================================
# 4. One streaming request
# ============================================================

async def stream_request(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: int,
    max_output_tokens: int,
) -> dict:
    """
    Measure one client-observed streaming request.

    TTFT:
        Time from request initiation to first content.

    TPOT:
        Time between first and last content divided by
        completion_tokens - 1.

    E2E:
        Time from request initiation until stream ends.
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    start = time.perf_counter()

    first_content_time = None
    last_content_time = None

    content_chunks = 0
    usage = None
    finish_reason = None

    async with client.stream(
        "POST",
        ENDPOINT,
        json=payload,
    ) as response:

        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                continue

            event = json.loads(data)

            if event.get("usage") is not None:
                usage = event["usage"]

            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

                delta = choice.get("delta") or {}
                content = delta.get("content")

                if content:
                    now = time.perf_counter()

                    if first_content_time is None:
                        first_content_time = now

                    last_content_time = now
                    content_chunks += 1

    end = time.perf_counter()

    if first_content_time is None:
        raise RuntimeError(
            f"Request {request_id}: no streamed content"
        )

    if usage is None:
        raise RuntimeError(
            f"Request {request_id}: missing token usage"
        )

    input_tokens = int(usage["prompt_tokens"])
    output_tokens = int(usage["completion_tokens"])

    if output_tokens <= 0:
        raise RuntimeError(
            f"Request {request_id}: zero output tokens"
        )

    if output_tokens > max_output_tokens:
        raise RuntimeError(
            f"Request {request_id}: output exceeds max_tokens"
        )

    ttft_ms = (
        first_content_time - start
    ) * 1000

    e2e_ms = (end - start) * 1000

    tpot_ms = None

    if output_tokens > 1:
        tpot_ms = (
            (last_content_time - first_content_time)
            / (output_tokens - 1)
            * 1000
        )

    return {
        "request_id": request_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "max_output_tokens": max_output_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "content_chunks": content_chunks,
    }


# ============================================================
# 5. Benchmark one workload
# ============================================================

async def benchmark_run(
    prompts: list[str],
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """
    Run one closed-loop benchmark.

    The number of in-flight requests cannot exceed
    the configured concurrency.

    The warmup prompt is excluded from measurement.
    """

    if len(prompts) != REQUESTS_PER_RUN + 1:
        raise ValueError("Unexpected prompt count")

    semaphore = asyncio.Semaphore(concurrency)

    timeout = httpx.Timeout(
        timeout=300.0,
        connect=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:

        # Dedicated warmup request.
        await stream_request(
            client,
            prompts[-1],
            request_id=-1,
            max_output_tokens=output_tokens,
        )

        async def worker(index, prompt):
            async with semaphore:
                return await stream_request(
                    client,
                    prompt,
                    request_id=index,
                    max_output_tokens=output_tokens,
                )

        start = time.perf_counter()

        requests = await asyncio.gather(*[
            worker(i, prompt)
            for i, prompt in enumerate(prompts[:-1])
        ])

        duration = time.perf_counter() - start

    # --------------------------------------------------------
    # Validate every request
    # --------------------------------------------------------

    for request in requests:
        if request["input_tokens"] != input_tokens:
            raise ValueError(
                "Input token mismatch: "
                f"expected={input_tokens}, "
                f"actual={request['input_tokens']}"
            )

    total_input = sum(
        r["input_tokens"] for r in requests
    )

    total_output = sum(
        r["output_tokens"] for r in requests
    )

    ttft = [r["ttft_ms"] for r in requests]
    e2e = [r["e2e_ms"] for r in requests]

    tpot = [
        r["tpot_ms"]
        for r in requests
        if r["tpot_ms"] is not None
    ]

    actual_outputs = [
        r["output_tokens"]
        for r in requests
    ]

    max_token_reached = sum(
        r["output_tokens"] == output_tokens
        for r in requests
    )

    summary = {
        "request_count": len(requests),

        "duration_s": duration,

        "total_input_tokens": total_input,
        "total_output_tokens": total_output,

        "output_throughput_tps": (
            total_output / duration
        ),

        "input_throughput_tps": (
            total_input / duration
        ),

        "requests_per_sec": (
            len(requests) / duration
        ),

        "ttft_mean_ms": safe_mean(ttft),
        "ttft_p95_ms": percentile(ttft, 0.95),

        "tpot_mean_ms": safe_mean(tpot),
        "tpot_p95_ms": (
            percentile(tpot, 0.95)
            if tpot else None
        ),

        "e2e_mean_ms": safe_mean(e2e),
        "e2e_p95_ms": percentile(e2e, 0.95),

        "actual_output_mean": safe_mean(actual_outputs),
        "actual_output_min": min(actual_outputs),
        "actual_output_max": max(actual_outputs),

        "max_tokens_reached_fraction": (
            max_token_reached / len(requests)
        ),
    }

    return {
        "summary": summary,
        "requests": requests,
    }


# ============================================================
# 6. Aggregate repeated runs
# ============================================================

METRICS = [
    "output_throughput_tps",
    "input_throughput_tps",
    "requests_per_sec",
    "ttft_mean_ms",
    "ttft_p95_ms",
    "tpot_mean_ms",
    "tpot_p95_ms",
    "e2e_mean_ms",
    "e2e_p95_ms",
    "actual_output_mean",
    "max_tokens_reached_fraction",
]


def aggregate_results(rows: list[dict]) -> list[dict]:
    """
    Aggregate repeated benchmark runs using
    arithmetic mean and sample standard deviation.
    """

    groups = {}

    for row in rows:
        key = (
            row["workload"],
            int(row["input_tokens"]),
            int(row["max_output_tokens"]),
            int(row["concurrency"]),
        )

        groups.setdefault(key, []).append(row)

    aggregated = []

    for key, group in sorted(groups.items()):

        workload, input_tokens, output_tokens, concurrency = key

        result = {
            "workload": workload,
            "input_tokens": input_tokens,
            "max_output_tokens": output_tokens,
            "concurrency": concurrency,
            "repeats": len(group),
        }

        for metric in METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
                and row.get(metric) != ""
            ]

            result[f"{metric}_mean"] = (
                statistics.mean(values)
                if values else None
            )

            result[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else None
            )

        aggregated.append(result)

    return aggregated


# ============================================================
# 7. Automatic analysis
# ============================================================

def build_analysis(aggregated: list[dict]) -> dict:
    """
    Organize benchmark results and derive simple
    within-workload concurrency scaling ratios.
    """

    workloads = {}

    for row in aggregated:
        workloads.setdefault(
            row["workload"], []
        ).append(row)

    analysis = {
        "workloads": workloads,
        "scaling": {},
        "notes": [
            "The load generator uses fixed concurrency.",
            "This is a closed-loop benchmark.",
            "TTFT/TPOT are client-observed metrics.",
            "Input throughput includes all input tokens.",
            "Output throughput counts generated tokens only.",
            "max_tokens is an upper bound, not guaranteed output.",
            "P95 means are averaged across benchmark repetitions.",
            "No GPU profiler data was collected.",
        ],
    }

    for workload, rows in workloads.items():

        ordered = sorted(
            rows,
            key=lambda r: r["concurrency"],
        )

        baseline = next(
            (
                r for r in ordered
                if r["concurrency"] == 1
            ),
            None,
        )

        if baseline is None:
            continue

        baseline_tps = baseline[
            "output_throughput_tps_mean"
        ]

        scaling = []

        for row in ordered:
            throughput = row[
                "output_throughput_tps_mean"
            ]

            speedup = throughput / baseline_tps

            scaling.append({
                "concurrency": row["concurrency"],
                "throughput_speedup": speedup,
                "scaling_efficiency": (
                    speedup / row["concurrency"]
                ),
            })

        analysis["scaling"][workload] = scaling

    return analysis


# ============================================================
# 8. Analyze an existing results directory
# ============================================================

def analyze_directory(output_dir: Path) -> None:

    summary_path = output_dir / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    with summary_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No benchmark results found")

    aggregated = aggregate_results(rows)

    save_csv(
        output_dir / "aggregated.csv",
        aggregated,
    )

    analysis = build_analysis(aggregated)

    save_json(
        output_dir / "analysis.json",
        analysis,
    )

    print("\n=== AGGREGATED RESULTS ===")

    for row in aggregated:
        throughput = row[
            "output_throughput_tps_mean"
        ]

        ttft = row["ttft_mean_ms_mean"]
        tpot = row["tpot_mean_ms_mean"]

        output_mean = row[
            "actual_output_mean_mean"
        ]

        print(
            f"{row['workload']:12s} "
            f"C={row['concurrency']:2d} | "
            f"Throughput={throughput:8.2f} tok/s | "
            f"TTFT={ttft:7.2f} ms | "
            f"TPOT={tpot:7.2f} ms | "
            f"Output={output_mean:.1f}"
        )

    print(f"\nReports saved to: {output_dir}")


# ============================================================
# 9. Full benchmark suite
# ============================================================

def run_experiment() -> None:

    environment = validate_server()

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_dir = (
        RESULTS_ROOT
        / f"prefill_decode_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "model": MODEL,
        "workloads": WORKLOADS,
        "concurrency": CONCURRENCY_LEVELS,
        "requests_per_run": REQUESTS_PER_RUN,
        "repeats": REPEATS,
        "prefix_caching": EXPECTED_PREFIX_CACHING,
    }

    save_json(output_dir / "config.json", config)
    save_json(output_dir / "environment.json", environment)

    total_runs = (
        len(WORKLOADS)
        * len(CONCURRENCY_LEVELS)
        * REPEATS
    )

    print(f"\nStarting {total_runs} benchmark runs")
    print(f"Results: {output_dir}", flush=True)

    rows = []
    completed = 0

    # Keep the same deterministic prompt set for
    # repeated experiments of each input length.
    prompt_cache = {}

    for workload, (
        input_tokens,
        max_output_tokens,
    ) in WORKLOADS.items():

        if input_tokens not in prompt_cache:

            print(
                f"\nGenerating {input_tokens}-token prompts...",
                flush=True,
            )

            prompt_cache[input_tokens] = (
                generate_token_length_prompts(
                    count=REQUESTS_PER_RUN + 1,
                    target_tokens=input_tokens,
                    seed=42 + input_tokens,
                    model=MODEL,
                )
            )

        prompts = prompt_cache[input_tokens]

        for concurrency in CONCURRENCY_LEVELS:

            for repeat in range(1, REPEATS + 1):

                completed += 1

                print(
                    f"\n[{completed}/{total_runs}] "
                    f"{workload} | "
                    f"Input={input_tokens} | "
                    f"OutputMax={max_output_tokens} | "
                    f"C={concurrency} | "
                    f"Repeat={repeat}",
                    flush=True,
                )

                report = asyncio.run(
                    benchmark_run(
                        prompts=prompts,
                        concurrency=concurrency,
                        input_tokens=input_tokens,
                        output_tokens=max_output_tokens,
                    )
                )

                summary = report["summary"]

                row = {
                    "workload": workload,
                    "input_tokens": input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **summary,
                }

                rows.append(row)

                filename = (
                    f"{workload}_c{concurrency}_r{repeat}.json"
                )

                save_json(
                    output_dir / "runs" / filename,
                    {
                        "config": {
                            "workload": workload,
                            "input_tokens": input_tokens,
                            "max_output_tokens": max_output_tokens,
                            "concurrency": concurrency,
                            "repeat": repeat,
                        },
                        **report,
                    },
                )

                # Save progress after every completed run.
                save_csv(
                    output_dir / "summary.csv",
                    rows,
                )

                print(
                    "  SUCCESS | "
                    f"Throughput="
                    f"{summary['output_throughput_tps']:.2f} tok/s | "
                    f"TTFT={summary['ttft_mean_ms']:.2f} ms | "
                    f"TPOT={summary['tpot_mean_ms']:.2f} ms | "
                    f"Mean Output="
                    f"{summary['actual_output_mean']:.1f}",
                    flush=True,
                )

    print(
        f"\nCompleted {completed}/{total_runs} runs"
    )

    analyze_directory(output_dir)


# ============================================================
# 10. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--analyze",
        type=Path,
        default=None,
        help="Reanalyze a previous experiment directory",
    )

    args = parser.parse_args()

    if args.analyze is not None:
        analyze_directory(args.analyze)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
