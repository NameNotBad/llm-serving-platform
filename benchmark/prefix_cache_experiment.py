
"""
Prefix Caching Benchmark for vLLM.

Scenarios:
    off_unique: Cache OFF, independent prompt bodies
    on_unique:  Cache ON, independent prompt bodies
    on_shared:  Cache ON, shared prefix + unique suffix

Commands:
    python benchmark/prefix_cache_experiment.py --mode off
    python benchmark/prefix_cache_experiment.py --mode on
    python benchmark/prefix_cache_experiment.py --mode shared
    python benchmark/prefix_cache_experiment.py --mode report

Existing OFF/Unique ON results are preserved when
running --mode shared.
"""

import argparse
import asyncio
import csv
import json
import math
import statistics
import time

from pathlib import Path

import httpx

from environment import collect_environment
from prompt_generator import generate_token_length_prompts


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

INPUT_TOKENS = 512
OUTPUT_TOKENS = 64
CONCURRENCY_LEVELS = [4, 16]
REQUESTS_PER_RUN = 128
REPEATS = 3

RESULTS_DIR = Path("results/prefix_cache_v1")

SCENARIOS = (
    "off_unique",
    "on_unique",
    "on_shared",
)


# ============================================================
# General utilities
# ============================================================

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def percentile(values, q):
    values = sorted(values)

    if not values:
        raise ValueError("Cannot calculate an empty percentile")

    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return values[low]

    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


# ============================================================
# Server verification
# ============================================================

def verify_server(expected_cache):
    environment = collect_environment()
    server = environment["server"]["running_process"]

    if not server.get("verified"):
        raise RuntimeError(f"Server not verified: {server}")

    if not server.get("port_process_verified"):
        raise RuntimeError("Server port ownership not verified")

    if server.get("prefix_caching") is not expected_cache:
        raise RuntimeError(
            "Prefix caching mismatch: "
            f"expected={expected_cache}, "
            f"actual={server.get('prefix_caching')}"
        )

    if server.get("model") != MODEL:
        raise RuntimeError("Unexpected model")

    if server.get("max_model_len") != 2048:
        raise RuntimeError("Unexpected max model length")

    if server.get("dtype") != "float16":
        raise RuntimeError("Unexpected dtype")

    return environment


# ============================================================
# Correct token counting
# ============================================================

def count_chat_tokens(tokenizer, prompt):
    """
    Count actual input token IDs, including the chat template.

    Important:
        len(BatchEncoding) may count mapping fields,
        not tokens. Always access ["input_ids"].
    """

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    token_ids = encoded["input_ids"]

    # Normally a flat list of token IDs.
    # Handle a single batched sequence defensively.
    if (
        len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = token_ids[0]

    return len(token_ids)


# ============================================================
# Shared-prefix prompt generation
# ============================================================

def generate_shared_prompts(
    tokenizer,
    count,
    target_tokens,
    repeat,
):
    """
    Generate prompts with a long identical prefix.

    Structure:
        [common reference document]
        [unique request identifier]
        [variable-length padding]

    Full chat length is checked for every prompt.
    """

    start = time.perf_counter()

    print(
        "  Generating shared-prefix prompts...",
        flush=True,
    )

    prefix = (
        f"Technical reference document {repeat:06d}.\n"
        "Use this common reference to answer the request.\n"
    )

    reference_line = (
        "Inference serving involves prefill, decode, "
        "continuous batching, attention, KV cache storage, "
        "GPU scheduling, and request processing.\n"
    )

    # Build approximately 70% of the total input as
    # an identical shared prefix.
    prefix_budget = int(target_tokens * 0.70)

    while True:
        candidate = prefix + reference_line

        if count_chat_tokens(
            tokenizer, candidate
        ) > prefix_budget:
            break

        prefix = candidate

    prefix_tokens = count_chat_tokens(
        tokenizer, prefix
    )

    print(
        f"  Shared prefix length: "
        f"{prefix_tokens} full-chat tokens",
        flush=True,
    )

    if prefix_tokens < target_tokens * 0.5:
        raise ValueError(
            "Shared prefix is unexpectedly short"
        )

    def make_prompt(request_id, padding_count):
        suffix = (
            f"\nRequest identifier {request_id:06d}.\n"
            "Explain the relevant inference behavior."
        )

        padding = (
            " additional serving context" * padding_count
        )

        return prefix + suffix + padding

    prompts = []
    token_counts = []

    # Binary-search padding individually because different
    # request identifiers can tokenize differently.
    for request_id in range(count):

        low = 0
        high = target_tokens

        best_prompt = None
        best_difference = float("inf")

        while low <= high:
            middle = (low + high) // 2

            candidate = make_prompt(
                request_id, middle
            )

            actual = count_chat_tokens(
                tokenizer, candidate
            )

            difference = abs(actual - target_tokens)

            if difference < best_difference:
                best_prompt = candidate
                best_difference = difference

            if actual == target_tokens:
                break

            if actual < target_tokens:
                low = middle + 1
            else:
                high = middle - 1

        # Make small corrections if padding granularity
        # cannot achieve an exact token count.
        if best_difference != 0:
            for suffix in (
                " a",
                " the",
                " test",
                " additional",
                " a test",
                " technical details",
            ):
                candidate = best_prompt + suffix
                actual = count_chat_tokens(
                    tokenizer, candidate
                )

                difference = abs(actual - target_tokens)

                if difference < best_difference:
                    best_prompt = candidate
                    best_difference = difference

                if difference == 0:
                    break

        actual = count_chat_tokens(
            tokenizer, best_prompt
        )

        if abs(actual - target_tokens) > 3:
            raise ValueError(
                f"Request {request_id}: "
                f"expected {target_tokens}, got {actual}"
            )

        prompts.append(best_prompt)
        token_counts.append(actual)

    elapsed = time.perf_counter() - start

    print(
        f"  Generated {len(prompts)} prompts "
        f"in {elapsed:.2f}s",
        flush=True,
    )

    print(
        f"  Token range: "
        f"{min(token_counts)}–{max(token_counts)}",
        flush=True,
    )

    return prompts


# ============================================================
# Streaming HTTP request
# ============================================================

async def stream_request(client, prompt, request_id):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    start = time.perf_counter()
    first_content = None
    last_content = None
    content_chunks = 0
    usage = None

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
                content = (
                    choice.get("delta", {})
                    .get("content")
                )

                if content:
                    now = time.perf_counter()

                    if first_content is None:
                        first_content = now

                    last_content = now
                    content_chunks += 1

    end = time.perf_counter()

    if first_content is None:
        raise RuntimeError(
            f"Request {request_id}: no content"
        )

    if usage is None:
        raise RuntimeError(
            f"Request {request_id}: missing usage"
        )

    input_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]

    tpot = None

    if output_tokens > 1:
        tpot = (
            (last_content - first_content)
            / (output_tokens - 1)
            * 1000
        )

    return {
        "request_id": request_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": (first_content - start) * 1000,
        "tpot_ms": tpot,
        "e2e_ms": (end - start) * 1000,
        "content_chunks": content_chunks,
    }


# ============================================================
# Benchmark execution
# ============================================================

async def benchmark_run(prompts, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    timeout = httpx.Timeout(
        timeout=180.0,
        connect=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:

        print(
            "  Sending warmup request...",
            flush=True,
        )

        # One request to warm the serving path and
        # potentially populate reusable KV blocks.
        await stream_request(client, prompts[0], -1)

        print(
            "  Warmup complete. "
            "Starting timed requests.",
            flush=True,
        )

        async def worker(request_id, prompt):
            async with semaphore:
                return await stream_request(
                    client, prompt, request_id
                )

        start = time.perf_counter()

        requests = await asyncio.gather(*[
            worker(i, prompt)
            for i, prompt in enumerate(prompts)
        ])

        duration = time.perf_counter() - start

    input_counts = [r["input_tokens"] for r in requests]
    output_counts = [r["output_tokens"] for r in requests]

    # Validate measured input sizes from the vLLM API.
    for value in input_counts:
        if abs(value - INPUT_TOKENS) > 3:
            raise ValueError(
                f"Unexpected server input length: {value}"
            )

    ttft = [r["ttft_ms"] for r in requests]
    e2e = [r["e2e_ms"] for r in requests]
    tpot = [
        r["tpot_ms"]
        for r in requests
        if r["tpot_ms"] is not None
    ]

    total_output = sum(output_counts)

    return {
        "summary": {
            "request_count": len(requests),
            "duration_s": duration,
            "total_output_tokens": total_output,
            "throughput_tps": total_output / duration,
            "ttft_mean_ms": statistics.mean(ttft),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "tpot_mean_ms": (
                statistics.mean(tpot) if tpot else None
            ),
            "e2e_mean_ms": statistics.mean(e2e),
            "e2e_p95_ms": percentile(e2e, 0.95),
            "input_tokens_mean": statistics.mean(input_counts),
            "output_tokens_mean": statistics.mean(output_counts),
        },
        "requests": requests,
    }


# ============================================================
# Scenario runner
# ============================================================

def run_scenario(scenario, environment):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    rows = []

    for concurrency in CONCURRENCY_LEVELS:
        for repeat in range(1, REPEATS + 1):

            print(
                f"\n{scenario} | "
                f"C={concurrency} | "
                f"Repeat={repeat}",
                flush=True,
            )

            generation_start = time.perf_counter()

            if scenario == "on_shared":
                prompts = generate_shared_prompts(
                    tokenizer=tokenizer,
                    count=REQUESTS_PER_RUN,
                    target_tokens=INPUT_TOKENS,
                    repeat=repeat + concurrency * 100,
                )
            else:
                prompts = generate_token_length_prompts(
                    count=REQUESTS_PER_RUN,
                    target_tokens=INPUT_TOKENS,
                    seed=100000 * repeat + concurrency * 1000,
                    model=MODEL,
                )

            generation_s = (
                time.perf_counter() - generation_start
            )

            report = asyncio.run(
                benchmark_run(prompts, concurrency)
            )

            summary = report["summary"]

            row = {
                "scenario": scenario,
                "concurrency": concurrency,
                "repeat": repeat,
                **summary,
            }

            rows.append(row)

            report_path = (
                RESULTS_DIR
                / scenario
                / f"c{concurrency}_r{repeat}.json"
            )

            save_json(
                report_path,
                {
                    "scenario": scenario,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "prompt_generation_s": generation_s,
                    "summary": summary,
                    "requests": report["requests"],
                },
            )

            print(
                "  SUCCESS | "
                f"Throughput={summary['throughput_tps']:.2f} tok/s | "
                f"TTFT={summary['ttft_mean_ms']:.2f} ms | "
                f"TPOT={summary['tpot_mean_ms']:.2f} ms",
                flush=True,
            )

    csv_path = RESULTS_DIR / f"{scenario}.csv"

    with csv_path.open(
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

    save_json(
        RESULTS_DIR / scenario / "environment.json",
        environment,
    )

    print(f"\nSaved: {csv_path}")


# ============================================================
# Comparison report
# ============================================================

def generate_report():
    metrics = (
        "throughput_tps",
        "ttft_mean_ms",
        "tpot_mean_ms",
        "e2e_p95_ms",
    )

    data = {}

    for scenario in SCENARIOS:
        path = RESULTS_DIR / f"{scenario}.csv"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing scenario results: {path}"
            )

        with path.open(
            newline="",
            encoding="utf-8",
        ) as file:
            data[scenario] = list(csv.DictReader(file))

    rows = []

    for concurrency in CONCURRENCY_LEVELS:
        for scenario in SCENARIOS:

            subset = [
                record
                for record in data[scenario]
                if int(record["concurrency"]) == concurrency
            ]

            if len(subset) != REPEATS:
                raise ValueError(
                    f"Incomplete scenario: {scenario}, "
                    f"C={concurrency}, runs={len(subset)}"
                )

            row = {
                "scenario": scenario,
                "concurrency": concurrency,
                "repeats": len(subset),
            }

            for metric in metrics:
                values = [
                    float(record[metric])
                    for record in subset
                ]

                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = (
                    statistics.stdev(values)
                    if len(values) > 1
                    else None
                )

            rows.append(row)

    output = RESULTS_DIR / "comparison.csv"

    with output.open(
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

    print("\n=== PREFIX CACHE COMPARISON ===")

    for row in rows:
        print(
            f"{row['scenario']:12s} "
            f"C={row['concurrency']:2d} | "
            f"Throughput={row['throughput_tps_mean']:.2f} | "
            f"TTFT={row['ttft_mean_ms_mean']:.2f} ms"
        )

    print(f"\nSaved: {output}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("off", "on", "shared", "report"),
        required=True,
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "report":
        generate_report()
        return

    environment = verify_server(
        expected_cache=(args.mode != "off")
    )

    if args.mode == "off":
        scenarios = ["off_unique"]
    elif args.mode == "on":
        scenarios = ["on_unique", "on_shared"]
    else:
        scenarios = ["on_shared"]

    for scenario in scenarios:
        run_scenario(scenario, environment)


if __name__ == "__main__":
    main()
