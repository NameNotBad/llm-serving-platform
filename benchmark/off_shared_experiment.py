
"""
Run the missing OFF + Shared prefix control group.

Reuse the existing benchmark implementation and
shared-prefix generator to keep experimental settings
consistent with ON + Shared.

Usage:
    python benchmark/off_shared_experiment.py
"""

import asyncio
import csv
import time

from pathlib import Path

from transformers import AutoTokenizer

from environment import collect_environment

from prefix_cache_experiment import (
    MODEL,
    INPUT_TOKENS,
    REQUESTS_PER_RUN,
    CONCURRENCY_LEVELS,
    REPEATS,
    RESULTS_DIR,
    benchmark_run,
    generate_shared_prompts,
    save_json,
)


def main():
    # Confirm that the currently running server has
    # explicitly disabled prefix caching.
    environment = collect_environment()
    server = environment["server"]["running_process"]

    if not server.get("verified"):
        raise RuntimeError("vLLM server is not verified")

    if not server.get("port_process_verified"):
        raise RuntimeError("Server port ownership not verified")

    if server.get("prefix_caching") is not False:
        raise RuntimeError(
            "This experiment requires Prefix Caching OFF"
        )

    if (
        server.get("model") != MODEL
        or server.get("dtype") != "float16"
        or server.get("max_model_len") != 2048
    ):
        raise RuntimeError(
            "Server configuration does not match baseline"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    scenario = "off_shared"
    rows = []

    for concurrency in CONCURRENCY_LEVELS:
        for repeat in range(1, REPEATS + 1):

            print(
                f"\n{scenario} | "
                f"C={concurrency} | Repeat={repeat}",
                flush=True,
            )

            start = time.perf_counter()

            # Identical generation procedure and seed
            # as the existing ON + Shared experiment.
            prompts = generate_shared_prompts(
                tokenizer=tokenizer,
                count=REQUESTS_PER_RUN,
                target_tokens=INPUT_TOKENS,
                repeat=repeat + concurrency * 100,
            )

            generation_s = time.perf_counter() - start

            report = asyncio.run(
                benchmark_run(prompts, concurrency)
            )

            summary = report["summary"]

            rows.append({
                "scenario": scenario,
                "concurrency": concurrency,
                "repeat": repeat,
                **summary,
            })

            save_json(
                RESULTS_DIR
                / scenario
                / f"c{concurrency}_r{repeat}.json",
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
                f"  SUCCESS | "
                f"Throughput="
                f"{summary['throughput_tps']:.2f} tok/s | "
                f"TTFT={summary['ttft_mean_ms']:.2f} ms",
                flush=True,
            )

    csv_path = RESULTS_DIR / f"{scenario}.csv"
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print(f"\nCompleted {len(rows)} runs")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
