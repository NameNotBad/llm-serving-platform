
"""
GPU profiling experiment for vLLM 0.32.0.

Profiles two workloads:
    prefill_heavy: 1024 input / 64 output
    decode_heavy:  128 input / 512 output

Requirements:
    - vLLM server started with --profiler-config
    - Prefix caching disabled
    - Existing benchmark/environment.py
    - Existing benchmark/prompt_generator.py

Usage:
    python benchmark/gpu_profile_experiment.py
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from environment import collect_environment
from prompt_generator import generate_token_length_prompts


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://127.0.0.1:8000"

RESULTS_DIR = Path("results/gpu_profiles")

CONCURRENCY = 8
REQUESTS_PER_WORKLOAD = 8

WORKLOADS = {
    "prefill_heavy": {
        "input_tokens": 1024,
        "output_tokens": 64,
    },
    "decode_heavy": {
        "input_tokens": 128,
        "output_tokens": 512,
    },
}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def validate_server() -> dict:
    """Verify the current server process."""

    environment = collect_environment()

    server = environment["server"]["running_process"]

    if not server.get("verified"):
        raise RuntimeError(
            "vLLM process verification failed"
        )

    if not server.get("port_process_verified"):
        raise RuntimeError(
            "vLLM port ownership verification failed"
        )

    if server.get("model") != MODEL:
        raise RuntimeError("Unexpected model")

    if server.get("prefix_caching") is not False:
        raise RuntimeError(
            "Prefix caching must be OFF"
        )

    if server.get("max_model_len") != 2048:
        raise RuntimeError(
            "Unexpected max_model_len"
        )

    if server.get("dtype") != "float16":
        raise RuntimeError(
            "Unexpected dtype"
        )

    # Verify that profiling was enabled at startup.
    args = server.get("startup_args", [])

    if "--profiler-config" not in args:
        raise RuntimeError(
            "Server was not started with --profiler-config"
        )

    return environment


async def profile_control(
    client: httpx.AsyncClient,
    action: str,
    prefix: str | None = None,
) -> None:
    """Start or stop the vLLM profiler."""

    if action not in ("start", "stop"):
        raise ValueError("Invalid profiling action")

    endpoint = f"{BASE_URL}/{action}_profile"

    kwargs = {}

    if action == "start" and prefix is not None:
        kwargs["json"] = {
            "profile_prefix": prefix,
        }

    response = await client.post(
        endpoint,
        timeout=300.0,
        **kwargs,
    )

    response.raise_for_status()

    print(
        f"  Profiler {action}: HTTP {response.status_code}",
        flush=True,
    )


async def generate_request(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: int,
    max_tokens: int,
) -> dict:
    """Execute a request during profiling."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()

    response = await client.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=180.0,
    )

    response.raise_for_status()

    elapsed = time.perf_counter() - started

    data = response.json()

    usage = data.get("usage") or {}
    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(
            f"Request {request_id}: missing choices"
        )

    if "prompt_tokens" not in usage:
        raise RuntimeError(
            f"Request {request_id}: missing prompt usage"
        )

    if "completion_tokens" not in usage:
        raise RuntimeError(
            f"Request {request_id}: missing completion usage"
        )

    return {
        "request_id": request_id,
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "finish_reason": choices[0].get("finish_reason"),
        "e2e_ms": elapsed * 1000,
    }


async def execute_workload(
    client: httpx.AsyncClient,
    prompts: list[str],
    max_output_tokens: int,
) -> list[dict]:
    """Execute requests with bounded concurrency."""

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def worker(index: int, prompt: str) -> dict:
        async with semaphore:
            return await generate_request(
                client,
                prompt,
                index,
                max_output_tokens,
            )

    return await asyncio.gather(*[
        worker(index, prompt)
        for index, prompt in enumerate(prompts)
    ])


async def run_workload(
    client: httpx.AsyncClient,
    name: str,
    config: dict,
) -> dict:
    """Profile one workload in its own sampling session."""

    input_tokens = config["input_tokens"]
    output_tokens = config["output_tokens"]

    print(
        f"\n=== {name} ===\n"
        f"Input={input_tokens}, "
        f"OutputMax={output_tokens}, "
        f"Concurrency={CONCURRENCY}",
        flush=True,
    )

    # Build prompts before starting the profiler.
    prompts = generate_token_length_prompts(
        count=REQUESTS_PER_WORKLOAD + 1,
        target_tokens=input_tokens,
        seed=2026 + input_tokens,
        model=MODEL,
    )

    # Warmup is excluded from the profiling window.
    print("  Warming up...", flush=True)

    await generate_request(
        client,
        prompts[-1],
        -1,
        output_tokens,
    )

    print("  Starting GPU profile...", flush=True)

    profile_started = False
    records = None

    # This identifies output files for this workload.
    profile_prefix = (
        f"{name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )

    try:
        await profile_control(
            client,
            "start",
            prefix=profile_prefix,
        )
        profile_started = True

        records = await execute_workload(
            client,
            prompts[:-1],
            output_tokens,
        )

    finally:
        # Always attempt to flush the profiler,
        # even if inference raises an exception.
        if profile_started:
            print("  Stopping GPU profile...", flush=True)

            await profile_control(client, "stop")

    assert records is not None

    # Validate actual workload sizes.
    for record in records:
        if record["input_tokens"] != input_tokens:
            raise ValueError(
                "Unexpected input length: "
                f"{record['input_tokens']}"
            )

        if not (
            1 <= record["output_tokens"] <= output_tokens
        ):
            raise ValueError(
                "Unexpected output length: "
                f"{record['output_tokens']}"
            )

    total_output = sum(
        record["output_tokens"]
        for record in records
    )

    mean_output = (
        total_output / len(records)
    )

    print(
        f"  Completed {len(records)} requests | "
        f"Mean output={mean_output:.1f} tokens",
        flush=True,
    )

    report = {
        "workload": name,
        "config": config,
        "concurrency": CONCURRENCY,
        "profile_prefix": profile_prefix,
        "request_count": len(records),
        "mean_output_tokens": mean_output,
        "requests": records,
    }

    save_json(
        RESULTS_DIR / f"{profile_prefix}.json",
        report,
    )

    return report


async def main_async() -> None:
    """Run the complete profiling experiment."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment = validate_server()

    save_json(
        RESULTS_DIR / "environment.json",
        environment,
    )

    timeout = httpx.Timeout(
        timeout=300.0,
        connect=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:

        # Check HTTP server health.
        health = await client.get(
            f"{BASE_URL}/health"
        )
        health.raise_for_status()

        reports = []

        for name, config in WORKLOADS.items():
            report = await run_workload(
                client,
                name,
                config,
            )
            reports.append(report)

    save_json(
        RESULTS_DIR / "profile_index.json",
        {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "reports": reports,
        },
    )

    print(
        "\nGPU profiling experiment completed.\n"
        f"Results directory: {RESULTS_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main_async())
