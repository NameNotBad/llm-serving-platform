
import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8080"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


async def get_status(client):
    response = await client.get(
        f"{BASE_URL}/gateway/status"
    )
    response.raise_for_status()
    return response.json()


async def send_request(client, request_id, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed explanation of "
                    "GPU memory management, CUDA memory "
                    "allocation, and KV cache management. "
                    f"Request ID: {request_id}."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    started = time.perf_counter()

    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
        ) as response:
            status_code = response.status_code

            queue_wait = response.headers.get(
                "X-Gateway-Queue-Wait-Ms"
            )

            error_type = None
            done_received = False
            error_body = []

            async for line in response.aiter_lines():
                if status_code == 200:
                    if line.strip() == "data: [DONE]":
                        done_received = True
                else:
                    error_body.append(line)

            if status_code != 200:
                try:
                    error_type = json.loads(
                        "".join(error_body)
                    ).get("error", {}).get("type")
                except (ValueError, TypeError):
                    error_type = "unknown"

            return {
                "request_id": request_id,
                "status_code": status_code,
                "error_type": error_type,
                "done_received": done_received,
                "queue_wait_ms": (
                    float(queue_wait)
                    if queue_wait is not None
                    else None
                ),
                "duration_s": (
                    time.perf_counter() - started
                ),
            }

    except Exception as exc:
        return {
            "request_id": request_id,
            "status_code": None,
            "error_type": type(exc).__name__,
            "done_received": False,
            "queue_wait_ms": None,
            "duration_s": (
                time.perf_counter() - started
            ),
        }


async def monitor_status(client, stop_event, interval):
    samples = []
    errors = []

    while not stop_event.is_set():
        try:
            status = await get_status(client)

            samples.append({
                "elapsed_s": time.perf_counter(),
                **status,
            })

        except Exception as exc:
            errors.append(str(exc))

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=interval,
            )
        except asyncio.TimeoutError:
            pass

    return samples, errors


async def run(args):
    limits = httpx.Limits(
        max_connections=args.requests + 10,
        max_keepalive_connections=args.requests + 10,
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0),
        limits=limits,
    ) as client:

        before = await get_status(client)

        stop_event = asyncio.Event()

        monitor_task = asyncio.create_task(
            monitor_status(
                client,
                stop_event,
                args.interval,
            )
        )

        try:
            tasks = [
                asyncio.create_task(
                    send_request(
                        client,
                        request_id=i,
                        max_tokens=args.max_tokens,
                    )
                )
                for i in range(args.requests)
            ]

            results = await asyncio.gather(*tasks)

        finally:
            stop_event.set()
            samples, monitor_errors = await monitor_task

        after = await get_status(client)

    max_inflight = max(
        (s["inflight"] for s in samples),
        default=0,
    )

    max_queued = max(
        (s["queued"] for s in samples),
        default=0,
    )

    max_allowed_inflight = after["max_inflight"]
    max_allowed_queue = after["max_queue"]

    status_counts = Counter(
        str(r["status_code"])
        for r in results
    )

    error_counts = Counter(
        r["error_type"]
        for r in results
        if r["error_type"] is not None
    )

    counter_deltas = {
        key: after[key] - before[key]
        for key in [
            "admitted_total",
            "rejected_total",
            "timeout_total",
        ]
    }

    invariants = {
        "inflight_within_limit": all(
            0 <= s["inflight"] <= max_allowed_inflight
            for s in samples
        ),
        "queue_within_limit": all(
            0 <= s["queued"] <= max_allowed_queue
            for s in samples
        ),
        "slots_released": (
            after["inflight"] == 0
            and after["queued"] == 0
        ),
        "all_success_streams_complete": all(
            r["done_received"]
            for r in results
            if r["status_code"] == 200
        ),
        "no_client_errors": all(
            r["status_code"] is not None
            for r in results
        ),
        "no_monitor_errors": not monitor_errors,
    }

    report = {
        "config": vars(args),
        "status_before": before,
        "status_after": after,
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "counter_deltas": counter_deltas,
        "max_observed_inflight": max_inflight,
        "max_observed_queued": max_queued,
        "samples_collected": len(samples),
        "monitor_errors": monitor_errors,
        "invariants": invariants,
        "samples": samples,
        "requests": results,
    }

    print("\n=== Gateway Overload Test ===")
    print(f"Requests: {args.requests}")
    print(f"Samples collected: {len(samples)}")
    print(f"HTTP status distribution: {dict(status_counts)}")
    print(f"Error types: {dict(error_counts)}")

    print("\n--- Observed Capacity ---")
    print(
        f"Max in-flight: {max_inflight} "
        f"/ {max_allowed_inflight}"
    )
    print(
        f"Max queued:    {max_queued} "
        f"/ {max_allowed_queue}"
    )

    print("\n--- Counter Deltas ---")
    for key, value in counter_deltas.items():
        print(f"{key}: +{value}")

    print("\n--- Final State ---")
    print(f"In-flight: {after['inflight']}")
    print(f"Queued:    {after['queued']}")

    print("\n--- Invariant Checks ---")
    for key, passed in invariants.items():
        print(
            f"{key}: {'PASS' if passed else 'FAIL'}"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = Path("results/gateway")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"overload_test_{timestamp}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--requests",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    if (
        args.requests <= 0
        or args.max_tokens <= 0
        or args.interval <= 0
    ):
        parser.error("All parameters must be positive")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
