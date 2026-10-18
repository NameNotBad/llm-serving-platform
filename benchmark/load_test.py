
import argparse
import asyncio
import time

import httpx
import numpy as np

URL = "http://127.0.0.1:8000/generate"

PAYLOAD = {
    "prompt": "Explain KV cache in Transformer inference in two sentences.",
    "max_new_tokens": 64,
}


async def send_request(client, semaphore):
    async with semaphore:
        start = time.perf_counter()

        response = await client.post(URL, json=PAYLOAD)
        latency = time.perf_counter() - start

        response.raise_for_status()
        result = response.json()

        return {
            "latency": latency,
            "output_tokens": result["output_tokens"],
            "generation_ms": result["generation_ms"],
            "queue_ms": result["queue_ms"],
        }


async def benchmark(concurrency, total_requests):
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warm up the model
        warmup = await client.post(URL, json=PAYLOAD)
        warmup.raise_for_status()

        start = time.perf_counter()

        tasks = [
            send_request(client, semaphore)
            for _ in range(total_requests)
        ]

        results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start

    latencies = [r["latency"] for r in results]
    queue_times = [r["queue_ms"] for r in results]
    generation_times = [r["generation_ms"] for r in results]
    total_tokens = sum(r["output_tokens"] for r in results)

    print("\n=== Benchmark Results ===")
    print(f"Concurrency: {concurrency}")
    print(f"Total requests: {total_requests}")
    print(f"Total duration: {elapsed:.2f} s")
    print(f"Throughput: {total_tokens / elapsed:.2f} tokens/s")
    print(f"Requests/sec: {total_requests / elapsed:.2f}")
    print(f"P50 latency: {np.percentile(latencies, 50):.3f} s")
    print(f"P95 latency: {np.percentile(latencies, 95):.3f} s")
    print(f"Mean latency: {np.mean(latencies):.3f} s")
    print(f"Mean queue wait: {np.mean(queue_times):.2f} ms")
    print(f"P95 queue wait: {np.percentile(queue_times, 95):.2f} ms")
    print(f"Mean generation: {np.mean(generation_times):.2f} ms")

    print("\n=== Per-Request Breakdown ===")
    for i, r in enumerate(results, 1):
        print(
            f"Request {i:02d} | "
            f"E2E: {r['latency'] * 1000:.0f} ms | "
            f"Queue: {r['queue_ms']:.0f} ms | "
            f"Generation: {r['generation_ms']:.0f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")

    asyncio.run(benchmark(args.concurrency, args.requests))
