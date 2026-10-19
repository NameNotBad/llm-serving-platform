
import argparse
import asyncio
import time

import httpx
import numpy as np

PROMPT = "Explain KV cache in Transformer inference in two sentences."

HF_URL = "http://127.0.0.1:8000/generate"
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"


def build_request(backend):
    if backend == "hf":
        return HF_URL, {
            "prompt": PROMPT,
            "max_new_tokens": 64,
        }

    return VLLM_URL, {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 64,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "top_k": -1,
    }


async def send_request(client, semaphore, backend):
    url, payload = build_request(backend)

    async with semaphore:
        start = time.perf_counter()

        response = await client.post(url, json=payload)
        latency = time.perf_counter() - start

        response.raise_for_status()
        result = response.json()

        if backend == "hf":
            output_tokens = result["output_tokens"]
            queue_ms = result["queue_ms"]
            generation_ms = result["generation_ms"]
        else:
            output_tokens = result["usage"]["completion_tokens"]
            queue_ms = None
            generation_ms = None

            content = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0]["finish_reason"]

            if output_tokens != 40:
                print(
                    f"\n[DEBUG] Output tokens: {output_tokens}\n"
                    f"Finish reason: {finish_reason}\n"
                    f"Content: {content}\n"
                )

        return {
            "latency": latency,
            "output_tokens": output_tokens,
            "queue_ms": queue_ms,
            "generation_ms": generation_ms,
        }


async def benchmark(backend, concurrency, total_requests):
    semaphore = asyncio.Semaphore(concurrency)
    url, payload = build_request(backend)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warmup is excluded from the measurements.
        warmup = await client.post(url, json=payload)
        warmup.raise_for_status()

        start = time.perf_counter()

        tasks = [
            send_request(client, semaphore, backend)
            for _ in range(total_requests)
        ]

        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    latencies = [r["latency"] for r in results]
    total_tokens = sum(r["output_tokens"] for r in results)

    print("\n=== Benchmark Results ===")
    print(f"Backend: {backend}")
    print(f"Concurrency: {concurrency}")
    print(f"Total requests: {total_requests}")
    print(f"Total output tokens: {total_tokens}")
    print(f"Total duration: {elapsed:.2f} s")
    print(f"Throughput: {total_tokens / elapsed:.2f} tokens/s")
    print(f"Requests/sec: {total_requests / elapsed:.2f}")
    print(f"P50 latency: {np.percentile(latencies, 50):.3f} s")
    print(f"P95 latency: {np.percentile(latencies, 95):.3f} s")
    print(f"Mean latency: {np.mean(latencies):.3f} s")

    if backend == "hf":
        queue_times = [r["queue_ms"] for r in results]
        generation_times = [r["generation_ms"] for r in results]

        print(f"Mean queue wait: {np.mean(queue_times):.2f} ms")
        print(f"P95 queue wait: {np.percentile(queue_times, 95):.2f} ms")
        print(f"Mean generation: {np.mean(generation_times):.2f} ms")

    print("\n=== Per-Request Breakdown ===")
    for i, r in enumerate(results, 1):
        line = (
            f"Request {i:02d} | "
            f"E2E: {r['latency'] * 1000:.0f} ms | "
            f"Output tokens: {r['output_tokens']}"
        )

        if backend == "hf":
            line += (
                f" | Queue: {r['queue_ms']:.0f} ms"
                f" | Generation: {r['generation_ms']:.0f} ms"
            )

        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default="hf",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")

    asyncio.run(benchmark(args.backend, args.concurrency, args.requests))
