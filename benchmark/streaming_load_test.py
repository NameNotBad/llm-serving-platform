
import argparse
import asyncio
import json
import time

import httpx
import numpy as np


URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "Explain KV cache in Transformer inference.",
    "Explain the difference between prefill and decode.",
    "What is continuous batching in LLM serving?",
    "How does PagedAttention manage GPU memory?",
    "What causes high time-to-first-token latency?",
    "Explain the throughput and latency trade-off.",
    "How does tensor parallelism work?",
    "What is the difference between quantization and pruning?",
]


def make_payload(prompt, max_tokens):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def send_request(client, semaphore, request_id, max_tokens):
    prompt = PROMPTS[request_id % len(PROMPTS)]
    payload = make_payload(prompt, max_tokens)

    async with semaphore:
        start = time.perf_counter()
        first_token_time = None
        last_token_time = None
        output_tokens = None
        content_chunks = 0

        async with client.stream(
            "POST", URL, json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                event = json.loads(data)

                if event.get("usage") is not None:
                    output_tokens = event["usage"]["completion_tokens"]

                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")

                    if content:
                        now = time.perf_counter()

                        if first_token_time is None:
                            first_token_time = now

                        last_token_time = now
                        content_chunks += 1

        end = time.perf_counter()

        if first_token_time is None:
            raise RuntimeError(
                f"Request {request_id} returned no content"
            )

        if output_tokens is None:
            raise RuntimeError(
                f"Request {request_id} returned no usage"
            )

        ttft = first_token_time - start
        e2e = end - start

        tpot = None
        if output_tokens > 1:
            tpot = (
                last_token_time - first_token_time
            ) / (output_tokens - 1)

        return {
            "request_id": request_id,
            "ttft": ttft,
            "tpot": tpot,
            "e2e": e2e,
            "output_tokens": output_tokens,
            "content_chunks": content_chunks,
        }


async def benchmark(concurrency, total_requests, max_tokens):
    semaphore = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(
        timeout=120.0,
        limits=limits,
    ) as client:

        # Warm up the model before measuring.
        await send_request(
            client,
            semaphore,
            request_id=0,
            max_tokens=max_tokens,
        )

        start = time.perf_counter()

        tasks = [
            send_request(
                client,
                semaphore,
                request_id=i,
                max_tokens=max_tokens,
            )
            for i in range(total_requests)
        ]

        results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start

    ttfts = np.array([r["ttft"] for r in results])
    e2es = np.array([r["e2e"] for r in results])
    tpots = np.array([
        r["tpot"]
        for r in results
        if r["tpot"] is not None
    ])

    total_tokens = sum(r["output_tokens"] for r in results)

    print("\n=== Concurrent Streaming Benchmark ===")
    print(f"Concurrency: {concurrency}")
    print(f"Requests: {total_requests}")
    print(f"Total output tokens: {total_tokens}")
    print(f"Total duration: {elapsed:.2f} s")
    print(f"Throughput: {total_tokens / elapsed:.2f} tokens/s")
    print(f"Requests/sec: {total_requests / elapsed:.2f}")

    print("\n--- TTFT ---")
    print(f"Mean: {np.mean(ttfts) * 1000:.2f} ms")
    print(f"P50:  {np.percentile(ttfts, 50) * 1000:.2f} ms")
    print(f"P95:  {np.percentile(ttfts, 95) * 1000:.2f} ms")

    if len(tpots) > 0:
        print("\n--- TPOT ---")
        print(f"Mean: {np.mean(tpots) * 1000:.2f} ms/token")
        print(f"P95:  {np.percentile(tpots, 95) * 1000:.2f} ms/token")

    print("\n--- E2E ---")
    print(f"Mean: {np.mean(e2es) * 1000:.2f} ms")
    print(f"P50:  {np.percentile(e2es, 50) * 1000:.2f} ms")
    print(f"P95:  {np.percentile(e2es, 95) * 1000:.2f} ms")

    print("\n--- Per-Request ---")
    for r in results:
        tpot_ms = (
            f"{r['tpot'] * 1000:.2f}"
            if r["tpot"] is not None
            else "N/A"
        )

        print(
            f"R{r['request_id'] + 1:02d} | "
            f"TTFT={r['ttft'] * 1000:.1f}ms | "
            f"TPOT={tpot_ms}ms | "
            f"E2E={r['e2e'] * 1000:.1f}ms | "
            f"Tokens={r['output_tokens']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concurrency", type=int, default=1
    )
    parser.add_argument(
        "--requests", type=int, default=8
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64
    )

    args = parser.parse_args()

    if (
        args.concurrency < 1
        or args.requests < 1
        or args.max_tokens < 1
    ):
        parser.error("All parameters must be positive")

    asyncio.run(
        benchmark(
            args.concurrency,
            args.requests,
            args.max_tokens,
        )
    )
