
import asyncio
import json
import time

import httpx


URL = "http://127.0.0.1:8000/v1/chat/completions"

PAYLOAD = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Explain KV cache in Transformer inference.",
        }
    ],
    "max_tokens": 64,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}


async def main():
    start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    output_tokens = None
    chunks = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", URL, json=PAYLOAD) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                event = json.loads(data)
                usage = event.get("usage")

                if usage is not None:
                    output_tokens = usage["completion_tokens"]

                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")

                    if content:
                        now = time.perf_counter()

                        if first_token_time is None:
                            first_token_time = now

                        last_token_time = now
                        chunks += 1

    end = time.perf_counter()

    if first_token_time is None:
        raise RuntimeError("No output tokens received")

    ttft = first_token_time - start
    e2e = end - start

    if output_tokens is not None and output_tokens > 1:
        tpot = (
            last_token_time - first_token_time
        ) / (output_tokens - 1)
    else:
        tpot = None

    print("\n=== Streaming Metrics ===")
    print(f"TTFT: {ttft * 1000:.2f} ms")
    print(f"E2E: {e2e * 1000:.2f} ms")
    print(f"Output tokens: {output_tokens}")
    print(f"Content chunks: {chunks}")

    if tpot is not None:
        print(f"Average TPOT: {tpot * 1000:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
