
import asyncio
import json
import time

import httpx


BASE_URL = "http://127.0.0.1:8080"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


async def status(client):
    response = await client.get(
        f"{BASE_URL}/gateway/status"
    )
    response.raise_for_status()
    return response.json()


async def cancel_stream(client, request_id):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a very long and detailed tutorial "
                    "about CUDA memory allocation, GPU kernels, "
                    "KV cache, and inference optimization. "
                    f"Request ID: {request_id}."
                ),
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "stream": True,
    }

    started = time.perf_counter()
    content_received = False

    async with client.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
    ) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            event = json.loads(data)

            for choice in event.get("choices", []):
                text = choice.get(
                    "delta", {}
                ).get("content")

                if text:
                    content_received = True
                    break

            if content_received:
                # Exit before generation is complete.
                break

    return {
        "request_id": request_id,
        "received_content": content_received,
        "duration_s": time.perf_counter() - started,
    }


async def main():
    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=20,
        ),
    ) as client:
        before = await status(client)

        print("Before:", before)

        results = await asyncio.gather(
            *(cancel_stream(client, i) for i in range(8)),
            return_exceptions=True,
        )

        print("\nCancelled streams:")
        for result in results:
            print(result)

        print("\nChecking slot cleanup...")

        for attempt in range(20):
            current = await status(client)

            print(
                f"Check {attempt + 1:02d}: "
                f"inflight={current['inflight']}, "
                f"queued={current['queued']}"
            )

            if (
                current["inflight"] == 0
                and current["queued"] == 0
            ):
                print("\nPASS: All slots released")
                return

            await asyncio.sleep(0.25)

        print("\nFAIL: Slots not released within 5 seconds")


if __name__ == "__main__":
    asyncio.run(main())
