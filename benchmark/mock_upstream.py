
import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse


app = FastAPI(title="Mock Upstream")


@app.post("/v1/chat/completions")
async def chat_completions():
    async def generate():
        for i in range(100):
            chunk = {
                "id": "mock-deadline-test",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": f"Token-{i} "
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(chunk)
                + "\n\n"
            )

            # Continuous streaming: no long idle gap.
            await asyncio.sleep(0.5)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
