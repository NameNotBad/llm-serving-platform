
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (
    JSONResponse,
    Response,
    StreamingResponse,
)
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.background import BackgroundTask

from gateway.metrics import (
    INFLIGHT_REQUESTS,
    QUEUED_REQUESTS,
    record_queue_wait,
    record_request,
    record_timeout,
    render_metrics,
)


# ============================================================
# Configuration
# ============================================================

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

MAX_INFLIGHT_REQUESTS = int(
    os.getenv("MAX_INFLIGHT_REQUESTS", "8")
)

MAX_QUEUE_SIZE = int(
    os.getenv("MAX_QUEUE_SIZE", "16")
)

QUEUE_TIMEOUT_SEC = float(
    os.getenv("QUEUE_TIMEOUT_SEC", "5")
)

UPSTREAM_HEADER_TIMEOUT_SEC = float(
    os.getenv("UPSTREAM_HEADER_TIMEOUT_SEC", "15")
)

STREAM_IDLE_TIMEOUT_SEC = float(
    os.getenv("STREAM_IDLE_TIMEOUT_SEC", "15")
)

REQUEST_DEADLINE_SEC = float(
    os.getenv("REQUEST_DEADLINE_SEC", "120")
)

if MAX_INFLIGHT_REQUESTS < 1:
    raise ValueError("MAX_INFLIGHT_REQUESTS must be >= 1")

if MAX_QUEUE_SIZE < 0:
    raise ValueError("MAX_QUEUE_SIZE must be >= 0")

for name, value in [
    ("QUEUE_TIMEOUT_SEC", QUEUE_TIMEOUT_SEC),
    (
        "UPSTREAM_HEADER_TIMEOUT_SEC",
        UPSTREAM_HEADER_TIMEOUT_SEC,
    ),
    ("STREAM_IDLE_TIMEOUT_SEC", STREAM_IDLE_TIMEOUT_SEC),
    ("REQUEST_DEADLINE_SEC", REQUEST_DEADLINE_SEC),
]:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


# ============================================================
# Request Metrics Lifecycle
# ============================================================

class RequestMetrics:
    """
    Records exactly one final result per request.

    Final duration includes streaming time.
    """

    def __init__(self):
        self.started = time.perf_counter()
        self.finished = False

    def finish(self, result):
        if self.finished:
            return

        self.finished = True

        duration = (
            time.perf_counter() - self.started
        )

        record_request(result, duration)


# ============================================================
# Admission Controller
# ============================================================

class AdmissionController:
    def __init__(self, max_inflight, max_queue):
        self.max_inflight = max_inflight
        self.max_queue = max_queue

        self.inflight = 0
        self.queued = 0

        self.admitted_total = 0
        self.rejected_total = 0
        self.timeout_total = 0

        self.header_timeout_total = 0
        self.stream_timeout_total = 0
        self.stream_error_total = 0
        self.request_deadline_total = 0

        self.condition = asyncio.Condition()

        INFLIGHT_REQUESTS.set(0)
        QUEUED_REQUESTS.set(0)

    async def acquire(self, timeout):
        started = time.perf_counter()

        async with self.condition:
            if (
                self.inflight < self.max_inflight
                and self.queued == 0
            ):
                self.inflight += 1
                self.admitted_total += 1

                INFLIGHT_REQUESTS.set(
                    self.inflight
                )

                return "admitted", 0.0

            if self.queued >= self.max_queue:
                self.rejected_total += 1
                return "full", 0.0

            self.queued += 1
            QUEUED_REQUESTS.set(self.queued)

            try:
                await asyncio.wait_for(
                    self.condition.wait_for(
                        lambda: (
                            self.inflight < self.max_inflight
                        )
                    ),
                    timeout=timeout,
                )

                self.inflight += 1
                self.admitted_total += 1

                INFLIGHT_REQUESTS.set(
                    self.inflight
                )

                waited_ms = (
                    time.perf_counter() - started
                ) * 1000

                return "admitted", waited_ms

            except asyncio.TimeoutError:
                self.timeout_total += 1

                record_timeout("queue")

                waited_ms = (
                    time.perf_counter() - started
                ) * 1000

                return "timeout", waited_ms

            finally:
                self.queued -= 1
                QUEUED_REQUESTS.set(self.queued)

    async def release(self):
        async with self.condition:
            if self.inflight <= 0:
                raise RuntimeError(
                    "AdmissionController slot underflow"
                )

            self.inflight -= 1

            INFLIGHT_REQUESTS.set(
                self.inflight
            )

            self.condition.notify(1)

    async def increment(self, name):
        async with self.condition:
            setattr(
                self,
                name,
                getattr(self, name) + 1,
            )

        timeout_types = {
            "header_timeout_total": "header",
            "stream_timeout_total": "stream_idle",
            "request_deadline_total": "request_deadline",
        }

        if name in timeout_types:
            record_timeout(timeout_types[name])

    async def snapshot(self):
        async with self.condition:
            return {
                "inflight": self.inflight,
                "queued": self.queued,
                "max_inflight": self.max_inflight,
                "max_queue": self.max_queue,
                "admitted_total": self.admitted_total,
                "rejected_total": self.rejected_total,
                "timeout_total": self.timeout_total,
                "header_timeout_total": (
                    self.header_timeout_total
                ),
                "stream_timeout_total": (
                    self.stream_timeout_total
                ),
                "stream_error_total": (
                    self.stream_error_total
                ),
                "request_deadline_total": (
                    self.request_deadline_total
                ),
            }


# ============================================================
# Request Lease
# ============================================================

class RequestLease:
    def __init__(self, controller):
        self.controller = controller
        self.released = False
        self.lock = asyncio.Lock()

    async def release(self):
        async with self.lock:
            if self.released:
                return

            await self.controller.release()
            self.released = True


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    controller = AdmissionController(
        max_inflight=MAX_INFLIGHT_REQUESTS,
        max_queue=MAX_QUEUE_SIZE,
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=5.0,
            read=None,
            write=30.0,
            pool=10.0,
        ),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    ) as client:
        app.state.http_client = client
        app.state.admission = controller

        print(
            "Gateway started. "
            f"Upstream={VLLM_BASE_URL}, "
            f"MaxInflight={MAX_INFLIGHT_REQUESTS}, "
            f"MaxQueue={MAX_QUEUE_SIZE}, "
            f"QueueTimeout={QUEUE_TIMEOUT_SEC}s, "
            f"HeaderTimeout={UPSTREAM_HEADER_TIMEOUT_SEC}s, "
            f"StreamIdleTimeout={STREAM_IDLE_TIMEOUT_SEC}s, "
            f"RequestDeadline={REQUEST_DEADLINE_SEC}s",
            flush=True,
        )

        yield

    print("Gateway stopped.", flush=True)


app = FastAPI(
    title="LLM Serving Gateway",
    lifespan=lifespan,
)


# ============================================================
# Response Helpers
# ============================================================

def error_response(
    status_code,
    message,
    error_type,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
            }
        },
    )


def sse_error_chunk(message, error_type):
    payload = json.dumps({
        "error": {
            "message": message,
            "type": error_type,
        }
    })

    return f"data: {payload}\n\n".encode("utf-8")


# ============================================================
# Monitoring Endpoints
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstream": VLLM_BASE_URL,
    }


@app.get("/gateway/status")
async def gateway_status(request: Request):
    return await (
        request.app.state.admission.snapshot()
    )


@app.get("/metrics")
async def metrics():
    return Response(
        content=render_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# Streaming Proxy
# ============================================================

async def stream_upstream(
    upstream_response: httpx.Response,
    lease: RequestLease,
    controller: AdmissionController,
    deadline_at: float,
    request_metrics: RequestMetrics,
):
    is_sse = (
        "text/event-stream"
        in upstream_response.headers.get(
            "content-type", ""
        ).lower()
    )

    iterator = upstream_response.aiter_raw().__aiter__()

    # Fallback result if client disconnects.
    result = "client_disconnect"

    try:
        while True:
            remaining = (
                deadline_at - time.monotonic()
            )

            if remaining <= 0:
                await controller.increment(
                    "request_deadline_total"
                )

                result = "request_deadline"

                if is_sse:
                    yield sse_error_chunk(
                        "Gateway request deadline exceeded",
                        "request_deadline_exceeded",
                    )

                break

            wait_sec = min(
                STREAM_IDLE_TIMEOUT_SEC,
                remaining,
            )

            try:
                async with asyncio.timeout(
                    wait_sec
                ):
                    chunk = await anext(iterator)

            except StopAsyncIteration:
                if upstream_response.status_code >= 400:
                    result = "upstream_http_error"
                else:
                    result = "success"

                break

            except TimeoutError:
                deadline_exceeded = (
                    time.monotonic() >= deadline_at
                )

                if deadline_exceeded:
                    await controller.increment(
                        "request_deadline_total"
                    )

                    result = "request_deadline"
                    message = (
                        "Gateway request deadline exceeded"
                    )
                    error_type = (
                        "request_deadline_exceeded"
                    )

                else:
                    await controller.increment(
                        "stream_timeout_total"
                    )

                    result = "stream_timeout"
                    message = (
                        "Upstream stream idle timeout"
                    )
                    error_type = "stream_idle_timeout"

                if is_sse:
                    yield sse_error_chunk(
                        message,
                        error_type,
                    )

                break

            except httpx.RequestError:
                await controller.increment(
                    "stream_error_total"
                )

                result = "stream_error"

                if is_sse:
                    yield sse_error_chunk(
                        "Upstream streaming connection failed",
                        "upstream_stream_error",
                    )

                break

            if chunk:
                yield chunk

    finally:
        try:
            await upstream_response.aclose()
        finally:
            try:
                await lease.release()
            finally:
                request_metrics.finish(result)


# ============================================================
# Chat Completion Endpoint
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_metrics = RequestMetrics()

    client: httpx.AsyncClient = (
        request.app.state.http_client
    )

    controller: AdmissionController = (
        request.app.state.admission
    )

    body = await request.body()

    status, queue_wait_ms = await controller.acquire(
        timeout=QUEUE_TIMEOUT_SEC
    )

    record_queue_wait(
        queue_wait_ms / 1000.0
    )

    if status == "full":
        request_metrics.finish(
            "rejected"
        )

        return error_response(
            status_code=503,
            message="Gateway request queue is full",
            error_type="gateway_overloaded",
        )

    if status == "timeout":
        request_metrics.finish(
            "queue_timeout"
        )

        return error_response(
            status_code=503,
            message="Gateway queue wait timeout",
            error_type="queue_timeout",
        )

    lease = RequestLease(controller)

    deadline_at = (
        time.monotonic() + REQUEST_DEADLINE_SEC
    )

    try:
        upstream_request = client.build_request(
            method="POST",
            url=(
                f"{VLLM_BASE_URL}"
                "/v1/chat/completions"
            ),
            content=body,
            headers={
                "content-type": "application/json",
            },
        )

        header_wait_sec = min(
            UPSTREAM_HEADER_TIMEOUT_SEC,
            max(
                0.0,
                deadline_at - time.monotonic(),
            ),
        )

        async with asyncio.timeout(
            header_wait_sec
        ):
            upstream_response = await client.send(
                upstream_request,
                stream=True,
            )

    except TimeoutError:
        deadline_exceeded = (
            time.monotonic() >= deadline_at
        )

        if deadline_exceeded:
            await controller.increment(
                "request_deadline_total"
            )

            result = "request_deadline"
            message = (
                "Gateway request deadline exceeded"
            )
            error_type = "request_deadline_exceeded"

        else:
            await controller.increment(
                "header_timeout_total"
            )

            result = "header_timeout"
            message = (
                "Upstream response header timeout"
            )
            error_type = "upstream_header_timeout"

        await lease.release()
        request_metrics.finish(result)

        return error_response(
            status_code=504,
            message=message,
            error_type=error_type,
        )

    except httpx.TimeoutException:
        await controller.increment(
            "header_timeout_total"
        )

        await lease.release()
        request_metrics.finish(
            "header_timeout"
        )

        return error_response(
            status_code=504,
            message="Upstream request timed out",
            error_type="upstream_timeout",
        )

    except httpx.RequestError:
        await lease.release()
        request_metrics.finish(
            "upstream_error"
        )

        return error_response(
            status_code=502,
            message="Upstream service unavailable",
            error_type="upstream_error",
        )

    except BaseException:
        try:
            await lease.release()
        finally:
            request_metrics.finish(
                "cancelled"
            )

        raise

    response_headers = {
        "X-Gateway-Queue-Wait-Ms": (
            f"{queue_wait_ms:.3f}"
        )
    }

    content_type = upstream_response.headers.get(
        "content-type"
    )

    if content_type:
        response_headers["content-type"] = content_type

    async def cleanup():
        try:
            await upstream_response.aclose()
        finally:
            try:
                await lease.release()
            finally:
                # Fallback when the stream does not
                # complete its normal lifecycle.
                request_metrics.finish(
                    "client_disconnect"
                )

    return StreamingResponse(
        stream_upstream(
            upstream_response,
            lease,
            controller,
            deadline_at,
            request_metrics,
        ),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(cleanup),
    )
