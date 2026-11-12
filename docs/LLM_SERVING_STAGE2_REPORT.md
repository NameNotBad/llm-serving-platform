
# LLM Serving Platform — Stage 2 Technical Report

## Production-Oriented Gateway, Admission Control, and Observability

## 1. Executive Summary

Stage 2 extends the GPU inference and benchmarking platform developed
in Stage 1 with a production-oriented HTTP gateway and monitoring stack.

The system uses:

- vLLM as the inference backend
- FastAPI as the API gateway
- HTTPX for asynchronous upstream request forwarding
- An asyncio-based admission controller
- Prometheus for metrics collection
- Grafana for operational dashboards
- Docker Compose for monitoring infrastructure

The goal is to study the engineering challenges involved in serving
LLMs under concurrent load, including bounded resource utilization,
request queueing, timeout enforcement, streaming lifecycle management,
and operational visibility.

The implementation is a local, single-node prototype. It is not
presented as a fully hardened production service.

## 2. System Architecture

The request path is:

    Client / Load Generator
              |
              v
    FastAPI Gateway :8080
       |
       |-- Admission Controller
       |     |-- Max In-flight: 8
       |     |-- Max Queue: 16
       |     |-- Queue Timeout: 5s
       |
       |-- Timeout Policies
       |     |-- Header Timeout: 15s
       |     |-- Stream Idle Timeout: 15s
       |     |-- Request Deadline: 120s
       |
       |-- Streaming HTTPX Proxy
       |     |-- SSE Forwarding
       |     |-- Client Disconnect Cleanup
       |     |-- Upstream Error Handling
       |
       v
    vLLM Backend :8000
              |
              v
    Qwen2.5-1.5B-Instruct

The monitoring path is:

    Gateway /metrics
           |
           v
    Prometheus :9090
           |
           v
    Grafana :3000

The Gateway and vLLM backend run directly in WSL2.
Prometheus and Grafana run as Docker Compose services.

## 3. Development Environment

| Component | Configuration |
|---|---|
| Operating system | Windows + WSL2 Ubuntu 24.04 |
| GPU | NVIDIA RTX 4070 Ti SUPER, 16 GB |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Model precision | FP16 |
| Model context limit | 2048 tokens |
| Inference server | vLLM |
| API gateway | FastAPI + HTTPX |
| Metrics client | prometheus-client 0.26.0 |
| Prometheus | v3.5.0 |
| Grafana | 12.4.0 |
| Container management | Docker Compose |

The local backend is exposed on port 8000, and the gateway
is exposed on port 8080.

Monitoring services use ports 9090 and 3000.

## 4. Streaming API Gateway

### 4.1 Implementation

The gateway forwards requests to the vLLM-compatible endpoint:

    POST /v1/chat/completions

The implementation uses a shared HTTPX AsyncClient created during
the FastAPI lifespan.

A shared client allows connection reuse and avoids creating a
new HTTP connection pool for every request.

The gateway forwards upstream response bytes through FastAPI's
StreamingResponse.

The request slot remains occupied until the streaming lifecycle
has ended and cleanup has been performed.

### 4.2 Gateway A/B Performance Experiment

The experiment compared direct vLLM access with routing through
the FastAPI gateway.

Workload:

- Input length: approximately 128 tokens
- Output length: 64 tokens
- Concurrency: 1, 4, 8, and 16
- Repetitions: 3 per configuration
- Requests: 32 per run
- Total: 24 experimental runs and 768 measured requests

Selected average results:

| Concurrency | Direct TTFT (ms) | Gateway TTFT (ms) | Direct Throughput (tok/s) | Gateway Throughput (tok/s) |
|---|---:|---:|---:|---:|
| 1 | 15.52 | 16.03 | 149.81 | 150.60 |
| 4 | 33.54 | 34.42 | 556.07 | 538.47 |
| 8 | 54.20 | 52.27 | 1032.98 | 1033.95 |
| 16 | 93.88 | 83.74 | 1861.00 | 1848.39 |

The gateway did not cause a consistent, substantial performance
regression in these measured configurations.

However, individual differences cannot be attributed exclusively
to intrinsic gateway overhead. GPU scheduling, run-to-run variation,
network effects, and experimental ordering can affect the results.

The experiment should therefore be interpreted as an initial
performance comparison rather than a definitive overhead bound.

## 5. Admission Control

### 5.1 Motivation

Without admission control, a gateway can accept requests faster
than the inference backend can process them.

This can lead to:

- Excessive concurrent requests
- Unbounded queue growth
- Increased tail latency
- Poor overload behavior
- Resource exhaustion

The gateway implements bounded concurrency and waiting capacity.

Default configuration:

    MAX_INFLIGHT_REQUESTS=8
    MAX_QUEUE_SIZE=16
    QUEUE_TIMEOUT_SEC=5

### 5.2 Admission Rules

An incoming request follows these rules:

1. If an execution slot is available and there are no waiting
   requests, the request is admitted immediately.

2. Otherwise, the request enters the waiting queue if capacity
   is available.

3. If the queue is full, the gateway returns HTTP 503 with the
   error type `gateway_overloaded`.

4. If a queued request waits beyond the queue timeout, the gateway
   returns HTTP 503 with the error type `queue_timeout`.

5. A successfully admitted request holds its execution slot until
   the upstream response lifecycle ends.

The concurrency controller uses asyncio.Condition and a request
lease abstraction to manage capacity.

The current implementation does not guarantee strict FIFO fairness.

### 5.3 Overload Test

The benchmark submitted 32 concurrent streaming requests.

Configuration:

- Maximum in-flight requests: 8
- Maximum queued requests: 16
- Queue timeout: 5 seconds
- Output limit: 512 tokens

Observed results:

| Outcome | Requests |
|---|---:|
| Successful HTTP 200 | 16 |
| Queue Full HTTP 503 | 8 |
| Queue Timeout HTTP 503 | 8 |
| Total | 32 |

Observed capacity:

    Max In-flight: 8 / 8
    Max Queued:    16 / 16

Final state:

    In-flight: 0
    Queued:    0

The benchmark performed 165 status samples in the final monitoring
test and reported six passing invariant checks.

The samples showed no capacity violations at observed timestamps.

Discrete monitoring cannot prove that every instantaneous state
transition was observed.

## 6. Streaming Lifecycle and Cleanup

### 6.1 Request Lease

Each admitted request receives a RequestLease.

Its purpose is to ensure that the concurrency slot is released
at most once.

Cleanup is performed when streaming completes, fails, or is
interrupted.

### 6.2 Client Disconnect Test

Eight concurrent streaming requests were started.

Each test client closed its stream after receiving the first
content chunk.

Observed results:

- All eight clients received content before disconnecting.
- The gateway returned to zero in-flight requests.
- No waiting requests remained.

The test validates slot cleanup for the exercised HTTP disconnect
path.

It does not prove that the vLLM backend immediately stops consuming
GPU resources after every client disconnect.

### 6.3 Upstream Connection Failure Test

A separate gateway instance was configured to connect to an
unavailable upstream service.

Observed response:

    HTTP 502 Bad Gateway
    error.type = upstream_error

Final state:

    In-flight: 0
    Queued:    0

The admitted request slot was released after connection failure.

This test does not cover every possible mid-stream transport failure.

## 7. Timeout Policies

The gateway supports multiple timeout mechanisms.

| Timeout | Default | Purpose |
|---|---:|---|
| Connection timeout | 5s | Establish upstream connection |
| Header timeout | 15s | Wait for upstream response headers |
| Streaming idle timeout | 15s | Wait for the next upstream chunk |
| Request deadline | 120s | Limit upstream request execution time |
| Queue timeout | 5s | Bound admission waiting time |

### 7.1 Streaming Idle Timeout

The gateway resets the idle timeout while waiting for each
upstream data chunk.

A mock server was configured to:

1. Return an initial SSE chunk.
2. Pause for five seconds.
3. Attempt to return another chunk.

The test gateway used:

    STREAM_IDLE_TIMEOUT_SEC=2

Observed output:

    stream_idle_timeout

The gateway recorded:

    stream_timeout_total = 1
    inflight = 0
    queued = 0

This validated idle timeout detection and slot cleanup.

### 7.2 Request Deadline

Request deadline enforcement begins after successful admission.

The implementation records an absolute deadline using
time.monotonic() and limits individual upstream read waits
to the smaller of:

- Remaining request deadline
- Streaming idle timeout

A mock backend returned a chunk every 0.5 seconds.

Test configuration:

    REQUEST_DEADLINE_SEC=3
    STREAM_IDLE_TIMEOUT_SEC=15

The gateway forwarded several chunks and then emitted:

    request_deadline_exceeded

Observed counters:

    request_deadline_total = 1
    stream_timeout_total = 0
    inflight = 0
    queued = 0

The result confirmed that total execution time can be limited
even when the upstream continues producing output.

### 7.3 Streaming Error Semantics

Once HTTP 200 response headers have been sent, the gateway
cannot replace the status code with HTTP 504.

For an SSE response, the implementation can emit a gateway-specific
error payload.

Example:

    data: {"error":{"type":"request_deadline_exceeded",...}}

This payload is not a standard OpenAI Chat Completion chunk.

Clients must not interpret an HTTP 200 response alone as proof
that the entire generation succeeded.

### 7.4 Remaining Timeout Limitations

The current request deadline primarily bounds upstream waiting.

It does not implement a strict end-to-end deadline covering every
possible slow-client write or cancellation race.

Header timeout enforcement is implemented but has not yet received
the same dedicated fault-injection coverage as streaming idle
timeout and request deadline.

## 8. Prometheus Metrics

### 8.1 Metrics Endpoint

The gateway exposes:

    GET /metrics

Metrics are exported in the Prometheus text exposition format.

### 8.2 Metric Definitions

| Metric | Type | Description |
|---|---|---|
| gateway_inflight_requests | Gauge | Current admitted requests |
| gateway_queued_requests | Gauge | Current waiting requests |
| gateway_requests_total | Counter | Requests grouped by result |
| gateway_request_duration_seconds | Histogram | Wall-clock request duration by result |
| gateway_queue_wait_seconds | Histogram | Admission wait duration |
| gateway_timeouts_total | Counter | Timeout counts by category |

The request duration includes streaming time and is finalized when
the request lifecycle completes.

It is not the same as model-only inference latency.

The queue wait histogram currently includes zero-duration
observations for immediately admitted or rejected requests.

### 8.3 Result Labels

Request outcomes include labels such as:

    success
    rejected
    queue_timeout
    header_timeout
    stream_timeout
    request_deadline
    upstream_error
    upstream_http_error
    stream_error
    client_disconnect

This allows dashboards to separate successful inference latency
from rejection and timeout latency.

### 8.4 Metrics Accuracy Validation

In the final 32-request overload experiment:

| Metric | Expected | Observed |
|---|---:|---:|
| Success requests | 16 | 16 |
| Rejected requests | 8 | 8 |
| Queue timeouts | 8 | 8 |
| Total | 32 | 32 |

The Prometheus request counters matched the benchmark results.

After the experiment, the gateway reported:

    inflight = 0
    queued = 0

## 9. Latency Distribution Analysis

A separate 32-request overload benchmark produced the following
client-side latency statistics:

| Metric | All Requests | Successful | Failed |
|---|---:|---:|---:|
| Mean | 4.068s | 5.605s | 2.531s |
| P50 | 4.394s | 5.599s | 2.530s |
| P95 | 7.452s | 7.452s | 5.030s |
| P99 | 7.452s | 7.452s | 5.031s |

Successful requests took longer on average because they completed
model generation, whereas queue-full requests were rejected quickly.

These statistics describe one small experimental workload.
P99 estimates based on only 16 successful samples are not stable
enough to represent a production service-level objective.

The Prometheus dashboard instead uses histogram_quantile()
on bucketed time series.

Histogram quantiles are approximations and may differ substantially
from percentiles calculated using raw client observations,
particularly with sparse traffic or coarse bucket boundaries.

## 10. Prometheus and Grafana Deployment

### 10.1 Monitoring Services

Docker Compose runs:

- Prometheus on localhost:9090
- Grafana on localhost:3000

Prometheus scrapes gateway metrics every five seconds.

A persistent Docker volume stores Prometheus time-series data.

A separate volume stores Grafana state.

### 10.2 WSL2 Networking

The gateway runs directly in WSL2 and is not part of the
Docker Compose network.

In this environment, direct container access through
host.docker.internal did not return the expected gateway routes.

Access through the WSL IP address succeeded.

Docker Compose uses an extra_hosts mapping:

    wsl-gateway -> current WSL IP

Prometheus targets:

    http://wsl-gateway:8080/metrics

Because WSL IP addresses can change after a restart, the environment
variable WSL_IP should be refreshed before recreating the container.

This is a local development workaround rather than a portable,
production networking architecture.

### 10.3 Dashboard as Code

Grafana automatically loads its data source and dashboard from
version-controlled configuration files.

The dashboard includes six panels:

1. Request Throughput (RPS)
2. Successful Request Latency (P50/P95/P99)
3. In-flight Requests
4. Queue Depth
5. Queue Wait P95
6. Request Outcomes

A final overload run produced visible activity in all six panels.

Prometheus observed a peak of eight in-flight requests and
sixteen queued requests.

The dashboard is available at:

    http://localhost:3000/d/llm-serving-platform

### 10.4 Monitoring Limitations

Prometheus scrapes every five seconds.

Short-lived concurrency peaks may occur between scrapes and
therefore may not appear in Gauge time series.

Request rates and histogram quantiles rely on rolling windows
and require enough recent observations to be meaningful.

The monitoring setup is intended for local, single-process use.
Multiple gateway workers would require revised metrics handling.

## 11. Reproducing the Platform

### 11.1 Start vLLM

Activate the vLLM environment:

    source ~/venvs/llm-serving-vllm/bin/activate

Start the inference server:

    vllm serve Qwen/Qwen2.5-1.5B-Instruct \
      --host 127.0.0.1 \
      --port 8000 \
      --dtype float16 \
      --max-model-len 2048

### 11.2 Start the Gateway

From the project root:

    source .venv/bin/activate

Install gateway dependencies:

    python -m pip install -r gateway/requirements.txt

Start the gateway:

    python -m uvicorn gateway.app:app \
      --host 0.0.0.0 \
      --port 8080

Binding to 0.0.0.0 is used here to permit access from the
local Docker monitoring environment.

The development gateway does not implement authentication.
Do not expose it to untrusted networks.

### 11.3 Start Monitoring

From the project root:

    export WSL_IP=$(hostname -I | awk '{print $1}')

    docker compose \
      -f monitoring/docker-compose.yml \
      up -d

Validate services:

    curl http://127.0.0.1:8080/gateway/status

    curl http://127.0.0.1:8080/metrics

    curl http://127.0.0.1:9090/-/healthy

Grafana:

    http://localhost:3000

Prometheus:

    http://localhost:9090

### 11.4 Run the Overload Benchmark

From the project virtual environment:

    python benchmark/gateway_overload_test.py \
      --requests 32 \
      --max-tokens 512 \
      --interval 0.05

The benchmark generates a local JSON report under:

    results/gateway/

The results directory is intentionally Git-ignored.

## 12. Known Limitations and Future Improvements

The current prototype has several important limitations.

### Request Scheduling

- Strict FIFO fairness is not guaranteed.
- Admission control is based on request counts, not token budgets.
- Prefill and decode workloads do not receive separate capacity
  allocation policies.

### Streaming and Cancellation

- Mid-stream errors use nonstandard SSE error payloads.
- Strict end-to-end deadlines are not implemented.
- Some cancellation and disconnect races remain untested.
- Backend-side GPU cancellation behavior is not fully validated.

### API Compatibility

- The primary gateway implementation focuses on
  /v1/chat/completions.
- Authentication, authorization and tenant isolation are absent.
- API compatibility is not exhaustively validated for every
  OpenAI-compatible request mode.

### Observability

- Metrics are maintained in a single Python process.
- Gauge scraping can miss short-lived peaks.
- Quantiles depend on histogram bucket resolution.
- Success does not constitute independent validation of the
  semantic quality of generated model output.

### Deployment

- The WSL IP mapping is environment-specific.
- Restarting WSL can require refreshing the address mapping.
- Gateway and model serving remain local processes.
- Production security, horizontal scaling, and service discovery
  are outside the current implementation.

## 13. Key Engineering Takeaways

Stage 2 demonstrates several practical lessons:

1. A streaming proxy must manage the full response lifecycle,
   not just the initial HTTP request.

2. Bounded queues and concurrency limits provide explicit
   overload behavior instead of unlimited request accumulation.

3. Queue timeout, stream idle timeout and request deadline
   solve different operational problems.

4. Cancellation and error paths are critical to preventing
   concurrency slot leaks.

5. Monitoring metrics must distinguish successful requests
   from rejected and failed requests.

6. Client-side benchmark percentiles and Prometheus histogram
   quantiles serve different analytical purposes.

7. Infrastructure reproducibility requires both configuration
   as code and attention to local networking assumptions.

## 14. Conclusion

Stage 2 adds a production-oriented API gateway and operational
monitoring layer to the Stage 1 GPU inference platform.

The implementation has been validated with streaming requests,
overload tests, cancellation tests, upstream connection failures,
mock timeout injection, metrics reconciliation, and a Grafana
dashboard receiving live traffic.

It provides a useful foundation for further work in LLM serving
reliability, scheduling, scalability and inference infrastructure.

The current system should be regarded as a tested local prototype,
not a fully production-certified LLM serving platform.
