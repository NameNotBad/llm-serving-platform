# LLM Serving Platform

A single-GPU LLM inference serving and performance engineering project built with **vLLM, PyTorch, and Python**.

The project focuses on serving large language models through an OpenAI-compatible API, developing reproducible performance benchmarks, evaluating inference optimizations, and diagnosing GPU execution bottlenecks.

## 1. Project Overview

The platform provides an experimental environment for investigating:

- LLM inference latency and throughput
- Concurrent request processing and batching behavior
- Time to First Token (TTFT) and Time per Output Token (TPOT)
- Prefill-heavy versus decode-heavy workloads
- Prefix caching effectiveness
- GPU kernel execution and hardware utilization

The project emphasizes **controlled experiments, reproducibility, and evidence-based optimization**.

### Technology Stack

| Component | Technology |
|---|---|
| Inference Engine | vLLM 0.32.0 |
| Model | Qwen2.5-1.5B-Instruct |
| GPU | NVIDIA RTX 4070 Ti SUPER, 16 GB |
| OS | Ubuntu 24.04 on WSL2 |
| Model Precision | FP16 |
| Inference API | OpenAI-compatible REST API |
| Load Testing | Python, asyncio, HTTPX |
| GPU Trace Profiling | PyTorch Profiler |
| Hardware Profiling | NVIDIA Nsight Compute 2026.3.1 |
| Data Analysis | Python, CSV, JSON, Matplotlib |

## 2. Architecture

```text
                     LLM Serving Platform
                              |
                +-------------+-------------+
                |                           |
         Benchmark Client              vLLM Server
                |                           |
         Async HTTPX                 Qwen2.5-1.5B
                |                           |
      Streaming Requests              GPU Inference
                |                           |
     +----------+----------+       +--------+---------+
     |          |          |       |                  |
   TTFT        TPOT     Throughput  Prefill          Decode
     |          |          |       |                  |
     +----------+----------+       +--------+---------+
                |                           |
         Performance Logs            GPU Execution
                |                           |
         CSV / JSON Results          PyTorch Profiler
                |                           |
         Statistical Analysis        CUDA Kernel Traces
                |                           |
         Performance Charts          Nsight Compute
                |                           |
                +-------------+-------------+
                              |
                    Bottleneck Analysis
```

## 3. Performance Benchmarking

The benchmarking suite evaluates inference performance under different input lengths and concurrency levels.

### Metrics

| Metric | Description |
|---|---|
| TTFT | Time between submitting a request and receiving its first content token |
| TPOT | Average time per generated token after the first token |
| E2E Latency | Total request completion time |
| Output Throughput | Generated tokens per second |
| Request Throughput | Completed requests per second |
| P95 Latency | 95th-percentile request latency |

The benchmark client uses asynchronous HTTP requests and streaming response parsing.

Experiment configurations, runtime environment information, per-request results, and aggregated summaries are saved for reproducibility.

### Benchmark Design

The baseline benchmark evaluates:

- Input lengths: 128, 512, and 1024 tokens
- Concurrency levels: 1, 4, 8, and 16
- Repeated measurements for each configuration

The workload generator uses deterministic prompts and tokenizer-based length validation.

Measurements are conducted using a fixed-concurrency, closed-loop load generator.

## 4. Prefix Caching Optimization

We evaluated vLLM prefix caching with a **2×2 controlled experiment**:

| Workload | Cache OFF | Cache ON |
|---|---|---|
| Unique Prompts | Baseline | Cache-enabled control |
| Shared Prefix | Cache-disabled control | Cache-enabled shared-prefix workload |

The shared-prefix workload uses identical deterministic input generation under both cache configurations.

### Results

**Shared-prefix workload, concurrency = 16**

| Metric | Cache OFF | Cache ON | Improvement |
|---|---:|---:|---:|
| Mean TTFT | 188.58 ms | 87.31 ms | **53.70% reduction** |
| Output Throughput | 1,155.59 tok/s | 1,610.27 tok/s | **39.35% increase** |

At concurrency = 4, shared-prefix caching reduced mean TTFT by 55.43% and increased output throughput by 11.68%.

Unique-prompt workloads exhibited substantially smaller differences, indicating that caching benefits depend strongly on prefix reuse.

**Interpretation:** Prefix caching is most useful when requests share reusable token prefixes. These results are specific to the evaluated model, GPU, request distribution, and cache-warmup conditions.

## 5. Prefill vs. Decode Workload Analysis

We evaluated four workload configurations across concurrency levels 1, 4, 8, and 16.

| Workload | Input Tokens | Max Output Tokens |
|---|---:|---:|
| Short Input / Short Output | 128 | 64 |
| Long Input / Short Output | 1024 | 64 |
| Short Input / Long Output | 128 | 512 |
| Long Input / Long Output | 1024 | 512 |

The experiment completed **48 benchmark runs**.

### Results at Concurrency 16

| Workload | Output Throughput | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|
| 128 / 64 | 1,729.34 tok/s | 76.41 ms | 8.14 ms |
| 1024 / 64 | 803.12 tok/s | 268.49 ms | 15.88 ms |
| 128 / 512 | **1,988.64 tok/s** | 72.08 ms | 7.91 ms |
| 1024 / 512 | 1,433.09 tok/s | 274.56 ms | 10.61 ms |

### Key Observations

**Longer input sequences increase TTFT.**

Processing 1024-token prompts incurs substantially greater first-token latency than processing 128-token prompts.

**Longer outputs can improve aggregate output throughput.**

The fixed costs associated with prefill and request processing are amortized across more generated tokens. Higher output throughput does not necessarily imply shorter request completion latency.

**Concurrency interacts with workload composition.**

Long input sequences combined with high concurrency can substantially increase observed TPOT.

These measurements characterize serving behavior but do not independently establish whether the GPU is compute-bound or memory-bound.

## 6. GPU Profiling and Bottleneck Analysis

We developed an offline GPU profiling workflow combining:

1. PyTorch Profiler traces collected from the vLLM server
2. GPU activity and kernel-family analysis
3. NVIDIA Nsight Compute hardware counter collection
4. Scheduler and warp stall investigation

### PyTorch Profiler Results

| Metric | Prefill-heavy | Decode-heavy |
|---|---:|---:|
| Input / Output Tokens | 1024 / 64 | 128 / 512 |
| Concurrency | 8 | 8 |
| GPU Activity Coverage | 96.59% | 96.54% |
| GEMM / CUTLASS Duration | 633.77 ms | 2,840.14 ms |
| FlashAttention Duration | 67.32 ms | 202.41 ms |

GEMM-related kernels accounted for the largest proportion of recorded GPU kernel execution time in both workloads.

GPU activity coverage measures the fraction of the captured activity span containing at least one recorded GPU event; it is not equivalent to SM utilization.

### Nsight Compute Hardware Analysis

A standalone 2048×2048 FP16 Tensor Core GEMM was analyzed using hardware performance counters.

| Metric | Observed Value |
|---|---:|
| Kernel Duration | 213.73 μs |
| SM Throughput | 46.11% |
| DRAM Throughput | 12.03% |
| Achieved Occupancy | 16.66% |
| Scheduler Cycles with No Eligible Warp | 93.21% |
| Issue Slots Busy | 6.53% |

The profiling results showed limited eligible warp availability and substantial math pipeline waiting in the selected standalone GEMM kernel.

**Important:** Hardware counter results were collected from standalone PyTorch GEMM operations, not directly from vLLM inference kernels. They are presented as an example of hardware-level performance diagnosis rather than proof of the complete serving system's bottleneck.

For detailed methodology, results, and limitations, see [GPU Profiling Report](docs/GPU_PROFILING_REPORT.md).

## 7. Repository Organization

```text
llm-serving-platform/
├── benchmark/
│   ├── benchmark_suite.py
│   ├── streaming_load_test.py
│   ├── prompt_generator.py
│   ├── aggregate_results.py
│   ├── plot_results.py
│   ├── environment.py
│   ├── prefix_cache_experiment.py
│   ├── off_shared_experiment.py
│   ├── prefix_cache_report.py
│   ├── prefill_decode_experiment.py
│   ├── gpu_profile_experiment.py
│   ├── analyze_gpu_trace.py
│   └── analyze_gpu_trace_v2.py
├── docs/
│   └── GPU_PROFILING_REPORT.md
├── results/              # Generated locally; Git-ignored
└── README.md
```

The repository contains the experimental implementation and analysis tools. Generated raw traces and large benchmarking outputs are excluded from version control.

## 8. Running the Platform

The following commands assume that the model dependencies have already been installed in a compatible vLLM environment and that the repository is opened from its root directory.

### Start vLLM

```bash
~/venvs/llm-serving-vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --no-enable-prefix-caching
```

### Run the Benchmark Suite

```bash
python benchmark/benchmark_suite.py
```

### Analyze Prefill and Decode Workloads

```bash
python benchmark/prefill_decode_experiment.py
```

### Analyze Existing GPU Traces

```bash
python benchmark/analyze_gpu_trace_v2.py \
  --results results/gpu_profiles
```

GPU trace analysis requires previously collected PyTorch Profiler trace files.

### Run Prefix Caching Experiments

The prefix caching workflow requires restarting the vLLM server with the corresponding cache configuration.

For example, to enable prefix caching:

```bash
~/venvs/llm-serving-vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching
```

Run the corresponding experiment modes documented in the benchmark scripts.

## 9. Reproducibility and Limitations

The project records experiment configurations, model settings, selected software versions, GPU information, and statistical summaries.

Important limitations include:

- Single-GPU experimental environment
- Fixed-concurrency, closed-loop benchmarking
- Three repetitions per performance configuration
- Deterministic workload generation
- Potential warm-cache and experiment-order effects
- Client-observed streaming latency measurements
- Profiling instrumentation overhead
- Limited hardware-counter measurements on standalone kernels
- No production distributed-serving or multi-node evaluation

Performance numbers should not be assumed to generalize to other models, GPUs, batching policies, or request distributions.

## 10. Future Work

Potential next steps include:

- GPU-aware request scheduling and batching experiments
- Dynamic batching and queueing analysis
- Quantization and precision trade-off evaluations
- Request arrival-rate and tail-latency benchmarking
- Server-level metrics and observability
- Multi-GPU and distributed inference
- Hardware profiling of representative vLLM kernels

## 11. Production-Oriented API Gateway (Stage 2)

Stage 2 extends the GPU inference platform with a FastAPI
gateway designed to study LLM serving reliability under
concurrent workloads.

### Gateway Architecture

```text
Client / Load Generator
         |
         v
FastAPI Gateway :8080
    |
    |-- Admission Control
    |     |-- Max In-flight: 8
    |     |-- Max Queue: 16
    |     |-- Queue Timeout: 5s
    |
    |-- Timeout Policies
    |     |-- Header Timeout: 15s
    |     |-- Stream Idle Timeout: 15s
    |     |-- Request Deadline: 120s
    |
    |-- Prometheus Metrics
    |
    v
vLLM Backend :8000
         |
         v
Qwen2.5-1.5B-Instruct
```

The gateway uses a shared HTTPX AsyncClient to forward
OpenAI-compatible chat completion requests.

Streaming responses preserve upstream SSE data while enforcing
bounded concurrency, queue capacity, and timeout policies.

### Admission Control and Overload Protection

The gateway supports:

- Bounded concurrent requests
- Bounded waiting queues
- HTTP 503 rejection when the queue is full
- Queue timeout enforcement
- Streaming lifecycle cleanup
- Client disconnect handling
- Upstream connection error handling

In a 32-request concurrent overload experiment:

| Outcome | Requests |
|---|---:|
| Successful requests | 16 |
| Queue-full rejection | 8 |
| Queue timeout | 8 |

The maximum observed concurrency was 8/8, and maximum observed
queue depth was 16/16.

All six benchmark invariant checks passed, and both in-flight
and queued request counts returned to zero after completion.

### Timeout Policies

| Policy | Default |
|---|---:|
| Queue timeout | 5 seconds |
| Upstream header timeout | 15 seconds |
| Streaming idle timeout | 15 seconds |
| Request deadline | 120 seconds |

Streaming idle timeout and request deadline were validated
using mock upstream fault-injection experiments.

The current implementation does not guarantee a strict
end-to-end deadline in every slow-client scenario.

### Gateway Performance Evaluation

A direct-vLLM versus Gateway A/B benchmark covered:

- 4 concurrency levels: 1, 4, 8, 16
- 3 repetitions per configuration
- 32 requests per run
- 768 measured requests overall

Across the measured configurations, Gateway routing did not
introduce a consistent, substantial performance regression.

See the Stage 2 report for the full experimental data,
methodology, and limitations.

## 12. Observability and Monitoring

Stage 2 includes a Prometheus and Grafana monitoring stack.

### Monitoring Architecture

```text
Gateway :8080/metrics
          |
          v
Prometheus :9090
          |
          v
Grafana :3000
```

The monitoring stack uses Docker Compose and configuration
files stored in the repository.

### Available Metrics

| Metric | Type |
|---|---|
| gateway_inflight_requests | Gauge |
| gateway_queued_requests | Gauge |
| gateway_requests_total | Counter |
| gateway_request_duration_seconds | Histogram |
| gateway_queue_wait_seconds | Histogram |
| gateway_timeouts_total | Counter |

Request duration histograms support result-based filtering
so successful inference latency can be monitored separately
from rejected and timed-out requests.

### Grafana Dashboard

The automatically provisioned dashboard includes six panels:

1. Request Throughput (RPS)
2. Successful Request Latency (P50/P95/P99)
3. In-flight Requests
4. Queue Depth
5. Queue Wait P95
6. Request Outcomes

Dashboard URL:

http://localhost:3000/d/llm-serving-platform

Prometheus URL:

http://localhost:9090

These URLs are available after starting the monitoring stack
on the local development machine.

## 13. Running Stage 2

### Start vLLM

```bash
source ~/venvs/llm-serving-vllm/bin/activate

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048
```

### Start the Gateway

Open a second terminal:

```bash
source .venv/bin/activate

python -m pip install -r gateway/requirements.txt

python -m uvicorn gateway.app:app \
  --host 0.0.0.0 \
  --port 8080
```

The gateway is unauthenticated and should only be used
on a trusted development network.

### Start Prometheus and Grafana

Open a third terminal:

```bash
export WSL_IP=$(hostname -I | awk '{print $1}')

docker compose \
  -f monitoring/docker-compose.yml \
  up -d
```

The Compose configuration maps the current WSL IP to
`wsl-gateway` for Prometheus scraping.

If WSL networking changes, refresh WSL_IP and recreate
the monitoring containers.

### Run the Gateway Overload Benchmark

```bash
python benchmark/gateway_overload_test.py \
  --requests 32 \
  --max-tokens 512 \
  --interval 0.05
```

Benchmark output is saved under `results/gateway/`.

### Check Service Health

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8080/gateway/status

curl http://127.0.0.1:8080/metrics
```

## 14. Stage 2 Technical Report

For implementation details, experimental methodology,
failure injection, metrics validation, and known limitations:

[Stage 2 Technical Report](docs/LLM_SERVING_STAGE2_REPORT.md)

The Stage 2 implementation is a tested local prototype intended
for learning and demonstrating LLM serving systems engineering.
It is not a fully production-hardened deployment.

## Summary

This project explores the design, optimization, and operational reliability of a single-node LLM serving platform.

**Stage 1 — Inference Performance Engineering** focuses on vLLM inference serving, performance benchmarking, prefix caching, prefill/decode workload analysis, and GPU profiling using PyTorch Profiler and NVIDIA Nsight Compute.

**Stage 2 — Gateway Reliability and Observability** extends the platform with a FastAPI streaming gateway, admission control, bounded queueing, timeout policies, request lifecycle management, and Prometheus/Grafana monitoring.

Together, these stages demonstrate practical experience across LLM inference performance, GPU bottleneck analysis, concurrent request handling, overload protection, and production-oriented observability.

The platform has been validated through controlled benchmarks, failure-injection experiments, and monitoring integration tests. It remains a single-node development prototype rather than a fully production-hardened distributed serving system.