
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


REGISTRY = CollectorRegistry()


INFLIGHT_REQUESTS = Gauge(
    "gateway_inflight_requests",
    "Current number of admitted requests.",
    registry=REGISTRY,
)


QUEUED_REQUESTS = Gauge(
    "gateway_queued_requests",
    "Current number of queued requests.",
    registry=REGISTRY,
)


REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total Gateway requests by final result.",
    ["result"],
    registry=REGISTRY,
)


REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds",
    "Gateway request wall-clock duration by result.",
    ["result"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        15.0,
        30.0,
        60.0,
        120.0,
        float("inf"),
    ),
    registry=REGISTRY,
)


QUEUE_WAIT = Histogram(
    "gateway_queue_wait_seconds",
    "Time spent waiting for admission.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        float("inf"),
    ),
    registry=REGISTRY,
)


TIMEOUTS_TOTAL = Counter(
    "gateway_timeouts_total",
    "Gateway timeouts by category.",
    ["type"],
    registry=REGISTRY,
)


def record_request(result: str, duration_seconds: float):
    REQUESTS_TOTAL.labels(
        result=result
    ).inc()

    REQUEST_DURATION.labels(
        result=result
    ).observe(
        max(0.0, duration_seconds)
    )


def record_queue_wait(seconds: float):
    QUEUE_WAIT.observe(
        max(0.0, seconds)
    )


def record_timeout(timeout_type: str):
    TIMEOUTS_TOTAL.labels(
        type=timeout_type
    ).inc()


def render_metrics():
    return generate_latest(REGISTRY)
