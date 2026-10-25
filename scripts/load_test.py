from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run_load_test(url: str, requests: int, concurrency: int, api_key: str) -> dict:
    endpoint = url.rstrip("/") + "/v1/predict"
    headers = {"Content-Type": "application/json", "X-Request-ID": "playerpulse-load-test"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = json.dumps(
        {
            "player_ref": None,
            "sessions_7d": 2,
            "sessions_30d": 8,
            "playtime_7d_hours": 1.5,
            "playtime_30d_hours": 5.0,
            "days_since_last_session": 2,
        }
    ).encode("utf-8")

    def send_one(index: int):
        request = Request(
            endpoint, data=payload, headers={**headers, "X-Request-ID": f"load-{index}"}
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=30) as response:
                response.read()
                return (time.perf_counter() - started) * 1_000, response.status, None
        except HTTPError as error:
            return (time.perf_counter() - started) * 1_000, error.code, str(error.reason)
        except URLError as error:
            return (time.perf_counter() - started) * 1_000, 0, str(error.reason)

    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(send_one, range(requests)))
    wall_seconds = time.perf_counter() - wall_started
    latencies = [item[0] for item in results]
    errors = [item for item in results if item[1] < 200 or item[1] >= 300]
    report = {
        "url": url,
        "hardware": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_mb": None,
        "request_count": requests,
        "concurrency": concurrency,
        "elapsed_seconds": wall_seconds,
        "throughput_requests_per_second": requests / wall_seconds if wall_seconds else None,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "error_count": len(errors),
        "error_rate": len(errors) / requests if requests else 0.0,
        "first_errors": [item[2] for item in errors[:5]],
    }
    try:
        import psutil

        report["memory_mb"] = round(psutil.Process().memory_info().rss / 1_048_576, 1)
    except ImportError:
        pass
    try:
        with urlopen(url.rstrip("/") + "/v1/model/info", timeout=8) as response:
            model = json.load(response)
        report["model_version"] = model["model_version"]
        report["model_warning"] = model["warning"]
    except Exception as error:
        report["model_info_error"] = str(error)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small bounded concurrent PlayerPulse inference test."
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--api-key", default=os.getenv("PREDICTION_API_KEY", ""))
    arguments = parser.parse_args()
    if arguments.requests < 1 or arguments.requests > 10_000:
        parser.error("--requests must be from 1 to 10000.")
    if arguments.concurrency < 1 or arguments.concurrency > 100:
        parser.error("--concurrency must be from 1 to 100.")
    print(
        json.dumps(
            run_load_test(
                arguments.url, arguments.requests, arguments.concurrency, arguments.api_key
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
