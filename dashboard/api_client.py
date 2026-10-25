from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
if API_BASE_URL and not API_BASE_URL.startswith(("http://", "https://")):
    API_BASE_URL = f"https://{API_BASE_URL}"
API_KEY = os.getenv("API_KEY") or os.getenv("PREDICTION_API_KEY", "")


class ApiUnavailable(RuntimeError):
    pass


def _request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    headers = {"Accept": "application/json", "X-Request-ID": "dashboard"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{API_BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=12) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("detail", error.reason)
        except (ValueError, AttributeError):
            detail = error.reason
        raise ApiUnavailable(f"The API returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError, OSError, ValueError) as error:
        raise ApiUnavailable(
            f"Could not retrieve dashboard data from {API_BASE_URL}: {error}"
        ) from error


def get_health() -> dict:
    return _request("/ready")


def get_model_info() -> dict:
    return _request("/v1/model/info")


def get_metrics(window_days: int = 90) -> dict:
    query = urlencode({"window_days": window_days})
    return _request(f"/v1/metrics/summary?{query}")


def get_predictions(total_limit: int = 5_000) -> tuple[list[dict], int]:
    predictions = []
    offset = 0
    total = 0
    while offset < total_limit:
        page_size = min(500, total_limit - offset)
        response = _request(f"/v1/predictions?limit={page_size}&offset={offset}")
        page = response.get("predictions", [])
        predictions.extend(page)
        total = response.get("total", len(predictions))
        offset += len(page)
        if not page or offset >= total:
            break
    return predictions, total


def post_prediction(payload: dict) -> dict:
    return _request("/v1/predict", method="POST", body=payload)
