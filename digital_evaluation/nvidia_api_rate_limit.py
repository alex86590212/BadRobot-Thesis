"""NVIDIA integrate API helpers: sliding-window RPM limit + 429 backoff retries."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import requests

_DEFAULT_WINDOW_S = 60.0
_DEFAULT_MAX_REQUESTS = int(os.environ.get("NVIDIA_MAX_RPM", "40"))
_DEFAULT_MAX_RETRIES = int(os.environ.get("NVIDIA_MAX_RETRIES", "12"))

_timestamps: deque[float] = deque()


def acquire_rate_slot(
    max_requests: int | None = None,
    window_s: float = _DEFAULT_WINDOW_S,
) -> None:
    """
    Block until a new request is allowed under a sliding window.

    At most `max_requests` calls may occur in any consecutive `window_s` second interval.
    """
    global _timestamps
    cap = max_requests if max_requests is not None else _DEFAULT_MAX_REQUESTS
    now = time.monotonic()

    while True:
        while _timestamps and (now - _timestamps[0]) >= window_s:
            _timestamps.popleft()

        if len(_timestamps) < cap:
            _timestamps.append(now)
            return

        wait_s = _timestamps[0] + window_s - now
        if wait_s > 0:
            time.sleep(wait_s)
        now = time.monotonic()


def post_chat_completions_with_limits(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: float,
    max_rpm: int | None = None,
    window_s: float = _DEFAULT_WINDOW_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> requests.Response:
    """
    POST to chat completions with RPM throttle and automatic retry on HTTP 429.

    NVIDIA may also enforce token-per-minute limits; pairing this with moderate
    `max_tokens` in the JSON body reduces spurious 429s.
    """
    cap = max_rpm if max_rpm is not None else _DEFAULT_MAX_REQUESTS
    backoff_s = 1.0
    last: requests.Response | None = None

    for _ in range(max_retries):
        acquire_rate_slot(max_requests=cap, window_s=window_s)
        last = requests.post(url, headers=headers, json=json_body, timeout=timeout)
        if last.status_code != 429:
            return last

        retry_after = last.headers.get("Retry-After")
        if retry_after is not None:
            try:
                wait_s = float(retry_after)
            except ValueError:
                wait_s = max(_DEFAULT_WINDOW_S, 1.0)
        else:
            wait_s = min(120.0, backoff_s)
            backoff_s = min(backoff_s * 2.0, 90.0)

        time.sleep(max(wait_s, 0.5))

    assert last is not None
    last.raise_for_status()
    return last
