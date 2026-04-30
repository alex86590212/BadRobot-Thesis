#!/usr/bin/env python3
"""Run gated BadRobot digital evaluation via DO-AI endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

# BadRobot-Thesis (repo root) and experiment package root
_SCRIPT = Path(__file__).resolve()
_NSG_ROOT = _SCRIPT.parents[1]
_REPO_ROOT = _NSG_ROOT.parents[1]  # BadRobot-Thesis (parent of experiments/)
for p in (_NSG_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nsg.runner_core import (  # noqa: E402
    default_rules_path,
    iter_experiment,
    load_malicious_queries,
    load_safe_queries,
)


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_REQUESTS_PER_MINUTE = 40
_MIN_REQUEST_INTERVAL_S = 60.0 / MAX_REQUESTS_PER_MINUTE
_LAST_REQUEST_TS = 0.0


def _throttle_requests() -> None:
    global _LAST_REQUEST_TS
    now = time.monotonic()
    wait_s = _MIN_REQUEST_INTERVAL_S - (now - _LAST_REQUEST_TS)
    if wait_s > 0:
        time.sleep(wait_s)
    _LAST_REQUEST_TS = time.monotonic()


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 16384,
    temperature: float = 0.70,
    top_p: float = 1.00,
    reasoning_effort: str = "high",
    stream: bool = False,
    timeout_s: int = 120,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    _throttle_requests()
    response = requests.post(base_url, headers=headers, json=payload, timeout=timeout_s)
    response.raise_for_status()
    if stream:
        chunks = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = ((event.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                chunks.append(delta)
        return "".join(chunks)

    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"Missing choices in response: {body}")
    content = (choices[0].get("message") or {}).get("content")
    if content is None:
        raise ValueError(f"Missing message.content in response: {body}")
    return content


class _RequestsCompatClient:
    """Minimal OpenAI client compatibility for runner_core.iter_experiment."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
        stream: bool,
        timeout_s: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._reasoning_effort = reasoning_effort
        self._stream = stream
        self._timeout_s = timeout_s
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent: "_RequestsCompatClient") -> None:
            self.completions = _RequestsCompatClient._Completions(parent)

    class _Completions:
        def __init__(self, parent: "_RequestsCompatClient") -> None:
            self._parent = parent

        def create(self, *, model: str, messages: list[dict[str, str]]) -> Any:
            content = _post_chat_completion(
                base_url=self._parent._base_url,
                api_key=self._parent._api_key,
                model=model,
                messages=messages,
                max_tokens=self._parent._max_tokens,
                temperature=self._parent._temperature,
                top_p=self._parent._top_p,
                reasoning_effort=self._parent._reasoning_effort,
                stream=self._parent._stream,
                timeout_s=self._parent._timeout_s,
            )
            return type(
                "Resp",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Msg", (), {"content": content})()},
                        )()
                    ]
                },
            )()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-symbolic gated red-team sweep (DO-AI)")
    parser.add_argument("--model", type=str, default="mistralai/mistral-medium-3.5-128b")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DOAI_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.environ.get("DOAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--attack_method",
        type=str,
        choices=["contextual jailbreak", "safety misalignment", "conceptual deception"],
        default="safety misalignment",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["malicious", "safe", "both"],
        default="malicious",
        help="Which query set to run",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max queries per split (0 = all)")
    parser.add_argument("--out", type=Path, default=_NSG_ROOT / "outputs" / "gated_run_doai.jsonl")
    parser.add_argument("--rules", type=Path, default=None, help="Override path to rsafety YAML")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--top_p", type=float, default=1.00)
    parser.add_argument("--reasoning_effort", type=str, default="high")
    parser.add_argument("--stream", type=bool, default=False)
    parser.add_argument("--timeout_s", type=int, default=120)
    args = parser.parse_args()

    if not args.api_key:
        print("Set DOAI_API_KEY or pass --api_key", file=sys.stderr)
        sys.exit(1)

    rules_path = args.rules if args.rules else default_rules_path()
    if not rules_path.is_file():
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        sys.exit(1)

    client = _RequestsCompatClient(
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        stream=args.stream,
        timeout_s=args.timeout_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def progress(i: int, n: int) -> None:
        print(f"\r[{i}/{n}]", end="", flush=True)

    all_rows: list[dict] = []
    if args.split in ("malicious", "both"):
        qs = load_malicious_queries(_REPO_ROOT)
        if args.limit:
            qs = qs[: args.limit]
        print(f"Malicious queries: {len(qs)}")
        all_rows.extend(iter_experiment(client, args.model, qs, args.attack_method, rules_path, "malicious", progress))
        print()

    if args.split in ("safe", "both"):
        safe_path = _NSG_ROOT / "data" / "safe_control_instructions.jsonl"
        qs = load_safe_queries(safe_path)
        if args.limit:
            qs = qs[: args.limit]
        print(f"Safe control queries: {len(qs)}")
        all_rows.extend(iter_experiment(client, args.model, qs, args.attack_method, rules_path, "safe", progress))
        print()

    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} records to {args.out.resolve()}")


if __name__ == "__main__":
    main()
