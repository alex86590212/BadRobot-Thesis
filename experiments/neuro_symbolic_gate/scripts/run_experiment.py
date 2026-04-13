#!/usr/bin/env python3
"""Run gated BadRobot digital evaluation (same LLM + system prompt as baseline)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-symbolic gated red-team sweep")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
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
    parser.add_argument("--out", type=Path, default=_NSG_ROOT / "outputs" / "gated_run.jsonl")
    parser.add_argument("--rules", type=Path, default=None, help="Override path to rsafety YAML")
    args = parser.parse_args()

    from openai import OpenAI  # noqa: WPS433

    if not args.api_key:
        print("Set OPENAI_API_KEY or pass --api_key", file=sys.stderr)
        sys.exit(1)

    rules_path = args.rules if args.rules else default_rules_path()
    if not rules_path.is_file():
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
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
