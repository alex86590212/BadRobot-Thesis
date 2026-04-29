#!/usr/bin/env python3
"""Evaluate ScienceQA accuracy on OpenAI-compatible API with and without NSG gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI


_SCRIPT = Path(__file__).resolve()
_NSG_ROOT = _SCRIPT.parents[1]
_REPO_ROOT = _NSG_ROOT.parents[1]
for p in (_NSG_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nsg.plan_parser import parse_model_response  # noqa: E402
from nsg.runner_core import default_rules_path  # noqa: E402
from nsg.safety_gate import evaluate_parsed_plan  # noqa: E402


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _chat_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Missing message.content in OpenAI response")
    return content.strip()


def _idx_to_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def _extract_answer_letter(text: str, num_choices: int) -> str | None:
    max_letter = _idx_to_letter(max(0, num_choices - 1))
    m = re.search(rf"\b([A-{max_letter}])\b", text.upper())
    if m:
        return m.group(1)
    return None


def _build_prompt(row: dict[str, Any]) -> str:
    question = (row.get("question") or "").strip()
    hint = (row.get("hint") or "").strip()
    lecture = (row.get("lecture") or "").strip()
    choices = row.get("choices") or []

    lines: list[str] = []
    lines.append("You are solving a science multiple-choice question.")
    lines.append("Return only the final answer letter (A, B, C, D, E, ...).")
    lines.append("")
    if hint:
        lines.append(f"Hint: {hint}")
        lines.append("")
    if lecture:
        lines.append(f"Background: {lecture}")
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Choices:")
    for i, choice in enumerate(choices):
        lines.append(f"{_idx_to_letter(i)}) {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _evaluate_one(
    *,
    row: dict[str, Any],
    client: OpenAI,
    model: str,
    max_tokens: int,
    rules_path: Path,
    blocked_policy: str,
) -> dict[str, Any]:
    prompt = _build_prompt(row)
    messages = [{"role": "user", "content": prompt}]
    raw = _chat_completion(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    choices = row.get("choices") or []
    gold_idx = int(row.get("answer", -1))
    gold_letter = _idx_to_letter(gold_idx) if 0 <= gold_idx < len(choices) else None
    pred_letter = _extract_answer_letter(raw, len(choices)) if choices else None
    correct_no_gate = (pred_letter == gold_letter) if (pred_letter and gold_letter) else False

    parsed = parse_model_response(raw)
    gate = evaluate_parsed_plan(parsed, rules_path=rules_path)
    is_blocked = gate.status.value != "allowed"

    if is_blocked and blocked_policy == "abstain":
        correct_with_gate = None
    else:
        correct_with_gate = False if is_blocked else correct_no_gate

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": row.get("question"),
        "choices": choices,
        "gold_index": gold_idx,
        "gold_letter": gold_letter,
        "raw_response": raw,
        "pred_letter": pred_letter,
        "correct_no_gate": bool(correct_no_gate),
        "gate_status": gate.status.value,
        "gate_violations": gate.violations,
        "gate_confirmation_requests": gate.confirmation_requests,
        "parse_error": parsed.parse_error,
        "correct_with_gate": correct_with_gate,
        "blocked": is_blocked,
    }


def _safe_rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate derek-thomas/ScienceQA with and without NSG gate via OpenAI API."
    )
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument(
        "--blocked_policy",
        type=str,
        choices=["incorrect", "abstain"],
        default="incorrect",
        help="How to score blocked outputs for gated accuracy.",
    )
    parser.add_argument("--rules", type=Path, default=None, help="Override NSG rules path")
    parser.add_argument(
        "--out_jsonl",
        type=Path,
        default=_NSG_ROOT / "outputs" / "scienceqa_openai_eval.jsonl",
    )
    parser.add_argument(
        "--out_summary",
        type=Path,
        default=_NSG_ROOT / "outputs" / "scienceqa_openai_eval_summary.json",
    )
    args = parser.parse_args()

    rules_path = args.rules if args.rules else default_rules_path()
    if not rules_path.is_file():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    ds = load_dataset("derek-thomas/ScienceQA", split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    total = len(ds)
    no_gate_correct = 0
    with_gate_correct = 0
    with_gate_scored = 0
    blocked_count = 0
    parse_error_count = 0

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds, start=1):
            result = _evaluate_one(
                row=row,
                client=client,
                model=args.model,
                max_tokens=args.max_tokens,
                rules_path=rules_path,
                blocked_policy=args.blocked_policy,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            no_gate_correct += int(result["correct_no_gate"])
            blocked_count += int(result["blocked"])
            parse_error_count += int(bool(result["parse_error"]))

            if result["correct_with_gate"] is not None:
                with_gate_scored += 1
                with_gate_correct += int(bool(result["correct_with_gate"]))

            if i % 10 == 0 or i == total:
                print(f"\r[{i}/{total}] processed", end="", flush=True)
    print()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "derek-thomas/ScienceQA",
        "split": args.split,
        "num_examples": total,
        "model": args.model,
        "base_url": args.base_url,
        "blocked_policy": args.blocked_policy,
        "accuracy_no_gate": _safe_rate(no_gate_correct, total),
        "accuracy_with_gate": _safe_rate(with_gate_correct, with_gate_scored),
        "with_gate_scored_examples": with_gate_scored,
        "blocked_count": blocked_count,
        "blocked_rate": _safe_rate(blocked_count, total),
        "parse_error_count": parse_error_count,
        "parse_error_rate": _safe_rate(parse_error_count, total),
        "rules_path": str(rules_path),
        "results_jsonl": str(args.out_jsonl.resolve()),
    }
    with args.out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote per-example results to: {args.out_jsonl.resolve()}")
    print(f"Wrote summary to: {args.out_summary.resolve()}")


if __name__ == "__main__":
    main()
