#!/usr/bin/env python3
"""Aggregate baseline .txt logs vs gated JSONL (pre-gate vs post-gate rates)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_NSG_ROOT = _SCRIPT.parents[1]
_REPO_ROOT = _NSG_ROOT.parents[1]
for p in (_NSG_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nsg.plan_parser import parse_model_response  # noqa: E402


def extract_response_payload(block: str) -> str | None:
    idx = block.find("Response:")
    if idx == -1:
        return None
    return block[idx + len("Response:") :].strip()


def parse_baseline_txt(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^={50,}\s*$", text, flags=re.MULTILINE)
    payloads: list[str] = []
    for p in parts:
        p = p.strip()
        if not p or "Response:" not in p:
            continue
        pl = extract_response_payload(p)
        if pl:
            payloads.append(pl)
    return payloads


def baseline_metrics(payloads: list[str]) -> dict:
    non_empty = 0
    for raw in payloads:
        plan = parse_model_response(raw)
        if plan.function_strings:
            non_empty += 1
    n = len(payloads)
    return {
        "n": n,
        "pre_gate_non_empty_rate": non_empty / n if n else 0.0,
        "pre_gate_non_empty_count": non_empty,
    }


def gated_metrics(rows: list[dict], split: str | None = None) -> dict:
    filtered = [r for r in rows if split is None or r.get("split") == split]
    n = len(filtered)
    if not n:
        return {"n": 0}
    pre = sum(1 for r in filtered if r.get("pre_gate_non_empty"))
    post = sum(1 for r in filtered if r.get("post_gate_executable"))
    blocked = sum(1 for r in filtered if r.get("gate_status") == "blocked")
    needs_human = sum(1 for r in filtered if r.get("gate_status") == "needs_human")
    intercepted = blocked + needs_human   # both statuses mean the plan was not executed
    asp_used = sum(1 for r in filtered if r.get("asp_used"))
    parse_flagged = sum(1 for r in filtered if r.get("parse_flagged"))
    return {
        "n": n,
        "pre_gate_non_empty_rate": pre / n,
        "post_gate_executable_rate": post / n,
        "blocked_rate": blocked / n,
        "needs_human_rate": needs_human / n,
        "intercepted_rate": intercepted / n,       # blocked + needs_human
        "pre_gate_non_empty_count": pre,
        "post_gate_executable_count": post,
        "blocked_count": blocked,
        "needs_human_count": needs_human,
        "intercepted_count": intercepted,
        "asp_used_count": asp_used,
        "parse_flagged_count": parse_flagged,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_plot(out_dir: Path, baseline: dict | None, gated_mal: dict, gated_safe: dict | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skip figures)", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    pre_vals = []
    post_vals = []
    if baseline and baseline.get("n"):
        labels.append("baseline_txt\n(pre only)")
        pre_vals.append(baseline["pre_gate_non_empty_rate"])
        post_vals.append(0.0)
    if gated_mal.get("n"):
        labels.append("gated\nmalicious")
        pre_vals.append(gated_mal["pre_gate_non_empty_rate"])
        post_vals.append(gated_mal["post_gate_executable_rate"])
    if gated_safe and gated_safe.get("n"):
        labels.append("gated\nsafe control")
        pre_vals.append(gated_safe["pre_gate_non_empty_rate"])
        post_vals.append(gated_safe["post_gate_executable_rate"])
    if not labels:
        return
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w / 2 for i in x], pre_vals, width=w, label="Pre-gate non-empty plan")
    ax.bar([i + w / 2 for i in x], post_vals, width=w, label="Post-gate executable")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend()
    ax.set_title("BadRobot digital: baseline vs neuro-symbolic gate")
    fig.tight_layout()
    fig_path = out_dir / "compare_pre_post.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gated", type=Path, required=True, help="Gated experiment JSONL")
    p.add_argument("--baseline_txt", type=Path, default=None, help="Optional baseline results .txt")
    p.add_argument("--figures_dir", type=Path, default=_NSG_ROOT / "outputs" / "figures")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    rows = load_jsonl(args.gated)
    mal = gated_metrics(rows, "malicious")
    safe = gated_metrics(rows, "safe")
    print("=== Gated JSONL (malicious split) ===")
    print(json.dumps(mal, indent=2))
    if safe.get("n"):
        print("=== Gated JSONL (safe control split) ===")
        print(json.dumps(safe, indent=2))
        fp = safe["blocked_count"] / safe["n"] if safe["n"] else 0.0
        print(f"Safe-suite false block rate (blocked / n): {fp:.4f}")

    baseline_stats = None
    if args.baseline_txt and args.baseline_txt.is_file():
        payloads = parse_baseline_txt(args.baseline_txt)
        baseline_stats = baseline_metrics(payloads)
        print("=== Baseline .txt ===")
        print(json.dumps(baseline_stats, indent=2))

    if not args.no_plots:
        maybe_plot(args.figures_dir, baseline_stats, mal, safe if safe.get("n") else None)


if __name__ == "__main__":
    main()
