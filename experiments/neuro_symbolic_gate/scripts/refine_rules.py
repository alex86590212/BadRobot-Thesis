#!/usr/bin/env python3
"""
Continual R_safety rule refinement CLI.

Loads one or more JSONL experiment outputs, builds the E⁺/E⁻ experience
buffer, asks an LLM to propose new rules for missed cases, runs the symbolic
verifier, presents verified candidates for human oversight, and writes
accepted rules back to rsafety_v0.yaml.

Usage (from BadRobot-Thesis root):

  # After running run_experiment.py with --split both:
  python3 experiments/neuro_symbolic_gate/scripts/refine_rules.py \\
      --gated experiments/neuro_symbolic_gate/outputs/gated_run.jsonl \\
      --model gpt-4o-mini \\
      --fp_threshold 0.10

  # Accept all verified rules without interactive prompt (CI / scripted):
  python3 ... --auto_accept
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_NSG_ROOT = _SCRIPT.parents[1]
_REPO_ROOT = _NSG_ROOT.parents[1]
for p in (_NSG_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nsg.experience_buffer import ExperienceBuffer  # noqa: E402
from nsg.rule_refiner import (  # noqa: E402
    FP_THRESHOLD,
    accept_with_human_oversight,
    apply_to_yaml,
    propose_rules,
    verify_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual R_safety rule refinement (neuro-symbolic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gated",
        type=Path,
        required=True,
        nargs="+",
        metavar="JSONL",
        help="One or more JSONL files produced by run_experiment.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for rule proposals (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--fp_threshold",
        type=float,
        default=FP_THRESHOLD,
        help=f"Max acceptable false-positive rate for a proposed rule (default: {FP_THRESHOLD})",
    )
    parser.add_argument(
        "--e_minus_limit",
        type=int,
        default=30,
        help="Max E⁻ examples shown to LLM per refinement round (default: 30)",
    )
    parser.add_argument(
        "--safe_limit",
        type=int,
        default=20,
        help="Max safe examples shown to LLM per refinement round (default: 20)",
    )
    parser.add_argument(
        "--auto_accept",
        action="store_true",
        help="Automatically accept all verified candidates without interactive prompt",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Propose and verify rules but do not write to YAML",
    )
    parser.add_argument(
        "--out_report",
        type=Path,
        default=None,
        help="Optional JSON path to write the refinement report",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Set OPENAI_API_KEY or pass --api_key", file=sys.stderr)
        sys.exit(1)

    # ── 1. Build experience buffer ────────────────────────────────────────────
    print(f"\n[refine] Loading {len(args.gated)} JSONL file(s)…")
    buffer = ExperienceBuffer.from_jsonl_files(args.gated)
    summary = buffer.summary()
    print(json.dumps(summary, indent=2))

    if not buffer.e_minus:
        print("[refine] E⁻ is empty — no missed dangerous actions to learn from.  Done.")
        sys.exit(0)

    # ── 2. LLM proposes candidate rules ──────────────────────────────────────
    from openai import OpenAI  # noqa: WPS433
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print(f"\n[refine] Querying {args.model} for rule proposals…")
    candidates = propose_rules(
        buffer,
        client,
        model=args.model,
        e_minus_limit=args.e_minus_limit,
        safe_limit=args.safe_limit,
    )
    print(f"[refine] Received {len(candidates)} candidate rule(s).")

    if not candidates:
        print("[refine] No candidates produced.  Done.")
        sys.exit(0)

    # ── 3. Symbolic verifier ──────────────────────────────────────────────────
    print("\n[refine] Running symbolic verifier…")
    candidates = verify_candidates(
        candidates,
        buffer,
        fp_threshold=args.fp_threshold,
    )

    # ── 4. Human oversight ────────────────────────────────────────────────────
    candidates = accept_with_human_oversight(
        candidates,
        auto_accept_verified=args.auto_accept,
    )

    # ── 5. Write accepted rules back to YAML ─────────────────────────────────
    n_written = 0
    if not args.dry_run:
        yaml_path = _NSG_ROOT / "rules" / "rsafety_v0.yaml"
        n_written = apply_to_yaml(candidates, yaml_path=yaml_path)
    else:
        print("[refine] --dry_run: YAML not modified.")

    # ── 6. Optional report ────────────────────────────────────────────────────
    if args.out_report:
        report = {
            "buffer_summary": summary,
            "candidates": [
                {
                    "rule_id": c.rule_id,
                    "section": c.section,
                    "description": c.description,
                    "patterns": c.patterns,
                    "primitive": c.primitive,
                    "e_minus_caught": c.e_minus_caught,
                    "fp_rate": c.fp_rate,
                    "verified": c.verified,
                    "accepted": c.accepted,
                }
                for c in candidates
            ],
            "rules_written": n_written,
        }
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[refine] Report written to {args.out_report.resolve()}")

    print("\n[refine] Done.")


if __name__ == "__main__":
    main()
