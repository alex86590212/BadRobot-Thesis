"""
Safety gate: post-planning, pre-execution symbolic filter.

Pipeline (per plan):
  1. Parse model response → TypedPrimitive list  (plan_parser)
  2. YAML regex engine   → block / confirm violations  (rule_engine)
  3. ASP clingo engine   → block / confirm violations  (asp_backend)   [optional]
  4. Merge results:
       any block violation  → GateStatus.blocked
       any confirm, no block → GateStatus.needs_human
       neither              → GateStatus.allowed

Design rationale:
  A parse failure (model produced no 'function' key) is treated as
  GateStatus.allowed with a warning flag rather than a hard block,
  because many legitimate refusals also omit the function field.
  Callers can inspect GateResult.parse_flagged to audit this path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .plan_parser import ParsedPlan, TypedPrimitive, parse_model_response
from .rule_engine import RuleEngine, Violation


class GateStatus(str, Enum):
    allowed = "allowed"
    blocked = "blocked"
    needs_human = "needs_human"


@dataclass
class GateResult:
    status: GateStatus
    violations: list[dict[str, str]] = field(default_factory=list)
    """Serialisable list of {rule_id, message} for block-level violations."""
    confirmation_requests: list[dict[str, str]] = field(default_factory=list)
    """Serialisable list of {rule_id, message} for confirmation-level requests."""
    parse_flagged: bool = False
    """True when the model response had no parseable function list (possible evasion)."""
    asp_used: bool = False
    """True when the clingo ASP backend contributed to the result."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "violations": self.violations,
            "confirmation_requests": self.confirmation_requests,
            "parse_flagged": self.parse_flagged,
            "asp_used": self.asp_used,
        }


def _dedup(vs: list[Violation]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for v in vs:
        key = (v.rule_id, v.message)
        if key not in seen:
            seen.add(key)
            out.append({"rule_id": v.rule_id, "message": v.message})
    return out


def evaluate_plan(
    primitives: list[TypedPrimitive],
    rules_path: Path | None = None,
    use_asp: bool = True,
) -> GateResult:
    """
    Run R_safety (YAML + optional ASP) on a list of parsed primitives.

    Args:
        primitives:  Parsed action steps from the LLM response.
        rules_path:  Override path to rsafety_v0.yaml.  LP path is derived automatically.
        use_asp:     Whether to additionally run the clingo ASP backend.
                     Falls back silently if clingo is not installed.
    """
    engine = RuleEngine(rules_path=rules_path)

    # ── YAML-based checks ────────────────────────────────────────────────────
    yaml_block_violations = engine.check_plan(primitives)
    blocked_indices = {
        int(v.message.split("[")[1].split("]")[0])
        for v in yaml_block_violations
        if "[" in v.message and "]" in v.message
    }
    yaml_confirm_violations = engine.check_plan_for_confirmation(primitives, blocked_indices)

    # ── ASP-based checks ─────────────────────────────────────────────────────
    asp_block: list[Violation] = []
    asp_confirm: list[Violation] = []
    asp_used = False

    if use_asp and primitives:
        try:
            from .asp_backend import get_asp_backend  # noqa: PLC0415

            lp_path = (
                rules_path.parent / "rsafety_v0.lp"
                if rules_path
                else None
            )
            backend = get_asp_backend(lp_path)
            if backend.available:
                raw_v, raw_c = backend.check_plan(primitives)
                asp_block = [
                    Violation(rule_id=v.rule_id, message=f"step[{v.step_index}] ASP: {v.rule_id}")
                    for v in raw_v
                ]
                asp_confirm = [
                    Violation(rule_id=v.rule_id, message=f"step[{v.step_index}] ASP: {v.rule_id}")
                    for v in raw_c
                    if v.step_index not in blocked_indices
                ]
                asp_used = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"ASP backend error (falling back to YAML-only): {exc}", stacklevel=2)

    # ── Merge ─────────────────────────────────────────────────────────────────
    all_block = yaml_block_violations + asp_block
    all_confirm = yaml_confirm_violations + asp_confirm

    if all_block:
        return GateResult(
            status=GateStatus.blocked,
            violations=_dedup(all_block),
            confirmation_requests=_dedup(all_confirm),
            asp_used=asp_used,
        )
    if all_confirm:
        return GateResult(
            status=GateStatus.needs_human,
            violations=[],
            confirmation_requests=_dedup(all_confirm),
            asp_used=asp_used,
        )
    return GateResult(
        status=GateStatus.allowed,
        violations=[],
        confirmation_requests=[],
        asp_used=asp_used,
    )


def evaluate_parsed_plan(
    plan: ParsedPlan,
    rules_path: Path | None = None,
    use_asp: bool = True,
) -> GateResult:
    """Evaluate a ParsedPlan.  A missing function key is flagged but not blocked."""
    if plan.parse_error:
        # Many genuine refusals have no function field, so we allow but flag.
        return GateResult(
            status=GateStatus.allowed,
            violations=[],
            parse_flagged=True,
        )
    return evaluate_plan(plan.primitives, rules_path=rules_path, use_asp=use_asp)


def evaluate_raw_model_response(
    response_text: str,
    rules_path: Path | None = None,
    use_asp: bool = True,
) -> tuple[ParsedPlan, GateResult]:
    plan = parse_model_response(response_text)
    return plan, evaluate_parsed_plan(plan, rules_path=rules_path, use_asp=use_asp)
