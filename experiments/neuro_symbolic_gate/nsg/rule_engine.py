from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .plan_parser import TypedPrimitive


@dataclass
class Violation:
    rule_id: str
    message: str


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required for rules/*.yaml (pip install pyyaml "
            "or pip install -r experiments/neuro_symbolic_gate/requirements.txt)"
        ) from e
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RuleEngine:
    """Loads R_safety and evaluates TypedPrimitive entries.

    Produces two disjoint violation lists per plan:
      • block violations   → plan must be blocked immediately
      • confirm violations → plan should be paused for human confirmation
        (only for steps that did not already trigger a block violation)
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        if rules_path is None:
            rules_path = Path(__file__).resolve().parent.parent / "rules" / "rsafety_v0.yaml"
        self._path = rules_path
        data = _load_yaml(rules_path)
        self._structural = list(data.get("structural_rules") or [])
        self._lexical = list(data.get("lexical_rules") or [])
        self._confirm_rules = list(data.get("human_confirmation_rules") or [])
        self._known_safe = set(data.get("known_safe_primitives") or [])

    # ── block-level checks ────────────────────────────────────────────────────

    def check_primitive(self, p: TypedPrimitive, index: int) -> list[Violation]:
        violations: list[Violation] = []
        haystack = (p.raw + " " + p.text_blob).lower()

        for rule in self._structural:
            prim = rule.get("primitive")
            if prim and p.name != prim:
                continue
            for pat in rule.get("patterns") or []:
                if re.search(pat, p.raw, re.DOTALL | re.IGNORECASE):
                    violations.append(
                        Violation(
                            rule_id=rule["id"],
                            message=f"step[{index}] {p.name}: {rule.get('description', pat)}",
                        )
                    )
                    break

        for rule in self._lexical:
            for pat in rule.get("patterns") or []:
                if re.search(pat, haystack, re.DOTALL | re.IGNORECASE):
                    violations.append(
                        Violation(
                            rule_id=rule["id"],
                            message=f"step[{index}] {p.name}: {rule.get('description', pat)}",
                        )
                    )
                    break

        return violations

    def check_plan(self, primitives: list[TypedPrimitive]) -> list[Violation]:
        all_v: list[Violation] = []
        for i, p in enumerate(primitives):
            all_v.extend(self.check_primitive(p, i))
        return all_v

    # ── confirmation-level checks ─────────────────────────────────────────────

    def check_primitive_for_confirmation(
        self, p: TypedPrimitive, index: int, already_blocked: bool
    ) -> list[Violation]:
        """Return confirmation-required violations for a step.

        Never fires for a step that already triggered a block violation.
        """
        if already_blocked:
            return []

        confirms: list[Violation] = []
        haystack = (p.raw + " " + p.text_blob).lower()

        for rule in self._confirm_rules:
            # unknown_primitive sentinel: primitive not in the known-safe list
            if rule.get("unknown_primitive"):
                if self._known_safe and p.name not in self._known_safe:
                    confirms.append(
                        Violation(
                            rule_id=rule["id"],
                            message=(
                                f"step[{index}] {p.name}: "
                                f"{rule.get('description', 'unknown primitive — confirm before execution')}"
                            ),
                        )
                    )
                continue

            prim = rule.get("primitive")
            if prim and p.name != prim:
                continue
            for pat in rule.get("patterns") or []:
                if re.search(pat, haystack, re.DOTALL | re.IGNORECASE):
                    confirms.append(
                        Violation(
                            rule_id=rule["id"],
                            message=f"step[{index}] {p.name}: {rule.get('description', pat)}",
                        )
                    )
                    break

        return confirms

    def check_plan_for_confirmation(
        self,
        primitives: list[TypedPrimitive],
        blocked_indices: set[int],
    ) -> list[Violation]:
        """Return confirmation violations for steps that are not already blocked."""
        all_c: list[Violation] = []
        for i, p in enumerate(primitives):
            all_c.extend(
                self.check_primitive_for_confirmation(p, i, already_blocked=(i in blocked_indices))
            )
        return all_c
