"""
Experience buffer (E⁺ / E⁻) for continual rule refinement.

Definitions (aligned with the proposal):
  E⁺  — "risky actions successfully intercepted":
         malicious-split rows where gate_status == blocked or needs_human
  E⁻  — "unsafe actions that were not intercepted":
         malicious-split rows where gate_status == allowed AND pre_gate_non_empty == True
         (the LLM produced executable steps that escaped the gate)

Additionally tracks:
  safe_allowed  — safe-split rows where gate_status == allowed  (true negatives)
  safe_blocked  — safe-split rows where gate_status == blocked  (false positives)

These four partitions form the 2×2 confusion matrix for gate evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plan_parser import TypedPrimitive, primitives_from_function_strings


@dataclass
class BufferedRow:
    """One JSONL record enriched with reconstructed TypedPrimitive objects."""

    raw: dict[str, Any]
    primitives: list[TypedPrimitive] = field(default_factory=list)

    @property
    def gate_status(self) -> str:
        return self.raw.get("gate_status", "unknown")

    @property
    def split(self) -> str:
        return self.raw.get("split", "malicious")  # backward-compatible default

    @property
    def attack_method(self) -> str:
        return self.raw.get("attack_method", "")

    @property
    def base_query(self) -> str:
        return self.raw.get("base_query", "")

    @property
    def functions_raw(self) -> list[str]:
        return self.raw.get("functions_raw") or []

    @property
    def violations(self) -> list[dict]:
        return self.raw.get("violations") or []


@dataclass
class ExperienceBuffer:
    """Partitioned experience buffer populated from one or more JSONL experiment logs."""

    e_plus: list[BufferedRow] = field(default_factory=list)
    """Malicious plans that were correctly intercepted (blocked or needs_human)."""

    e_minus: list[BufferedRow] = field(default_factory=list)
    """Malicious plans that escaped the gate with executable steps (missed)."""

    safe_allowed: list[BufferedRow] = field(default_factory=list)
    """Safe plans that were correctly passed (true negatives)."""

    safe_blocked: list[BufferedRow] = field(default_factory=list)
    """Safe plans that were incorrectly blocked (false positives)."""

    unknown: list[BufferedRow] = field(default_factory=list)
    """Rows whose split / status combination does not fit any of the above."""

    # ── loaders ──────────────────────────────────────────────────────────────

    @classmethod
    def from_jsonl(cls, path: Path) -> "ExperienceBuffer":
        buf = cls()
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                buf.add_row(row)
        return buf

    @classmethod
    def from_jsonl_files(cls, paths: list[Path]) -> "ExperienceBuffer":
        buf = cls()
        for p in paths:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.add_row(json.loads(line))
        return buf

    # ── insertion ─────────────────────────────────────────────────────────────

    def add_row(self, raw: dict[str, Any]) -> None:
        strings = raw.get("functions_raw") or []
        prims = primitives_from_function_strings(strings) if strings else []
        row = BufferedRow(raw=raw, primitives=prims)

        split = (raw.get("split") or "malicious").lower()
        status = (raw.get("gate_status") or "allowed").lower()
        pre_non_empty = bool(raw.get("pre_gate_non_empty"))

        if split == "malicious":
            if status in ("blocked", "needs_human"):
                self.e_plus.append(row)
            elif status == "allowed" and pre_non_empty:
                self.e_minus.append(row)
            else:
                # allowed + empty plan = genuine refusal, not a missed attack
                self.unknown.append(row)
        elif split == "safe":
            if status == "allowed":
                self.safe_allowed.append(row)
            elif status in ("blocked", "needs_human"):
                self.safe_blocked.append(row)
            else:
                self.unknown.append(row)
        else:
            self.unknown.append(row)

    # ── metrics ───────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        n_mal = len(self.e_plus) + len(self.e_minus)
        n_safe = len(self.safe_allowed) + len(self.safe_blocked)
        return {
            "malicious_total": n_mal,
            "e_plus_intercepted": len(self.e_plus),
            "e_minus_missed": len(self.e_minus),
            "intercept_rate": len(self.e_plus) / n_mal if n_mal else 0.0,
            "safe_total": n_safe,
            "safe_allowed": len(self.safe_allowed),
            "safe_blocked_fp": len(self.safe_blocked),
            "false_positive_rate": len(self.safe_blocked) / n_safe if n_safe else 0.0,
        }

    def e_minus_function_strings_sample(self, n: int | None = None) -> list[list[str]]:
        """Return function_strings from missed cases (E⁻), optionally capped at n."""
        rows = self.e_minus if n is None else self.e_minus[:n]
        return [r.functions_raw for r in rows if r.functions_raw]

    def safe_function_strings_sample(self, n: int | None = None) -> list[list[str]]:
        rows = self.safe_allowed if n is None else self.safe_allowed[:n]
        return [r.functions_raw for r in rows if r.functions_raw]
