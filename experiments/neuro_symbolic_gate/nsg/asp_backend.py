"""
ASP backend for the Neuro-Symbolic Gate (clingo-based).

Architecture (inspired by CLMASP, Lin et al. 2024):
  Neural side  → LLM generates action plan (TypedPrimitive list)
  Symbolic side → clingo derives violation/2 and needs_confirm/2 atoms

Each TypedPrimitive is compiled into ground ASP facts:
  action(Index, primitive_name).
  keyword(Index, word).   # one atom per lowercase token in the call string

The LP file (rsafety_v0.lp) encodes safety constraints as Horn rules;
clingo's unique answer set contains all derived violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .plan_parser import TypedPrimitive

_DEFAULT_LP = Path(__file__).resolve().parent.parent / "rules" / "rsafety_v0.lp"


@dataclass
class ASPViolation:
    rule_id: str
    step_index: int
    kind: str  # "violation" | "needs_confirm"


# ── token extraction ──────────────────────────────────────────────────────────

def _safe_atom(token: str) -> str | None:
    """Return a valid clingo atom name from a raw token, or None if not convertible."""
    token = re.sub(r"[^a-z0-9_]", "", token.lower())
    if not token or not token[0].isalpha() or len(token) < 3:
        return None
    return token


def _keywords_from_blob(text_blob: str) -> set[str]:
    raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text_blob)
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "into",
        "you", "your", "please", "now", "can", "will", "are", "not",
        "its", "has", "but", "all",
    }
    out: set[str] = set()
    for t in raw_tokens:
        atom = _safe_atom(t)
        if atom and atom not in stop:
            out.add(atom)
    return out


# ── fact generation ───────────────────────────────────────────────────────────

def _build_facts(primitives: list[TypedPrimitive]) -> str:
    lines: list[str] = []
    for i, p in enumerate(primitives):
        name = _safe_atom(p.name) or "unknown"
        lines.append(f"action({i}, {name}).")
        for kw in _keywords_from_blob(p.text_blob):
            lines.append(f"keyword({i}, {kw}).")
    return "\n".join(lines)


# ── clingo solver ─────────────────────────────────────────────────────────────

class ASPBackend:
    """
    Thin wrapper around clingo.  Falls back gracefully if clingo is not installed.
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        self._rules_path = rules_path or _DEFAULT_LP
        self._lp_text: str | None = None
        self._available: bool | None = None

    def _load_lp(self) -> str:
        if self._lp_text is None:
            self._lp_text = self._rules_path.read_text(encoding="utf-8")
        return self._lp_text

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import clingo  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def check_plan(
        self,
        primitives: list[TypedPrimitive],
    ) -> tuple[list[ASPViolation], list[ASPViolation]]:
        """
        Run the LP safety check on the action plan.

        Returns:
            (violations, confirmations)
            violations    → gate should block
            confirmations → gate should pause for human confirmation
        Raises RuntimeError if clingo unavailable (caller should catch and degrade gracefully).
        """
        if not self.available:
            raise RuntimeError(
                "clingo is not installed.  Run: pip install clingo\n"
                "The gate will fall back to YAML-only regex rules."
            )
        import clingo  # type: ignore

        facts = _build_facts(primitives)
        lp = self._load_lp()
        full_program = facts + "\n" + lp

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], full_program)
        ctl.ground([("base", [])])

        violations: list[ASPViolation] = []
        confirmations: list[ASPViolation] = []

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for model in handle:
                for sym in model.symbols(shown=True):
                    if len(sym.arguments) != 2:
                        continue
                    idx = sym.arguments[0].number
                    rid = sym.arguments[1].string  # quoted string constant
                    entry = ASPViolation(rule_id=rid, step_index=idx, kind=sym.name)
                    if sym.name == "violation":
                        violations.append(entry)
                    elif sym.name == "needs_confirm":
                        confirmations.append(entry)

        return violations, confirmations


# ── module-level singleton ────────────────────────────────────────────────────

_backend: ASPBackend | None = None


def get_asp_backend(rules_path: Path | None = None) -> ASPBackend:
    global _backend
    if _backend is None or (rules_path is not None and rules_path != _backend._rules_path):
        _backend = ASPBackend(rules_path=rules_path)
    return _backend
