"""
Continual rule refinement for R_safety (NeSyC-inspired).

Pipeline
--------
1. Load E⁺ / E⁻ from the experience buffer.
2. LLM proposes candidate rule additions based on E⁻ patterns.
3. Symbolic verifier tests each candidate:
     a. Coverage    : must catch ≥ 1 E⁻ example.
     b. Regression  : must not remove any existing E⁺ intercepts.
     c. Specificity : false-positive rate on safe split ≤ FP_THRESHOLD.
4. Verified candidates are shown to a human operator for final acceptance.
5. Accepted rules are merged into rsafety_v0.yaml and saved.

References:
  NeSyC (Choi et al., ICLR 2025) — continual neuro-symbolic learner.
  CLMASP (Lin et al., 2024)      — LLM + ASP for robotic task planning.
"""

from __future__ import annotations

import json
import re
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .experience_buffer import ExperienceBuffer
from .plan_parser import primitives_from_function_strings
from .rule_engine import RuleEngine

_YAML_PATH = Path(__file__).resolve().parent.parent / "rules" / "rsafety_v0.yaml"

# Maximum acceptable false-positive rate for a proposed rule.
FP_THRESHOLD = 0.10


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class CandidateRule:
    section: str          # "lexical_rules" | "structural_rules"
    rule_id: str
    description: str
    patterns: list[str]
    primitive: str | None = None   # only for structural_rules

    # Verification results (filled in by the verifier)
    e_minus_caught: int = 0
    fp_count: int = 0
    fp_rate: float = 0.0
    regression_safe: bool = True   # True = no existing E⁺ case lost
    verified: bool = False
    accepted: bool = False

    def to_yaml_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": self.rule_id,
            "description": self.description,
            "patterns": self.patterns,
        }
        if self.primitive:
            entry["primitive"] = self.primitive
        return entry


# ── LLM proposal ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a safety rule engineer for a neuro-symbolic robot safety gate.
    The gate checks LLM-generated action plans (lists of function call strings)
    against a YAML rule base (R_safety) before physical execution.

    You will be shown:
      • The current rule base (YAML text).
      • Dangerous action plans that SLIPPED THROUGH the gate (E⁻ — missed cases).
      • Benign action plans that MUST NOT be blocked (safe control examples).

    Your task: propose 1–4 new YAML rules that would catch more of the missed
    dangerous plans without introducing false positives on the safe examples.

    Rules use Python-compatible regex patterns applied case-insensitively to the
    raw function call string (e.g. "vlm_move(\"human_wrist\", 0.9)").

    Respond ONLY with a JSON object of the form:
    {
      "proposed_rules": [
        {
          "section":     "lexical_rules",
          "id":          "R_LEX_PROPOSED_001",
          "description": "one-line description",
          "patterns":    ["(?i)(regex_pattern)"]
        },
        {
          "section":     "structural_rules",
          "id":          "R_STRUCT_PROPOSED_001",
          "description": "one-line description",
          "primitive":   "primitive_name",
          "patterns":    ["(?i)(regex_pattern)"]
        }
      ]
    }
    Do NOT propose rules that would match the safe control examples.
""")


def _format_function_samples(samples: list[list[str]], label: str, limit: int = 20) -> str:
    lines = [f"=== {label} (showing up to {limit}) ==="]
    for i, fns in enumerate(samples[:limit]):
        lines.append(f"[{i}] {', '.join(fns)}")
    return "\n".join(lines)


def propose_rules(
    buffer: ExperienceBuffer,
    client: Any,
    model: str = "gpt-4o-mini",
    e_minus_limit: int = 30,
    safe_limit: int = 20,
) -> list[CandidateRule]:
    """Ask the LLM to propose new YAML rules based on E⁻ patterns.

    Args:
        buffer:       Populated ExperienceBuffer.
        client:       openai.OpenAI client instance.
        model:        Model to use for proposal.
        e_minus_limit: Max E⁻ examples to include in the prompt.
        safe_limit:   Max safe examples to include in the prompt.

    Returns:
        List of CandidateRule objects (unverified).
    """
    e_minus_samples = buffer.e_minus_function_strings_sample(e_minus_limit)
    safe_samples = buffer.safe_function_strings_sample(safe_limit)

    if not e_minus_samples:
        print("[refiner] No E⁻ examples in buffer — nothing to propose rules for.")
        return []

    current_yaml = _YAML_PATH.read_text(encoding="utf-8")

    user_message = "\n\n".join([
        "## Current R_safety rule base (YAML)\n```yaml\n" + current_yaml + "\n```",
        _format_function_samples(e_minus_samples, "MISSED DANGEROUS plans (E⁻)", e_minus_limit),
        _format_function_samples(safe_samples, "SAFE plans (must NOT be blocked)", safe_limit),
    ])

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    raw_json = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        print(f"[refiner] JSON parse error from LLM: {exc}")
        return []

    candidates: list[CandidateRule] = []
    for entry in data.get("proposed_rules") or []:
        try:
            c = CandidateRule(
                section=str(entry.get("section", "lexical_rules")),
                rule_id=str(entry.get("id", "R_PROPOSED_???")),
                description=str(entry.get("description", "")),
                patterns=[str(p) for p in entry.get("patterns") or []],
                primitive=entry.get("primitive") or None,
            )
            candidates.append(c)
        except Exception as exc:  # noqa: BLE001
            print(f"[refiner] Skipped malformed candidate: {entry!r} — {exc}")
    return candidates


# ── symbolic verifier ─────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _engine_with_extra_rule(base_yaml: dict, candidate: CandidateRule) -> RuleEngine:
    """Return a temporary RuleEngine with one extra rule injected."""
    import tempfile, yaml as _yaml  # type: ignore

    merged = deepcopy(base_yaml)
    section_key = candidate.section  # "lexical_rules" or "structural_rules"
    merged.setdefault(section_key, []).append(candidate.to_yaml_entry())

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        _yaml.dump(merged, tmp, default_flow_style=False)
        tmp_path = Path(tmp.name)
    try:
        engine = RuleEngine(rules_path=tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return engine


def verify_candidates(
    candidates: list[CandidateRule],
    buffer: ExperienceBuffer,
    fp_threshold: float = FP_THRESHOLD,
    yaml_path: Path = _YAML_PATH,
) -> list[CandidateRule]:
    """
    Symbolic verifier: test each candidate against the experience buffer.

    Three checks:
      1. Coverage    — candidate catches ≥ 1 E⁻ example.
      2. Specificity — FP rate on safe split ≤ fp_threshold.
      3. Regression  — existing E⁺ cases are still intercepted by the
                       ORIGINAL rules (candidate must not remove coverage,
                       which is impossible for additive rules — always True).

    Returns the same list with verification fields populated.
    Candidates that fail coverage or specificity have verified=False.
    """
    base_yaml = _load_yaml(yaml_path)
    base_engine = RuleEngine(rules_path=yaml_path)

    # Pre-compute which E⁺ cases the base engine already intercepts.
    e_plus_intercept_ids: set[int] = set()
    for idx, row in enumerate(buffer.e_plus):
        if row.primitives and base_engine.check_plan(row.primitives):
            e_plus_intercept_ids.add(idx)

    for c in candidates:
        # Validate that patterns compile
        try:
            for pat in c.patterns:
                re.compile(pat)
        except re.error as exc:
            print(f"[verifier] {c.rule_id}: invalid regex — {exc}")
            c.verified = False
            continue

        try:
            aug_engine = _engine_with_extra_rule(base_yaml, c)
        except Exception as exc:  # noqa: BLE001
            print(f"[verifier] {c.rule_id}: engine build failed — {exc}")
            c.verified = False
            continue

        # 1. Coverage: does the new rule catch any E⁻ example?
        caught = 0
        for row in buffer.e_minus:
            if row.primitives and aug_engine.check_plan(row.primitives):
                caught += 1
        c.e_minus_caught = caught

        # 2. Specificity: false-positive rate on safe examples.
        fp = 0
        for row in buffer.safe_allowed:
            if row.primitives and aug_engine.check_plan(row.primitives):
                fp += 1
        safe_n = len(buffer.safe_allowed)
        c.fp_count = fp
        c.fp_rate = fp / safe_n if safe_n else 0.0

        # 3. Regression: since we only ADD rules, existing E⁺ coverage is preserved.
        c.regression_safe = True

        c.verified = (c.e_minus_caught >= 1) and (c.fp_rate <= fp_threshold)

        status = "✓ PASS" if c.verified else "✗ FAIL"
        print(
            f"[verifier] {c.rule_id}: {status}  "
            f"E⁻ caught={c.e_minus_caught}  FP={c.fp_count}/{safe_n} ({c.fp_rate:.2%})"
        )

    return candidates


# ── human oversight ───────────────────────────────────────────────────────────

def accept_with_human_oversight(
    candidates: list[CandidateRule],
    auto_accept_verified: bool = False,
) -> list[CandidateRule]:
    """
    Present verified candidates to the human operator for final acceptance.

    If auto_accept_verified is True, verified candidates are accepted without
    interactive prompt (useful in non-interactive / CI contexts).

    Returns the list with accepted=True on accepted entries.
    """
    verified = [c for c in candidates if c.verified]
    if not verified:
        print("[oversight] No verified candidates to review.")
        return candidates

    print("\n" + "=" * 70)
    print("HUMAN OVERSIGHT: Review proposed rule additions to R_safety")
    print("=" * 70)

    for c in verified:
        print(f"\nRule ID   : {c.rule_id}")
        print(f"Section   : {c.section}")
        print(f"Primitive : {c.primitive or '(any)'}")
        print(f"Description: {c.description}")
        print(f"Patterns  : {c.patterns}")
        print(f"Verification — E⁻ caught: {c.e_minus_caught}  FP rate: {c.fp_rate:.2%}")

        if auto_accept_verified:
            c.accepted = True
            print("  → Auto-accepted (auto_accept_verified=True)")
        else:
            ans = input("  Accept this rule? [y/N] ").strip().lower()
            c.accepted = ans in ("y", "yes")
            print(f"  → {'Accepted' if c.accepted else 'Rejected'}")

    accepted_count = sum(1 for c in candidates if c.accepted)
    print(f"\n[oversight] Accepted {accepted_count}/{len(verified)} verified candidates.")
    return candidates


# ── YAML writer ───────────────────────────────────────────────────────────────

def apply_to_yaml(
    candidates: list[CandidateRule],
    yaml_path: Path = _YAML_PATH,
    backup: bool = True,
) -> int:
    """
    Merge accepted candidates into the YAML rule base and write it back.

    Args:
        candidates: Candidate list (only accepted=True entries are written).
        yaml_path:  Path to rsafety_v0.yaml.
        backup:     If True, writes a .bak file before overwriting.

    Returns:
        Number of rules actually written.
    """
    import yaml  # type: ignore

    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        print("[apply] No accepted candidates — YAML unchanged.")
        return 0

    data = _load_yaml(yaml_path)
    if backup:
        bak = yaml_path.with_suffix(".yaml.bak")
        bak.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[apply] Backup written to {bak}")

    for c in accepted:
        section = c.section
        data.setdefault(section, []).append(c.to_yaml_entry())
        print(f"[apply] Added {c.rule_id} to {section}")

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[apply] Wrote {len(accepted)} new rule(s) to {yaml_path}")
    return len(accepted)
