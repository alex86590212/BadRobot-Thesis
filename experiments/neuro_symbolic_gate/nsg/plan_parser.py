from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TypedPrimitive:
    """One entry from the model's `function` list."""

    name: str
    raw: str
    """Lowercased concatenation of quoted string args + raw for lexical rules."""
    text_blob: str = field(default="")


@dataclass
class ParsedPlan:
    function_strings: list[str]
    primitives: list[TypedPrimitive]
    parse_error: str | None = None


def extract_function_list_inner(response_text: str) -> str | None:
    """Return the inside of the first 'function': [...] list, or None."""
    m = re.search(r'''["']function["']\s*:\s*''', response_text)
    if not m:
        return None
    i = m.end()
    while i < len(response_text) and response_text[i] in " \t\n\r":
        i += 1
    if i >= len(response_text) or response_text[i] != "[":
        return None
    start = i
    depth = 0
    for j in range(start, len(response_text)):
        c = response_text[j]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return response_text[start + 1 : j]
    return None


def split_top_level_commas(inner: str) -> list[str]:
    """Split comma-separated call strings inside [...], respecting quotes and nesting."""
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_bracket = 0
    in_squote = False
    in_dquote = False
    escape = False
    i = 0
    while i < len(inner):
        c = inner[i]
        if escape:
            buf.append(c)
            escape = False
            i += 1
            continue
        if c == "\\" and (in_squote or in_dquote):
            escape = True
            buf.append(c)
            i += 1
            continue
        if in_dquote:
            if c == '"':
                in_dquote = False
            buf.append(c)
            i += 1
            continue
        if in_squote:
            if c == "'":
                in_squote = False
            buf.append(c)
            i += 1
            continue
        if c == '"':
            in_dquote = True
            buf.append(c)
            i += 1
            continue
        if c == "'":
            in_squote = True
            buf.append(c)
            i += 1
            continue
        if c == "(":
            depth_paren += 1
            buf.append(c)
        elif c == ")":
            depth_paren -= 1
            buf.append(c)
        elif c == "[":
            depth_bracket += 1
            buf.append(c)
        elif c == "]":
            depth_bracket -= 1
            buf.append(c)
        elif c == "," and depth_paren == 0 and depth_bracket == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        part = "".join(buf).strip()
        if part:
            parts.append(part)
    return parts


def _extract_quoted_literals(s: str) -> str:
    """Concatenate contents of '...' and \"...\" literals for keyword scans."""
    chunks: list[str] = []
    for m in re.finditer(r'''(["'])((?:\\.|(?!\1).)*)\1''', s, re.DOTALL):
        chunks.append(m.group(2))
    return " ".join(chunks).lower()


def _primitive_name(raw_call: str) -> str:
    raw_call = raw_call.strip()
    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", raw_call)
    return m.group(1) if m else "unknown"


def normalize_call_entry(e: str) -> str:
    """Strip one layer of matching outer quotes if the payload still looks like a call."""
    e = e.strip()
    if len(e) >= 2 and e[0] == e[-1] and e[0] in "\"'" and "(" in e[1:-1]:
        return e[1:-1].strip()
    return e


def parse_function_inner_to_primitives(inner: str | None) -> tuple[list[str], list[TypedPrimitive], str | None]:
    if inner is None:
        return [], [], "missing_function_key"
    inner_stripped = inner.strip()
    if not inner_stripped:
        return [], [], None
    entries = [normalize_call_entry(x) for x in split_top_level_commas(inner_stripped)]
    primitives: list[TypedPrimitive] = []
    for e in entries:
        name = _primitive_name(e)
        literals = _extract_quoted_literals(e)
        blob = (literals + " " + e.lower()).strip()
        primitives.append(TypedPrimitive(name=name, raw=e.strip(), text_blob=blob))
    return entries, primitives, None


def parse_model_response(response_text: str) -> ParsedPlan:
    """Parse raw LLM message content into structured plan."""
    inner = extract_function_list_inner(response_text)
    strings, primitives, err = parse_function_inner_to_primitives(inner)
    return ParsedPlan(function_strings=strings, primitives=primitives, parse_error=err)


def primitives_from_function_strings(strings: list[str]) -> list[TypedPrimitive]:
    """Reconstruct TypedPrimitive objects from already-split function call strings.

    Used by the experience buffer and rule refiner to re-evaluate stored JSONL rows
    without re-querying the LLM.
    """
    primitives: list[TypedPrimitive] = []
    for s in strings:
        name = _primitive_name(s)
        literals = _extract_quoted_literals(s)
        blob = (literals + " " + s.lower()).strip()
        primitives.append(TypedPrimitive(name=name, raw=s.strip(), text_blob=blob))
    return primitives
