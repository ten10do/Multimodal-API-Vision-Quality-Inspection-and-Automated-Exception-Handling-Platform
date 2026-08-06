"""Deterministic numeric grounding (9H).

Every critical number in the final answer must be supported by the tool
evidence. This validator walks the evidence bundle, collects all numeric
values, then scans the LLM answer for numbers. Any number that cannot be
found in the evidence is replaced with "[insufficient evidence]" and listed
as a limitation -- so the surfaced answer never contains an unsupported
critical numeric claim (acceptance target: unsupported rate = 0).
"""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)  # keep full precision; rounding happens on variants
    return None


def _flatten(values: Any, out: list[float]) -> None:
    if isinstance(values, dict):
        for v in values.values():
            _flatten(v, out)
    elif isinstance(values, (list, tuple)):
        for v in values:
            _flatten(v, out)
    else:
        n = _normalize(values)
        if n is not None and math.isfinite(n):
            out.append(n)


def collect_evidence_numbers(evidence: list[dict]) -> set[float]:
    """All numeric values present in the evidence bundle.

    The supported set also includes scaled variants (x100 for percentages,
    /100, and absolute values) so that an answer correctly phrasing an
    evidence fraction as "92.4%" or a -0.037 delta as "下降 3.7pp" is still
    considered grounded -- while a genuinely absent number (no evidence
    value, scale or sign matches) is flagged."""
    nums: list[float] = []
    for ev in evidence:
        _flatten(ev, nums)
    supported: set[float] = set()
    for n in nums:
        for variant in (n, -n, round(n * 100, 2), round(-n * 100, 2), round(n / 100, 2)):
            supported.add(round(variant, 2))
    return supported


def extract_answer_numbers(text: str) -> list[float]:
    out = []
    for token in _NUMBER_RE.findall(text or ""):
        try:
            out.append(round(float(token), 2))
        except ValueError:
            continue
    return out


def ground_answer(answer: str, evidence: list[dict]) -> tuple[str, list[str]]:
    """Return (grounded_answer, unsupported_notes).

    Unsupported numbers are replaced inline with '[insufficient evidence]'
    and a limitation note is produced for each distinct value."""
    supported = collect_evidence_numbers(evidence)
    numbers = extract_answer_numbers(answer)
    if not numbers:
        return answer, []
    notes: list[str] = []
    seen: set[float] = set()
    for n in numbers:
        if n in supported:
            continue
        if n in seen:
            continue
        seen.add(n)
        notes.append(f"数字 {n:g} 无证据支持，已标注为 insufficient evidence")
    if not notes:
        return answer, []

    def _repl(match: re.Match) -> str:
        try:
            n = round(float(match.group(0)), 2)
        except ValueError:
            return match.group(0)
        return "[insufficient evidence]" if n not in supported else match.group(0)

    grounded = _NUMBER_RE.sub(_repl, answer)
    return grounded, notes
