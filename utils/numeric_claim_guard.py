"""Detect and neutralize numerical claims not supported by forecast evidence.

The language model may paraphrase a forecast with rounded values, so matching
uses a small absolute/relative tolerance. Numbers that cannot be aligned to
the deterministic evidence are replaced with a provenance-safe placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")


@dataclass(frozen=True)
class NumericClaimAudit:
    supported: tuple[float, ...]
    unsupported: tuple[float, ...]

    @property
    def blocked_count(self) -> int:
        return len(self.unsupported)


_TYPED_LABELS = {
    "horizon": re.compile(r"(?:预测\s*(?:步数|长度|步)?|forecast\s*horizon)\s*[:=为约]?\s*", re.I),
    "start": re.compile(r"(?:起点|起|start)\s*[:=为约]?\s*", re.I),
    "end": re.compile(r"(?:终点|止|end)\s*[:=为约]?\s*", re.I),
    "min": re.compile(r"(?:最低|最小|min(?:imum)?)\s*[:=为约]?\s*", re.I),
    "max": re.compile(r"(?:最高|最大|max(?:imum)?)\s*[:=为约]?\s*", re.I),
}
_UNSUPPORTED_TYPED_LABEL = re.compile(r"(?:风险评分|风险分数|增长率|risk\s*score|growth\s*rate)\s*[:=为约]?\s*", re.I)


def audit_typed_numeric_claims(response: str, evidence_fields: dict[str, float], *, rel_tol: float = 0.02, abs_tol: float = 0.02) -> list[float]:
    """Return numeric claims whose *field* is not supported by typed evidence.

    Token matching alone cannot distinguish ``最高=25`` from ``风险评分=25``.
    This small typed layer catches field swaps and unit-like claims when the
    response uses the known evidence labels.  Unlabeled prose remains covered
    by the original conservative token guard.
    """
    bad: list[float] = []
    text = str(response or "")
    for field, pattern in _TYPED_LABELS.items():
        for match in pattern.finditer(text):
            number = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?", text[match.end():])
            if not number:
                continue
            token = number.group(0)
            try: value = float(token.rstrip("%"))
            except ValueError: continue
            expected = evidence_fields.get(field)
            if expected is None or not _close(value, [float(expected)], rel_tol=rel_tol, abs_tol=abs_tol):
                bad.append(value)
    for match in _UNSUPPORTED_TYPED_LABEL.finditer(text):
        number = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?", text[match.end():])
        if number:
            try: bad.append(float(number.group(0).rstrip("%")))
            except ValueError: pass
    return bad


def extract_numeric_claims(text: str) -> list[tuple[str, float]]:
    """Extract numeric tokens and parsed values from a generated response."""
    claims: list[tuple[str, float]] = []
    for match in _NUMBER_RE.finditer(str(text or "")):
        token = match.group(0)
        try:
            value = float(token.rstrip("%"))
        except ValueError:
            continue
        claims.append((token, value))
    return claims


def _close(value: float, evidence: list[float], *, rel_tol: float, abs_tol: float) -> bool:
    for ref in evidence:
        scale = max(abs(ref), abs(value), 1.0)
        if abs(value - ref) <= max(abs_tol, rel_tol * scale):
            return True
    return False


def audit_numeric_claims(
    response: str,
    evidence: str,
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.02,
) -> NumericClaimAudit:
    """Audit response numbers against numbers present in deterministic evidence."""
    evidence_values = [value for _, value in extract_numeric_claims(evidence)]
    supported: list[float] = []
    unsupported: list[float] = []
    for _, value in extract_numeric_claims(response):
        if _close(value, evidence_values, rel_tol=rel_tol, abs_tol=abs_tol):
            supported.append(value)
        else:
            unsupported.append(value)
    return NumericClaimAudit(tuple(supported), tuple(unsupported))


def enforce_numeric_claims(
    response: str,
    evidence: str,
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.02,
    typed_evidence: dict[str, float] | None = None,
) -> tuple[str, NumericClaimAudit]:
    """Replace unsupported numeric tokens while preserving the text response."""
    audit = audit_numeric_claims(response, evidence, rel_tol=rel_tol, abs_tol=abs_tol)
    typed_bad = audit_typed_numeric_claims(response, typed_evidence or {}, rel_tol=rel_tol, abs_tol=abs_tol) if typed_evidence else []
    if typed_bad:
        audit = NumericClaimAudit(audit.supported, tuple(list(audit.unsupported) + typed_bad))
    if not audit.unsupported:
        return response, audit

    unsupported_iter = iter(audit.unsupported)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            value = float(token.rstrip("%"))
        except ValueError:
            return token
        if any(abs(value - blocked) <= max(abs_tol, rel_tol * max(abs(value), abs(blocked), 1.0))
               for blocked in audit.unsupported):
            return "[数值已隐藏：无对应预测证据]"
        return token

    return _NUMBER_RE.sub(replace, str(response or "")), audit
