"""Shared contracts for LLM-assisted baseline experiments.

Every adapter writes one row per manifest sample.  Rows remain useful when a
third-party implementation is unavailable: the error and comparability
fields make the missing coverage explicit instead of silently dropping it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def extract_json(text: str | None) -> dict[str, Any]:
    """Parse a JSON object from a strict or fenced model response."""
    if not text:
        return {}
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        pass
    for match in re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", str(text), re.DOTALL):
        try:
            value = json.loads(match)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    text = str(text)
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                        return value if isinstance(value, dict) else {}
                    except json.JSONDecodeError:
                        break
    return {}


def fit_horizon(values: Any, horizon: int) -> list[float]:
    """Make a prediction exactly as long as the manifest target."""
    if not isinstance(values, (list, tuple)):
        return []
    result = [float(value) for value in values]
    if len(result) >= horizon:
        return result[:horizon]
    if result:
        return result + [result[-1]] * (horizon - len(result))
    return [0.0] * horizon


@dataclass
class BaselineRecord:
    """Canonical one-sample output shared by all agentic baselines."""

    id: str
    dimension: str
    baseline: str
    protocol_status: str = "controlled_adaptation"
    comparability: str = "forecast_only"
    dataset: str = "?"
    domain: str = "?"
    predicted_values: list[float] = field(default_factory=list)
    predicted_p10: list[float] | None = None
    predicted_p50: list[float] | None = None
    predicted_p90: list[float] | None = None
    forecast_metrics: dict[str, float] | None = None
    controller_output: dict[str, Any] | None = None
    revision_output: dict[str, Any] | None = None
    controller_source: str | None = None
    numerical_source: str | None = None
    context_alignment: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    error: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def write_records(output_dir: str | Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Persist per-dimension JSON and a canonical JSONL stream atomically."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_dimension: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_dimension.setdefault(str(row["dimension"]), []).append(row)
    for dimension, rows in by_dimension.items():
        destination = root / dimension / "results.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stream = root / "all_results.jsonl"
    stream.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
