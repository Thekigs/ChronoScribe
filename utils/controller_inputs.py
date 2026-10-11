"""Stable language-controller inputs shared by the full system and ablations.

Keeping this representation in one module prevents an ablation from receiving
less task or history information than the reference ChronoScribe controller.
"""

from __future__ import annotations

import numpy as np


def request_text(sample: dict, limit: int = 1800) -> str:
    """Extract user-facing task text without assuming a benchmark prompt schema."""
    contents = [
        message["content"]
        for message in sample.get("prompt", [])
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    text = "\n".join(contents).strip()
    return text[-limit:] if text else "Forecast the requested future values and explain the result."


def history_summary(values, max_points: int = 128) -> str:
    """Summarize global, sampled, and recent history for controller decisions."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return "No finite historical observations are available."
    indices = np.linspace(0, len(finite) - 1, min(max_points, len(finite)), dtype=int)
    sampled = finite[indices]
    recent = finite[-min(64, len(finite)):]
    slope = float(np.polyfit(np.arange(len(finite)), finite, 1)[0]) if len(finite) > 1 else 0.0
    encode = lambda sequence: ",".join(f"{value:.5g}" for value in sequence)
    return (
        f"length={len(finite)}; mean={finite.mean():.5g}; std={finite.std():.5g}; "
        f"min={finite.min():.5g}; max={finite.max():.5g}; linear_slope={slope:.5g}\n"
        f"evenly_sampled_history=[{encode(sampled)}]\n"
        f"recent_history=[{encode(recent)}]"
    )
