"""Comparable LLM-assisted forecasting baseline adapters.

The package deliberately keeps baseline execution separate from the released
ChronoScribe runners.  It provides a common manifest-keyed record contract,
local Qwen+TSFM controlled baselines, and an external-command adapter for
official third-party implementations.
"""

from .contracts import BaselineRecord, extract_json, fit_horizon

__all__ = ["BaselineRecord", "extract_json", "fit_horizon"]
