"""Adapter contract for official third-party agent implementations.

The adapter never pretends that an unavailable repository was reproduced.  A
caller supplies a command that accepts one JSON request and writes one JSON
response.  The response is validated and annotated with the exact protocol
status before it enters the shared result stream.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import BaselineRecord, fit_horizon
from scoring.detailed_scoring import compute_all_forecast_metrics


class ExternalAgentAdapter:
    """Run a user-provided official or adapted agent command per sample."""

    def __init__(self, name: str, command: Sequence[str], protocol_status: str,
                 timeout_s: int = 900):
        if not command:
            raise ValueError("An external command is required")
        if protocol_status not in {"official_reproduction", "controlled_adaptation", "qualitative_only"}:
            raise ValueError(f"Invalid protocol_status: {protocol_status}")
        self.name = name
        self.command = [str(item) for item in command]
        self.protocol_status = protocol_status
        self.timeout_s = int(timeout_s)

    def run_one(self, sample: dict[str, Any], dimension: str, work_dir: Path) -> dict[str, Any]:
        started = time.time()
        gt = sample.get("ground_truth", {})
        horizon = len(gt.get("values", [])) or int(sample.get("horizon", 24))
        row = BaselineRecord(
            id=str(sample["id"]), dimension=dimension, baseline=self.name,
            protocol_status=self.protocol_status, comparability="forecast_only",
            dataset=str(sample.get("dataset", "?")), domain=str(sample.get("domain", "?")),
        )
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(sample["id"]))
        request = work_dir / f"{dimension}__{safe_id}.request.json"
        response = work_dir / f"{dimension}__{safe_id}.response.json"
        request.parent.mkdir(parents=True, exist_ok=True)
        # Never pass the target window to an external runner.  The in-memory
        # caller retains it only for post-hoc scoring.
        execution_sample = {key: value for key, value in sample.items() if key != "ground_truth"}
        request.write_text(json.dumps({
            "sample": execution_sample,
            "dimension": dimension,
            "horizon": horizon,
            "output_path": str(response),
            "contract": {
                "no_future_values_in_input": True,
                "required_prediction_field": "predicted_values",
                "required_length": horizon,
            },
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            command = [item.replace("{request}", str(request)).replace("{response}", str(response)) for item in self.command]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=self.timeout_s)
            if completed.returncode != 0:
                raise RuntimeError(f"external_exit_{completed.returncode}: {completed.stderr[-300:]}")
            if not response.exists():
                raise RuntimeError("external_response_missing")
            payload = json.loads(response.read_text(encoding="utf-8"))
            values = payload.get("predicted_values")
            if not isinstance(values, list):
                raise RuntimeError("external_predicted_values_missing")
            row.predicted_values = fit_horizon(values, horizon)
            row.predicted_p10 = fit_horizon(payload.get("predicted_p10", values), horizon)
            row.predicted_p50 = fit_horizon(payload.get("predicted_p50", values), horizon)
            row.predicted_p90 = fit_horizon(payload.get("predicted_p90", values), horizon)
            if gt.get("values"):
                row.forecast_metrics = compute_all_forecast_metrics(
                    np.asarray(gt["values"], dtype=float),
                    np.asarray(row.predicted_values, dtype=float),
                    naive_forecast=gt.get("naive_forecast"),
                )
            row.controller_output = payload.get("controller_output")
            row.revision_output = payload.get("revision_output")
            row.numerical_source = payload.get("numerical_source", self.name)
            row.audit = {
                "command": command,
                "stdout_tail": completed.stdout[-500:],
                "stderr_tail": completed.stderr[-500:],
                "future_values_provided_to_agent": False,
                "ground_truth_removed_from_request": True,
            }
        except Exception as exc:
            row.error = str(exc)[:300]
        row.latency_s = round(time.time() - started, 3)
        return row.to_dict()
