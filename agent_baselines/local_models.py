"""Lazy local Qwen and TSFM loaders for controlled baseline runs.

No model is loaded at import time.  This is intentional: configuration and
static validation must work while another MPS job is using the machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import time

BASE = Path(__file__).resolve().parents[1]


def device_memory(device: str) -> dict[str, int]:
    """Allocated vs driver memory; RSS alone misses Metal allocations."""
    import torch
    if device != "mps":
        return {}
    return {name: int(getattr(torch.mps, name)())
            for name in ("current_allocated_memory", "driver_allocated_memory")
            if hasattr(torch.mps, name)}


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenController:
    """Small deterministic Qwen+LoRA wrapper used by controlled baselines."""

    def __init__(self, device: str, max_new_tokens: int = 256,
                 enable_thinking: bool = True, release_mps_cache: bool = True):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = enable_thinking
        self.release_mps_cache = release_mps_cache
        base = AutoModelForCausalLM.from_pretrained(
            str(BASE / "models/Qwen3.5-9B"), torch_dtype=torch.float16, device_map="cpu"
        )
        self.model = PeftModel.from_pretrained(base, str(BASE / "models/checkpoint-7500")).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(BASE / "models/Qwen3.5-9B"))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
        import torch

        budget = int(max_new_tokens or self.max_new_tokens)
        started = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=budget, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        eos_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        audit = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "max_new_tokens": budget,
            "generated_tokens": int(len(generated)),
            "ended_with_eos": bool(len(generated) and generated[-1].item() in eos_ids),
            "budget_exhausted": bool(len(generated) >= budget and generated[-1].item() not in eos_ids),
            "latency_s": time.perf_counter() - started,
            "device": self.device,
            "enable_thinking": self.enable_thinking,
            "memory_before_release": device_memory(self.device),
        }
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        del generated, output, inputs
        if self.device == "mps" and self.release_mps_cache:
            torch.mps.empty_cache()
        audit["memory_after_release"] = device_memory(self.device)
        audit["total_latency_with_cleanup_s"] = time.perf_counter() - started
        return text, audit


class LocalSpecialist:
    """Interchangeable local numerical specialist with Chronos-2 and TimesFM."""

    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        self.model = self._load(name, device)

    @staticmethod
    def _load(name: str, device: str):
        import torch

        if name == "chronos2":
            from chronos import Chronos2Pipeline
            return Chronos2Pipeline.from_pretrained(
                str(BASE / "models/chronos-2"), device_map=device, dtype=torch.float32
            )
        if name == "timesfm":
            import timesfm
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            model.model.to(device)
            model.model.device = torch.device(device)
            model.model.device_count = 1
            model.compile(timesfm.ForecastConfig(
                max_context=1024, max_horizon=720, normalize_inputs=True, per_core_batch_size=1
            ))
            return model
        raise ValueError(f"Unsupported controlled specialist: {name}")

    @staticmethod
    def _fit(values: list[float], horizon: int) -> list[float]:
        if len(values) >= horizon:
            return [float(x) for x in values[:horizon]]
        return [float(x) for x in values] + ([float(values[-1])] * (horizon - len(values)) if values else [0.0] * horizon)

    def forecast(self, context: np.ndarray, horizon: int) -> tuple[list[float], list[float], list[float]]:
        import torch

        context = np.asarray(context, dtype=np.float32)
        if self.name == "chronos2":
            tensor = torch.tensor(context, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
            quantiles, medians = self.model.predict_quantiles(
                tensor, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
            )
            if quantiles and len(quantiles[0].shape) == 3:
                values = quantiles[0]
                n_quantiles = values.shape[2]
                return tuple(
                    self._fit(values[0, :, index].detach().cpu().tolist(), horizon)
                    for index in (0, n_quantiles // 2, n_quantiles - 1)
                )
            median = medians[0].squeeze().detach().cpu().tolist()
            median = [median] if isinstance(median, float) else median
            median = self._fit(median, horizon)
            return median, median, median
        padded = context[-1024:]
        if len(padded) < 1024:
            padded = np.pad(padded, (1024 - len(padded), 0), mode="edge")
        point, _ = self.model.forecast(horizon=horizon, inputs=[padded])
        median = self._fit(point[0].tolist(), horizon)
        return median, median, median
