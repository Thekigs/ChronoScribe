# Usage and runnable workflows

## Choose a workflow

| Goal | Start with | External inputs |
| --- | --- | --- |
| Check a clean source clone | `README.md` validation commands | None |
| Run the complete ChronoScribe system | [Paired benchmark](#paired-benchmark) | Qwen3.5-9B, the reference LoRA adapter, Chronos-2, and permitted datasets |
| Compare numerical specialists | [Specialist comparisons](#specialist-comparisons) | The selected specialist checkpoints or packages |
| Reproduce an external-agent comparison | [AGENTIC_BASELINE_PROTOCOL.md](AGENTIC_BASELINE_PROTOCOL.md) | The external runner, its documented inputs, and any separately supplied result bundle |

Run every command below from the repository root. The source-only checks are
the only workflow that works before model and dataset setup.

## Install

```bash
conda env create -f environment.yml
conda activate ChronoScribe
pip install -r requirements.txt
pip install -e .
python scripts/download_models.py
python scripts/download_data.py
```

Use Python 3.10 or 3.11. The reference environment supports the Qwen/LoRA controller, Chronos-2, TimesFM, Sundial, Aurora adapters, scoring, and audits. MOMENT requires `requirements-moment.txt` in a separate environment because its official dependencies conflict with the reference Torch/Transformers versions. The archival Moirai runner requires `requirements-moirai.txt` in a separate environment because Uni2TS 2.0.0 pins Datasets 2.17.x while the active environment uses patched Datasets 5.0.1.

Local Qwen3.5-9B, Chronos-2, and LoRA files are expected under `models/` and are ignored by version control. The downloader does not download the reference LoRA adapter, and it does not bypass registration or separate terms. A clean source checkout can run the validation checks, but the full adaptive benchmark requires the external inputs listed in `REPRODUCIBILITY_STATUS.md`.

## Paired benchmark

Use the fixed manifest for every comparable system:

```bash
python runners/run_chronoscribe.py --manifest benchmark/evaluation_manifest.json
python runners/run_chronoscribe.py --policy simple_fixed --fixed-context-length 1024 \
  --manifest benchmark/evaluation_manifest.json \
  --output-dir results/simple_fixed1024
python runners/ablate_nolora.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_noc2.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_c2only.py --manifest benchmark/evaluation_manifest.json
```

Use `--dims` and `--max-samples` for smoke tests. A smoke test is a subset of the same protocol and must not be presented as a full result.

## Specialist comparisons

`runners/run_qwen_specialist.py` keeps the controller decision, manifest, history resolver, horizon, and output schema fixed while changing the specialist:

```bash
python runners/run_qwen_specialist.py --specialist chronos2 --device mps \
  --manifest benchmark/evaluation_manifest.json --output-dir results/chronos2_run
python runners/run_qwen_specialist.py --specialist timesfm --device mps \
  --manifest benchmark/evaluation_manifest.json --output-dir results/timesfm_run
python runners/run_qwen_specialist.py --specialist sundial --device mps \
  --manifest benchmark/evaluation_manifest.json --output-dir results/sundial_run
python runners/run_qwen_specialist.py --specialist aurora --device cpu \
  --manifest benchmark/evaluation_manifest.json --output-dir results/aurora_run
```

These are forecast-focused rows. They should be scored with the operational scorer and should not be merged into a TLS table unless the specialist also implements the same language-response protocol.

## Forecast-only scoring

```bash
python tools/score_operational_forecasts.py \
  --baseline FreshAdaptive \
  --model FreshAdaptive results/e2e_fresh_adaptive_strict_forecastdims_20261002 \
  --model ParserFixed1024 results/p0_parser_fixed1024_mps_v2 \
  --model Chronos2Fixed512 results/qwen_chronos2_fixed512 \
  --model TimesFM results/qwen_timesfm_matched \
  --model Sundial results/qwen_sundial_strict_v2 \
  --model Aurora results/qwen_aurora_strict_v2 \
  --output scores/operational_forecasting_verified_20261003.json
python tools/export_operational_table.py
```

The export step reads the JSON written by the preceding scorer. It should be
run only after that scorer completes successfully.

This protocol uses only source-resolved held-out targets, history-based scale denominators, exact horizons, and common IDs. It reports MAE, RMSE, sMAPE, history-based seasonal MASE, Direction Accuracy, medians, coverage, and paired bootstrap intervals.

## External and controlled agent baselines

Read `docs/AGENTIC_BASELINE_PROTOCOL.md` before running a fusion or revision experiment. `agent_baselines/` provides `simple_fusion`, bounded `revision_control`, and an external JSON adapter. The local ChronoSteer-style result is a controlled adaptation, not an official ChronoSteer reproduction.

## Score a complete run

```bash
python scoring/score_results.py \
  --results-dir results/chronoscribe \
  --benchmark-dir benchmark \
  --output-dir scores/chronoscribe \
  --model-name chronoscribe
```

The complete scorer generates per-sample metrics, dimension summaries, `final_scores.json`, and `scorecard.md`. It rejects records that cannot be matched to the benchmark by `(dimension, id)`.

## Interactive mode and local API

```bash
python runners/chat.py --temperature 0.5 --max-tokens 1024
python runners/server.py --port 8000
```

The interactive path can answer ordinary language requests. When usable numerical history exists, the controller may select the specialist path through `use_chronos2`. The local service exposes an OpenAI-compatible chat-completions endpoint with model ID `chronoscribe`; it binds to `127.0.0.1` and disables streaming and cross-origin access by default. Pass `--allow-origin https://your-trusted-origin.example` only for a trusted browser client.

## Validation before a rerun

```bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python tools/evaluate_numeric_guard_adversarial.py
python -m unittest tests.test_training
```
