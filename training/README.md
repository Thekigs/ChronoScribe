# ChronoScribe SFT Data Builder

`build_chronoscribe_sft.py` converts a conventional wide time-series CSV into
paired chat records for the ChronoScribe language controller. It supports CSV
files with one timestamp column and one or more numeric measurement columns.

## Record construction

Every usable history window produces two records with a shared `pair_id`:

1. A directive record maps the user request and a deterministic history summary
   to the exact executable JSON accepted by `runners/run_chronoscribe.py`:
   `use_chronos2`, `prediction_length`, and `context_length`.
2. A response record maps the request plus forecast-derived evidence to a
   `trend` and `analysis` JSON response. It never asks the controller to emit
   the forecast trajectory.

The input prompt contains only the historical prefix. Future values merely
define a valid temporal window and are never serialized into a directive input
or target. Records with missing values in the required window are rejected
rather than silently resampled or imputed. By default, the two system prompts
and compact evidence string are exactly those used by the current reference
runner.

## Example

```bash
python -B training/build_chronoscribe_sft.py \
  --input-csv data/sft/provenance_disjoint_series.csv \
  --output-dir data/sft/provenance_disjoint_series \
  --dataset-name provenance_disjoint_series \
  --domain energy \
  --horizons 24 48 96 \
  --context-candidates 128 256 512 \
  --stride 48
```

The command writes `train.jsonl`, `validation.jsonl`, and an auditable
`manifest.json`. The split unit is `pair_id`, so the directive and response
for a window are never separated.

## Evidence provenance

When only a CSV is available, the builder creates deterministic
seasonal-naive P10/P50/P90 proxy quantiles from the historical prefix. This
creates interface-consistent response records but does **not** reproduce the
reference adapter's specialist evidence. Exact training reproduction requires
Chronos-2 quantiles for each same history/horizon pair, then the same evidence
and response construction contract.

Use `--response-evidence-format full_quantiles` only after updating the
inference runner to pass the corresponding P10/P50/P90 fields to the language
controller. The default `current_runner` uses the exact compact evidence string
currently sent by `runners/run_chronoscribe.py`.

The command refuses source names that match the fixed evaluation suite unless
`--allow-evaluation-source` is provided. That override is intended for schema
tests only; reported LoRA training must remain provenance-disjoint from the
evaluation sources.

## Validation

Run the offline test suite with:

```bash
python -B -m unittest tests.test_training
```

It builds a small synthetic CSV and verifies deterministic output, paired
splitting, the live runner's directive/response field names, evidence alignment,
and the absence of a forecast-trajectory target.
