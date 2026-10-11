# Reproducibility status of the public source release

This repository is a source release, not a self-contained result archive. A clean clone can validate the benchmark contracts and run the unit tests, but it cannot reproduce the paper's full numbers until the external models, adapter, and permitted datasets are supplied.

The JSONL files under `benchmark/` are derived evaluation records rather than raw
datasets. They can still be subject to upstream redistribution terms. Review
`benchmark/README.md` and the source-specific terms before publishing this
directory; technical validation does not constitute license clearance.

## What works from a clean clone

After installing Python 3.10 or 3.11 and `requirements/requirements.txt`, these checks require no model weights or raw datasets:

```bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python tools/evaluate_numeric_guard_adversarial.py
python -m unittest tests.test_training
```

The release currently validates 8,120 benchmark records, 460 paired-manifest records, 23 dimensions, and the Chinese documentation companions.

## What must be supplied separately

- The Qwen3.5-9B base model and the Chronos-2 specialist can be downloaded with `scripts/download_models.py`.
- `scripts/download_models.py` pins Qwen3.5-9B to `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Chronos-2 to `29ec3766d36d6f73f0696f85560a422f50e8498c`; do not replace these revisions when comparing against the documented reference setup.
- The reference LoRA adapter is not included and has no public download URL in this release. Place a compatible adapter under `models/checkpoint-7500/` before running the adaptive controller. Without it, the full ChronoScribe adaptive runner cannot reproduce the reported controller decisions.
- `scripts/download_data.py` retrieves permitted public sources, but SWaT and Yahoo S5 require manual registration. Dataset availability and redistribution terms can change by provider.
- TimesFM, Sundial, and Aurora require their own upstream checkpoints or package downloads. The source release contains the Aurora/Sundial integration code, not their weights. The archival Moirai runner uses `requirements/moirai.txt` because Uni2TS 2.0.0 pins Datasets 2.17.x; it must not be installed into the active Datasets 5.0.1 environment.
- Exact paper numbers additionally require the same manifest digest, model revisions, prompt/configuration, framework versions, device, precision, generation settings, and failure policy recorded in the run metadata.

## Reproduction levels

| Level | Result | Required inputs |
|---|---|---|
| Contract and metadata checks | Reproducible from this repository | Python environment only |
| Numeric-guard and training-builder tests | Reproducible from this repository | Python environment only |
| A small specialist smoke test | Reproducible after setup | Permitted data plus Chronos-2 weights |
| Full fixed or adaptive benchmark | Conditional | All resolved datasets, Qwen base model, compatible LoRA adapter, Chronos-2, and hardware |
| Exact published score tables | Not guaranteed from source alone | All above plus the original external result bundle and pinned revisions |

The scripts create `results/` and `scores/` locally. Those directories are intentionally ignored and are not evidence that a clean checkout already contains the result files referenced by the historical evidence documents. `tools/validate_experiment_registry.py` is only applicable when a separately supplied result bundle contains `scores/experiment_registry_20261003.json` and its authoritative files.
