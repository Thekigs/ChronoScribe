# Reproducibility and release guide

This guide defines what the local artifact can reproduce and which inputs remain external. It also prevents historical score directories from being mistaken for one current experiment.

## Artifact boundary

The repository contains benchmark records, canonical IDs, the fixed paired manifest, runners, specialist adapters, component ablations, scoring code, and audit tools. It does not contain private model checkpoints, API credentials, raw LoRA training records, or restricted upstream datasets. The training builder is a reproducible data-construction method, not the original private corpus.

## Fixed evaluation contract

- Source suite: 8,120 records across 23 dimensions.
- Paired manifest: 460 records, 20 per dimension, seed `2026`.
- Pairing key: `(dimension, id)`.
- Manifest: `benchmark/evaluation_manifest.json`.
- Current manifest SHA-256: `25eb2519ed22cc61697fe2faab4337e85cdc3070f23d66952b075f2c7d650bea`.
- Current verified forecast pool: 212 source-aligned targets.
- Current all-model forecast intersection: 85 rows.

Every run must record the manifest digest, model revision, device, framework versions, generation settings, controller source, target-alignment status, and failure policy. A result generated before canonical IDs or the fixed manifest is historical unless it is re-matched and audited.

## Target and leakage rules

The benchmark supplies the target horizon. During execution, the controller and specialist receive history only; the held-out target is used after prediction for scoring. A target that cannot be located in the resolved source series is excluded from the verified forecast table. No runner may turn `tail_before_gt`, an extracted future field, or an observed tail into a publication target.

The numeric-claim guard checks forecast-related numerical statements against typed evidence. The case-study audit separately checks that future metrics are not injected into response-generation inputs.

## Required disclosures

Any paper or release using these results must state dataset sources and licenses, full-suite and paired counts, manifest digest, data-adaptation provenance, prompt and output contracts, specialist checkpoint and quantile settings, hardware/software, device, generation budget, run count, latency boundary, failure policy, raw metrics, aggregation rules, and confidence-interval procedure.

The project is an empirical systems study. Unless formal claims and proofs are added, the theoretical-contribution branch of a reproducibility checklist should be marked `no`.

## Validation commands

```bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python tools/evaluate_numeric_guard_adversarial.py
python -m unittest tests.test_training
```

These public-release checks do not download weights or run costly inference. They verify canonical IDs, manifest membership, suite metadata, Chinese Markdown companions, and local artifacts that must not enter a public archive. The experiment-registry validator is intentionally separate because the source release excludes the result registry and score files.

## Public release rules

- Keep `.env`, `.env.*`, local checkpoints, private adapters, raw restricted datasets, `results/`, and generated `scores/` out of a source archive unless a result bundle is intentionally published with its provenance.
- Do not publish serialized trainer state containing private filesystem paths.
- Do not relicense third-party model weights or datasets under the project license.
- Include model cards, dataset citations, the manifest digest, the experiment registry, and its supporting evidence files with any released result bundle.
