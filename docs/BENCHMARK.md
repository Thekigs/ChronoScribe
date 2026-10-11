# ChronoScribe Evaluation Suite

The ChronoScribe Evaluation Suite evaluates systems that receive natural-language requests and historical observations. It is broader than a pure forecasting benchmark, so complete-system TLS and verified forecast-only scoring are reported separately.

## Suite and paired protocol

The source suite contains 8,120 records across 23 dimensions. `benchmark/evaluation_manifest.json` selects 20 records per dimension, 460 records in total, with seed `2026`. The current manifest digest is `25eb2519ed22cc61697fe2faab4337e85cdc3070f23d66952b075f2c7d650bea`.

Every paired run must use the manifest and match records by `(dimension, id)`. The suite count and paired count are different quantities and must be reported separately. Outputs generated before the manifest and canonical-ID migration are historical until re-matched.

## Dimension catalogue

| ID | Group | Evaluation focus |
| --- | --- | --- |
| D1 | Forecasting | Multi-domain short-horizon forecasting |
| D2 | Forecasting | Long-horizon degradation |
| D3 | Forecasting | Multivariate forecasting |
| D4 | Forecasting | Cross-domain zero-shot transfer |
| D5 | Forecasting | Few-shot data efficiency |
| D6 | Temporal reasoning | Trend judgement |
| D7 | Temporal reasoning | Anomaly detection and explanation |
| D8 | Temporal reasoning | Periodicity identification |
| D9 | Forecasting | Noise, missingness, and shift robustness |
| D10 | Uncertainty | Quantile intervals and calibration |
| D11 | Instruction following | JSON validity and field completeness |
| D12 | Instruction following | Multi-turn refinement |
| D13 | Temporal reasoning | Causal attribution |
| D14 | Forecasting | Renewable-energy forecasting |
| D15 | Temporal reasoning | Time-series classification |
| D16 | Instruction following | Workflow planning and execution output |
| D17 | Forecasting | Progressive missing-data robustness |
| D18 | Efficiency | Latency, throughput, memory, and token budget |
| D19 | Temporal reasoning | Multivariate contribution attribution |
| D20 | Forecasting | Error behavior across horizons |
| D21 | Instruction following | End-to-end prediction and explanation compliance |
| D22 | Forecasting | Extreme-outlier robustness |
| D23 | Instruction following | Extended multi-turn consistency |

The historical TLS grouping assigns forecasting 60%, temporal reasoning 15%, instruction following 15%, uncertainty 5%, and efficiency 5%. This weighted score summarizes a complete workflow; the current primary forecasting table does not use TLS.

## Data sources and construction

The suite draws from ETT, Electricity, Traffic, Weather, Exchange Rate, Illness, M5, renewable energy, NAB, Yahoo S5, UEA/UCR, MONSTER, MSL, SMAP, SMD, PSM, and SWaT. Dataset licenses and access terms remain those of the upstream sources. The benchmark records combine historical windows, natural-language requests, and ground truth or task labels. Templates vary horizons, context requests, output formats, reasoning requirements, uncertainty requests, and robustness conditions.

No system under test should receive future observations. For verified forecast-only scoring, a manifest target must also be locatable in the resolved raw source; otherwise the row is excluded rather than assigned a guessed target.

## Manifest generation

```bash
python -m utils.evaluation_manifest --benchmark-dir benchmark \
  --output benchmark/evaluation_manifest.json --samples-per-dim 20 --seed 2026
```

IDs are canonicalized as `<dimension>__<source-file>__<row-number>__<legacy-id>`. Dimension-qualified IDs prevent collisions from legacy files. Do not silently replace a published manifest; create a new protocol when membership changes.

## Evaluation boundaries

The benchmark supplies the target horizon and all systems are scored on the same target window. The controller may configure context length and route selection, but this protocol does not establish unconstrained horizon planning. Invalid structured output, fallback routing, missing target alignment, and controller cache misses must be recorded separately from successful adaptive decisions.
