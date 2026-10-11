# Scoring and reporting policy

ChronoScribe keeps two result families separate: complete-system scores that include language, uncertainty, and efficiency tasks, and verified forecast-only scores that compare numerical predictions under a common target protocol. The second family is the primary evidence for forecasting claims.

## Complete-system TLS

The historical Task-Level Score is:

```text
TLS = 0.60 F + 0.15 R + 0.15 I + 0.05 U + 0.05 E
```

`F`, `R`, `I`, `U`, and `E` are dimension aggregates mapped to `[0,100]`. TLS is a summary of the complete workflow; it is not forecasting accuracy. It should be reported with the raw dimension scores, coverage, and the reference-bound sensitivity analysis. The legacy 65.5 and 10.6-point narrative is protocol-specific and is not the current forecast-only conclusion.

## Verified forecast-only protocol

`tools/score_operational_forecasts.py` is the authoritative forecast-only scorer. It:

- matches records by `(dimension, id)`;
- uses only manifest targets that can be located in the resolved source series;
- constructs scale denominators from pre-origin history;
- requires the exact target horizon;
- excludes tail approximations and pseudo targets;
- reports common-model and pairwise coverage explicitly; and
- never adds text, latency, memory, or TLS weights.

The primary metric vector is MAE, RMSE, sMAPE, history-based seasonal MASE, and Direction Accuracy. WAPE, history-lag-1 MASE, normalized errors, medians, and per-dimension summaries remain available as diagnostics. The current all-model intersection is 85 rows; the verified target pool is 212 rows before model-intersection filtering.

## Aggregation

Metrics are first computed per sample, then summarized by mean and median. Paired comparisons use the same `(dimension, id)` set and report candidate-minus-baseline differences with sample- and dimension-level bootstrap intervals. A lower value is better for MAE, RMSE, sMAPE, MASE, and WAPE; a higher value is better for Direction Accuracy and interval coverage.

Raw errors are not directly comparable across domains with different units. The CSV export therefore includes both means and medians, while the paper should include coverage and per-dimension results whenever scale heterogeneity can change the interpretation.

On the current 85-row all-model intersection, no single method wins every
metric. Persistence is strong on raw mean MAE/RMSE because of large-scale
outliers, Aurora is a strong but less balanced learned specialist, and
TimesFM/FreshAdaptive trade wins across sMAPE, MASE, and direction. This is why
the primary table should show the full metric vector rather than a single
selected headline number.

## Uncertainty and response quality

When a specialist supplies P10/P50/P90, retain the quantiles and score coverage, interval width, calibration, and CRPS-like quantities. A point-only system must be marked as missing interval output rather than receiving an inferred interval. The numeric-claim guard audits numerical statements against typed evidence; it does not certify all qualitative reasoning.

## Efficiency boundary

Efficiency is meaningful for local end-to-end runs when hardware, software, initialization policy, precision, batch size, and timeout are fixed. Remote API baselines have provider-dependent latency and token accounting, so efficiency is not a fair universal ranking across commercial models. Report API latency only when the same measurement boundary and provider conditions are documented.

## Interpretation rules

A higher TLS does not mean a system is best at every capability. Do not select a single favorable metric after seeing results. Current forecast-only evidence shows trade-offs among FreshAdaptive, fixed Qwen policies, Chronos-2, TimesFM, Sundial, Aurora, Persistence, and SeasonalNaive; the full five-metric vector is required for an honest comparison.
