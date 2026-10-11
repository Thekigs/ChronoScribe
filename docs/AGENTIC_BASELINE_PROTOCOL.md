# LLM-assisted forecasting baseline protocol
> **Source-only release:** `scores/` and `results/` are excluded from this checkout. Paths to those directories refer to a separately supplied experiment result bundle and are not required for the contract checks. See [`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md).


This protocol covers local fusion/revision controls and the external methods named in the review: ChronoSteer, TimeSeriesScientist, and last-mile forecasting agents. It is separate from the complete-system TLS runner because forecast-only agents do not necessarily implement the same response contract.

## Shared conditions

- Use `benchmark/evaluation_manifest.json` and report its digest.
- Resolve history with `utils.benchmark_data.load_raw` and `build_context`.
- Keep held-out targets out of every controller, event, routing, and revision input.
- Use the same horizon, sample IDs, forecast metrics, and failure accounting.
- Report raw forecast metrics before any F/R/I/U/E/TLS aggregate.
- Record device, model calls, token budgets, latency, revision operations, and numerical provenance.
- Use `tools/score_operational_forecasts.py` for forecast-only claims; it excludes unverified target alignments and pseudo targets.

## Baseline labels

`official_reproduction` means the cited implementation and checkpoint are executed with their documented inputs. `controlled_adaptation` means the mechanism is tested under the shared manifest and backbone without claiming the official implementation. `qualitative_only` means there is no common input/task protocol and no numeric rank is assigned.

## Completed local controls

The local controls are:

```text
simple_fusion      fixed context → TSFM → Qwen response
revision_control   fixed context → TSFM → Qwen revision → bounded forecast
```

The bounded revision control isolates the operation central to ChronoSteer-style systems. It uses the same manifest, Chronos-2, Qwen LoRA controller, and strict scale/offset/trend bounds. It changed only two trajectories and did not produce a stable aggregate improvement over the fixed baseline. It is recorded as a mechanism control in:

- `results/agentic/revision_chronos2_fixed1024_v2_full/all_results.jsonl`
- `scores/agentic/paired_strict_revision_vs_simple_fixed1024.json`
- `scores/agentic/chronosteer_style_comparison.json`
- `results/agentic/simple_fusion_chronos2_fixed1024_thinking_full_20261006/all_results.jsonl`
- `scores/agentic/revision_vs_strict_simple_fixed1024_thinking_20261006.json`

These forecast-only rows must not be merged into the six-component TLS table.
The paired report is stratified in `scores/agentic/revision_protocol_strata_20261005.json`:
the all-forecast intent-to-treat stratum retains invalid-directive no-op
fallbacks, while valid-directive and exact-target-alignment strata are
diagnostic only. This prevents a selected-success-only revision claim.

The strict simple-fusion rerun uses the historical Qwen generation setting
with thinking enabled, a requested context length of 1024, and the same
Chronos-2 specialist. It contains 220 unique manifest rows, 179 forecast
rows, and one explicitly recorded `no_usable_history` loader error. The raw
Chronos-2 trajectories are exactly equal to the revision control's raw
trajectories on all 179 common forecast rows (maximum absolute difference
0.0). This is the authoritative same-language-interface fixed-policy
baseline; the older directory named `e2e_full_simple_fixed1024_mps` is not
used for this strict comparison because its recorded context alignment was
not uniformly 1024.

The legacy adaptive result is retained only as protocol-sensitivity evidence:
`scores/p1_adaptive_protocol_audit_20261006.json` reports the complete paired
coverage while explicitly recording that its adaptive controller rows are
historical/unverifiable. The completed fresh thinking-enabled run in
`results/controller_fresh_thinking_full_20261006` contains 220 manifest rows,
144 valid decisions, and 124 no-clamp paired forecast rows. Its fixed-context
frontier is recorded in `scores/p1_fresh_thinking_adaptive_fixed_frontier_20261006.json`;
adaptive routing has no universal advantage over fixed-512, fixed-1024, or
full-history. The verified-target fixed-control table is separately recorded
in `scores/fresh_thinking_all_fixed_controls_verified_20261006.json`.

## Official external methods

ChronoSteer’s official repository, released alignment checkpoints, BGE-M3 encoder, Chronos-Bolt backbone, and MTSFBench-300 `rev-test-reply.json` were executed through the documented `--only_test` entry point in an isolated compatible environment. The released test file supplies precomputed context embeddings, so this is checkpoint inference reproduction rather than a rerun of the original external LLM/API instruction-generation step. The official status is `official_checkpoint_reproduction_completed`; the aggregate and grouped results are recorded in `scores/agentic/chronosteer_official_reproduction_20261005.json`. Because its benchmark and pre-forecast event-context protocol differ from the current ChronoScribe manifest, this result is an official external reference and is not merged into the paired ChronoScribe ranking. The same-manifest local result remains `controlled_adaptation`.

TimeSeriesScientist requires an external LLM API and its graph-specific input/output protocol. An adapter exists for a matched single-sample probe, but no multi-sample official numeric comparison is promoted without an authorized data-transfer and API protocol. Last-mile forecasting agents require business or event context absent from the current manifest, so they remain qualitative-only.

The production controller uses greedy decoding. The completed MPS diagnostic
probe (`scores/sampling_variance_probe_20261005.json`) evaluates 10 matched
samples with three stochastic repeats at temperature 0.7/top-p 0.9 and reports
run-level variance and invalid-directive rate. A separate thinking-enabled
pilot (`scores/sampling_variance_probe_thinking_20261006.json`) covers four
matched samples with the same three-repeat design. The consolidated protocol
status is `scores/sampling_variance_protocol_summary_20261006.json`; neither
probe replaces the production score or creates a stochastic TLS ranking.

The exact status is recorded in `scores/agentic/external_protocol_audit.json` and the method profiles in `configs/official_agent_profiles.json`.

## External adapter contract

An external runner must write JSON containing `predicted_values` and may also write quantiles, controller output, revision output, and numerical source. The adapter removes `ground_truth` before sending a request, checks horizon alignment, records command/stdout/stderr, and calculates raw forecast metrics only after the external program returns.

```bash
python tools/run_agentic_baselines.py --baseline external \
  --external-name chronosteer_official \
  --external-status official_reproduction \
  --external-command 'python ./external_runner.py --request {request} --response {response}' \
  --output-dir results/agentic/chronosteer_official
```

## Paired scoring

```bash
python tools/score_agentic_baselines.py \
  --baseline results/e2e_full_simple_fixed1024_mps \
  --candidate results/agentic/revision_chronos2_fixed1024_v2_full \
  --output scores/agentic/paired_revision_vs_simple_fixed1024_new.json
```

The scorer reports common-row counts, means, paired differences, and deterministic bootstrap intervals. It does not create a TLS ranking for a forecast-only external agent.
