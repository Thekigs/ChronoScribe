# External-agent comparison policy
> **Source-only release:** `scores/` and `results/` are excluded from this checkout. Paths to those directories refer to a separately supplied experiment result bundle and are not required for the contract checks. See [`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md).


We separate external-agent comparisons into two evidence layers so that a
different input protocol is never presented as a paired benchmark result.

| Layer | Method | Input protocol | Use in the paper |
|---|---|---|---|
| Primary paired comparison | ChronoSteer-style bounded revision | Current manifest, same Qwen controller, Chronos-2, no matched event text; revision is bounded and audited | Mechanism control against a simple Qwen+TSFM pipeline; may be discussed beside the main forecast table, but is not called official ChronoSteer |
| Official external reference | Official ChronoSteer checkpoint inference | Released ChronoSteer checkpoints, Chronos-Bolt, BGE-M3, and MTSFBench-300 `rev-test-reply.json` with its precomputed context embeddings | Separate reference table; demonstrates that released checkpoint inference was executed, but does not establish a cross-benchmark rank against ChronoScribe |
| Not protocol-aligned | TimeSeriesScientist | External LLM/API and graph-specific agent contract | Related-work/protocol discussion only until an authorized common input contract is available |
| Not protocol-aligned | Last-mile forecasting agent | Business/event context and retrieval/memory inputs absent from the manifest | Qualitative boundary case only; no fabricated numeric rank |

## Why the separation is necessary

ChronoSteer's official data contains pre-forecast textual context and uses a
different benchmark. The released test file supplies the context embeddings,
so the checkpoint reproduction does not rerun the original external LLM/API
instruction-generation step. Removing that context or inventing it for the
current manifest would not be an official reproduction. Conversely, comparing
the official MTSFBench-300 numbers directly with ChronoScribe's manifest-level
metrics would conflate dataset, horizon, scaling, context, and metric-unit
effects.

The fair claim is therefore two-part:

1. On the current manifest, allowing a Qwen controller to issue a bounded
   ChronoSteer-style revision does not produce a stable improvement over the
   simple fixed Qwen+TSFM pipeline (`scores/agentic/paired_strict_revision_vs_simple_fixed1024.json`).
2. The released ChronoSteer checkpoints run successfully on their official
   MTSFBench-300 test protocol (using the released precomputed context
   embeddings)
   (`scores/agentic/chronosteer_official_reproduction_20261005.json`).

These are complementary claims, not one merged leaderboard.

## Manuscript-ready wording

> We evaluate LLM-assisted forecasting agents under two explicitly separated
> protocols. First, a same-manifest ChronoSteer-style bounded-revision control
> uses the same Qwen interface and Chronos-2 specialist as ChronoScribe; it is
> a mechanism control, not an official reproduction. Second, we execute the
> released ChronoSteer checkpoints on the authors' MTSFBench-300
> `rev-test-reply` protocol. We report the latter as an external reference,
> rather than a paired rank, because its benchmark and pre-forecast event
> context differ from our manifest. TimeSeriesScientist and Last-mile are
> retained as protocol-bound related baselines because their required external
> API or business/event context is not present in our benchmark.

The corresponding evidence and exact execution environments are recorded in
`scores/agentic/external_protocol_audit.json` and
`scores/agentic/chronosteer_official_reproduction_20261005.json`.
