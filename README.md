# ChronoScribe

ChronoScribe is a grounded time-series forecasting framework that separates language interaction from numerical prediction. A language controller interprets a request and emits a typed execution directive; a time-series foundation model produces the numerical trajectory; deterministic evidence is derived from that trajectory; and the language model writes the final response from the evidence.

~~~text
natural-language request
        |
typed directive: route, horizon, context length
        |
time-series foundation model
        |
deterministic evidence record
        |
forecast-grounded response
~~~

## Start here

Run these checks first. They work from a clean clone and do not need model
weights or raw datasets:

~~~bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python -m unittest discover -v
~~~

A clean checkout should report 8,120 validated records, 460 paired records,
23 dimensions, and two passing tests.

If those commands pass, the source package is internally consistent. To run a
model-backed smoke test, install the reference environment, download the
permitted models and data, and then use the smoke command in
[docs/USAGE.md](docs/USAGE.md). To reproduce the reported paper tables, read
[docs/REPRODUCIBILITY_STATUS.md](docs/REPRODUCIBILITY_STATUS.md) first: the private LoRA
adapter, some registered datasets, specialist checkpoints, and the external
result bundle are not part of this repository.

The reference configuration uses a LoRA-adapted Qwen3.5-9B controller with Chronos-2. TimesFM and Sundial are specialist baselines, and Aurora is an additional boundary-case specialist. The controller, specialist, evidence, and scoring contracts are documented in [docs/MODEL.md](docs/MODEL.md).

## What this release contains

This source release contains the benchmark records, fixed evaluation manifest, reference runners, scoring code, audit tools, training-data builder, model integration code, configuration files, bilingual documentation, and release validation checks.

The release intentionally excludes model weights, the private LoRA adapter, raw datasets that must be downloaded or registered separately, generated results, scorecards, local secrets, and paper-submission artifacts. The committed README files under data/ and models/ describe those external inputs and their licensing boundaries.

## Claim boundary

The strongest supported mechanism claim is that a language-generated context configuration can affect forecasting quality when the specialist and target horizon are held fixed. The benchmark supplies the target horizon to every system; the controller repeats it for schema compliance. The controller does not receive held-out future values and does not generate or revise the numerical trajectory.

The current evidence does not support a universal claim that adaptive routing beats every fixed policy or every specialist. Strict paired analysis shows a clear aggregate advantage over fixed-512, but only a small MAE/RMSE advantage over fixed-1024; other metrics and full-history comparisons are mixed. Report the complete metric vector, coverage, and protocol label with every comparison.

## Why this project is useful

ChronoScribe is designed around five practical contributions:

1. **Grounded language control.** The language model interprets the request and
   selects an execution route and context length; the specialist remains the
   only component that produces numerical forecast values.
2. **A typed execution contract.** The controller emits a schema-checked
   directive with explicit route, horizon, and context fields. Every system is
   evaluated on the same manifest target window.
3. **Evidence-grounded responses.** The runner derives deterministic evidence
   from the specialist trajectory, and the numeric-claim guard blocks response
   numbers that are not supported by that evidence.
4. **Protocol-controlled comparisons.** Paired IDs, source-resolved targets,
   history-only scaling, and separate forecast-only scoring make it possible to
   distinguish a routing effect from a specialist or response-format effect.
5. **A reproducible training boundary.** The public SFT builder creates paired
   directive/evidence records, keeps future values out of controller inputs,
   and rejects evaluation-source leakage by default.

## Results at a glance

The table below lists every model and component ablation in the historical 23-dimension complete-system panel. All rows use the same F/TLS scoring protocol. The final column is the score margin of ChronoScribe over that row.

| System / model | F | TLS | ChronoScribe advantage |
| --- | ---: | ---: | ---: |
| **ChronoScribe (Qwen3.5-9B + Chronos-2)** | **73.7** | **65.545** | — |
| Fixed-1024 controller | 59.9 | 60.090 | **F +13.8 / TLS +5.455** |
| Fixed-512 controller | 59.5 | 59.640 | **F +14.2 / TLS +5.905** |
| Fixed-256 controller | 59.3 | 59.640 | **F +14.4 / TLS +5.905** |
| Fixed-128 controller | 58.4 | 59.095 | **F +15.3 / TLS +6.450** |
| Fixed-64 controller | 56.3 | 57.820 | **F +17.4 / TLS +7.725** |
| ChronoScribe without LoRA | 56.5 | 51.650 | **F +17.2 / TLS +13.895** |
| Chronos-2-only ablation | 37.6 | 35.335 | **F +36.1 / TLS +30.210** |
| No-Chronos-2 ablation | 23.9 | 25.470 | **F +49.8 / TLS +40.075** |
| Gemini 3.1 Flash | 51.8 | 54.995 | **F +21.9 / TLS +10.550** |
| MiniMax-M3 | 51.8 | 50.020 | **F +21.9 / TLS +15.525** |
| GPT-5.5 | 51.5 | 51.690 | **F +22.2 / TLS +13.855** |
| Grok-4.3 | 49.4 | 50.260 | **F +24.3 / TLS +15.285** |
| GPT-5.6-sol | 49.1 | 50.920 | **F +24.6 / TLS +14.625** |
| Claude Fable 5 | 48.4 | 48.900 | **F +25.3 / TLS +16.645** |
| GLM-5.2 | 47.9 | 49.275 | **F +25.8 / TLS +16.270** |
| Claude Sonnet 5 | 47.6 | 50.040 | **F +26.1 / TLS +15.505** |
| GPT-5.4 | 46.5 | 50.255 | **F +27.2 / TLS +15.290** |
| DeepSeek-V4 Flash | 46.0 | 50.335 | **F +27.7 / TLS +15.210** |
| Gemini 3.5 Flash | 46.0 | 48.000 | **F +27.7 / TLS +17.545** |
| TimesFM | 44.3 | 49.085 | **F +29.4 / TLS +16.460** |
| Qwen3.7 Max | 43.6 | 46.900 | **F +30.1 / TLS +18.645** |
| Gemini 3.8 Flash | 43.6 | 45.900 | **F +30.1 / TLS +19.645** |
| Kimi K3 | 42.7 | 45.000 | **F +31.0 / TLS +20.545** |
| Claude Opus 4.8 | 41.1 | 46.185 | **F +32.6 / TLS +19.360** |
| Grok-4.6 | 40.8 | 45.400 | **F +32.9 / TLS +20.145** |
| DeepSeek-V4 Pro | 40.0 | 47.315 | **F +33.7 / TLS +18.230** |

`F` is the forecast component; `TLS` is the complete-workflow score that also includes reasoning, instruction following, uncertainty, and efficiency. These values are a historical complete-system snapshot. Specialist-only 11-dimension panels are kept separate in [docs/SCORING.md](docs/SCORING.md) rather than mixed into this table.

## Evaluation snapshot

- Full benchmark: 8,120 source records across 23 dimensions.
- Fixed paired manifest: [benchmark/evaluation_manifest.json](benchmark/evaluation_manifest.json), containing 460 records.
- Manifest file SHA-256: `25eb2519ed22cc61697fe2faab4337e85cdc3070f23d66952b075f2c7d650bea`.
- Benchmark membership digest: `a3ab95f613cfd2cc99f63465cd7a2b4af31d200a941456259eccbfb27847cca5`.
- The verified forecast-only protocol uses source-resolved targets, history-only scaling, raw metrics, and common-ID paired comparisons.

Detailed score definitions and result provenance are documented in [docs/SCORING.md](docs/SCORING.md) and [docs/REPRODUCIBILITY_STATUS.md](docs/REPRODUCIBILITY_STATUS.md).

## Installation

Python 3.10 or 3.11 is supported. The reference Conda file pins Python 3.10; Python 3.11 can use the same package pins when compatible platform wheels are available.

~~~bash
conda env create -f configs/environment.yml
conda activate ChronoScribe
pip install -r requirements/requirements.txt
pip install -e .
~~~

The editable install exposes the reusable source packages to Python tooling.
The model and specialist dependencies remain in the separate requirements
files because some baseline stacks are intentionally incompatible.

The main environment covers the Qwen/LoRA controller, Chronos-2, TimesFM, scoring, auditing, and the local API. MOMENT and archival Moirai/Uni2TS use incompatible dependency lines and must be installed separately:

~~~bash
pip install -r requirements/moment.txt
pip install -r requirements/moirai.txt
~~~

The reference environment can use CUDA, Apple MPS, or CPU according to the selected runner. Install only the environment required by the workflow you are reproducing.

## External inputs

| Input | Location or command | Required for |
| --- | --- | --- |
| Qwen3.5-9B | `python scripts/download_models.py --qwen-only` | Live language controller |
| Chronos-2 | `python scripts/download_models.py --c2-only` | Reference numerical specialist |
| Reference LoRA adapter | `models/checkpoint-7500/` | Full adaptive controller |
| Raw and registered datasets | `python scripts/download_data.py` plus [docs/REPRODUCIBILITY_STATUS.md](docs/REPRODUCIBILITY_STATUS.md) | Dataset-backed runs |
| TimesFM, Sundial, Aurora checkpoints | Upstream packages/checkpoints described in [docs/USAGE.md](docs/USAGE.md) | Specialist baselines |
| MOMENT and Moirai dependencies | `requirements/moment.txt` and `requirements/moirai.txt` | Isolated archival baselines |

The model downloader pins the Qwen and Chronos-2 snapshots used by the documented reference setup. It does not download the private LoRA adapter or bypass upstream registration and license requirements. The data downloader exits nonzero when an automatic source fails or produces incomplete output.

## Run the paired benchmark

All comparable systems must use the same manifest:

~~~bash
python runners/run_chronoscribe.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_nolora.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_noc2.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_c2only.py --manifest benchmark/evaluation_manifest.json
~~~

Use a small subset for a smoke run:

~~~bash
python runners/run_chronoscribe.py \
  --manifest benchmark/evaluation_manifest.json \
  --dims D1_short_term_forecasting,D10_uncertainty
~~~

After a run completes, score the generated records:

~~~bash
python scoring/score_results.py \
  --results-dir results/chronoscribe \
  --benchmark-dir benchmark \
  --output-dir scores/chronoscribe \
  --model-name chronoscribe
~~~

For forecast-only comparisons that exclude text, latency, and TLS weights, first
read the full command in [docs/USAGE.md](docs/USAGE.md):

~~~bash
python tools/score_operational_forecasts.py --help
~~~

After the scorer successfully writes
`scores/operational_forecasting_verified_20261003.json`, export its compact CSV
table:

~~~bash
python tools/export_operational_table.py
~~~

The operational scorer excludes targets that cannot be aligned to the resolved source history and never turns a tail approximation into a verified future target.

## Local API

Start the local OpenAI-compatible service with:

~~~bash
python runners/server.py --port 8000
~~~

The default bind address is `127.0.0.1`. CORS is disabled unless an exact origin is supplied with `--allow-origin`. Request fields, error behavior, and example payloads are documented in [docs/USAGE.md](docs/USAGE.md).

## Repository layout

~~~text
ChronoScribe/
├── benchmark/         # 23 dimensions, source records, metadata, fixed manifest
├── configs/           # sanitized controller and baseline configurations
├── data/              # README and local upstream data, ignored when downloaded
├── docs/              # framework, benchmark, scoring, usage, and evidence docs
├── models/            # model integration code, model cards, ignored weights
├── runners/           # reference, API, specialist, and ablation runners
├── scripts/           # dataset and model download entry points
├── scoring/           # metric computation and paired aggregation
├── tests/             # offline regression tests
├── tools/             # audits, normalization, scoring, export, validation
├── training/          # provenance-disjoint SFT builder and tests
├── utils/             # manifest, data, evidence, and guard contracts
└── agent_baselines/   # controlled revision and external-adapter contracts
~~~

The repository is packaged with `pyproject.toml`; use `pip install -e .` after
installing the requirements for the workflow you want to reproduce. The
existing `runners/` and `tools/` paths remain the canonical command-line entry
points.


## Documentation and evidence policy

| Start with | Read this for |
| --- | --- |
| [Usage and runnable workflows](docs/USAGE.md) | Installation, smoke tests, benchmark commands, scoring, and the local API |
| [Reproducibility status](docs/REPRODUCIBILITY_STATUS.md) | What a clean clone can run and which models, data, or result files must be supplied separately |
| [Model and claim boundary](docs/MODEL.md) | The controller/specialist contract and claims supported by the current protocol |
| [Benchmark and task catalogue](docs/BENCHMARK.md) | Manifest membership, dimensions, source data, and target-leakage rules |
| [Scoring and metric policy](docs/SCORING.md) | The complete-system TLS score and the verified forecast-only metrics |
| [Reproducibility and release rules](docs/REPRODUCIBILITY.md) | Required disclosures and what may enter a public result bundle |
| [External and controlled agent baselines](docs/AGENTIC_BASELINE_PROTOCOL.md) | How local controls and external-agent comparisons are separated |

The smaller READMEs under `benchmark/`, `data/`, `models/`, and `training/`
explain the files in those directories and the conditions for using them.

Every English Markdown document has a same-name .zh-CN.md companion. When a result bundle is supplied separately, use its registry and the protocol labels before quoting any generated result in a paper.

## License and citation

Project code is released under the MIT License. Third-party models, datasets, and upstream code retain their own licenses and terms; see [docs/legal/THIRD_PARTY_NOTICES.md](docs/legal/THIRD_PARTY_NOTICES.md).

For academic use, cite the metadata in [CITATION.cff](CITATION.cff). Do not publish .env, private adapters, local checkpoints, restricted raw datasets, generated score directories, or paper-submission artifacts as source code.
