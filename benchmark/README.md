# Benchmark records and upstream data terms

The JSONL files in this directory are derived evaluation records. Depending on
the dimension, a record can include a dataset or series identifier, prompts,
summary statistics, target values, anomaly metadata, classification labels, or
other expected-answer fields. They are not the raw files stored under `data/`,
but derived values can still be covered by the terms of the upstream dataset.

Before publishing this directory to a public repository, review the terms for
every upstream source represented here. In particular, Yahoo S5 and other
registration-controlled or research-only sources may restrict redistribution.
If a source does not permit redistribution of derived values, remove the
affected records and regenerate `evaluation_manifest.json` and any dependent
digests before publishing. Do not treat a passing local validation check as a
license clearance.

The records are retained in this source release because the public checks and
benchmark contracts refer to them. The exact paper scores are not included;
see `../docs/REPRODUCIBILITY_STATUS.md` for the external inputs required for a full
run.
