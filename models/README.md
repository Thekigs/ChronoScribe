# Local model artifacts

This public source release contains only lightweight integration code and configuration. Model weights, caches, the Qwen base model, Chronos-2 files, and the private LoRA adapter are excluded by `.gitignore`.

| Directory | Role in ChronoScribe | How to obtain the runtime files |
|---|---|---|
| `Qwen3.5-9B/` | language controller and response model | Run `python scripts/download_models.py --qwen-only` |
| `checkpoint-7500/` | reference LoRA adapter | Obtain the compatible adapter separately and place it here; it is not included in this release |
| `chronos-2/` | reference numerical specialist | Run `python scripts/download_models.py --c2-only` |
| `sundial-base-128m/` | specialist baseline | Obtain `thuml/sundial-base-128m` from its upstream release and place the files here |
| `aurora/` | additional boundary-case specialist | Obtain the Aurora checkpoint from the [upstream Aurora release](https://github.com/decisionintelligence/Aurora) and place the files here |

The runners use local paths for Chronos-2, Sundial, and Aurora so that the model revision is explicit. TimesFM, Moirai, and MOMENT load their upstream packages/checkpoints directly. See [`REPRODUCIBILITY_STATUS.md`](../docs/REPRODUCIBILITY_STATUS.md) for the exact boundary between source checks and full benchmark reproduction.

`scripts/download_models.py` pins Qwen3.5-9B to revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Chronos-2 to revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`. It writes these identifiers to
`chronoscribe_download.json` beside each downloaded model so a later run cannot
silently accept a different snapshot.
