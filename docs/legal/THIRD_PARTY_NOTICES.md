# Third-party models and datasets

ChronoScribe code is released under the MIT License in `LICENSE`. Model weights and datasets are not bundled in this source release. Download or obtain them from their upstream providers and follow each provider's terms.

| Component | Upstream terms | Release handling |
|---|---|---|
| Qwen3.5-9B | See the upstream Qwen model card and license | Download locally with `scripts/download_models.py`; do not commit weights |
| Chronos-2 | See the Amazon/Chronos upstream model card and license | Download locally with `scripts/download_models.py`; do not commit weights |
| Sundial | Apache-2.0 model card | Local integration code only; obtain weights upstream |
| Aurora | MIT model card | Local integration code only; obtain weights upstream |
| Benchmark datasets | Terms vary by dataset and provider | `data/` is excluded; use `scripts/download_data.py` and comply with registration and redistribution terms |

Some sources, including SWaT and Yahoo S5, require registration and are intentionally not redistributed. The JSONL records under `benchmark/` are derived evaluation artifacts and may contain target values, summary values, labels, or anomaly metadata from upstream datasets. Read `benchmark/README.md` before publishing them; check the corresponding upstream terms before redistributing any affected record, even when the raw dataset is absent.
