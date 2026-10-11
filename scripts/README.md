# Download scripts

This directory contains the two optional download entry points used to fetch
external inputs that are excluded from the source release:

- `python scripts/download_models.py` downloads the pinned Qwen3.5-9B and Chronos-2 snapshots.
- `python scripts/download_data.py` retrieves permitted public benchmark sources.

Both scripts write into the repository's `models/` or `data/` directories and
never commit downloaded artifacts. Read the reproducibility and licensing
notes before downloading registered or restricted sources.
