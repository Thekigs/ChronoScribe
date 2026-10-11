# Runner map

The runner paths are stable command-line entry points documented in
`README.md` and `docs/USAGE.md`.

- `run_chronoscribe.py`, `ablate_*.py`: reference and ablation workflows.
- `run_*_specialist.py`, `run_timesfm.py`, `run_moment.py`, `run_moirai.py`:
  specialist comparisons.
- `server.py`, `chat.py`, `run_api.py`: local service and interactive clients.
- `_ablate_fixed_c.py`: shared fixed-context experiment implementation.

Keep model-specific imports inside the runner that needs them so offline
validation remains usable without model weights.
