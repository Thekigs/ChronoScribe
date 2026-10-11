# Tooling map

The scripts in this directory keep their historical paths so commands in the
reproducibility documentation remain stable. Use this map when looking for a
new tool; future additions should prefer the indicated group.

| Group | Typical scripts | Purpose |
| --- | --- | --- |
| `audit_*`, `evaluate_*` | `audit_*`, `evaluate_numeric_guard.py` | Release, provenance, and contract checks |
| `score_*`, `export_*` | `score_operational_forecasts.py`, `export_operational_table.py` | Scoring and result export |
| `run_*` | `run_agentic_baselines.py`, `run_controller_forecasts.py` | Experiment and baseline execution |
| `normalize_*`, `consolidate_*`, `merge_*` | `normalize_benchmark_ids.py` | Data and result preparation |
| `build_*`, `generate_*`, `annotate_*` | `build_adaptive_deployment_control.py` | Artifact and report generation |

Scripts are intentionally standalone command-line entry points. Move a script
into a subdirectory only together with a compatibility wrapper or a documented
command migration.
