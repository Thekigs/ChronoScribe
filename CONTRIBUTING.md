# Contributing to ChronoScribe

Thank you for helping improve ChronoScribe. Small documentation fixes,
reproducibility reports, benchmark-quality improvements, and focused code
changes are all welcome.

## Before opening a pull request

1. Read the relevant documentation in `docs/` and check the repository's
   [security policy](SECURITY.md).
2. Keep benchmark membership, evaluation manifests, and provenance boundaries
   explicit. Do not add private data, credentials, model weights, or generated
   score artifacts.
3. Install the environment required by the workflow you are changing, then
   install the source package with `pip install -e .`.
4. Run the offline checks from the README:

   ```bash
   python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
   python tools/validate_release.py
   python -m unittest discover -v
   ```

## Pull requests

Describe the user-visible or reproducibility impact, list the commands used
for validation, and call out any changes to benchmark records, manifests,
metrics, or external inputs. Keep unrelated formatting and generated files
out of the change.

## Reporting security issues

Follow [SECURITY.md](SECURITY.md) and use a private report for credentials,
data-access issues, or vulnerabilities.
