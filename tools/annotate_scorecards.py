#!/usr/bin/env python3
"""Annotate historical generated scorecards with the current protocol pointer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = (
    "> This generated scorecard is protocol-bound. For current forecast-only claims, "
    "use `scores/operational_forecasting_verified_20261003.json` and "
    "`scores/experiment_registry_20261003.json`.\n"
)


def main() -> None:
    changed = 0
    for path in sorted((ROOT / "scores").rglob("scorecard.md")):
        text = path.read_text(encoding="utf-8")
        if "experiment_registry_20261003.json" in text:
            continue
        lines = text.splitlines()
        if not lines or not lines[0].startswith("# "):
            continue
        lines.insert(1, "")
        lines.insert(2, NOTE.rstrip())
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        changed += 1
    print(f"annotated {changed} scorecards")


if __name__ == "__main__":
    main()
