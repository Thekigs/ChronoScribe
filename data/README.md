# Local data artifacts

`data/` contains locally downloaded upstream datasets and files used by the source resolver. It is ignored from public source releases because upstream licenses and redistribution terms differ.

The benchmark records and fixed paired selection live under `benchmark/`; `data/` is only the raw/local source layer. Forecast-only scoring accepts a row only when its manifest target can be located in the resolved source series. See `docs/BENCHMARK.md` and `docs/REPRODUCIBILITY.md` for provenance and licensing rules.
