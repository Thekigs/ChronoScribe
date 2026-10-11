# Dependency profiles

These files keep the reproducible environments for the public source release
together without mixing incompatible specialist stacks.

- `requirements.txt` is the active ChronoScribe workflow.
- `moment.txt` is the isolated MOMENT baseline environment.
- `moirai.txt` is the isolated archival Moirai environment.

The project source itself is installed with `pip install -e .` after selecting
the dependency profile required by the workflow.
