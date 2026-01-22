# hub-topology

Component context:
> hub_topology

This component is scaffolded following the OEDISI best practice guide.

## Layout

- `component/` — main Python package (`hub_topology`) with an entry point.
- `tests/unit/` — unit tests for core logic.
- `tests/integration/` — integration/system tests.
- `examples/` — example configs, `component_definition.json`, sample data.
- `Dockerfile` / `docker-compose.yml` — for replicable integration test runs.

## Quick Start

Create and activate a virtual environment, then run:

```bash
pip install -e .
pytest tests/unit
```

To run integration tests (once implemented):

```bash
docker-compose -f docker-compose.yml up --build
```

## Updating Docs

- Keep this `README.md` up to date with usage and configuration examples.
- Keep `best_practice_guide.md` in sync with lessons learned.
- Maintain `component_definition.json` to reflect inputs/outputs and configuration.
