#!/usr/bin/env python3
"""
Scaffold a new OEDISI-style component/module using best practices.

Usage:
    python generate_component.py "short description of what this module does"

This script will create a new component layout under:
    ./oedisi-example/<component_name>/

The component_name is derived from the context string (lowercased, dashed).
"""

import os
import re
import sys
import textwrap
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """
    Convert free-form text into a filesystem-safe component name.

    Examples:
        "Grid Simulation Federate" -> "grid-simulation-federate"
        "MyComponent" -> "mycomponent"
    """
    text = text.strip().lower()
    # Replace non-alphanumeric with dashes
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "oedisi-component"


def to_package_name(component_name: str) -> str:
    """
    Convert component name to a Python package name.
    Example:
        "grid-simulation-federate" -> "grid_simulation_federate"
    """
    return component_name.replace("-", "_")


# ---------------------------------------------------------------------------
# Templates using best-practice guide content
# ---------------------------------------------------------------------------

BEST_PRACTICE_GUIDE = textwrap.dedent("""
    # Best Practice Guide for OEDISI Component Development

    - Purpose: Practical, repeatable steps for building a new OEDISI component.
    - Audience: Component developers and integrators working with the `oedisi` framework.

    ## Repository & Project Setup
    - Prefer location: Create the new component inside the `oedisi-example` repository.
      Keeping components together simplifies testing, version control, and CI/integration workflows.
    - If adding to `oedisi-example`: Create a top-level directory `oedisi-example/<component-name>/`
      and follow the recommended layout.
    - Required docs: Add `LICENSE.md` and `CONTRIBUTORS.json` that describe commit/PR expectations
      and how to run tests locally.
    - Alternate location: For independently versioned components, create a dedicated repository
      under the `openEDI` organization (not preferred).
    - Prohibited: Avoid private or internal-only repositories.

    ## Recommended Component Layout
    - `component/` — source package (python module, go module, etc.).
    - `tests/unit/` — unit tests for core logic.
    - `tests/integration/` — system-level tests that run the component with other federates or a minimal orchestrator.
    - `examples/` — example configs, `component_definition.json`, `input_mapping.json`, sample data.
    - `Dockerfile` / `docker-compose.yml` — for replicable integration test runs.
    - `README.md` — usage, configuration, and quickstart (also update central docs).

    ## Scaffolding Steps
    - Create the directory under `oedisi-example/` and initialize git (or create a repository under `openEDI`).
    - Add `.gitignore`, `LICENSE.md`, and `README.md`.
    - Scaffold a minimal package in `component/` with a clear entry point, configuration loader,
      and a simple run loop that can be exercised by tests.
    - Add `component_definition.json` describing inputs/outputs and include an example `input_mapping.json` in `examples/`.

    ## Testing Strategy
    - Unit tests: Keep logic pure and unit-testable. Use `pytest` (or the project standard).
      Put tests in `tests/unit/`.
    - Mock dependencies: Abstract external systems (HELICS, databases) with adapter interfaces
      so unit tests can use fast mocks.
    - Integration tests: Put end-to-end tests in `tests/integration/`. Use a local HELICS broker
      or `docker-compose` to run multiple federates for system tests.
    - Test data & fixtures: Store small representative datasets in `examples/` or `tests/fixtures/`
      so CI runs quickly.

    ## Continuous Integration
    - GitHub Actions: Create workflows to run unit tests on each PR and a separate workflow/job
      for integration tests (nightly or on-demand if they are long-running).
    - Integration job config: Use `services`, `docker-compose`, or reusable workflows to
      bring up HELICS and dependent services.
    - Dependency caching: Cache dependencies to speed up CI runs.

    ## Documentation & Discovery
    - Live docs: Keep this best practice guide and the component `README.md` up to date
      as you add features.
    - Component metadata: Maintain `component_definition.json` to reflect actual inputs,
      outputs, and configuration.
    - Examples: Provide at least one runnable example in `examples/` demonstrating a typical
      run and expected outputs.

    ## Collaboration & Avoiding Silos
    - Don't build in a silo: Avoid side-loading models or making large, undocumented changes in a single component.
    - Start minimal: Implement a small, well-documented component that demonstrates the required
      inputs/outputs and includes a `component_definition.json`. Use this artifact to propose
      core API or schema changes.
    - Communicate early: Open an issue or RFC in the appropriate `openEDI` repo describing the change,
      expected APIs, and link to the minimal example so reviewers can run and test it.
    - Design for extensibility: Prefer plug-ins or optional configuration over hard-coded changes.
      Provide fallbacks so the component can run even if core features are not yet available.

    ## Release & Contribution Workflow
    - Use semantic or conventional commits to help automate changelogs.
    - Use PR templates and require at least one reviewer for new components or breaking changes.
    - PR checklist: unit tests pass, integration smoke-tested (if applicable), docs updated,
      and `component_definition.json` validated.

    ## Checklist (Quick Start)
    - Create component scaffold under `oedisi-example/<component-name>/`.
    - Add `component_definition.json` and an example `helics_config.json` in `examples/`.
    - Implement unit tests in `tests/unit/` and run them locally:

      ```bash
      pytest tests/unit
      ```

    - Add an integration test in `tests/integration/` and provide `docker-compose.yml` to run it:

      ```bash
      docker-compose -f docker-compose.yml up --build
      ```

    - Update the component `README.md` and this best practice guide with usage and test instructions.
""").strip() + "\n"


def render_readme(component_name: str, context: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f"""
    # {component_name}

    Component context:
    > {context}

    This component is scaffolded following the OEDISI best practice guide.

    ## Layout

    - `component/` — main Python package (`{pkg_name}`) with an entry point.
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
    """).strip() + "\n"


def render_license() -> str:
    year = datetime.utcnow().year
    return textwrap.dedent(f"""
    MIT License

    Copyright (c) {year} openEDI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """).strip() + "\n"


def render_contributors() -> str:
    return textwrap.dedent("""
    [
      {
        "name": "Example Contributor",
        "email": "example@example.com",
        "role": "maintainer",
        "notes": "Update this file with real contributors and PR expectations."
      }
    ]
    """).strip() + "\n"


def render_component_definition(context: str) -> str:
    return textwrap.dedent(f"""
    {{
      "name": "example-component",
      "description": "{context}",
      "version": "0.1.0",
      "inputs": [
        {{
          "name": "example_input",
          "type": "float",
          "description": "Example input value"
        }}
      ],
      "outputs": [
        {{
          "name": "example_output",
          "type": "float",
          "description": "Example output value"
        }}
      ],
      "configuration": {{
        "example_param": {{
          "type": "float",
          "default": 1.0,
          "description": "Example configuration parameter"
        }}
      }}
    }}
    """).strip() + "\n"


def render_input_mapping() -> str:
    return textwrap.dedent("""
    {
      "mappings": [
        {
          "source": "helics_topic/input",
          "target": "example_input"
        }
      ]
    }
    """).strip() + "\n"


def render_dockerfile(component_name: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f"""
    FROM python:3.13-slim

    WORKDIR /app

    COPY . /app

    RUN pip install --no-cache-dir -e .[dev]

    CMD ["pytest", "tests/unit"]
    """).strip() + "\n"


def render_docker_compose(component_name: str) -> str:
    return textwrap.dedent(f"""
    version: "3.9"
    services:
      {component_name}:
        build: .
        command: pytest tests/integration
        volumes:
          - ./:/app
    """).strip() + "\n"


def render_pyproject(component_name: str, context: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f"""
    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "{component_name}"
    version = "0.1.0"
    description = "{context}"
    authors = [{{ name = "openEDI", email = "info@example.com" }}]
    readme = "README.md"
    requires-python = ">=3.13"
    dependencies = []

    [project.optional-dependencies]
    dev = ["pytest"]

    [tool.setuptools.packages.find]
    where = ["component"]
    """).strip() + "\n"


def render_init_py(component_name: str, context: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f'''
    """
    {pkg_name}

    Context:
        {context}

    This package provides a minimal OEDISI component skeleton following the best
    practice guide:
    - Clear entry point (`main` function and CLI).
    - Simple run loop that can be tested.
    - Configuration placeholder for future extension.
    """

    from .main import main

    __all__ = ["main"]
    ''').strip() + "\n"


def render_main_py(component_name: str, context: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f'''
    """
    Main entry point for the `{pkg_name}` component.

    This is intentionally minimal so it can be easily unit-tested and integrated
    in system-level tests.

    Key ideas:
    - Keep core logic pure and testable (see `run_step`).
    - Add adapters around external systems (HELICS, databases, etc.).
    - Expand configuration handling as the component evolves.
    """

    from dataclasses import dataclass
    from typing import Any, Dict


    @dataclass
    class Config:
        \"\"\"Configuration for the `{pkg_name}` component.\"\"\"

        example_param: float = 1.0


    def run_step(value: float, config: Config) -> float:
        \"\"\"Pure function representing one step of component logic.

        Args:
            value: Input value.
            config: Component configuration.

        Returns:
            Transformed value (example: scaled by `example_param`).
        \"\"\"
        return value * config.example_param


    def run_loop(initial_value: float, steps: int, config: Config) -> float:
        \"\"\"Simple run loop to be replaced/extended with real simulation logic.\"\"\"
        current = initial_value
        for _ in range(steps):
            current = run_step(current, config)
        return current


    def load_config(raw: Dict[str, Any]) -> Config:
        \"\"\"Load configuration from a dictionary.

        In real usage, this could parse JSON/YAML or environment variables.
        \"\"\"
        return Config(
            example_param=float(raw.get("example_param", 1.0)),
        )


    def main() -> None:
        \"\"\"CLI entry point.

        For now this just runs a minimal demo and prints the result, but it is
        structured so that integration with HELICS or other orchestrators can
        be added later without changing the core logic.
        \"\"\"
        print("Running {pkg_name} component")
        config = load_config({{"example_param": 1.5}})
        result = run_loop(initial_value=1.0, steps=3, config=config)
        print(f"Final result: {{result}}")


    if __name__ == "__main__":
        main()
    ''').strip() + "\n"


def render_unit_test(component_name: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f'''
    import {pkg_name}  # type: ignore
    from {pkg_name}.main import run_step, Config


    def test_run_step_scales_value():
        cfg = Config(example_param=2.0)
        assert run_step(3.0, cfg) == 6.0
    ''').strip() + "\n"


def render_integration_test(component_name: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f'''
    import subprocess
    import sys


    def test_component_cli_runs():
        # Smoke test: run the module as a script and ensure it exits successfully.
        result = subprocess.run(
            [sys.executable, "-m", "{pkg_name}.main"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Final result" in result.stdout
    ''').strip() + "\n"


def render_gitignore() -> str:
    return textwrap.dedent("""
    __pycache__/
    *.pyc
    .pytest_cache/
    .venv/
    .env/
    dist/
    build/
    *.egg-info/
    .DS_Store
    """).strip() + "\n"


def render_server(component_name: str) -> str:
    pkg_name = to_package_name(component_name)
    return textwrap.dedent(f'''
    from functools import cache
    import traceback
    import requests
    import logging
    import socket
    import json
    import os

    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from {pkg_name} import run_simulator
    from fastapi.responses import JSONResponse
    import uvicorn

    from oedisi.componentframework.system_configuration import ComponentStruct
    from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames
    from oedisi.types.common import BrokerConfig

    app = FastAPI()


    @cache
    def kubernetes_service():
        if "KUBERNETES_SERVICE_NAME" in os.environ:
            # works with kurenetes
            return os.environ["KUBERNETES_SERVICE_NAME"]
        elif "SERVICE_NAME" in os.environ:
            return os.environ["SERVICE_NAME"]  # works with minikube
        else:
            return None


    def build_url(host: str, port: int, enpoint: list):

        if kubernetes_service():
            logging.info("Containers running in docker-compose environment")
            url = f"http://{{host}}.{{kubernetes_service()}}:{{port}}/"
        else:
            logging.info("Containers running in kubernetes environment")
            url = f"http://{{host}}:{{port}}/"
        url = url + "/".join(enpoint)
        logging.info(f"Built url {{url}}")
        return url


    @app.get("/")
    async def read_root():
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        response = HeathCheck(
            hostname=hostname,
            host_ip=host_ip
        ).dict()
        return JSONResponse(response, 200)


    @app.post("/run")
    async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
        logging.info(broker_config)
        feeder_host = broker_config.feeder_host
        feeder_port = broker_config.feeder_port
        url = build_url(feeder_host, feeder_port, ['sensor'])
        logging.info(f"Making a request to url - {{url}}")
        try:
            reply = requests.get(url)
            sensor_data = reply.json()
            if not sensor_data:
                msg = "empty sensor list"
                raise HTTPException(404, msg)
            logging.info(f"Received sensor data {{sensor_data}}")
            logging.info("Writing sensor data to sensors.json")
            with open("sensors.json", "w") as outfile:
                json.dump(sensor_data, outfile)

            background_tasks.add_task(run_simulator, broker_config)
            response = ServerReply(
                detail=f"Task sucessfully added."
            ).dict()
            return JSONResponse(response, 200)
        except Exception as e:
            err = traceback.format_exc()
            raise HTTPException(500, str(err))


    @app.post("/configure")
    async def configure(component_struct: ComponentStruct):
        component = component_struct.component
        params = component.parameters
        params["name"] = component.name
        links = {{}}
        for link in component_struct.links:
            links[link.target_port] = f"{{link.source}}/{{link.source_port}}"
        json.dump(links, open(DefaultFileNames.INPUT_MAPPING.value, "w"))
        json.dump(params, open(DefaultFileNames.STATIC_INPUTS.value, "w"))
        response = ServerReply(
            detail=f"Sucessfully updated configuration files."
        ).dict()
        return JSONResponse(response, 200)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ['PORT']))
    ''').strip() + "\n"

# ---------------------------------------------------------------------------
# Main scaffolding logic
# ---------------------------------------------------------------------------


def create_file(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"  created {path}")
    else:
        print(f"  skipped (exists) {path}")


def scaffold_component(context: str) -> None:
    component_name = slugify(context)
    pkg_name = to_package_name(component_name)

    base_dir = Path(".") / component_name
    print(f"Scaffolding component at: {base_dir}")

    # Directories
    component_dir = base_dir / "component" / pkg_name
    tests_unit_dir = base_dir / "tests" / "unit"
    tests_integration_dir = base_dir / "tests" / "integration"
    examples_dir = base_dir / "examples"

    for d in [component_dir, tests_unit_dir, tests_integration_dir, examples_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ensured directory {d}")

    # Top-level files
    create_file(base_dir / "README.md", render_readme(component_name, context))
    create_file(base_dir / "LICENSE.md", render_license())
    create_file(base_dir / "CONTRIBUTORS.json", render_contributors())
    create_file(base_dir / "best_practice_guide.md", BEST_PRACTICE_GUIDE)
    create_file(base_dir / ".gitignore", render_gitignore())
    create_file(base_dir / "pyproject.toml",
                render_pyproject(component_name, context))
    create_file(base_dir / "Dockerfile", render_dockerfile(component_name))
    create_file(base_dir / "docker-compose.yml",
                render_docker_compose(component_name))
    create_file(base_dir / "server.py", render_server(component_name))

    # Component package files
    create_file(component_dir / "__init__.py",
                render_init_py(component_name, context))
    create_file(component_dir / "main.py",
                render_main_py(component_name, context))

    # Tests
    create_file(
        tests_unit_dir / f"test_{pkg_name}.py",
        render_unit_test(component_name),
    )
    create_file(
        tests_integration_dir / f"test_{pkg_name}_integration.py",
        render_integration_test(component_name),
    )

    # Examples and metadata
    create_file(examples_dir / "component_definition.json",
                render_component_definition(context))
    create_file(examples_dir / "input_mapping.json", render_input_mapping())

    print("Done.")


def main_cli() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python oedisi_new.py \"short description / context for the module\"",
            file=sys.stderr,
        )
        sys.exit(1)

    context = sys.argv[1].strip()
    if not context:
        print("Error: context string must be non-empty.", file=sys.stderr)
        sys.exit(1)

    scaffold_component(context)


if __name__ == "__main__":
    main_cli()
