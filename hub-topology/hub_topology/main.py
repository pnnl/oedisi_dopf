"""
Main entry point for the `hub_topology` component.

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
    """Configuration for the `hub_topology` component."""

    example_param: float = 1.0


def run_step(value: float, config: Config) -> float:
    """Pure function representing one step of component logic.

    Args:
        value: Input value.
        config: Component configuration.

    Returns:
        Transformed value (example: scaled by `example_param`).
    """
    return value * config.example_param


def run_loop(initial_value: float, steps: int, config: Config) -> float:
    """Simple run loop to be replaced/extended with real simulation logic."""
    current = initial_value
    for _ in range(steps):
        current = run_step(current, config)
    return current


def load_config(raw: Dict[str, Any]) -> Config:
    """Load configuration from a dictionary.

    In real usage, this could parse JSON/YAML or environment variables.
    """
    return Config(
        example_param=float(raw.get("example_param", 1.0)),
    )


def main() -> None:
    """CLI entry point.

    For now this just runs a minimal demo and prints the result, but it is
    structured so that integration with HELICS or other orchestrators can
    be added later without changing the core logic.
    """
    print("Running hub_topology component")
    config = load_config({"example_param": 1.5})
    result = run_loop(initial_value=1.0, steps=3, config=config)
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
