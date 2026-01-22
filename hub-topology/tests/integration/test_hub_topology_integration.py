import subprocess
import sys


def test_component_cli_runs():
    # Smoke test: run the module as a script and ensure it exits successfully.
    result = subprocess.run(
        [sys.executable, "-m", "component.hub_topology.main"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "Final result" in result.stdout
