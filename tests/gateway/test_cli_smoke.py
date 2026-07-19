import subprocess
import sys


def test_gateway_module_help_works():
    completed = subprocess.run(
        [sys.executable, "-m", "rag_core.gateway.server", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0
    assert "stdio" in completed.stdout.lower()
