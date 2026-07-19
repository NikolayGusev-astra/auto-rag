import json
import subprocess
import sys

import pytest


def _stdio_client(client_name: str):
    def call(message: dict) -> dict:
        completed = subprocess.run(
            [sys.executable, "-m", "rag_core.gateway.server"],
            input=json.dumps(message) + "\n",
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return json.loads(completed.stdout.splitlines()[0])

    call.__name__ = f"{client_name}_client"
    return call


@pytest.fixture
def hermes_client():
    return _stdio_client("hermes")


@pytest.fixture
def codex_client():
    return _stdio_client("codex")


def test_hermes_and_codex_clients_receive_structured_evidence(hermes_client, codex_client):
    for client in (hermes_client, codex_client):
        response = client({"method": "search", "params": {"query": "deploy"}})

        assert "results" in response
        assert isinstance(response["results"], list)
        assert "runtime" in response
