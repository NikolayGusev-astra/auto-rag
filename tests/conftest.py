import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag_core'))


@pytest.fixture
def sample_query():
    return "настройка postgresql streaming replication"


@pytest.fixture
def malicious_query():
    return 'foo" OR summary!="'
