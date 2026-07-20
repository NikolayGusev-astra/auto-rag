def test_resolve_credential_reads_environment_and_allows_none(monkeypatch):
    from rag_core.gateway.secrets import resolve_credential

    monkeypatch.setenv("JIRA_TOKEN", "secret-from-environment")

    assert resolve_credential("env:JIRA_TOKEN") == "secret-from-environment"
    assert resolve_credential(None) is None
