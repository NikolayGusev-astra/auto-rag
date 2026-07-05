import pytest
from dcd_router import classify


class TestDCDRouter:
    @pytest.mark.parametrize("query,expected_domain", [
        ("настройка postgresql streaming replication", "database"),
        ("rust ownership borrow checker", "software-dev"),
        ("SSH key-based authentication disable password", "security"),
        ("nginx reverse proxy ssl letsencrypt", "security"),
        ("wireguard vpn config", "networking"),
        ("docker compose up production", "devops"),
        ("zfs pool create raidz2", "storage"),
        ("prometheus alert rule", "monitoring"),
        ("systemd service unit timer", "linux-admin"),
    ])
    def test_classify_domain(self, query, expected_domain):
        result = classify(query)
        assert result["domain"] == expected_domain, \
            f"Query: {query}\nExpected: {expected_domain}\nGot: {result['domain']}\nMatched: {result.get('keywords_matched', [])}"

    def test_empty_query(self):
        result = classify("")
        assert result["fallback"] is True
        assert result["confidence"] == 0.0

    def test_unknown_query(self):
        result = classify("random text about nothing specific")
        assert result["confidence"] < 0.3 or result["fallback"] is True

    def test_no_substring_false_positives(self):
        result = classify("ssh connection timeout")
        assert result["domain"] != "scripting", \
            f"'sh' matched inside 'ssh': {result}"

        result = classify("database based application")
        assert "scripting" not in result.get("keywords_matched", []), \
            f"'sed' matched inside 'based': {result}"
