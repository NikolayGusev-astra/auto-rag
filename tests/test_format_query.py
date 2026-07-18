import pytest
from rag_core.rag_mcp_client import MCPClient


class TestFormatQuery:
    @pytest.fixture
    def client(self):
        return MCPClient()

    @pytest.fixture
    def jira_template(self):
        return '/rest/api/2/search?jql={query_and3}&maxResults={max}'

    def test_benign_query(self, client, jira_template):
        result = client.format_query(jira_template, "postgresql config", 5)
        assert 'text~"postgresql"' in result
        assert 'text~"config"' in result
        assert "AND" in result

    def test_jql_injection_attempt(self, client, jira_template):
        malicious = 'foo" OR summary!="'
        result = client.format_query(jira_template, malicious, 5)
        assert '\\"' in result, f"Quotes not escaped: {result}"
        jql = result.split("jql=")[1].split("&")[0]
        import re
        opens = jql.count('text~"')
        matches = re.findall(r'text~"((?:[^"\\]|\\.)*)"', jql)
        assert opens == len(matches), f"Unbalanced quotes in: {jql}"

    def test_single_word_query(self, client, jira_template):
        result = client.format_query(jira_template, "postgresql", 5)
        assert 'text~"postgresql"' in result

    def test_confluence_template(self, client):
        template = '/rest/api/content/search?cql=text~"{query_first3}"&limit={max}'
        result = client.format_query(template, "nginx reverse proxy ssl", 5)
        assert "nginx" in result
        assert "reverse" in result
