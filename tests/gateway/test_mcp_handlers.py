import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.models import Evidence, EvidenceOrigin


@pytest.mark.asyncio
async def test_handler_returns_results_and_trace():
    class Connector:
        source = "local"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            return [
                Evidence(
                    id="doc#c0",
                    document_id="doc",
                    title="title",
                    text="body",
                    source=self.source,
                    origin=EvidenceOrigin.LOCAL_SNAPSHOT,
                )
            ]

    response = await handle_search(SearchRequest(query="q"), {"local": Connector()})

    assert "results" in response
    assert "trace" in response
    assert response["results"][0]["document_id"] == "doc"


@pytest.mark.asyncio
async def test_web_connector_not_called_when_include_web_is_false():
    class WebConnector:
        source = "web"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            raise AssertionError("web connector must not be searched by default")

    response = await handle_search(
        SearchRequest(query="q", include_web=False), {"web": WebConnector()}
    )

    assert response["results"] == []


@pytest.mark.asyncio
async def test_public_web_excluded_from_runtime_without_opt_in():
    class WebConnector:
        source = "public_web"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            raise AssertionError("public web must be opt-in")

    response = await handle_search(SearchRequest(query="q"), {"web": WebConnector()})

    assert "public_web" not in response["runtime"]["source_status"]
