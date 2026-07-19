"""Tests for compound query detection in rag_async."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_core"))

from rag_core.rag_async import _detect_compound


class TestCompoundDetection:
    def test_simple_product_query_not_compound(self):
        """Только продукт — не составной."""
        res = _detect_compound("установка АЛД Про", {"domain": "rusbitech", "collection": "x", "confidence": 0.5})
        assert res == []

    def test_simple_infra_query_not_compound(self):
        """Только инфра — не составной."""
        res = _detect_compound("настройка postgresql репликации", {"domain": "devops", "collection": "x", "confidence": 0.5})
        assert res == []

    def test_compound_ald_postgres(self):
        """альд + postgresql → 2 подзапроса (rusbitech + devops)."""
        res = _detect_compound("альд postgresql репликация", {"domain": "rusbitech", "collection": "x", "confidence": 0.5})
        assert len(res) == 2
        domains = {sq["domain"] for sq in res}
        assert "rusbitech" in domains
        assert "devops" in domains
        # product-часть хранит исходный запрос
        prod = [sq for sq in res if sq["domain"] == "rusbitech"][0]
        assert "альд" in prod["query"]

    def test_compound_astra_docker(self):
        """astra + docker → составной."""
        res = _detect_compound("astra linux docker compose", {"domain": "rusbitech", "collection": "x", "confidence": 0.5})
        assert len(res) == 2

    def test_compound_adds_infra_terms(self):
        """Infra-подзапрос включает извлечённые инфра-термины."""
        res = _detect_compound("альд postgresql patroni", {"domain": "rusbitech", "collection": "x", "confidence": 0.5})
        infra = [sq for sq in res if sq["domain"] == "devops"][0]
        assert "postgresql" in infra["query"]
        assert "patroni" in infra["query"]