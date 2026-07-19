from unittest import mock

from rag_core import rag_async


def test_cache_set_logs_explicit_dcd_without_global():
    """deslop: routing-лог получает DCD явным аргументом, без _LAST_DCD.

    Это исключает утечку маршрутизации между параллельными запросами.
    """
    dcd = {"domain": "database", "collection": "database", "confidence": 0.8}
    result = {"chunks": [], "_trace": ""}

    with mock.patch.object(rag_async, "_log_routing") as log_routing:
        rag_async._cache_set("test-explicit-dcd", result, dcd=dcd)

    log_routing.assert_called_once()
    assert log_routing.call_args.args[1] == dcd


def test_last_dcd_global_removed():
    """Регрессия: модуль больше не хранит DCD предыдущего запроса глобально."""
    assert not hasattr(rag_async, "_LAST_DCD")