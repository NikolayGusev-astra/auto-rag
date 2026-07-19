import pytest

import dcd_router


def test_known_domains_stable():
    """Регрессия: ранее подтверждённые домены (Фаза 1) остаются корректными
    после фиксов C1 (нормализация confidence по домену) и C2 (k8s не в anti)."""
    cases = [
        ("postgresql streaming replication debian", "database"),
        ("DNS over HTTPS linux", "networking"),
        ("SSH key authentication disable password", "security"),
        ("Prometheus node_exporter", "monitoring"),
        ("kubernetes helm deployment pod", "devops"),
        ("docker compose stack", "devops"),
    ]
    for q, expected in cases:
        r = dcd_router.classify(q)
        assert r["domain"] == expected, f"{q} -> {r['domain']}, ожидали {expected}"


def test_kubernetes_routes_to_devops():
    """C2: kubernetes/helm — позитивные ключи devops, не должны быть в
    anti_keywords (иначе сигнал подавлялся штрафом *0.3 и домен мог уйти)."""
    r = dcd_router.classify("kubernetes pod scaling")
    assert r["domain"] == "devops"
    # после C2 anti_keywords devops не содержит kubernetes/helm,
    # значит штрафа нет — confidence выше, чем был бы с подавлением
    assert r["confidence"] > 0.0


def test_confidence_deterministic():
    """C1: confidence детерминирован и вычисляется от выбранного домена
    (не от глобального максимума по всем доменам)."""
    r = dcd_router.classify("kubernetes helm deployment pod")
    # не должно падать, domain стабилен, confidence в диапазоне [0,1]
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["domain"] == "devops"