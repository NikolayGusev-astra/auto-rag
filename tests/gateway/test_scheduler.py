def test_interactive_priority_over_sync():
    from rag_core.gateway.scheduler import PriorityQueue

    queue = PriorityQueue()
    queue.put("sync", 3)
    queue.put("search", 1)

    assert queue.get() == "search"


def test_low_cpu_priority_applies_without_error():
    from rag_core.gateway.scheduler import apply_low_cpu_priority

    assert isinstance(apply_low_cpu_priority(), bool)
