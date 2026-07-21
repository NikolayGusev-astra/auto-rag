from rag_core.gateway.stage_timer import StageTimer


def test_stage_timer_basic() -> None:
    clock = iter((10.0, 10.125))
    timer = StageTimer(clock=lambda: next(clock))

    timer.start("local")
    timer.stop("local")

    assert timer.summary() == {
        "local": {"status": "completed", "duration_ms": 125.0},
    }


def test_stage_timer_skip() -> None:
    timer = StageTimer()

    timer.skip("web", reason="disabled")

    assert timer.summary()["web"] == {"status": "skipped", "reason": "disabled"}
    assert "duration_ms" not in timer.summary()["web"]


def test_stage_timer_summary() -> None:
    clock = iter((1.0, 1.02))
    timer = StageTimer(clock=lambda: next(clock))
    timer.start("corporate")
    timer.stop("corporate")
    timer.skip("browser_fallback", reason="not_needed")

    assert timer.summary() == {
        "corporate": {"status": "completed", "duration_ms": 20.0},
        "browser_fallback": {"status": "skipped", "reason": "not_needed"},
    }
