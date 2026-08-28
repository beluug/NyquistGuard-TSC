from experiment_dashboard import SystemMetrics


def test_gpu_query_is_cached(monkeypatch) -> None:
    calls = []

    def fake_query():
        calls.append(1)
        return 10.0, 1.0, 12.0, "test gpu"

    monkeypatch.setattr(SystemMetrics, "_query_gpu", staticmethod(fake_query))
    monitor = SystemMetrics(gpu_interval_seconds=60.0)
    assert monitor.gpu()[0] == 10.0
    assert monitor.gpu()[0] == 10.0
    assert len(calls) == 1
