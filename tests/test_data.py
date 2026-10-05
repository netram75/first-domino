# checks on the simulator
import random

from generate_data import MINUTES, make_pool, make_system


def test_same_seed_same_data():
    a, b = make_pool(5, "train", seed=3), make_pool(5, "train", seed=3)
    for name in a:
        assert a[name].equals(b[name])


def test_roots_are_real_services():
    pool = make_pool(30, "test", seed=4)
    services = pool["services"].groupby("incident_id").service.apply(set)
    for row in pool["labels"].itertuples():
        roots = row.root_causes.split()
        assert 1 <= len(roots) <= 2
        assert set(roots) <= services[row.incident_id]


def test_calls_only_go_down():
    # every call goes to a deeper level, so there are no loops
    for seed in range(20):
        system = make_system(random.Random(seed), "test")
        for a, b in system["edges"]:
            assert system["nodes"][a]["level"] < system["nodes"][b]["level"]


def test_everything_but_gateways_has_a_caller():
    for seed in range(20):
        system = make_system(random.Random(seed), "train")
        called = {b for _, b in system["edges"]}
        for name, info in system["nodes"].items():
            if info["kind"] != "gateway":
                assert name in called


def test_metrics_look_sane():
    m = make_pool(10, "train", seed=5)["metrics"]
    assert (m.groupby(["incident_id", "service"]).size() == MINUTES).all()
    ok = m.dropna()
    assert ok.error_rate.between(0, 1).all()
    assert ok.cpu.between(0, 100).all()
    assert (ok.latency_ms > 0).all()
