from hub_topology.main import run_step, Config


def test_run_step_scales_value():
    cfg = Config(example_param=2.0)
    assert run_step(3.0, cfg) == 6.0
