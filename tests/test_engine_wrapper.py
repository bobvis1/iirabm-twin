"""run_batch against the real engine (skipped if no engine is available; see conftest.py)."""

import os

import pytest

from twin.engine import EngineError, run_batch

pytestmark = pytest.mark.skipif(not os.environ.get("IIRABM_BIN"), reason="no engine binary")


def configs(n=6):
    return [
        dict(
            id=f"w{i}",
            seed=i + 1,
            inj_number=20 + 3 * i,
            num_steps=700,
            output_stride=51,
        )
        for i in range(n)
    ]


def test_one_worker_equals_four_workers_and_order_is_preserved():
    cfgs = configs()
    one = run_batch(cfgs, n_workers=1)
    four = run_batch(cfgs, n_workers=4, chunk_size=1)
    assert [r.id for r in one] == [c["id"] for c in cfgs] == [r.id for r in four]
    for a, b in zip(one, four, strict=True):
        assert a.died == b.died and a.steps_run == b.steps_run
        for name in a.series:
            assert (a.series[name] == b.series[name]).all()


def test_readings_grid_matches_step_labels():
    (res,) = run_batch(configs(1), n_workers=1)
    assert list(res.step[:12]) == list(range(51, 612 + 1, 51))
    assert res.value_at("TNF", 612) == res.series["TNF"][11]


def test_engine_error_names_the_config():
    with pytest.raises(EngineError) as err:
        run_batch([dict(id="bad-one", sede=1)], n_workers=1)
    assert err.value.config_id == "bad-one"
