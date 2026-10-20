import json
import os

import pyarrow.parquet as pq
import pytest

from twin import seeds
from twin.library import design, library_configs, write_shards
from twin.params import Prior

PRIOR = Prior(
    {
        "inj_number": (10, 50),
        "oxyHeal": (0.03, 0.30),
        "infectSpread": (2, 12),
        "numRecurInj": (1, 6),
        "numInfectRepeat": (1, 6),
    }
)


def test_every_theta_seed_pair_appears_exactly_once():
    thetas = design(PRIOR, 64)
    cfgs = library_configs(thetas, 0, 16)
    assert len(cfgs) == 64 * 16
    assert len({c["id"] for c in cfgs}) == len(cfgs)
    assert len({c["seed"] for c in cfgs}) == len(cfgs)
    assert all(seeds.role_of(c["seed"]) == "library" for c in cfgs)
    assert {c["output_stride"] for c in cfgs} == {51} and {c["num_steps"] for c in cfgs} == {5760}


def test_extending_the_design_keeps_earlier_points_and_ids_continue():
    a = design(PRIOR, 32)
    b = design(PRIOR, 64)
    assert a == b[:32]
    ext = library_configs(design(PRIOR, 32, start=32), 32, 4)
    assert ext[0]["id"] == "32/0" and ext[0]["seed"] == seeds.library_seed(32, 0)
    assert ext[0]["inj_number"] == library_configs(b[32:33], 32, 1)[0]["inj_number"]


def test_shards_partition_the_configs(tmp_path):
    cfgs = library_configs(design(PRIOR, 16), 0, 4)
    paths = write_shards(cfgs, tmp_path, 5)
    back = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert sorted(c["id"] for c in back) == sorted(c["id"] for c in cfgs)


@pytest.mark.skipif(not os.environ.get("IIRABM_BIN"), reason="no engine binary")
def test_simulate_job_writes_long_format_parquet(tmp_path):
    from twin.jobs.simulate import process_shard

    cfgs = library_configs(design(PRIOR, 2), 0, 2, num_steps=130)
    shard = write_shards(cfgs, tmp_path / "in", 1)[0]
    out = tmp_path / "out"
    out.mkdir()
    info = process_shard(shard, out, workers=2)
    assert info["configs"] == 4
    runs = pq.read_table(out / "runs-shard-000.parquet").to_pylist()
    assert sorted(r["id"] for r in runs) == ["0/0", "0/1", "1/0", "1/1"]
    assert {(r["theta_id"], r["replicate"]) for r in runs} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    series = pq.read_table(out / "series-shard-000.parquet")
    # One row per record: multiples of 51 up to the last completed step (a run that dies at step d
    # has no record at d).
    expected = sum(((r["death_step"] - 1) if r["died"] else r["steps_run"]) // 51 for r in runs)
    assert series.num_rows == expected > 0
    assert {"id", "step", "TNF", "oxy_deficit", "TH2"} <= set(series.column_names)
