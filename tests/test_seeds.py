import pytest

from twin import seeds


def test_ranges_do_not_overlap():
    items = sorted(seeds.RANGES.values())
    for (lo1, hi1), (lo2, hi2) in zip(items, items[1:], strict=False):
        assert lo1 < hi1 <= lo2 < hi2
    assert seeds.PATIENT == (1_000_000, 2_000_000)
    assert seeds.FUTURE == (2_000_000, 3_000_000)
    assert seeds.LIBRARY[0] == 1 and seeds.SEARCH[1] == 1_000_000


def test_helpers_stay_in_their_range():
    for k in range(16):
        assert seeds.role_of(seeds.library_seed(29_999, k)) == "library"
        assert seeds.role_of(seeds.library_seed(seeds.PILOT_THETA_OFFSET + 5, k)) == "library"
    for i in range(200):
        assert seeds.role_of(seeds.patient_seed(i)) == "patient"
        assert seeds.role_of(seeds.future_seed(f"test-{i:03d}", i)) == "future"
        assert seeds.role_of(seeds.calibration_seed("dev-001", i)) == "calibration"
        assert seeds.role_of(seeds.search_seed("dev-001", i)) == "search"


def test_library_seeds_are_unique_and_pilot_is_disjoint():
    lib = {
        seeds.library_seed(t, k) for t in range(seeds.MAX_LIBRARY_DESIGN_POINTS) for k in range(16)
    }
    assert len(lib) == seeds.MAX_LIBRARY_DESIGN_POINTS * 16
    pilot = {
        seeds.library_seed(seeds.PILOT_THETA_OFFSET + t, k) for t in range(1000) for k in range(16)
    }
    assert not (lib & pilot)


def test_derived_seeds_are_deterministic_and_distinct():
    assert seeds.patient_seed(3) == seeds.patient_seed(3)
    assert len({seeds.patient_seed(i) for i in range(50_000)}) == 50_000
    assert len({seeds.future_seed("test-001", r) for r in range(1000)}) == 1000
    assert len({seeds.calibration_seed("dev-001", k) for k in range(1000)}) == 1000
    assert len({seeds.search_seed("dev-001", k) for k in range(1000)}) == 1000
    assert seeds.future_seed("a", 1) != seeds.future_seed("b", 1)


def test_library_seed_validation():
    with pytest.raises(ValueError):
        seeds.library_seed(0, 16)
    with pytest.raises(ValueError):
        seeds.library_seed(10**6, 0)
    with pytest.raises(ValueError):
        seeds.role_of(3_000_000)
