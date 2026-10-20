"""Compare simulation Parquet output (runs-*.parquet, series-*.parquet) in two directories, value
for value and exactly. Used for the amd64-container vs SageMaker parity check.

    uv run python scripts/compare_results.py DIR_A DIR_B
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq


def load(d: Path, kind: str):
    files = sorted(d.glob(f"{kind}-*.parquet"))
    if not files:
        raise SystemExit(f"no {kind}-*.parquet in {d}")
    import pyarrow as pa

    t = pa.concat_tables([pq.read_table(f) for f in files])
    return t.sort_by([("id", "ascending")] + ([("step", "ascending")] if kind == "series" else []))


def main(a: Path, b: Path) -> int:
    ok = True
    for kind in ("runs", "series"):
        ta, tb = load(a, kind), load(b, kind)
        same = ta.equals(tb)
        print(
            f"{kind}: {ta.num_rows} rows vs {tb.num_rows} rows -> {'IDENTICAL' if same else 'DIFFERENT'}"
        )
        ok = ok and same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
