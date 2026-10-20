"""Record the actual AWS cost from Cost Explorer (which lags about a day) into
docs/results/actual_cost.json. Re-run after the lag; `render_results.py` quotes the file.

    uv run python scripts/cost_report.py [--since 2026-10-19]

The account also runs other projects (the sepsis demo), so the figure is the whole account over the
period unless resources carry an activated `project` cost-allocation tag (the tag grouping is shown too).
"""

from __future__ import annotations

import argparse
import datetime
import json

from twin.config import DOCS_DIR, boto_session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-19")
    args = ap.parse_args()
    ce = boto_session().client("ce", region_name="us-east-1")
    end = datetime.date.today() + datetime.timedelta(days=1)
    period = {"Start": args.since, "End": end.isoformat()}
    by_service = ce.get_cost_and_usage(TimePeriod=period, Granularity="MONTHLY", Metrics=["UnblendedCost"],
                                       GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}])  # fmt: skip
    daily = ce.get_cost_and_usage(TimePeriod=period, Granularity="DAILY", Metrics=["UnblendedCost"])
    services: dict[str, float] = {}
    for month in by_service["ResultsByTime"]:
        for g in month["Groups"]:
            services[g["Keys"][0]] = services.get(g["Keys"][0], 0.0) + float(
                g["Metrics"]["UnblendedCost"]["Amount"]
            )
    out = {
        "as_of": datetime.datetime.now(datetime.UTC).isoformat(timespec="minutes"),
        "since": args.since,
        "account_total_usd": round(sum(services.values()), 2),
        "by_service_usd": {k: round(v, 2) for k, v in sorted(services.items(), key=lambda kv: -kv[1]) if v >= 0.005},
        "daily_usd": {d["TimePeriod"]["Start"]: round(float(d["Total"]["UnblendedCost"]["Amount"]), 2) for d in daily["ResultsByTime"]},
        "note": "Cost Explorer lags about a day: the most recent days are incomplete. Whole account, not only this project.",
    }  # fmt: skip
    (DOCS_DIR / "results" / "actual_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
