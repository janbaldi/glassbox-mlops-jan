"""The dates report: every date column on one timeline, in order, by day and by time of
day. A new task leaves every date column out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import shared
from ml import cli, dates, names, plots, screening, stlite, taskconfig


def _h(x) -> pd.Series:
    return pd.to_timedelta(x, "h")


def _process(n: int = 90, seed: int = 1) -> pd.DataFrame:
    """A made-up process under made-up names, over several days."""
    rng = np.random.default_rng(seed)
    start = pd.Series(
        pd.Timestamp("2024-01-01 08:00")
        + pd.to_timedelta(np.arange(n) * 3, "D")
        + pd.to_timedelta(rng.integers(0, 120, n), "min")
    )
    midnight = start.dt.normalize()
    seeded = start + _h(2 + rng.normal(0, 0.2, n))
    check = seeded + _h(rng.normal(1, 0.3, n))
    check.iloc[::12] = seeded.iloc[::12] - _h(1)  # now and then before the seeding
    feed = midnight + pd.Timedelta(days=3) + _h(rng.normal(10, 1, n))
    feed[np.arange(n) >= 60] += pd.Timedelta(days=1)  # moved to day 4 from batch 60 on
    harvest = midnight + pd.Timedelta(days=5) + _h(rng.normal(9, 0.5, n))
    qc = harvest + _h(30)
    qc.iloc[:30] = pd.NaT  # recorded from batch 30 on
    return pd.DataFrame(
        {
            "key": [f"B{i:03d}" for i in range(n)],
            "start": start,
            "seed": seeded,
            "check": check,
            "feed": feed,
            "harvest": harvest,
            "harvest copy": harvest,
            # Around midnight: 23:00 to 01:00.
            "night sample": harvest + _h(14 + rng.normal(0, 0.6, n)),
            "qc": qc,
            "release": harvest.dt.normalize() + pd.Timedelta(days=5),
            # A time of day with no date, as a spreadsheet keeps one.
            "thaw": pd.Timestamp("1900-01-01") + _h(rng.uniform(7, 9, n)),
            # A lot's date: shared by batches started weeks apart.
            "lot made": pd.Timestamp("2023-10-01")
            + pd.to_timedelta((np.arange(n) // 15) * 40, "D"),
            "measured": rng.normal(size=n),
        }
    )


def _found():
    rows, report = dates.for_build(_process(), {}, "key", "start")
    return {row["Parameter"]: row for row in rows}, report


def test_every_date_column_is_put_in_order_by_day():
    rows, report = _found()
    assert report.anchor == "start"
    assert report.order == [
        "start",
        "seed",
        "check",
        "feed",
        "harvest",
        "harvest copy",
        "night sample",
        "qc",
        "release",
    ]
    assert "measured" not in report.kinds
    assert report.kinds["release"] == dates.DATE_ONLY
    assert report.kinds["thaw"] == dates.TIME_ONLY
    assert rows["lot made"]["Kind"] == dates.LOT and "lot made" in report.lots
    assert "thaw" not in report.order and "lot made" not in report.order
    # Phases: neighbours on the same usual day.
    assert report.phases == [
        (0, ["start", "seed", "check"]),
        (3, ["feed"]),
        (5, ["harvest", "harvest copy", "night sample"]),
        (6, ["qc"]),
        (10, ["release"]),
    ]
    assert rows["feed"]["Usual day"] == 3 and rows["feed"]["On usual day %"] == 66.7
    assert rows["harvest"]["Median"].startswith("day 5, 0")
    assert rows["seed"]["Median wait since previous (h)"] == 2.0
    assert rows["harvest"]["Same as"] == "harvest copy"
    # Every batch's place of every column, from the anchor's midnight.
    assert len(report.days["release"]) == 90 and report.days["release"][0] == 10.0


def test_each_column_says_how_it_sits_on_the_timeline():
    rows, report = _found()
    # Usually after seed, before it in one batch of twelve.
    assert rows["check"]["Before previous %"] == 8.9
    assert rows["feed"]["Drift ρ"] > 0.5, "moved to day 4 over the months"
    assert rows["qc"]["First recorded"] > rows["start"]["First recorded"]
    assert rows["lot made"]["Median hours from anchor"] < 0
    assert rows["thaw"]["Off the timeline because"].startswith("a time of day")
    days = dates.day_rows(report)
    assert len(days) == 90 and days[0]["release (day)"] == 10.0


def test_the_time_of_day_is_read_round_the_clock():
    rows, report = _found()
    # Times either side of midnight stay together: unwrapped past 24, read back as a clock.
    clock = pd.Series(report.clock["night sample"], dtype=float)
    assert clock.min() > 20 and clock.max() > 24
    assert rows["night sample"]["Median time of day"].startswith(("23", "00"))
    assert rows["thaw"]["Median time of day"].startswith("0")
    assert rows["release"]["Median time of day"] == ""
    shifted = dates.circular(pd.Series([23.5, 0.5, 23.0, 1.0]))
    assert list(shifted) == [23.5, 24.5, 23.0, 25.0]
    assert list(dates.circular(pd.Series([8.0, 9.0]))) == [8.0, 9.0]


def test_too_few_values_keep_a_column_off_the_timeline_and_nothing_breaks():
    rows, report = dates.for_build(_process(n=15), {}, "key", "start")
    assert report.order == [] and report.phases == []
    assert {r["Parameter"] for r in rows} >= {"start", "seed"}
    assert all(r["Off the timeline because"] for r in rows)
    for draw in (plots.date_timeline,):
        assert draw(report) is None
    _, empty = dates.for_build(pd.DataFrame({"key": ["a"], "x": [1.0]}), {}, "key", "start")
    assert empty.kinds == {} and plots.date_clock(empty) is None


def test_the_figures_draw_every_batch_light_enough_for_thousands():
    _, report = _found()
    for draw in (plots.date_timeline, plots.date_clock):
        block = draw(report)
        assert block["kind"] == "figure" and block["spec"]["data"], draw.__name__
    timeline = plots.date_timeline(report)["spec"]["data"]
    # Everything is WebGL: a browser draws WebGL above SVG, so the spreads must be too.
    assert {t["type"] for t in timeline} == {"scattergl"}
    clouds = [t for t in timeline if t.get("name") == "one batch"]
    # One WebGL cloud per column, holding every batch with a value; the hover names the batch.
    assert len(clouds) == len(report.order)
    assert sum(len(t["x"]) for t in clouds) == sum(
        v is not None for c in report.order for v in report.days[c]
    )
    assert clouds[0]["text"][0] == report.batches[0]
    # The spreads are three traces however many columns, not three per column.
    assert len(timeline) == len(report.order) + 3
    clock = plots.date_clock(report)["spec"]["data"]
    assert len(clock) == len(report.clock) + 3
    # The time of day is ordered by median, earliest at the top.
    rows = plots.date_clock(report)["spec"]["layout"]["yaxis"]["ticktext"]
    medians = [
        float(np.median([v for v in report.clock[c] if v is not None]))
        for c in (t.removesuffix(" (time only)") for t in rows)
    ]
    assert medians == sorted(medians) and rows[0] == "thaw (time only)"
    assert rows[-1] == "night sample"


# ---------------------------------------------------------------------------
# The build and a new task
# ---------------------------------------------------------------------------
def test_the_build_writes_the_dates_report_and_table(built):
    folder = shared(built["directory"], built["config"], "batch_table").parent
    payload = stlite.read_payload((folder / "dates_report.html").read_text("utf-8"))
    assert payload["title"] == "Batch table dates" and payload["help"]["title"] == "Method"
    assert [s["title"] for s in payload["sections"]] == [
        "The timeline",
        "The time of day",
        "Provenance",
    ]
    # The site in the sidebar; one picker over its manufacturing sites and lines, which
    # every section follows.
    assert payload["site"] == names.site("a")
    plant = f"{names.site('a')} plant"
    for section in payload["sections"][:2]:
        (chooser,) = section["blocks"]
        assert chooser["kind"] == "site_chooser" and chooser["group"] == "subsite"
        assert [o["label"] for o in chooser["options"]] == [
            f"{plant} · Line A",
            f"{plant} · Line B",
        ]
    timeline = payload["sections"][0]["blocks"][0]["options"]
    assert timeline[0]["blocks"][0]["kind"] == "figure"
    assert timeline[1]["blocks"][0]["text"].startswith("No date column has 20 values")
    book = pd.ExcelFile(folder / "dates.xlsx")
    assert {"Days", screening.ABOUT_SHEET} <= set(book.sheet_names)
    table = pd.read_excel(book, sheet_name=screening.EVIDENCE_SHEET)
    assert set(table["Manufacturing site"]) == {f"{plant} · Line A", f"{plant} · Line B"}
    assert names.col("date_start") in set(table["Parameter"])
    order = list(cli.BUILD_OUTPUTS)
    assert order.index("dates report") < order.index("missingness report")


def test_a_new_task_leaves_out_every_date_column(built, tmp_path):
    config = str(built["config_path"])
    table = pd.read_excel(
        shared(built["directory"], built["config"], "batch_table", "dates.xlsx"),
        sheet_name=screening.EVIDENCE_SHEET,
    )
    found = set(table["Parameter"])
    folder = tmp_path / "plain"
    assert cli.main(["--config", config, "task", "new", str(folder)]) == 0
    assert "Dates and times at this site" in (folder / "task.yaml").read_text("utf-8")
    excluded = dict(taskconfig.load(folder / "task.yaml").predictors.exclude)
    assert found <= set(excluded)
    dated = {p for p, why in excluded.items() if why.startswith("a date (")}
    assert dated and dated <= found
    # A date at any subsite is left out once: by its catalog role when it has one (the
    # catalog groups come first), else as a date, the reason saying where it was found.
    text = (folder / "task.yaml").read_text("utf-8")
    start = names.col("date_start")
    assert excluded[start] == "catalog Variable Role is 'Timestamp'"
    assert f"#   {start} (listed above)" in text.split("Dates and times at this site")[1]
    assert all("Line A" in why for why in (excluded[p] for p in dated))
    kept = tmp_path / "kept"
    assert cli.main(["--config", config, "task", "new", str(kept), "--keep-dates"]) == 0
    reasons = dict(taskconfig.load(kept / "task.yaml").predictors.exclude).values()
    assert not any(why.startswith("a date (") for why in reasons)
