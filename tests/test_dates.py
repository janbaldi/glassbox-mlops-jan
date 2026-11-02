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


def test_the_pairs_a_batch_reverses_are_named_with_their_batches():
    rows, report = _found()
    (swap,) = report.swaps
    assert (swap["Usually first"], swap["Usually second"]) == ("seed", "check")
    assert swap["Reversed %"] == 8.9 and swap["Neighbours"] == "yes"
    assert swap["Reversed in"].startswith("B000, B012, B024") and "3 more" in swap["Reversed in"]
    assert rows["check"]["Before previous %"] == 8.9 and rows["check"]["Swaps with"] == "seed"
    assert report.reversed_in[0] == ["check before seed"] and report.reversed_in[1] == []
    first, *_ = dates.batch_rows(report)
    assert first["Pairs reversed"] == 1 and first["Reversed"] == "check before seed"
    # Row before column, ties half.
    k = report.order.index
    assert report.before[k("seed")][k("check")] == 0.911
    assert report.before[k("harvest")][k("harvest copy")] == 0.5


def test_the_time_of_day_is_read_round_the_clock():
    rows, report = _found()
    # Times either side of midnight stay together: unwrapped past 24, read back as a clock.
    clock = pd.Series(report.clock["night sample"], dtype=float)
    assert clock.min() > 20 and clock.max() > 24
    assert rows["night sample"]["Median time of day"].startswith(("23", "00"))
    assert rows["thaw"]["Median time of day"].startswith("0")
    assert rows["release"]["Median time of day"] == ""
    said = "\n".join(report.findings)
    assert "Done at night: *night sample*" in said
    assert "any time of day" not in said
    shifted = dates.circular(pd.Series([23.5, 0.5, 23.0, 1.0]))
    assert list(shifted) == [23.5, 24.5, 23.0, 25.0]
    assert list(dates.circular(pd.Series([8.0, 9.0]))) == [8.0, 9.0]


def test_the_findings_say_what_the_dates_show():
    rows, report = _found()
    said = "\n".join(report.findings)
    assert "**day 0**: *start* → *seed* → *check*; **day 3**: *feed*" in said
    assert "*check* comes before *seed* in 9% of the batches with both (B000" in said
    assert "*harvest* and *harvest copy* hold the same timepoint" in said
    assert "*feed* comes later over the months" in said
    assert "*feed* has no fixed day: day 3 in 67%, day 4 in 33%" in said
    assert "*harvest copy* has no fixed day" not in said, "said once, of the first of the two"
    assert "*qc* is recorded from 2024-03-31 on" in said
    assert "The longest wait is between *qc* and *release*" in said
    assert "*lot made* is the date of a lot or a material" in said
    assert "Off the timeline: *thaw*" in said


def test_too_few_values_keep_a_column_off_the_timeline_and_nothing_breaks():
    rows, report = dates.for_build(_process(n=15), {}, "key", "start")
    assert report.order == [] and report.swaps == []
    assert {r["Parameter"] for r in rows} >= {"start", "seed"}
    assert all(r["Off the timeline because"] for r in rows)
    assert len(dates.batch_rows(report)) == 15
    for draw in (plots.date_timeline, plots.date_days, plots.date_order):
        assert draw(report) is None
    _, empty = dates.for_build(pd.DataFrame({"key": ["a"], "x": [1.0]}), {}, "key", "start")
    assert empty.kinds == {} and plots.date_calendar(empty) is None


def test_every_figure_draws_every_batch():
    _, report = _found()
    for draw in (
        plots.date_timeline,
        plots.date_days,
        plots.date_order,
        plots.date_clock,
        plots.date_calendar,
        plots.date_lots,
    ):
        block = draw(report)
        assert block["kind"] == "figure" and block["spec"]["data"], draw.__name__
    dots = plots.date_timeline(report)["spec"]["data"][0]
    assert len(dots["x"]) == sum(v is not None for c in report.order for v in report.days[c])
    calendar = plots.date_calendar(report)["spec"]["data"]
    broken = next(t for t in calendar if t["name"] == "out of order here")
    assert len(broken["x"]) == 2 * sum(len(p) for p in report.reversed_in)


# ---------------------------------------------------------------------------
# The build and a new task
# ---------------------------------------------------------------------------
def test_the_build_writes_the_dates_report_and_table(built):
    folder = shared(built["directory"], built["config"], "batch_table").parent
    payload = stlite.read_payload((folder / "dates_report.html").read_text("utf-8"))
    assert payload["title"] == "Batch table dates" and payload["help"]["title"] == "Method"
    assert [s["title"] for s in payload["sections"]][:7] == [
        "What the dates say",
        "Explore the timeline",
        "The timeline",
        "The order",
        "The time of day",
        "Over the calendar",
        "Every date column",
    ]
    viewer = payload["sections"][1]["blocks"][0]
    assert viewer["kind"] == "date_viewer"
    assert len(viewer["batch_rows"]) == len(viewer["batches"])
    assert [row[0] for row in viewer["column_rows"]] == viewer["order"]
    book = pd.ExcelFile(folder / "dates.xlsx")
    assert {"Reversed pairs", "Batches", "Days", screening.ABOUT_SHEET} <= set(book.sheet_names)
    table = pd.read_excel(book, sheet_name=screening.EVIDENCE_SHEET)
    assert set(table["Site"]) == {names.site("a")}
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
    kept = tmp_path / "kept"
    assert cli.main(["--config", config, "task", "new", str(kept), "--keep-dates"]) == 0
    reasons = dict(taskconfig.load(kept / "task.yaml").predictors.exclude).values()
    assert not any(why.startswith("a date (") for why in reasons)
