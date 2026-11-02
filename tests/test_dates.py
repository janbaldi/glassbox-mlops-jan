"""The dates report: the order of a batch's dates, the steps between them, and how the
batches cluster by their timing. A new task leaves every date column out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import shared
from ml import cli, dates, names, plots, screening, stlite, taskconfig


def _h(x) -> pd.Series:
    return pd.to_timedelta(x, "h")


def _process(n: int = 90, slower_from: int | None = 50, seed: int = 1) -> pd.DataFrame:
    """A made-up process under made-up names: start, seed, check, feed, harvest, release.
    From batch ``slower_from`` on, the feed step takes a day longer."""
    rng = np.random.default_rng(seed)
    start = pd.Series(
        pd.Timestamp("2024-01-01 08:00")
        + pd.to_timedelta(np.arange(n) * 3, "D")
        + pd.to_timedelta(rng.integers(0, 120, n), "min")
    )
    later = np.arange(n) >= (slower_from if slower_from is not None else n)
    seeded = start + _h(2 + rng.normal(0, 0.2, n))
    fed = seeded + _h(np.where(later, 96, 72) + rng.normal(0, 3, n))
    harvest = fed + _h(48 + rng.normal(0, 3, n))
    check = seeded + _h(rng.normal(1, 0.3, n))
    check.iloc[::12] = seeded.iloc[::12] - _h(1)  # now and then before the seeding
    return pd.DataFrame(
        {
            "key": [f"B{i:03d}" for i in range(n)],
            "start": start,
            "seed": seeded,
            "feed": fed,
            "harvest": harvest,
            "harvest copy": harvest,
            "release": harvest.dt.normalize() + pd.Timedelta(days=5),
            "check": check,
            # A time of day with no date, as a spreadsheet keeps one.
            "thaw": pd.Timestamp("1900-01-01") + _h(rng.uniform(7, 9, n)),
            # A lot's date: shared by batches started weeks apart.
            "lot made": pd.Timestamp("2023-10-01")
            + pd.to_timedelta((np.arange(n) // 15) * 40, "D"),
            "measured": rng.normal(size=n),
        }
    )


def test_the_order_the_steps_and_the_eras_are_found():
    rows, report = dates.for_build(_process(), {}, "key", "start")
    assert report.anchor == "start"
    assert report.order == ["start", "seed", "check", "feed", "harvest", "harvest copy", "release"]
    assert report.kinds["release"] == dates.DATE_ONLY
    assert report.kinds["thaw"] == dates.TIME_ONLY and "thaw" not in report.order
    assert "measured" not in report.kinds
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["harvest"]["Same as"] == "harvest copy"
    assert by_name["check"]["Out of order %"] == 8.9
    assert by_name["thaw"]["Median time of day"].startswith("0")
    assert by_name["release"]["Median time of day"] == ""
    # Two eras, split where the feed step lengthened, following one another in time.
    assert [(c.name, c.batches) for c in report.clusters] == [("Timing 1", 50), ("Timing 2", 40)]
    assert report.cluster_of[:50] == ["Timing 1"] * 50
    assert report.clusters[1].apart[0].startswith("check → feed 9")
    assert report.silhouette > dates.WEAK
    assert report.runs == 2 and report.runs_p < dates.P_MAX
    said = "\n".join(report.findings)
    assert "follow one another in time" in said
    assert "*check* comes before *seed* in 9%" in said
    assert "*harvest* and *harvest copy* hold the same timepoint" in said
    assert "*check → feed* lengthens over the months" in said


def test_a_lots_date_is_kept_off_the_timeline_and_told_as_an_age():
    rows, report = dates.for_build(_process(), {}, "key", "start")
    assert "lot made" not in report.order and "lot made" in report.shared
    assert report.not_placed["lot made"].startswith("shared by batches started weeks apart")
    assert not any("lot made" in step for step in report.clustered_on)
    row = next(r for r in rows if r["Parameter"] == "lot made")
    assert row["Position"] is None and row["Median hours from anchor"] < 0
    assert any("*lot made* is the date of a lot" in line for line in report.findings)


def test_one_timing_with_small_fixed_variations_makes_no_clusters():
    frame = _process(slower_from=None)
    # Every step varies by a few fixed minutes: cleanly split, and immaterial.
    frame["feed"] = frame["seed"] + _h(72 + (np.arange(len(frame)) % 3) * 0.1)
    frame["harvest"] = frame["feed"] + _h(48)
    frame = frame.drop(columns=["check", "release", "harvest copy"])
    _, report = dates.for_build(frame, {}, "key", "start")
    assert report.clusters == [] and "one timing" in report.unclustered
    assert set(report.cluster_of) == {""}


def test_the_weekday_a_batch_starts_on_can_go_with_its_cluster():
    frame = _process(slower_from=None, n=120)
    # Started on a Thursday or Friday, the feeding waits for the weekend to pass.
    weekend = frame["start"].dt.weekday.isin([3, 4])
    frame.loc[weekend, "feed"] += pd.Timedelta(hours=48)
    frame["harvest"] = frame["feed"] + _h(48)
    frame = frame.drop(columns=["harvest copy"])
    _, report = dates.for_build(frame, {}, "key", "start")
    assert len(report.clusters) >= 2
    assert report.weekdays_p is not None and report.weekdays_p < dates.P_MAX
    assert any("goes with its cluster" in line for line in report.findings)


def test_too_few_batches_make_no_clusters_and_nothing_breaks():
    rows, report = dates.for_build(_process(n=15), {}, "key", "start")
    assert report.clusters == [] and report.unclustered
    # Every column is listed; none has the values to be placed, so there are no steps.
    assert {r["Parameter"] for r in rows} >= {"start", "seed"}
    assert report.order == [] and dates.step_rows(report) == []
    assert len(dates.batch_rows(report)) == 15 and plots.timing_order(report) is None
    _, empty = dates.for_build(pd.DataFrame({"key": ["a"], "x": [1.0]}), {}, "key", "start")
    assert empty.kinds == {} and plots.timing_profiles(empty) is None


def test_the_tables_and_figures_hold_every_batch():
    _, report = dates.for_build(_process(), {}, "key", "start")
    steps = dates.step_rows(report)
    assert [s["Step"] for s in steps][:2] == ["start → seed", "seed → check"]
    assert next(s for s in steps if s["Step"] == "seed → check")["Negative %"] == 8.9
    # Every step that varies is clustered on; a copy of a column is 0 h, every time.
    assert {s["Step"] for s in steps if not s["Clustered on"]} == {"harvest → harvest copy"}
    batches = dates.batch_rows(report)
    assert len(batches) == 90 and batches[0]["Cluster"] == "Timing 1"
    for draw in (
        plots.timing_profiles,
        plots.timing_calendar,
        plots.step_calendar,
        plots.timing_order,
        plots.in_progress,
    ):
        block = draw(report)
        assert block["kind"] == "figure" and block["spec"]["data"], draw.__name__
    # Clusters keep their colour in every figure: Timing 1 is always the first hue.
    profile = plots.timing_profiles(report)["spec"]["data"]
    first = next(t for t in profile if t.get("name", "").startswith("Timing 1"))
    assert first["line"]["color"] == plots.TIMING_COLOURS[0]


# ---------------------------------------------------------------------------
# The build and a new task
# ---------------------------------------------------------------------------
def test_the_build_writes_the_dates_report_and_table(built):
    folder = shared(built["directory"], built["config"], "batch_table").parent
    payload = stlite.read_payload((folder / "dates_report.html").read_text("utf-8"))
    assert payload["title"] == "Batch table dates" and payload["help"]["title"] == "Method"
    titles = [s["title"] for s in payload["sections"]]
    assert titles[:5] == [
        "What the dates say",
        "Timing clusters",
        "The order of the date columns",
        "Steps between the columns",
        "How the batches relate in time",
    ]
    table = pd.read_excel(folder / "dates.xlsx", sheet_name=screening.EVIDENCE_SHEET)
    assert set(table["Site"]) == {names.site("a")}
    assert names.col("date_start") in set(table["Parameter"])
    assert {"Steps", "Batches", screening.ABOUT_SHEET} <= set(
        pd.ExcelFile(folder / "dates.xlsx").sheet_names
    )
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
