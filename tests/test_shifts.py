"""The shifts report: parameters whose value runs in groups over time, and which change together."""

import numpy as np
import pandas as pd

from conftest import shared
from dummy_data import c
from ml import cli, names, screening, shifts, stlite, taskconfig


def _lots(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    # Three batches a day; a lot lasts ten batches, so it mostly changes during a day.
    day = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n) // 3, "D")
    lot = np.arange(n) // 10
    frame = pd.DataFrame(
        {
            "id": range(n),
            "date": day + pd.to_timedelta(rng.integers(0, 600, n), "min"),  # the time is ignored
            "lot": [f"L{k}" for k in lot],
            # The lot's certificate value and expiry date: they change with it.
            "titre": rng.normal(10, 3, lot.max() + 1).round(2)[lot],
            "expiry": pd.Timestamp("2025-01-01") + pd.to_timedelta(lot * 30, "D"),
            "other lot": np.arange(n) // 7,  # its own lots: they change on other days
            "process": np.where(np.arange(n) < 150, "old", "new"),  # changed once
            "operator": rng.choice(["X", "Y"], n),  # both, side by side, every day
            "rare flag": np.where(np.arange(n) % 37 == 5, "Yes", "No"),  # now and then
            "measured": rng.normal(size=n),
            "harvest": day + pd.Timedelta(days=9),  # one value a day
            "lot copy": lot,
        }
    )
    frame.loc[10:30, "titre"] = np.nan
    return frame


def test_what_runs_in_groups_is_found_and_grouped_by_the_days_it_changes():
    rows, report = shifts.for_build(_lots(), "id", "date")
    assert set(report.analysed) == {"lot", "titre", "expiry", "other lot", "process", "lot copy"}
    assert report.tested == 10
    (group,) = report.groups
    assert set(group.members) == {"lot", "titre", "expiry", "lot copy"} and group.weakest > 0.9
    by_name = {row["Parameter"]: row for row in rows}
    # A lot that changes during a day is one change: 24 lots, 24 groups.
    assert by_name["lot"]["Groups"] == 24 and by_name["lot"]["Shift group"] == group.name
    assert by_name["process"]["Groups"] == 2 and by_name["process"]["Shift group"] == ""
    assert by_name["other lot"]["Shift group"] == ""
    # Each group: first and last day, value, batches; the change day is in both.
    assert report.segments["lot"][:2] == [
        ["2024-01-01", "2024-01-04", "L0", 10],
        ["2024-01-04", "2024-01-07", "L1", 10],
    ]
    assert report.segments["expiry"][1][2] == "2025-01-31"


def test_a_change_during_a_day_is_one_change_and_values_side_by_side_are_many():
    # Days 0, 1, 1, 2: the lot changes on day 1, which has both.
    assert shifts.count_groups(np.array([0, 1, 1, 2]), np.array([0, 0, 1, 1])) == 2
    assert shifts.segments(np.array([0, 1, 1, 2]), np.array([0, 0, 1, 1])) == [
        [0, 0, 1, 2],
        [1, 1, 2, 2],
    ]
    # Two values every day: only one group carries on, the other starts again each day.
    side_by_side = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 0, 1])
    assert shifts.count_groups(*side_by_side) == 4 == len(shifts.segments(*side_by_side))
    # Shuffles are counted the same way, a row each.
    assert list(shifts.count_groups(np.array([0, 1, 2]), np.array([[0, 0, 1], [0, 1, 0]]))) == [
        2,
        3,
    ]


def test_a_change_counts_on_the_next_day_both_have_a_value():
    days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-11"))
    gap = np.delete(days, 4)  # no value on the day the first one changes
    assert shifts.shared_shifts(days, days[[4]], gap, days[[5]]) == 1.0
    assert shifts.shared_shifts(days, days[[4]], gap, days[[7]]) == 0.0


def test_the_shifts_report_has_no_part_in_the_screening_register():
    assert "shifts with group" not in screening.REASONS
    assert all("shift" not in c.lower() for c in screening.PROPOSAL_COLUMNS)


def test_the_build_writes_the_shifts_report_and_table_and_finds_the_demo_lots(built):
    page = shared(built["directory"], built["config"], "batch_table", "shifts_report.html")
    payload = stlite.read_payload(page.read_text(encoding="utf-8"))
    viewer = payload["sections"][0]["blocks"][0]
    assert viewer["kind"] == "shift_viewer" and viewer["site_label"] == "Manufacturing site"
    # Site A's own batches: its plant whole, and per production line, each picked by level.
    plant = f"{names.site('a')} plant"
    line = names.optional_col("production_line")
    assert [(s["label"], s["levels"]) for s in viewer["sites"]] == [
        (plant, {"Manufacturing site": plant, line: "All"}),
        (f"{plant} · Line A", {"Manufacturing site": plant, line: "Line A"}),
        (f"{plant} · Line B", {"Manufacturing site": plant, line: "Line B"}),
    ]
    assert payload["site"] == names.site("a")
    # Site A records no work shift: the report says so.
    assert any("No shift split" in note for note in payload["subsite_notes"])
    for site in viewer["sites"]:
        if site["label"].endswith("Line B"):
            assert site["rows"] == [] and site["groups"] == {}, "too few batches"
            continue
        # The medium and cytokine lots change on the same days.
        assert {c("media_lot"), c("cytokine_lot")} in [set(g) for g in site["groups"].values()]
        grouped = {m for members in site["groups"].values() for m in members}
        for row in site["rows"]:
            assert site["segments"][row[0]], "every parameter in the table has its groups"
            assert row[0] not in grouped, "the parameter table leaves out the grouped ones"
    table = pd.read_excel(
        shared(built["directory"], built["config"], "batch_table", "shifts.xlsx"),
        sheet_name=screening.EVIDENCE_SHEET,
    )
    lots = table[table["Parameter"] == c("media_lot")]
    assert set(lots["Manufacturing site"]) == {plant, f"{plant} · Line A"}
    assert lots["Shift group"].str.startswith("Shifting").all()
    path = shared(built["directory"], built["config"], "batch_table", screening.EVIDENCE_FILE)
    evidence = pd.read_excel(path, sheet_name=screening.EVIDENCE_SHEET)
    assert not any("shift" in column.lower() for column in evidence.columns)


def test_a_new_task_can_leave_out_every_parameter_that_runs_in_groups(built, tmp_path):
    folder = tmp_path / "no_lots"
    config = str(built["config_path"])
    assert cli.main(["--config", config, "task", "new", str(folder), "--exclude-shifting"]) == 0
    text = (folder / "task.yaml").read_text(encoding="utf-8")
    assert "Run in groups over time at a manufacturing site, as the build of" in text
    excluded = dict(taskconfig.load(folder / "task.yaml").predictors.exclude)
    # The medium lot is listed under its catalog role and again for running in groups: the
    # task carries both reasons.
    why = excluded[c("media_lot")]
    assert why.startswith("catalog Variable Role is 'Material lot ID'; runs in groups over time: ")
    assert why.count("changes with Shifting") == 2, "the plant whole, and its line A"
    shifting = text.split("Run in groups over time at a manufacturing site")[1]
    assert f'column: "{c("media_lot")}"' in shifting
    assert f"at {names.site('a')} plant · Line A" in why and names.site("b") not in why
    # Without the option, nothing that runs in groups is listed.
    plain = tmp_path / "plain"
    assert cli.main(["--config", config, "task", "new", str(plain)]) == 0
    reasons = dict(taskconfig.load(plain / "task.yaml").predictors.exclude).values()
    assert not any(why.startswith("runs in groups over time") for why in reasons)
    assert "Run in groups over time" not in (plain / "task.yaml").read_text(encoding="utf-8")


def test_the_shifts_run_per_manufacturing_site_and_production_line():
    plant = names.col("manufacturing_site")
    line = names.optional_col("production_line")
    frame = _lots().rename(columns={"id": "Patient Lot/Batch #"})
    north, south = frame.iloc[:120].copy(), frame.iloc[120:].copy()
    north[plant] = ["P1"] * 60 + ["P2"] * 60
    south[plant] = "P2"
    cohorts = {"North": (north, None), "South": (south, None)}
    subsites, _ = cli._subsites(cohorts)
    found = cli._shifts_by_subsite(subsites, "date")
    assert set(found) == {"P1", "P2"}
    p1, p2 = found["P1"][1], found["P2"][1]
    # P2 makes batches for both sources.
    assert (p1.sources, p2.sources) == (["North"], ["North", "South"])
    assert p2.cohort.startswith("180 batches")
    assert p2.levels == {"Manufacturing site": "P2"}
    # The register has no part in it: every parameter takes part.
    assert "lot" in p1.analysed and "lot" in p2.analysed
    assert all(g.name.endswith(" @ P2") for g in p2.groups)
    # A production line splits its plant further; every plant is also analysed whole, and
    # a batch without a line only there.
    north[line] = ["L1"] * 30 + [None] * 30 + ["L2"] * 60
    subsites, _ = cli._subsites({"North": (north, None), "South": (south, None)})
    assert [(name, levels[line], len(batches)) for name, levels, batches, _, _ in subsites] == [
        ("P1", "All", 60),
        ("P1 · L1", "L1", 30),
        ("P2", "All", 180),
        ("P2 · L2", "L2", 60),
    ]
    # Without the plant column (nor the line), each data source is one plant.
    subsites, _ = cli._subsites(
        {s: (b.drop(columns=[plant, line], errors="ignore"), a) for s, (b, a) in cohorts.items()}
    )
    assert {name for name, *_ in subsites} == {"North", "South"}


def test_subsites_split_by_line_then_shift_and_say_what_they_could_not_use(caplog):
    plant = names.col("manufacturing_site")
    line = names.optional_col("production_line")
    shift = names.optional_col("work_shift")
    frame = _lots().rename(columns={"id": "Patient Lot/Batch #"})
    frame[plant] = "P1"
    frame[line] = ["A"] * 100 + [""] * 140  # a blank is only in its level's All
    frame[shift] = ["Day", "Night"] * 120
    subsites, notes = cli._subsites({"North": (frame, None)})
    # Every combination, "All" first: the plant whole, per shift, per line, per line and shift.
    assert [(name, len(batches)) for name, _, batches, _, _ in subsites] == [
        ("P1", 240),
        ("P1 · Day", 120),
        ("P1 · Night", 120),
        ("P1 · A", 100),
        ("P1 · A · Day", 50),
        ("P1 · A · Night", 50),
    ]
    assert subsites[4][1] == {"Manufacturing site": "P1", line: "A", shift: "Day"}
    assert notes == [] and "6 subsites for the shifts and dates reports" in caplog.text
    # A split column the batch table lacks, or that is blank everywhere, is said and warned.
    caplog.clear()
    subsites, notes = cli._subsites(
        {"North": (frame.drop(columns=shift).assign(**{line: None}), None)}
    )
    assert [name for name, *_ in subsites] == ["P1"]
    assert notes == [
        f"No line split: '{line}' is blank in every batch described.",
        f"No shift split: the batch table has no '{shift}' column (vocabulary key work_shift); "
        "check the name against the batch records.",
    ]
    assert all(note in caplog.text for note in notes)


def test_the_demo_sites_split_into_their_subsites(built):
    for config, plant, split, values in (
        ("config", f"{names.site('a')} plant", "production_line", ["All", "Line A", "Line B"]),
        ("config_b", f"{names.site('b')} plant", "work_shift", ["All", "Day", "Night"]),
    ):
        page = shared(built["directory"], built[config], "batch_table", "dates_report.html")
        payload = stlite.read_payload(page.read_text(encoding="utf-8"))
        (chooser,) = payload["sections"][0]["blocks"]
        levels = [option["levels"] for option in chooser["options"]]
        column = names.optional_col(split)
        assert [level[column] for level in levels] == values
        assert {level["Manufacturing site"] for level in levels} == {plant}
