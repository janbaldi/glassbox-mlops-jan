"""The identifiers report: parameters that name something rather than describe the batch.

They are found per site, by general signs, and shown first. A new task leaves the clear
ones out; the possible ones are screened out through their own track of the register.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import shared
from ml import cli, identifiers, names, screening, stlite, taskconfig

SITES = ("North", "South")


def _batches(n: int = 120) -> pd.DataFrame:
    """One column per case, under names no real batch table has: the rules are general."""
    rng = np.random.default_rng(0)
    day = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), "D")
    return pd.DataFrame(
        {
            "key": [f"B{i:03d}" for i in range(n)],
            "date": day,
            # A different value per batch, or nearly.
            "subject": [f"PX{1200 + i % 3:04d}-{i:03d}" for i in range(n)],
            "case": [f"INV-{i // 2:03d}" for i in range(n)],  # half as many values as batches
            "comment": [f"batch {i} was held by operator {rng.integers(9)}" for i in range(n)],
            "running": np.arange(1000, 1000 + n),
            "running as text": [str(5000 - i) for i in range(n)],
            # Shared by many batches, but its number counts with time, under any name.
            "buffer": [f"RX-{9000 + i // 15}" for i in range(n)],
            "buffer reference": [f"RX-{9000 + i // 15}" for i in range(n)],
            # Code-shaped and named as one, in no order.
            "filter lot": rng.choice(["F-77", "F-12", "F-40", "F-03"], n),
            # One sign each: code-shaped only, named only, a whole number named as one.
            "vessel": rng.choice([f"BR{k}" for k in range(1, 7)], n),
            "centre code": rng.choice(["ALPHA", "BETA", "GAMMA"], n),
            "kit serial": rng.integers(100000, 100010, n),
            # Joined from a source of identifiers, whatever it looks like.
            "reagent": rng.choice(["kit A", "kit B"], n),
            # Not identifiers.
            "vessels number": rng.integers(1, 4, n),  # "number of …" counts things
            "colour": rng.choice(["red", "green", "blue"], n),
            "score": rng.choice(["1", "2", "3"], n),
            "count": rng.permutation(np.arange(n)) * 1000 + 7,  # distinct, in no order
            "measured": rng.normal(size=n),
            "value as text": [f"{v:.2f}" for v in rng.normal(10, 2, n)],
            "site": "North",
            "flag": rng.choice([True, False], n),
            # Not judged.
            "harvest": (day + pd.Timedelta(days=9)).astype(str),
            "month": [d.strftime("%b-%Y") for d in day],
            "few": [None] * (n - 5) + ["a"] * 5,
        }
    )


def _found(frame=None, **extra):
    rows, report = identifiers.for_build(
        _batches() if frame is None else frame,
        {"case": "categorical", "score": "categorical"},
        "key",
        "date",
        sources={"reagent": "Raw materials and consumables"},
        **extra,
    )
    return {row["Parameter"]: row for row in rows}, report


def test_every_kind_of_identifier_is_found_by_its_signs_and_nothing_else():
    found, report = _found()
    assert {p: (r["Status"], r["Kind"]) for p, r in found.items()} == {
        "subject": ("identifier", "unique code"),
        "case": ("identifier", "near-unique code"),
        "running": ("identifier", "running number"),
        "running as text": ("identifier", "running number"),
        "buffer": ("identifier", "sequential code"),
        "buffer reference": ("identifier", "sequential code"),
        "filter lot": ("identifier", "lot or reference code"),
        "reagent": ("identifier", "lot or reference code"),
        "comment": ("identifier", "free text"),
        "vessel": ("possible identifier", "possible identifier"),
        "centre code": ("possible identifier", "possible identifier"),
        "kit serial": ("possible identifier", "possible identifier"),
    }
    # The signs say why, in words.
    assert found["buffer"]["Signs"] == (
        "code-shaped (A-9); its number counts with the date (|ρ| 0.99)"
    )
    assert found["filter lot"]["Signs"] == "code-shaped (A-9); named as one ('lot')"
    assert found["reagent"]["Signs"] == "from Raw materials and consumables"
    assert found["vessel"]["Signs"] == "code-shaped (A9)"
    assert found["centre code"]["Signs"] == "named as one ('code')"
    assert found["kit serial"]["Signs"] == "named as one ('serial'), whole numbers"
    # Clear identifiers first, by kind; the possible ones last.
    statuses = [r["Status"] for r in found.values()]
    assert statuses == sorted(statuses, key=lambda s: s == identifiers.POSSIBLE)
    assert report.not_analysed == {
        "harvest": "a date",
        "month": "a date",
        "few": "fewer than 20 values",
    }
    # The key and the date take no part; every other column is judged.
    assert report.tested == _batches().shape[1] - 2 - len(report.not_analysed)
    assert found["running as text"]["|ρ| with the date"] == 1.0
    assert found["case"]["Most batches with one value"] == 2
    assert found["subject"]["Examples"].startswith("PX1200-000 · PX1201-001")
    # The viewer lists each one's value per batch, in date order.
    assert report.values["running"][:2] == ["1000", "1001"]
    assert len(report.values["running"]) == len(report.batches) == len(report.dates)
    assert identifiers.evidence(found["filter lot"]) == (
        "lot or reference code: code-shaped (A-9); named as one ('lot')"
    )


def test_the_signs_are_general():
    assert identifiers.shape("XY-104") == identifiers.shape("QR-7") == "A-9"
    assert identifiers.named_as_identifier("Kit Lot #") == "#"
    assert identifiers.named_as_identifier("Batch_ID") in ("batch", "id")
    assert identifiers.named_as_identifier("Number of cycles", numeric=True) == ""
    assert identifiers.named_as_identifier("Catalogue number") == "number"
    codes = pd.Series(["AB-1", "AB-22", "CD-3"] * 10)
    assert identifiers.code_shaped(codes)
    assert not identifiers.code_shaped(pd.Series(["low", "high"] * 10)), "no digits"
    assert not identifiers.code_shaped(pd.Series(["CD4 low", "CD4 high"] * 10)), "not one token"
    assert not identifiers.code_shaped(pd.Series(["1", "22", "3"] * 10)), "numbers"
    assert identifiers.code_shaped(pd.Series(["20431", "10022"] * 10), categorical=True)
    # Two signs make an identifier, however the signs are typed.
    frame = pd.DataFrame({"k": range(30), "site code": ["20431", "10022", "30017"] * 10})
    rows, _ = identifiers.for_build(frame, {"site code": "categorical"}, "k", "")
    assert [(r["Status"], r["Signs"]) for r in rows] == [
        ("identifier", "code-shaped (9); named as one ('code')")
    ]
    # A short code is not read as a date.
    rows, report = identifiers.for_build(
        pd.DataFrame({"k": range(30), "m": ["M3", "T 12", "M4"] * 10}), {}, "k", ""
    )
    assert "m" not in report.not_analysed


def test_the_thresholds_decide_what_a_code_is():
    found, _ = _found(near_unique=0.6)
    assert "case" not in found or found["case"]["Kind"] != "near-unique code"
    found, _ = _found(unique=0.4, near_unique=0.3)
    assert found["case"]["Kind"] == identifiers.UNIQUE_CODE


def test_only_the_batches_analysed_are_judged():
    frame = _batches()
    # Ten batches are too few to judge any column by.
    analysed = pd.Series(np.arange(len(frame)) < 10, index=frame.index)
    found, report = _found(frame, analysed=analysed)
    assert found == {} and report.tested == 0 and len(report.batches) == 10


# ---------------------------------------------------------------------------
# The register's identifiers track
# ---------------------------------------------------------------------------
def _decision(number: int, action: str, parameter: str, **extra) -> screening.Decision:
    fields = dict.fromkeys(f.name for f in screening.Decision.__dataclass_fields__.values())
    fields = {k: "" for k in fields} | {
        "id": f"S-{number:04d}",
        "date": "2026-01-01",
        "author": "A. Reviewer",
        "action": action,
        "parameter": parameter,
        "site": "North",
        "reason": screening.IDENTIFIED,
    }
    return screening.Decision(**(fields | extra))


def test_the_identifiers_report_keeps_its_own_track_in_the_register():
    assert screening.TRACKS[0] == screening.IDENTIFIERS, "read first"
    assert screening.track_of("identifier") == screening.track_of("free text") == "identifiers"
    assert screening.track_of("retired") == "missingness"
    assert screening.track_of(screening.CORRELATED) == "correlation"
    log = [
        _decision(1, "exclude", "X"),
        _decision(2, "exclude", "X", reason="retired"),
        # Reinstated in the identifiers report: the missingness exclusion still stands.
        _decision(3, "reinstate", "X", report="identifiers"),
    ]
    assert [d.track for d in screening.excluded(log[:2], SITES)["X"]] == [
        "identifiers",
        "missingness",
    ]
    assert [d.id for d in screening.excluded(log, SITES)["X"]] == ["S-0002"]
    own, other = screening.at_site(log[:2], SITES, "North", "identifiers")
    assert own["X"]["Decision"] == "S-0001" and other["X"]["Report"] == "missingness"
    frame = screening.to_frame(
        [_decision(1, "exclude", "X", reason="retired", report="identifiers")]
    )
    with pytest.raises(screening.ScreeningError, match="not decided in the identifiers report"):
        screening.from_frame(frame, SITES)


def test_a_proposal_from_the_identifiers_report_records_what_it_was_found_to_be():
    evidence = pd.DataFrame(
        [
            ["North", "Patient", 0.0, "Complete", "2026-09-01", "", 100.0, "unique code, 9 of 9"],
            ["North", "Titre", 0.0, "Complete", "2026-09-01", "", 100.0, ""],
        ],
        columns=[
            "Site",
            "Parameter",
            "Missing %",
            "Pattern",
            "Last measured",
            "Group",
            screening.RECENT,
            screening.IDENTIFIER,
        ],
    )
    proposal = pd.DataFrame(
        [
            {c: "" for c in screening.PROPOSAL_COLUMNS}
            | {
                "Action": "exclude",
                "Parameter": "Patient",
                "Site": "North",
                "Reason": "identifier",
                "Report": "identifiers",
            }
        ]
    )
    (added,) = screening.add([], proposal, evidence, "digest", "A. Reviewer", SITES)
    assert added.track == "identifiers" and added.identifier == "unique code, 9 of 9"
    # The register keeps it, and an old register without the column still reads.
    frame = screening.to_frame([added])
    assert frame.loc[0, screening.IDENTIFIER] == "unique code, 9 of 9"
    assert screening.from_frame(frame, SITES) == [added]
    old = frame.drop(columns=screening.IDENTIFIER)
    assert screening.from_frame(old, SITES)[0].identifier == ""
    # A decision without it keeps the digest a register written before the column had.
    plain = _decision(1, "exclude", "Y", reason="retired")
    assert screening.digest([plain]) == screening.digest(
        screening.from_frame(screening.to_frame([plain]).drop(columns=screening.IDENTIFIER), SITES)
    )
    # The next build no longer finds it: worth a look.
    evidence.loc[0, screening.IDENTIFIER] = ""
    (finding,) = screening.review([added], evidence, SITES, today=pd.Timestamp("2026-02-01"))
    assert "the identifiers report no longer finds it" in finding["Why look again"]


def test_every_screening_report_shows_the_others_decisions():
    log = [
        _decision(1, "exclude", "Patient"),
        _decision(2, "exclude", "Old", reason="retired"),
        _decision(3, "exclude", "Twin", reason=screening.CORRELATED, represented_by="X"),
    ]

    def results(*parameters):
        rows = [{"Parameter": p} for p in parameters]
        return {"North": (rows, SimpleNamespace(screened={}, elsewhere={}, conflicts=[]))}

    found, missing, ranked = results("Patient"), results("Old", "Patient"), results("Twin")
    cli._show_register(log, "North", missing, ranked, found)
    here = found["North"][1]
    assert set(here.screened) == {"Patient"} and set(here.elsewhere) == {"Old", "Twin"}
    assert set(missing["North"][1].elsewhere) == {"Patient", "Twin"}
    assert set(ranked["North"][1].elsewhere) == {"Patient", "Old"}
    assert found["North"][0][0]["Register"] == "excluded S-0001 (identifiers: identifier)"


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------
def _table(built, config="config"):
    path = shared(built["directory"], built[config], "batch_table", "identifiers.xlsx")
    return pd.read_excel(path, sheet_name=screening.EVIDENCE_SHEET).fillna("")


def test_the_build_writes_the_identifiers_report_first_and_cites_it(built):
    folder = shared(built["directory"], built["config"], "batch_table").parent
    payload = stlite.read_payload((folder / "identifiers_report.html").read_text("utf-8"))
    assert payload["title"] == "Batch table identifiers"
    viewer = next(b for b in payload["sections"][0]["blocks"] if b["kind"] == "id_viewer")
    assert viewer["proposal"]["report"] == "identifiers"
    assert viewer["proposal"]["reasons"] == ["identifier", "free text"]
    (site,) = viewer["sites"]
    assert site["label"] == names.site("a")
    clear = {row[0] for row in site["rows"]}
    possible = {row[0] for row in site["possible_rows"]}
    table = _table(built)
    # The page and the table hold the same, split by status.
    assert clear == set(table.loc[table["Status"] == "identifier", "Parameter"])
    assert possible == set(table.loc[table["Status"] == "possible identifier", "Parameter"])
    assert names.col("patient_id") in clear
    # Every column the raw materials export brought is an identifier.
    raw = table[table["Signs"].str.contains("from Raw materials and consumables")]
    assert set(raw["Status"]) <= {"identifier"}
    # The batch key orders nothing and names every batch: not listed, nor judged.
    listed = clear | possible | {p for p, _ in site["not_analysed"]}
    assert "Patient Lot/Batch #" not in listed
    for parameter in clear | possible:
        assert len(site["values"][parameter]) == len(site["batches"])
    # The evidence `ml screening add` cites says what each was found to be.
    evidence = pd.read_excel(folder / screening.EVIDENCE_FILE, sheet_name=screening.EVIDENCE_SHEET)
    cited = evidence.set_index("Parameter")[screening.IDENTIFIER].fillna("")
    assert set(cited[cited != ""].index) == clear | possible
    assert cited[names.col("patient_id")].startswith("unique code: ")
    # Built before the missingness report.
    order = list(cli.BUILD_OUTPUTS)
    assert order.index("identifiers report") < order.index("missingness report")


def test_a_new_task_leaves_out_the_identifiers_but_not_the_possible_ones(built, tmp_path):
    config = str(built["config_path"])
    folder = tmp_path / "plain"
    assert cli.main(["--config", config, "task", "new", str(folder)]) == 0
    text = (folder / "task.yaml").read_text(encoding="utf-8")
    assert "Identifiers at this site, as the build of" in text
    excluded = dict(taskconfig.load(folder / "task.yaml").predictors.exclude)
    table = _table(built)
    clear = set(table.loc[table["Status"] == "identifier", "Parameter"])
    possible = set(table.loc[table["Status"] == "possible identifier", "Parameter"])
    screened = {p for p, why in excluded.items() if why.startswith("screened out by")}
    assert clear - screened <= set(excluded)
    assert all(excluded[p].startswith("identifier (") for p in clear - screened)
    assert not (possible - screened) & set(excluded), "the possible ones are for a person"
    # --keep-identifiers keeps them all.
    kept = tmp_path / "kept"
    assert cli.main(["--config", config, "task", "new", str(kept), "--keep-identifiers"]) == 0
    reasons = dict(taskconfig.load(kept / "task.yaml").predictors.exclude).values()
    assert not any(why.startswith("identifier (") for why in reasons)


def test_a_reinstated_identifier_stays_in_new_tasks(built, tmp_path, monkeypatch):
    table = _table(built)
    patient = names.col("patient_id")
    reinstated = [
        _decision(1, "exclude", patient, site=names.site("a")),
        _decision(2, "reinstate", patient, site=names.site("a")),
    ]
    monkeypatch.setattr(
        cli, "_register", lambda config: screening.Register(reinstated, "the register")
    )
    found, _ = cli._identifiers_for_new_task(built["config"])
    left_out = {p for p, _ in found}
    assert patient not in left_out
    assert left_out == set(table.loc[table["Status"] == "identifier", "Parameter"]) - {patient}


def test_a_site_with_nothing_clear_still_says_so(built):
    folder = shared(built["directory"], built["config_b"], "batch_table").parent
    payload = stlite.read_payload((folder / "identifiers_report.html").read_text("utf-8"))
    viewer = next(b for b in payload["sections"][0]["blocks"] if b["kind"] == "id_viewer")
    (site,) = viewer["sites"]
    assert site["label"] == names.site("b")
    assert f"{names.site('b')}: " in payload["lead"] and " identifiers and " in payload["lead"]
