"""The identifiers report: parameters that name a batch rather than describe it.

They are found per site, shown first, and screened out through their own track of the
screening register, which every task then applies.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import shared
from ml import cli, identifiers, names, screening, stlite

SITES = ("North", "South")


def _batches(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    day = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), "D")
    return pd.DataFrame(
        {
            "id": [f"B{i:03d}" for i in range(n)],
            "date": day,
            "patient": [f"PX{1200 + i % 3:04d}-{i:03d}" for i in range(n)],
            # Two batches per case: half as many values as batches.
            "case": [f"INV-{i // 2:03d}" for i in range(n)],
            "comment": [f"batch {i} was held by operator {rng.integers(9)}" for i in range(n)],
            "running": np.arange(1000, 1000 + n),
            "running as text": [str(5000 - i) for i in range(n)],
            # Whole numbers, nearly all distinct, but not in date order: a count.
            "count": rng.permutation(np.arange(n)) * 1000 + 7,
            "measured": rng.normal(size=n),
            "lot": [f"L{i // 10}" for i in range(n)],
            "operator": rng.choice(["X", "Y", "Z"], n),
            "site": "North",
            "harvest": (day + pd.Timedelta(days=9)).astype(str),
            "value as text": [f"{v:.2f}" for v in rng.normal(10, 2, n)],
            "few": [None] * (n - 5) + ["a"] * 5,
            "flag": rng.choice([True, False], n),
        }
    )


def test_codes_running_numbers_and_free_text_are_found_and_nothing_else():
    rows, report = identifiers.for_build(_batches(), {"case": "categorical"}, "id", "date")
    kinds = {row["Parameter"]: row["Kind"] for row in rows}
    assert kinds == {
        "patient": identifiers.UNIQUE_CODE,
        "case": identifiers.NEAR_UNIQUE_CODE,
        "comment": identifiers.FREE_TEXT,
        "running": identifiers.RUNNING_NUMBER,
        "running as text": identifiers.RUNNING_NUMBER,
    }
    # Listed by kind, in the order the report explains them.
    assert [row["Kind"] for row in rows] == sorted(
        kinds.values(), key=list(identifiers.KINDS).index
    )
    assert report.not_analysed == {"harvest": "a date", "few": "fewer than 20 values"}
    # The key and the date take no part; every other column is judged.
    assert report.tested == _batches().shape[1] - 2 - len(report.not_analysed)
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["running as text"]["|ρ| with the date"] == 1.0
    assert by_name["patient"]["Distinct %"] == 100.0
    assert by_name["case"]["Most batches with one value"] == 2
    assert by_name["patient"]["Examples"].startswith("PX1200-000 · PX1201-001")
    # The viewer lists each one's value per batch, in date order.
    assert report.values["running"][:2] == ["1000", "1001"]
    assert len(report.values["running"]) == len(report.batches) == len(report.dates)
    assert identifiers.evidence(by_name["running"]) == (
        "running number, 120 of 120 distinct, |ρ| 1.00 with the date"
    )


def test_the_thresholds_decide_what_a_code_is():
    frame = _batches()
    rows, _ = identifiers.for_build(frame, {}, "id", "date", unique=0.95, near_unique=0.6)
    assert "case" not in {row["Parameter"] for row in rows}
    rows, _ = identifiers.for_build(frame, {}, "id", "date", unique=0.4, near_unique=0.3)
    assert {row["Parameter"]: row["Kind"] for row in rows}["case"] == identifiers.UNIQUE_CODE


def test_only_the_batches_analysed_are_judged():
    frame = _batches()
    # Ten batches are too few to judge any column by.
    analysed = pd.Series(np.arange(len(frame)) < 10, index=frame.index)
    rows, report = identifiers.for_build(frame, {}, "id", "date", analysed=analysed)
    assert rows == [] and report.tested == 0 and len(report.batches) == 10


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
def test_the_build_writes_the_identifiers_report_first_and_cites_it(built):
    folder = shared(built["directory"], built["config"], "batch_table").parent
    page = folder / "identifiers_report.html"
    payload = stlite.read_payload(page.read_text(encoding="utf-8"))
    assert payload["title"] == "Batch table identifiers"
    viewer = next(b for b in payload["sections"][0]["blocks"] if b["kind"] == "id_viewer")
    assert viewer["proposal"]["report"] == "identifiers"
    assert viewer["proposal"]["reasons"] == ["identifier", "free text"]
    (site,) = viewer["sites"]
    assert site["label"] == names.site("a")
    rows = [dict(zip(site["headers"], row)) for row in site["rows"]]
    patient = names.col("patient_id")
    assert {row["Parameter"]: row["Kind"] for row in rows} == {patient: "unique code"}
    # The batch key and the date order the batches; they are not listed.
    listed = {row["Parameter"] for row in rows} | {p for p, _ in site["not_analysed"]}
    assert "Patient Lot/Batch #" not in listed
    assert len(site["values"][patient]) == len(site["batches"])
    # The evidence `ml screening add` cites says what it was found to be.
    evidence = pd.read_excel(folder / screening.EVIDENCE_FILE, sheet_name=screening.EVIDENCE_SHEET)
    cited = evidence.set_index("Parameter")[screening.IDENTIFIER].fillna("")
    assert cited[patient].startswith("unique code, ")
    assert (cited != "").sum() == 1
    # Built before the missingness report.
    order = list(cli.BUILD_OUTPUTS)
    assert order.index("identifiers report") < order.index("missingness report")


def test_a_site_with_nothing_found_still_says_so(built):
    folder = shared(built["directory"], built["config_b"], "batch_table").parent
    payload = stlite.read_payload((folder / "identifiers_report.html").read_text("utf-8"))
    viewer = next(b for b in payload["sections"][0]["blocks"] if b["kind"] == "id_viewer")
    (site,) = viewer["sites"]
    assert site["rows"] == [] and site["label"] == names.site("b")
    assert f"{names.site('b')} 0 of " in payload["lead"]
