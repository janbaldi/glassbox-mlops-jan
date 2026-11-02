"""Parameters that name a batch rather than describe it: identifiers and free text.

A patient code, a case number, a running batch number or a comment written for
one batch has a different value in nearly every batch. A model can learn which
batch is which from it, never what made a batch turn out as it did: it brings
no value to a task, and only makes a fit look better than it is. This module
finds them per site, over the batches the missingness report counts in its
statistics, so they can be screened out (the ``identifiers`` track of the
screening register) before anything else is read.

1. Every column takes part but the batch key, the date the batches are ordered
   by, and the dates (catalog type ``datetime``, or values that are dates). A
   column needs :data:`MIN_VALUES` batches with a value to be judged.
2. *Distinct %* is its distinct values over its batches with a value.
3. A column whose values are words or codes (not numbers, or typed categorical
   in the catalog) is
   - **free text** when its values are sentences (a median of at least
     :data:`TEXT_WORDS` words) and at least ``near_unique`` of them are distinct;
   - a **unique code** when at least ``unique`` of its values are distinct;
   - a **near-unique code** when at least ``near_unique`` are: as many categories
     as half the batches is more than any encoder can learn from, and a task
     would drop it (``max_cardinality_ratio``).
4. A numerical column is a **running number** when its values are whole numbers,
   at least ``unique`` of them distinct, and they rise or fall with the date
   (Spearman |ρ| of at least :data:`SEQUENCE_RHO`): a counter, which tells the
   model when a batch was made and nothing else.

A lot number shared by a run of batches is not an identifier here: it carries
what the lot was, and the shifts report looks at it. A column with one value
(constant) is not one either: it is not reported.

This is description only. Nothing here changes the batch table, and nothing is
written to the register: ``ml screening add`` does that, from a proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logger import log

MODULE = "identifiers"

#: Values a column needs before it is judged.
MIN_VALUES = 20
#: The share of distinct values that makes a code unique to a batch (``build.identifier_unique``).
UNIQUE = 0.95
#: The share that makes it near-unique, as a task's ``max_cardinality_ratio`` does
#: (``build.identifier_near_unique``).
NEAR_UNIQUE = 0.5
#: The median number of words that makes a value a sentence rather than a code.
TEXT_WORDS = 3
#: The |ρ| with the date that makes a whole number a running number.
SEQUENCE_RHO = 0.95
#: Values shown per parameter in the table, and the characters kept of each.
EXAMPLES, EXAMPLE_LENGTH = 3, 40
#: The characters of each value the viewer keeps.
VALUE_LENGTH = 120

#: What a column can be found to be, and the screening reason each is excluded with.
UNIQUE_CODE, NEAR_UNIQUE_CODE = "unique code", "near-unique code"
FREE_TEXT, RUNNING_NUMBER = "free text", "running number"
KINDS = {
    UNIQUE_CODE: "A code with a different value in (nearly) every batch: a patient code, a "
    "case number, a sample id. It names the batch.",
    NEAR_UNIQUE_CODE: "A code most batches have to themselves: more categories than half the "
    "batches, too many for any encoder to learn from. A task drops it anyway.",
    FREE_TEXT: "Sentences written for one batch: a comment, a description. No two batches "
    "share one, so a model can only memorise them.",
    RUNNING_NUMBER: "A whole number that counts up (or down) with the date: a running batch "
    "or order number. It tells a model when a batch was made, nothing else.",
}
#: The reason a proposal gives each kind in the screening register.
REASON_OF = {
    UNIQUE_CODE: "identifier",
    NEAR_UNIQUE_CODE: "identifier",
    RUNNING_NUMBER: "identifier",
    FREE_TEXT: "free text",
}

_CODE_TYPES = ("categorical", "boolean")


@dataclass
class IdentifierReport:
    #: Columns judged (enough values, not the key or a date), and those found.
    tested: int = 0
    analysed: list[str] = field(default_factory=list)
    #: The thresholds the kinds were decided with.
    unique: float = UNIQUE
    near_unique: float = NEAR_UNIQUE
    #: The batches judged, in date order, and each found column's value per batch
    #: (None where it has none): what the viewer lists when a row is clicked.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    values: dict[str, list[str | None]] = field(default_factory=dict)
    #: Columns not judged, and why.
    not_analysed: dict[str, str] = field(default_factory=dict)
    cohort: str = ""
    #: Set by the build: parameter → its standing exclusion at this site, made in this
    #: report (``screened``) or in another (``elsewhere``), and the representatives here
    #: that are excluded themselves.
    screened: dict[str, dict] = field(default_factory=dict)
    elsewhere: dict[str, dict] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)


def _text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, pd.Timestamp):
        return f"{value:%Y-%m-%d %H:%M}".removesuffix(" 00:00")
    return str(value).strip()


def _short(text: str, length: int) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"


def _mostly(parsed: pd.Series, present: int) -> bool:
    """Whether nine values in ten parsed."""
    return present > 0 and parsed.notna().sum() >= 0.9 * present


def _is_date(values: pd.Series, value_type: str) -> bool:
    if value_type == "datetime" or pd.api.types.is_datetime64_any_dtype(values):
        return True
    if pd.api.types.is_numeric_dtype(values):
        return False
    present = values.dropna()
    if present.map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64))).all():
        return len(present) > 0
    text = present.astype(str)
    # Numbers kept as text are numbers, however a date parser reads them ("12.5" is 5 December).
    if _mostly(pd.to_numeric(text, errors="coerce"), len(text)) or text.str.fullmatch(r"\d+").any():
        return False
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    return _mostly(parsed, len(text))


def kind_of(
    values: pd.Series,
    value_type: str,
    dates: pd.Series | None,
    unique: float = UNIQUE,
    near_unique: float = NEAR_UNIQUE,
) -> tuple[str, dict]:
    """What one column is, ``""`` when it is none of :data:`KINDS`, and the numbers that say so.

    ``values`` are the column over the batches judged, ``dates`` their dates (None
    when there are none).
    """
    present = values.dropna()
    n = len(present)
    if n == 0:
        return "", {}
    if pd.api.types.is_bool_dtype(values):
        return "", {}
    labels = present.map(_text)
    distinct = int(labels.nunique())
    ratio = distinct / n
    facts = {
        "Batches with a value": n,
        "Distinct values": distinct,
        "Distinct %": round(100 * ratio, 1),
        "Most batches with one value": int(labels.value_counts().iloc[0]),
    }
    numeric = pd.api.types.is_numeric_dtype(values)
    if not numeric and value_type not in _CODE_TYPES:
        # Numbers kept as text are numbers.
        parsed = pd.to_numeric(labels, errors="coerce")
        if _mostly(parsed, n):
            present, numeric = parsed.dropna(), True
    if numeric and value_type not in _CODE_TYPES:
        whole = bool(np.all(np.isclose(present, np.round(present))))
        if not whole or ratio < unique or dates is None:
            return "", facts
        both = pd.DataFrame({"value": present, "date": dates.reindex(present.index)}).dropna()
        if len(both) < MIN_VALUES or both["date"].nunique() < 2:
            return "", facts
        rho = both["value"].rank().corr(both["date"].rank())
        if pd.isna(rho) or abs(rho) < SEQUENCE_RHO:
            return "", facts
        return RUNNING_NUMBER, facts | {"|ρ| with the date": round(abs(float(rho)), 3)}
    words = float(labels.str.split().str.len().median())
    facts["Median words"] = words
    if words >= TEXT_WORDS and ratio >= near_unique:
        return FREE_TEXT, facts
    if ratio >= unique:
        return UNIQUE_CODE, facts
    if ratio >= near_unique:
        return NEAR_UNIQUE_CODE, facts
    return "", facts


def evidence(row: dict) -> str:
    """One found column, as the screening register cites it: ``unique code, 312 of 312
    distinct``."""
    rho = row.get("|ρ| with the date")
    return f"{row['Kind']}, {row['Distinct values']} of {row['Batches with a value']} distinct" + (
        f", |ρ| {rho:.2f} with the date" if rho else ""
    )


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    unique: float = UNIQUE,
    near_unique: float = NEAR_UNIQUE,
) -> tuple[list[dict], IdentifierReport]:
    """The columns that name a batch rather than describe it, one row each.

    ``analysed`` marks the batches judged (default: all), as for the missingness
    report; the others are left out. ``unique`` and ``near_unique`` are the shares of
    distinct values that make a code unique or near-unique.
    """
    if analysed is not None:
        frame = frame[analysed.reindex(frame.index).fillna(False).astype(bool)]
    report = IdentifierReport(unique=unique, near_unique=near_unique)
    if order_column in frame.columns:
        dates = pd.to_datetime(frame[order_column], errors="coerce")
        frame = (
            frame.assign(__date=dates).sort_values("__date", kind="stable").drop(columns="__date")
        )
        dates = dates.reindex(frame.index)
    else:
        dates = None
    report.batches = (
        frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    ) or [f"row {i}" for i in frame.index]
    report.dates = (
        dates.dt.strftime("%Y-%m-%d").fillna("").tolist() if dates is not None else []
    ) or [""] * len(frame)

    rows = []
    for column in frame.columns:
        if column in (id_column, order_column):
            continue
        values = frame[column]
        value_type = str(value_types.get(column, "")).strip()
        present = int(values.notna().sum())
        if present < MIN_VALUES:
            report.not_analysed[str(column)] = f"fewer than {MIN_VALUES} values"
            continue
        if _is_date(values, value_type):
            report.not_analysed[str(column)] = "a date"
            continue
        report.tested += 1
        kind, facts = kind_of(values, value_type, dates, unique, near_unique)
        if not kind:
            continue
        labels = values.dropna().map(_text)
        examples = list(dict.fromkeys(labels))[:EXAMPLES]
        rows.append(
            {
                "Parameter": str(column),
                "Kind": kind,
                "Catalog type": value_type,
                "Batches with a value": facts["Batches with a value"],
                "Distinct values": facts["Distinct values"],
                "Distinct %": facts["Distinct %"],
                "Most batches with one value": facts["Most batches with one value"],
                "Median words": facts.get("Median words"),
                "|ρ| with the date": facts.get("|ρ| with the date"),
                "Examples": " · ".join(_short(e, EXAMPLE_LENGTH) for e in examples),
            }
        )
        report.values[str(column)] = [
            None if pd.isna(v) else _short(_text(v), VALUE_LENGTH) for v in values
        ]
    order = list(KINDS)
    rows.sort(key=lambda r: (order.index(r["Kind"]), r["Parameter"]))
    report.analysed = [r["Parameter"] for r in rows]
    counts = {k: sum(r["Kind"] == k for r in rows) for k in KINDS}
    log.success(
        MODULE,
        f"{len(rows):,} of {report.tested:,} parameters name the batch rather than describe it "
        f"over {len(frame):,} batches",
        ", ".join(f"{n} {k}" for k, n in counts.items() if n) or "none found",
    )
    return rows, report
