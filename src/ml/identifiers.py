"""Parameters that name a thing rather than describe it: identifiers and free text.

A patient code, a case number, a running batch number, a raw material or
consumable lot, a reference to a document: each names something, and a model
can learn which batch (or which lot, or which month) is which from it, never
what made a batch turn out as it did. A comment written for one batch is the
same. This module finds them per site, over the batches the missingness report
counts in its statistics, before any other report is read.

Each column found has a *status*:

- **identifier**: clear. ``ml task new`` leaves it out of a new task
  (``predictors.exclude``, with why), unless the register reinstated it.
- **possible identifier**: one sign only, so it may as well be a category (an
  equipment code, a grade such as ``G2``). Listed for a person to decide: an
  exclusion in the register (the ``identifiers`` track) leaves it out of every
  task.

How a column is judged:

1. Every column takes part but the batch key, the date the batches are ordered
   by, and the dates (catalog type ``datetime``, or values that are dates). A
   column needs :data:`MIN_VALUES` batches with a value, and two distinct values.
2. *Distinct %* is its distinct values over its batches with a value.
3. The signs, each general:

   - **code-shaped**: nine values in ten are one token (no space) with a digit
     in it, and four in five share one shape (letters read as ``A``, digits as
     ``9``: ``XY-104`` and ``QR-7`` are both ``A-9``). Numbers are not codes,
     unless the catalog types the column categorical and they have
     :data:`CODE_DIGITS` digits or more.
   - **named as one**: a word of the column's name is one of :data:`ID_WORDS`
     (``lot``, ``code``, ``id``, ``serial``, …), or the name has a ``#``.
   - **from an identifier source**: the build joined it from a source that holds
     identifiers only (the raw materials and consumables export).
   - **counts with time**: the number in the value (its last run of digits, or
     the whole number) rises or falls with the date: Spearman |ρ| of at least
     :data:`TIME_RHO`, over at least :data:`MIN_VALUES` batches and
     :data:`MIN_NUMBERS` distinct numbers. A lot numbered in order does this,
     whether or not it runs in groups (the shifts report's question).

4. A column whose values are words or codes (not numbers, or typed categorical)
   is, in this order,

   - **free text** when its values are sentences (a median of at least
     :data:`TEXT_WORDS` words) and at least ``near_unique`` of them are distinct;
   - a **unique code** when at least ``unique`` of its values are distinct;
   - a **near-unique code** when at least ``near_unique`` are: more categories
     than half the batches, more than any encoder can learn from;
   - a **sequential code** (counts with time) or a **lot or reference code**
     when it comes from an identifier source, or shows two of the other signs
     (code-shaped, named as one, counts with time);
   - a **possible identifier** when it shows one of code-shaped or named as one.

5. A numerical column is a **running number** when its values are whole, at
   least ``unique`` distinct, and rise or fall with the date (|ρ| of at least
   :data:`SEQUENCE_RHO`); a **lot or reference code** when it comes from an
   identifier source; a **sequential code** when it is whole numbers, named as
   one and counts with time; a **possible identifier** when it is whole numbers
   and named as one. Any other number describes the batch.

All four statuses' kinds are identifiers; free text is proposed under its own
screening reason. A column with one value (constant) is not reported.

This is description only. Nothing here changes the batch table or writes to the
register.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logger import log

MODULE = "identifiers"

#: The build's sources whose every column is an identifier (a lot, a material or kit number),
#: by the name the build gives them.
SOURCES = ("Raw materials and consumables",)
#: Values a column needs before it is judged.
MIN_VALUES = 20
#: The share of distinct values that makes a code unique to a batch (``build.identifier_unique``).
UNIQUE = 0.95
#: The share that makes it near-unique, as a task's ``max_cardinality_ratio`` does
#: (``build.identifier_near_unique``).
NEAR_UNIQUE = 0.5
#: The median number of words that makes a value a sentence rather than a code.
TEXT_WORDS = 3
#: The |ρ| with the date that makes a distinct whole number a running number.
SEQUENCE_RHO = 0.95
#: The |ρ| with the date, and the distinct numbers, that make a code count with time.
TIME_RHO, MIN_NUMBERS = 0.8, 3
#: The share of values that must be one token with a digit, and share the commonest shape.
CODE_SHARE, SHAPE_SHARE = 0.9, 0.8
#: The digits a code made of digits alone needs (in a column the catalog types categorical).
CODE_DIGITS = 3
#: Words of a column's name that say it holds identifiers.
ID_WORDS = frozenset(
    {
        "lot",
        "lots",
        "batch",
        "id",
        "ids",
        "identifier",
        "uid",
        "code",
        "codes",
        "serial",
        "barcode",
        "ref",
        "reference",
        "sku",
        "number",
        "nr",
        "num",
    }
)
#: Of those, the ones that say so of a numerical column too ("number of …" counts things).
NUMERIC_ID_WORDS = ID_WORDS - {"number", "nr", "num"}
#: Values shown per parameter in the table, and the characters kept of each.
EXAMPLES, EXAMPLE_LENGTH = 3, 40
#: The characters of each value the viewer keeps.
VALUE_LENGTH = 120

IDENTIFIER, POSSIBLE = "identifier", "possible identifier"
UNIQUE_CODE, NEAR_UNIQUE_CODE = "unique code", "near-unique code"
FREE_TEXT, RUNNING_NUMBER = "free text", "running number"
SEQUENTIAL_CODE, LOT_CODE = "sequential code", "lot or reference code"
#: What a column can be found to be.
KINDS = {
    UNIQUE_CODE: "A code with a different value in (nearly) every batch: a patient code, a "
    "case number, a sample id. It names the batch.",
    NEAR_UNIQUE_CODE: "A code most batches have to themselves: more categories than half the "
    "batches, too many for any encoder to learn from.",
    RUNNING_NUMBER: "A whole number, distinct per batch, that counts up (or down) with the "
    "date: a running batch or order number. It says when a batch was made.",
    SEQUENTIAL_CODE: "A code whose number counts with the date, shared by a few batches or "
    "many: a lot numbered in order. It says when a batch was made, in other words.",
    LOT_CODE: "A code that names a lot, a material, a document or a piece of equipment: from "
    "an identifier source, or code-shaped and named as one.",
    FREE_TEXT: "Sentences written for one batch: a comment, a description. No two batches "
    "share one, so a model can only memorise them.",
    POSSIBLE: "One sign only: code-shaped values, or a name that says code, id, lot …. An "
    "equipment code or a grade looks like this, and may be a category worth keeping.",
}
#: The screening reason a proposal gives each kind.
REASON_OF = {kind: "identifier" for kind in KINDS} | {FREE_TEXT: "free text"}

_CODE_TYPES = ("categorical", "boolean")
_LETTERS, _DIGITS = re.compile(r"[^\W\d_]+"), re.compile(r"\d+")


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
    # A short code such as "M3" parses as a time on 1 January of year 1: not a date.
    return _mostly(parsed.where(parsed.dt.year >= 1900), len(text))


def shape(value: str) -> str:
    """A value's shape: letters read as ``A``, digits as ``9``, the rest kept."""
    return _DIGITS.sub("9", _LETTERS.sub("A", value))


def named_as_identifier(name: str, numeric: bool = False) -> str:
    """The word of a column's name that says it holds identifiers, or ``""``."""
    if "#" in name:
        return "#"
    words = set(re.findall(r"[a-z]+", name.lower()))
    found = words & (NUMERIC_ID_WORDS if numeric else ID_WORDS)
    return sorted(found)[0] if found else ""


def code_shaped(labels: pd.Series, categorical: bool = False) -> bool:
    """Whether the values look like codes: one token each with a digit in it, most of them
    of one shape. Plain numbers are not codes, unless ``categorical`` and long enough."""
    if labels.empty:
        return False
    token = ~labels.str.contains(r"\s")
    digit = labels.str.contains(r"\d")
    if (token & digit).mean() < CODE_SHARE:
        return False
    shapes = labels.map(shape)
    if shapes.value_counts(normalize=True).iloc[0] < SHAPE_SHARE:
        return False
    number = labels.str.fullmatch(r"[-+]?\d+([.,]\d+)?")
    if number.mean() >= CODE_SHARE:
        return bool(categorical and labels.str.len().median() >= CODE_DIGITS)
    return True


def _number_in(labels: pd.Series) -> pd.Series:
    """The last run of digits in each value, as a number."""
    return pd.to_numeric(labels.str.findall(r"\d+").str[-1], errors="coerce")


def _with_time(numbers: pd.Series, dates: pd.Series | None) -> float | None:
    """|ρ| of the numbers with the date, or None when there is too little to say."""
    if dates is None:
        return None
    both = pd.DataFrame({"n": numbers, "date": dates.reindex(numbers.index)}).dropna()
    if len(both) < MIN_VALUES or both["n"].nunique() < MIN_NUMBERS or both["date"].nunique() < 2:
        return None
    rho = both["n"].rank().corr(both["date"].rank())
    return None if pd.isna(rho) else round(abs(float(rho)), 3)


def judge(
    values: pd.Series,
    name: str,
    value_type: str = "",
    dates: pd.Series | None = None,
    source: str = "",
    unique: float = UNIQUE,
    near_unique: float = NEAR_UNIQUE,
) -> tuple[str, str, dict]:
    """What one column is: (status, kind, facts), with ``("", "", facts)`` when it is none.

    ``values`` are the column over the batches judged, ``dates`` their dates (None
    when there are none), ``source`` the identifier source it was joined from, if any.
    ``facts`` holds the numbers the report shows, and ``Signs`` the signs found.
    """
    present = values.dropna()
    n = len(present)
    if n == 0 or pd.api.types.is_bool_dtype(values):
        return "", "", {}
    labels = present.map(_text)
    distinct = int(labels.nunique())
    ratio = distinct / n
    facts: dict = {
        "Batches with a value": n,
        "Distinct values": distinct,
        "Distinct %": round(100 * ratio, 1),
        "Most batches with one value": int(labels.value_counts().iloc[0]),
    }
    if distinct < 2:
        return "", "", facts
    signs: list[str] = [f"from {source}"] if source else []

    def found(status: str, kind: str, rho: float | None = None) -> tuple[str, str, dict]:
        if rho is not None:
            facts["|ρ| with the date"] = rho
        facts["Signs"] = "; ".join(signs)
        return status, kind, facts

    categorical = value_type in _CODE_TYPES
    numeric = pd.api.types.is_numeric_dtype(values)
    if not numeric and not categorical:
        # Numbers kept as text are numbers.
        parsed = pd.to_numeric(labels, errors="coerce")
        if _mostly(parsed, n):
            present, numeric = parsed.dropna(), True
    if numeric and not categorical:
        whole = bool(np.all(np.isclose(present, np.round(present))))
        if not whole:
            return ("", "", facts) if not source else found(IDENTIFIER, LOT_CODE)
        rho = _with_time(present, dates)
        if ratio >= unique and rho is not None and rho >= SEQUENCE_RHO:
            signs.append(f"distinct whole numbers that count with the date (|ρ| {rho:.2f})")
            return found(IDENTIFIER, RUNNING_NUMBER, rho)
        if source:
            return found(IDENTIFIER, LOT_CODE)
        word = named_as_identifier(name, numeric=True)
        if not word:
            return "", "", facts
        signs.append(f"named as one ('{word}'), whole numbers")
        if rho is not None and rho >= TIME_RHO:
            signs.append(f"counts with the date (|ρ| {rho:.2f})")
            return found(IDENTIFIER, SEQUENTIAL_CODE, rho)
        return found(POSSIBLE, POSSIBLE)

    words = float(labels.str.split().str.len().median())
    if words >= TEXT_WORDS and ratio >= near_unique:
        signs.append(f"sentences (median {words:.0f} words), {ratio:.0%} distinct")
        return found(IDENTIFIER, FREE_TEXT)
    if ratio >= unique:
        signs.append(f"{ratio:.0%} distinct")
        return found(IDENTIFIER, UNIQUE_CODE)
    if ratio >= near_unique:
        signs.append(f"{ratio:.0%} distinct")
        return found(IDENTIFIER, NEAR_UNIQUE_CODE)
    shaped = code_shaped(labels, categorical)
    word = named_as_identifier(name)
    rho = _with_time(_number_in(labels), dates) if shaped else None
    timed = rho is not None and rho >= TIME_RHO
    if shaped:
        signs.append(f"code-shaped ({labels.map(shape).value_counts().index[0]})")
    if word:
        signs.append(f"named as one ('{word}')")
    if timed:
        signs.append(f"its number counts with the date (|ρ| {rho:.2f})")
    # Counted as integers: NumPy adds two True to True.
    if source or int(shaped) + int(bool(word)) + int(timed) >= 2:
        return found(IDENTIFIER, SEQUENTIAL_CODE if timed else LOT_CODE, rho if timed else None)
    if shaped or word:
        return found(POSSIBLE, POSSIBLE)
    return "", "", facts


def evidence(row: dict) -> str:
    """One found column, as the screening register cites it: ``lot or reference code: …``."""
    return f"{row['Kind']}: {row['Signs']}" if row.get("Signs") else row["Kind"]


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    unique: float = UNIQUE,
    near_unique: float = NEAR_UNIQUE,
    sources: dict[str, str] | None = None,
) -> tuple[list[dict], IdentifierReport]:
    """The columns that name a thing rather than describe it, one row each.

    ``analysed`` marks the batches judged (default: all), as for the missingness
    report; the others are left out. ``unique`` and ``near_unique`` are the shares of
    distinct values that make a code unique or near-unique. ``sources`` maps each
    column joined from a source of identifiers to that source's name.
    """
    sources = sources or {}
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
        if int(values.notna().sum()) < MIN_VALUES:
            report.not_analysed[str(column)] = f"fewer than {MIN_VALUES} values"
            continue
        if _is_date(values, value_type):
            report.not_analysed[str(column)] = "a date"
            continue
        report.tested += 1
        status, kind, facts = judge(
            values,
            str(column),
            value_type,
            dates,
            sources.get(column, ""),
            unique,
            near_unique,
        )
        if not status:
            continue
        labels = values.dropna().map(_text)
        examples = list(dict.fromkeys(labels))[:EXAMPLES]
        rows.append(
            {
                "Parameter": str(column),
                "Status": status,
                "Kind": kind,
                "Signs": facts["Signs"],
                "Catalog type": value_type,
                "Batches with a value": facts["Batches with a value"],
                "Distinct values": facts["Distinct values"],
                "Distinct %": facts["Distinct %"],
                "Most batches with one value": facts["Most batches with one value"],
                "|ρ| with the date": facts.get("|ρ| with the date"),
                "Examples": " · ".join(_short(e, EXAMPLE_LENGTH) for e in examples),
            }
        )
        report.values[str(column)] = [
            None if pd.isna(v) else _short(_text(v), VALUE_LENGTH) for v in values
        ]
    order = list(KINDS)
    rows.sort(key=lambda r: (r["Status"] == POSSIBLE, order.index(r["Kind"]), r["Parameter"]))
    report.analysed = [r["Parameter"] for r in rows]
    clear = sum(r["Status"] == IDENTIFIER for r in rows)
    log.success(
        MODULE,
        f"{clear:,} identifiers and {len(rows) - clear:,} possible identifiers among "
        f"{report.tested:,} parameters over {len(frame):,} batches",
        ", ".join(f"{n} {k}" for k in KINDS if (n := sum(r["Kind"] == k for r in rows)))
        or "none found",
    )
    return rows, report
