"""The batch table's dates and times, put in order on one timeline.

A date is not a predictor: a model fitted on one learns when a batch was made, and
says nothing about a batch made later. ``ml task new`` leaves every date column out
of a new task (``predictors.exclude``, with why). What the dates do tell is how a
batch went: what came after what, on which day, and at what time of day. This module
puts every date column of a site on one timeline, over the batches the missingness
report counts in its statistics.

1. **The date columns.** Every column the catalog types ``datetime``, or whose
   values are dates, but the batch key. Each is *date and time*, *date only* (nine
   values in ten at midnight) or *time of day only* (nine in ten before 1901: a
   spreadsheet keeps a time alone as a time on 1 January 1900).
2. **The anchor.** Each batch's timepoints are read from its anchor: the date the
   batches are ordered by (``build.date_column``), or the batch's first timepoint.
   *Day 0* is the anchor's calendar day; a timepoint is on *day n* when it falls
   on the n-th calendar day after it.
3. **Lot dates.** A date shared by batches started more than :data:`SHARED_DAYS`
   days apart is the date of a lot or a material (a vector's manufacture): it is
   told as its age at the anchor, and kept off the batch's timeline.
4. **The order.** The other columns with :data:`MIN_VALUES` values or more are
   sorted by their median hours from the anchor: the order the process runs in,
   as the data has it.
5. **Each column.** Its usual day (the commonest) and the share on it, its spread
   (10th to 90th percentile), the median wait since the column before it, how often
   it comes before that column, whether it is the same as another column, its time
   of day, the share on a weekend, when it was first and last recorded, and whether
   its place drifts over the months (Spearman ρ of its hours from the anchor with
   the anchor date): what dates.xlsx keeps.
6. **Phases.** Columns next to each other in the order with the same usual day
   form a phase: what the process does on that day.

This is description only. Nothing here changes the batch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .logger import log

MODULE = "dates"

#: Values a column needs to be placed on the timeline.
MIN_VALUES = 20
#: Two columns are the same when equal in this share of the batches with both.
SAME = 0.95
#: A date shared by batches whose anchors lie more than this many days apart (median over
#: its shared values) is a lot's or a material's date, not the batch's.
SHARED_DAYS = 7.0

DATE_TIME, DATE_ONLY, TIME_ONLY = "date and time", "date only", "time of day only"
KINDS = {
    DATE_TIME: "A date with the time of day: placed on the timeline to the minute.",
    DATE_ONLY: "A date (nine values in ten at midnight): placed on the timeline to the day.",
    TIME_ONLY: "A time of day with no date (kept as 1 January 1900): on the clock only.",
}
LOT = "lot or material date"
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class DatesReport:
    #: Set by the build: the subsite it is about, level by level (manufacturing site, then
    #: each split column), "All" where it takes every value of a level.
    levels: dict[str, str] = field(default_factory=dict)
    #: Every date column found, with its kind.
    kinds: dict[str, str] = field(default_factory=dict)
    #: The columns on the timeline, in order, and the ones off it with why.
    order: list[str] = field(default_factory=list)
    not_placed: dict[str, str] = field(default_factory=dict)
    anchor: str = ""
    cohort: str = ""
    #: The batches, in anchor order, with their anchor date.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    #: Per column on the timeline, per batch: days from the anchor's midnight (day 0 is
    #: the anchor's day), and its 10th, 25th, 50th, 75th and 90th percentiles.
    days: dict[str, list[float | None]] = field(default_factory=dict)
    quantiles: dict[str, list[float | None]] = field(default_factory=dict)
    #: Per column on the timeline: share of its batches on each day, from ``first_day``.
    first_day: int = 0
    on_day: dict[str, list[float]] = field(default_factory=dict)
    usual_day: dict[str, int] = field(default_factory=dict)
    #: Per column with a time of day, per batch: hours since midnight, unwrapped by
    #: :func:`circular` (a step done around midnight runs past 24).
    clock: dict[str, list[float | None]] = field(default_factory=dict)
    #: Lot and material dates, per batch: their age at the anchor, in days.
    lots: dict[str, list[float | None]] = field(default_factory=dict)
    #: The phases: (usual day, columns), in order.
    phases: list[tuple[int, list[str]]] = field(default_factory=list)


def _hours(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 3600.0


def _mostly(parsed: pd.Series, present: int) -> bool:
    """Whether nine values in ten parsed."""
    return present > 0 and parsed.notna().sum() >= 0.9 * present


def is_date(values: pd.Series, value_type: str) -> bool:
    """Whether a column holds dates: the catalog says ``datetime``, or nine values in ten read
    as dates. Numbers, and numbers kept as text, are not dates, however a parser reads them."""
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


def parse(values: pd.Series) -> pd.Series:
    """A column as timestamps; what does not read as one is missing."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(
        values.map(lambda v: v if pd.isna(v) else str(v)), errors="coerce", format="mixed"
    )


def kind_of(times: pd.Series) -> str:
    """Date and time, date only, or time of day only."""
    present = times.dropna()
    if present.empty:
        return DATE_ONLY
    if (present.dt.year < 1901).mean() >= 0.9:
        return TIME_ONLY
    if (present == present.dt.normalize()).mean() >= 0.9:
        return DATE_ONLY
    return DATE_TIME


def _shared(times: pd.Series, anchor: pd.Series) -> bool:
    """Whether the batches sharing a value of the column were started weeks apart."""
    both = pd.DataFrame({"t": times, "a": anchor}).dropna()
    spread = both.groupby("t")["a"].agg(lambda a: (a.max() - a.min()).total_seconds() / 86400)
    shared = spread[both.groupby("t").size() > 1]
    return bool(len(shared)) and float(shared.median()) > SHARED_DAYS


def _rho(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """Spearman ρ and its p-value over the pairs with both, or (None, None)."""
    both = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(both) < MIN_VALUES or both["x"].nunique() < 2 or both["y"].nunique() < 2:
        return None, None
    rho, p = stats.spearmanr(both["x"], both["y"])
    return (None, None) if pd.isna(rho) else (round(float(rho), 3), float(p))


def _round(value, digits: int = 2):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _clock(hours: float | None) -> str:
    if hours is None or pd.isna(hours):
        return ""
    minutes = int(round(hours * 60)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def circular(clock: pd.Series) -> pd.Series:
    """Hours of the day, cut at the emptiest hour of the clock rather than at midnight, so a
    step done around midnight keeps its times together: 23:00 and 01:00 are two hours
    apart, not twenty-two. Values may run past 24; take them modulo 24 to read them."""
    present = clock.dropna()
    if present.empty:
        return clock
    counts = np.bincount(np.floor(present).astype(int) % 24, minlength=24)
    # The start of the longest run of empty hours, going round the clock.
    best, cut = -1, 0
    for start in range(24):
        run = 0
        while run < 24 and counts[(start + run) % 24] == 0:
            run += 1
        if run > best:
            best, cut = run, start
    # Midnight inside the emptiest stretch: nothing straddles it, nothing moves.
    if best <= 0 or any((cut + h) % 24 == 0 for h in range(best)):
        return clock
    return clock.where(clock >= cut, clock + 24)


def day_text(value: float | None) -> str:
    """Days from the anchor's midnight, as *day 3, 14:05*."""
    if value is None or pd.isna(value):
        return ""
    day = int(np.floor(value))
    return f"day {day}, {_clock((value - day) * 24)}"


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
) -> tuple[list[dict], DatesReport]:
    """Every date column, one row each, and the timeline they make.

    ``analysed`` marks the batches described (default: all), as for the missingness
    report. ``order_column`` is the anchor when it holds dates.
    """
    if analysed is not None:
        frame = frame[analysed.reindex(frame.index).fillna(False).astype(bool)]
    report = DatesReport()
    times: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column == id_column:
            continue
        if is_date(frame[column], str(value_types.get(column, "")).strip()):
            times[str(column)] = parse(frame[column])
            report.kinds[str(column)] = kind_of(times[str(column)])
    if not times:
        log.info(MODULE, "No date columns to describe")
        return [], report

    # The anchor, and the batches in its order.
    if order_column in times and report.kinds[order_column] != TIME_ONLY:
        anchor, report.anchor = times[order_column], order_column
    else:
        placed = [c for c, k in report.kinds.items() if k != TIME_ONLY]
        anchor = (
            pd.concat([times[c] for c in placed], axis=1).min(axis=1)
            if placed
            else pd.Series(pd.NaT, index=frame.index)
        )
    order = anchor.sort_values(kind="stable", na_position="last").index
    frame, anchor = frame.loc[order], anchor.loc[order]
    times = {c: t.loc[order] for c, t in times.items()}
    midnight = anchor.dt.normalize()
    report.batches = (
        frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    ) or [f"row {i}" for i in frame.index]
    report.dates = anchor.dt.strftime("%Y-%m-%d").fillna("").tolist()
    when = anchor.map(lambda t: t.toordinal() if pd.notna(t) else np.nan).astype(float)

    # What goes on the timeline.
    hours: dict[str, pd.Series] = {}
    for column, kind in report.kinds.items():
        if kind in (DATE_TIME, TIME_ONLY):
            clock = circular(times[column].dt.hour + times[column].dt.minute / 60.0)
            report.clock[column] = [_round(v, 3) for v in clock]
        if kind == TIME_ONLY:
            report.not_placed[column] = "a time of day with no date: on the clock only"
        elif int(times[column].notna().sum()) < MIN_VALUES:
            report.not_placed[column] = f"fewer than {MIN_VALUES} values"
        elif column != report.anchor and _shared(times[column], anchor):
            report.not_placed[column] = (
                "shared by batches started weeks apart: the date of a lot or a material"
            )
            report.lots[column] = [_round(v / 24) for v in _hours(anchor - times[column])]
        else:
            hours[column] = _hours(times[column] - anchor)
    report.order = sorted(hours, key=lambda c: (hours[c].median(), c))
    days: dict[str, pd.Series] = {}
    index: dict[str, pd.Series] = {}
    for column in report.order:
        days[column] = _hours(times[column] - midnight) / 24.0
        index[column] = np.floor(days[column])
        report.days[column] = [_round(v, 4) for v in days[column]]
        report.quantiles[column] = [
            _round(days[column].quantile(q), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        ]
    if report.order:
        spread = pd.concat(index.values()).dropna()
        report.first_day = int(spread.min()) if len(spread) else 0
        last_day = int(spread.max()) if len(spread) else 0
        for column in report.order:
            counts = index[column].value_counts(normalize=True)
            report.on_day[column] = [
                round(float(counts.get(d, 0.0)), 4) for d in range(report.first_day, last_day + 1)
            ]
            if len(counts):
                report.usual_day[column] = int(counts.idxmax())

    # Phases: neighbours with the same usual day.
    for column in report.order:
        day = report.usual_day.get(column, 0)
        if report.phases and report.phases[-1][0] == day:
            report.phases[-1][1].append(column)
        else:
            report.phases.append((day, [column]))

    rows = [
        _row(column, times, hours, days, index, anchor, when, report) for column in report.kinds
    ]
    rows.sort(key=lambda r: (r["Position"] is None, r["Position"] or 0, r["Parameter"]))
    log.success(
        MODULE,
        f"{len(report.kinds):,} date columns, {len(report.order):,} on the timeline in "
        f"{len(report.phases)} phases",
        " → ".join(report.order[:8]) + (" …" if len(report.order) > 8 else ""),
    )
    return rows, report


def _row(column, times, hours, days, index, anchor, when, report) -> dict:
    """One date column, everything the report knows of it."""
    kind = report.kinds[column]
    t = times[column]
    present = t.dropna()
    n = len(t)
    position = report.order.index(column) + 1 if column in report.order else None
    row: dict = {
        "Parameter": column,
        "Kind": LOT if column in report.lots else kind,
        "Position": position,
        "Usual day": report.usual_day.get(column),
        "On usual day %": None,
        "Median": "",
        "Earliest (P10)": "",
        "Latest (P90)": "",
        "Median hours from anchor": None,
        "Median wait since previous (h)": None,
        "Before previous %": None,
        "Same as": "",
        "Median time of day": "",
        "Time of day P10–P90": "",
        "Weekend %": None,
        "Batches with a value": int(len(present)),
        "Coverage %": round(100 * len(present) / n, 1) if n else 0.0,
        "First recorded": "" if kind == TIME_ONLY or present.empty else f"{present.min():%Y-%m-%d}",
        "Last recorded": "" if kind == TIME_ONLY or present.empty else f"{present.max():%Y-%m-%d}",
        "Drift ρ": None,
        "Drift p": None,
        "Off the timeline because": report.not_placed.get(column, ""),
    }
    if column in report.clock:
        clock = pd.Series(report.clock[column], dtype=float)
        row["Median time of day"] = _clock(clock.median())
        row["Time of day P10–P90"] = f"{_clock(clock.quantile(0.1))}–{_clock(clock.quantile(0.9))}"
    if kind != TIME_ONLY and len(present):
        row["Weekend %"] = round(100 * float((present.dt.weekday >= 5).mean()), 1)
    if column in report.lots:
        age = pd.Series(report.lots[column], dtype=float)
        row["Median hours from anchor"] = _round(-24 * age.median())
        row["Drift ρ"], row["Drift p"] = _rho(when, age)
    if column in hours:
        h, d = hours[column], days[column]
        q = report.quantiles[column]
        row["Median"], row["Earliest (P10)"], row["Latest (P90)"] = (
            day_text(q[2]),
            day_text(q[0]),
            day_text(q[4]),
        )
        row["Median hours from anchor"] = _round(h.median())
        if column in report.usual_day:
            row["On usual day %"] = round(
                100 * float((index[column] == report.usual_day[column]).mean() / d.notna().mean()),
                1,
            )
        if column != report.anchor:
            row["Drift ρ"], row["Drift p"] = _rho(when, h)
        if position and position > 1:
            previous = report.order[position - 2]
            both = h.notna() & hours[previous].notna()
            if both.sum():
                wait = h[both] - hours[previous][both]
                row["Median wait since previous (h)"] = _round(wait.median())
                row["Before previous %"] = round(100 * float((wait < 0).mean()), 1)
    for other in report.kinds:
        if other == column:
            continue
        both = t.notna() & times[other].notna()
        if both.sum() >= MIN_VALUES and (t[both] == times[other][both]).mean() >= SAME:
            row["Same as"] = other
            break
    return row


def day_rows(report: DatesReport) -> list[dict]:
    """Each batch's place of every column on the timeline, in days from the anchor's
    midnight: what dates.xlsx keeps."""
    return [
        {
            "Batch": batch,
            "Anchor date": report.dates[i],
            **{f"{c} (day)": report.days[c][i] for c in report.order},
        }
        for i, batch in enumerate(report.batches)
    ]
