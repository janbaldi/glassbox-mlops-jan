"""The batch table's dates and times: their order, and how the batches relate over time.

A date is not a predictor: a model fitted on one learns when a batch was made, and
says nothing about a batch made later. ``ml task new`` leaves every date column
out of a new task (``predictors.exclude``, with why). What the dates do tell is how
a batch went: which step came after which, how long each took, and whether the
batches fall into groups that were run to a different timing. This module works
that out per site, over the batches the missingness report counts in its
statistics.

1. **The date columns.** Every column the catalog types ``datetime``, or whose
   values are dates, but the batch key. Each is *date and time*, *date only*
   (nine values in ten at midnight) or *time of day only* (nine in ten before
   1901: a spreadsheet keeps a time alone as a time on 1 January 1900). A time of
   day alone cannot be put on a batch's timeline; it is described, not placed.
2. **The anchor.** Each batch's timepoints are read as hours from its anchor:
   the date the batches are ordered by (``build.date_column``), or, without it,
   the batch's first timepoint.
3. **The order.** A date shared by batches started more than :data:`SHARED_DAYS`
   days apart is the date of a lot or a material (a vector's manufacture): it is
   described as an age at the anchor, and kept off the timeline. The other columns
   with :data:`MIN_VALUES` values or more are sorted by their median hours from
   the anchor: the order the process runs in, as the data has it. For each
   column: how often it comes before the column sorted before it (*Out of
   order*), whether it is the same as another column, the share on a weekend,
   its median time of day, and whether its hours from the anchor drift over the
   months (Spearman ρ with the anchor date, p < :data:`P_MAX`).
4. **The steps.** A step is two columns next to each other in that order; its
   duration is the hours between them, in each batch with both.
5. **Clusters of batches.** The batches are clustered on their step durations:
   the steps with a value in at least :data:`MIN_SHARE` of the batches, the
   batches with at least :data:`MIN_SHARE` of those steps. Each duration is
   measured as its deviation from the step's median in median lengths of the
   step (at least :data:`FLOOR_HOURS` hours; cut at ±:data:`CLIP`): a day late is
   much in a three-day step and little in a two-month one. A missing one counts
   as the median, and Ward's hierarchical clustering joins the batches; the number of clusters, 2 to
   :data:`MAX_CLUSTERS`, is the one with the highest mean silhouette. Below a
   silhouette of :data:`WEAK` the clusters are weak: the timing varies more
   than it groups. Clusters are named *Timing 1, 2, …* by their median anchor
   date. A step *sets a cluster apart* when the cluster's median differs from the
   other batches' by :data:`APART` of the step's length or more.
6. **How the batches relate over time.** Whether the clusters follow one
   another in time (eras: fewer runs of one cluster in date order than in
   :data:`PERMUTATIONS` shuffles, p < :data:`P_MAX`) or interleave; whether
   the weekday a batch starts on goes with its cluster (χ² test); how many
   batches are in progress at once (anchor to last timepoint); the days between
   consecutive starts.

This is description only. Nothing here changes the batch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_samples, silhouette_score

from .identifiers import is_date
from .logger import log

MODULE = "dates"

#: Values a column needs to be placed on the timeline, and batches to cluster.
MIN_VALUES = 20
#: The share of batches a step needs to take part in the clusters, and of those steps a
#: batch needs.
MIN_SHARE = 0.5
#: The most clusters tried, and the mean silhouette below which they are weak.
MAX_CLUSTERS, WEAK = 6, 0.25
#: A step's deviation is measured in its median lengths, at least this many hours, and cut
#: at :data:`CLIP` lengths, so one extreme batch does not make a cluster alone.
FLOOR_HOURS, CLIP = 24.0, 5.0
#: Two columns are the same when equal in this share of the batches with both.
SAME = 0.95
#: A column is out of order, worth naming, from this share of its batches; from
#: NO_ORDER, the two have no fixed order.
OUT_OF_ORDER, NO_ORDER = 0.05, 0.25
#: A column falls on a weekend notably more often than the anchor from this many points.
WEEKEND_POINTS = 5.0
#: A date shared by batches whose anchors lie more than this many days apart (median over
#: its shared values) is a lot's or a material's date, not the batch's.
SHARED_DAYS = 7.0
#: Significance for drift, eras and weekdays; shuffles for the eras.
P_MAX, PERMUTATIONS = 0.01, 500
#: A step sets a cluster apart when its median differs from the other batches' by this
#: share of the step's length (at least FLOOR_HOURS).
APART = 0.1

DATE_TIME, DATE_ONLY, TIME_ONLY = "date and time", "date only", "time of day only"
KINDS = {
    DATE_TIME: "A date with the time of day: placed on the timeline to the minute.",
    DATE_ONLY: "A date (nine values in ten at midnight): placed on the timeline to the day.",
    TIME_ONLY: "A time of day with no date (kept as 1 January 1900): described, not placed.",
}
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class Cluster:
    name: str
    batches: int
    first: str
    last: str
    #: Median hours from the anchor to the batch's last timepoint.
    span: float | None
    silhouette: float
    #: The steps that set it apart, in words, most first.
    apart: list[str] = field(default_factory=list)
    #: Median hours from the anchor of each timeline column, in ``order``.
    profile: list[float | None] = field(default_factory=list)


@dataclass
class DatesReport:
    #: Every date column found, with its kind; the ones on the timeline, sorted.
    kinds: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    #: The columns found but not placed, and why.
    not_placed: dict[str, str] = field(default_factory=dict)
    #: Of those, the dates of a lot or material: their age at the anchor, in hours per batch.
    shared: dict[str, list[float | None]] = field(default_factory=dict)
    anchor: str = ""
    cohort: str = ""
    #: The batches, in date order, with their anchor date and hours from it per column.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    hours: dict[str, list[float | None]] = field(default_factory=dict)
    #: Share of batches with both where the row column comes before the other (ties half).
    before: list[list[float | None]] = field(default_factory=list)
    #: The steps, each (from, to), and their hours per batch; those clustered on.
    steps: list[tuple[str, str]] = field(default_factory=list)
    step_hours: dict[str, list[float | None]] = field(default_factory=dict)
    clustered_on: list[str] = field(default_factory=list)
    #: Per batch: its cluster ("" when not clustered), silhouette and span.
    cluster_of: list[str] = field(default_factory=list)
    silhouette_of: list[float | None] = field(default_factory=list)
    span_of: list[float | None] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    silhouette: float | None = None
    #: Why there are no clusters, when there are none.
    unclustered: str = ""
    #: In date order: runs of one cluster, what shuffles give, and the p-value.
    runs: int | None = None
    runs_expected: float | None = None
    runs_p: float | None = None
    #: Batches starting per weekday, per cluster, and the χ² p-value.
    weekdays: dict[str, list[int]] = field(default_factory=dict)
    weekdays_p: float | None = None
    #: Per day from the first anchor to the last timepoint: batches in progress.
    progress_days: list[str] = field(default_factory=list)
    progress: list[int] = field(default_factory=list)
    #: Days between consecutive starts.
    gaps: list[float] = field(default_factory=list)
    #: What the report says in words, most telling first.
    findings: list[str] = field(default_factory=list)


def _hours(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 3600.0


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


def _rho(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """Spearman ρ and its p-value over the pairs with both, or (None, None)."""
    both = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(both) < MIN_VALUES or both["x"].nunique() < 2 or both["y"].nunique() < 2:
        return None, None
    rho, p = stats.spearmanr(both["x"], both["y"])
    return (None, None) if pd.isna(rho) else (round(float(rho), 3), float(p))


def _shared(times: pd.Series, anchor: pd.Series) -> bool:
    """Whether the batches sharing a value of the column were started weeks apart."""
    both = pd.DataFrame({"t": times, "a": anchor}).dropna()
    spread = both.groupby("t")["a"].agg(lambda a: (a.max() - a.min()).total_seconds() / 86400)
    shared = spread[both.groupby("t").size() > 1]
    return bool(len(shared)) and float(shared.median()) > SHARED_DAYS


def _round(value, digits: int = 2):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _time_of_day(times: pd.Series) -> str:
    present = times.dropna()
    if present.empty:
        return ""
    minutes = float((present.dt.hour * 60 + present.dt.minute).median())
    return f"{int(minutes // 60):02d}:{int(minutes % 60):02d}"


def _cluster(scaled: pd.DataFrame) -> tuple[np.ndarray | None, float | None, str]:
    """Ward clusters of the rows, by the best mean silhouette: labels 1…k, the silhouette,
    or None and why not."""
    n = len(scaled)
    if n < MIN_VALUES:
        return None, None, f"fewer than {MIN_VALUES} batches with enough of the steps"
    if scaled.drop_duplicates().shape[0] < 3:
        return None, None, "the batches' step durations hardly differ"
    tree = linkage(scaled.to_numpy(), method="ward")
    best: tuple[float, np.ndarray] | None = None
    for k in range(2, min(MAX_CLUSTERS, n - 1) + 1):
        labels = fcluster(tree, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(scaled.to_numpy(), labels))
        if best is None or score > best[0] + 1e-9:
            best = (score, labels)
    if best is None:
        return None, None, "no split of the batches into two or more clusters"
    return best[1], round(best[0], 3), ""


def _runs(labels: list[str], rng: np.random.Generator) -> tuple[int, float, float]:
    """Runs of one label in this order, their mean over shuffles, and the one-sided p of so
    few."""
    codes = pd.factorize(pd.Series(labels))[0]
    count = int(1 + np.count_nonzero(np.diff(codes)))
    shuffled = rng.permuted(np.tile(codes, (PERMUTATIONS, 1)), axis=1)
    chance = 1 + np.count_nonzero(np.diff(shuffled, axis=1), axis=1)
    p = (1 + int((chance <= count).sum())) / (PERMUTATIONS + 1)
    return count, round(float(chance.mean()), 1), p


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
) -> tuple[list[dict], DatesReport]:
    """Every date column, one row each, and the order, steps and clusters of the batches.

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
        report.anchor = ""
    order = anchor.sort_values(kind="stable", na_position="last").index
    frame, anchor = frame.loc[order], anchor.loc[order]
    times = {c: t.loc[order] for c, t in times.items()}
    report.batches = (
        frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    ) or [f"row {i}" for i in frame.index]
    report.dates = anchor.dt.strftime("%Y-%m-%d").fillna("").tolist()
    when = anchor.map(lambda t: t.toordinal() if pd.notna(t) else np.nan).astype(float)

    # The timeline: hours from the anchor, sorted by their median.
    hours: dict[str, pd.Series] = {}
    for column, kind in report.kinds.items():
        count = int(times[column].notna().sum())
        if kind == TIME_ONLY:
            report.not_placed[column] = "a time of day with no date"
        elif count < MIN_VALUES:
            report.not_placed[column] = f"fewer than {MIN_VALUES} values"
        elif column != report.anchor and _shared(times[column], anchor):
            report.not_placed[column] = (
                "shared by batches started weeks apart: the date of a lot or a material"
            )
            report.shared[column] = [_round(v) for v in _hours(anchor - times[column])]
        else:
            hours[column] = _hours(times[column] - anchor)
    report.order = sorted(hours, key=lambda c: (hours[c].median(), c))
    report.hours = {c: [_round(v) for v in hours[c]] for c in report.order}

    k = len(report.order)
    before = np.full((k, k), np.nan)
    for i, a in enumerate(report.order):
        for j, b in enumerate(report.order):
            both = hours[a].notna() & hours[b].notna()
            if i != j and both.sum():
                d = hours[a][both] - hours[b][both]
                before[i, j] = float(((d < 0) + 0.5 * (d == 0)).mean())
    report.before = [[_round(v, 3) for v in row] for row in before]

    # One row per date column.
    rows = []
    position = {c: i + 1 for i, c in enumerate(report.order)}
    for column, kind in report.kinds.items():
        t = times[column]
        present = t.dropna()
        row: dict = {
            "Parameter": column,
            "Kind": kind,
            "Position": position.get(column),
            "Batches with a value": int(len(present)),
            "Coverage %": round(100 * len(present) / len(frame), 1) if len(frame) else 0.0,
            "First": "" if kind == TIME_ONLY or present.empty else f"{present.min():%Y-%m-%d}",
            "Last": "" if kind == TIME_ONLY or present.empty else f"{present.max():%Y-%m-%d}",
            "Median hours from anchor": None,
            "Hours P25": None,
            "Hours P75": None,
            "Out of order %": None,
            "Same as": "",
            "Weekend %": None
            if kind == TIME_ONLY or present.empty
            else round(100 * float((present.dt.weekday >= 5).mean()), 1),
            "Median time of day": "" if kind == DATE_ONLY else _time_of_day(t),
            "Drift ρ": None,
            "Drift p": None,
            "Not placed because": report.not_placed.get(column, ""),
        }
        if column in report.shared:
            age = -pd.Series(report.shared[column], index=frame.index, dtype=float)
            row["Median hours from anchor"] = _round(age.median())
            row["Hours P25"] = _round(age.quantile(0.25))
            row["Hours P75"] = _round(age.quantile(0.75))
            row["Drift ρ"], row["Drift p"] = _rho(when, age)
        if column in hours:
            h = hours[column]
            row["Median hours from anchor"] = _round(h.median())
            row["Hours P25"] = _round(h.quantile(0.25))
            row["Hours P75"] = _round(h.quantile(0.75))
            if column != report.anchor:
                row["Drift ρ"], row["Drift p"] = _rho(when, h)
            i = position[column] - 1
            if i > 0:
                previous = report.order[i - 1]
                both = hours[column].notna() & hours[previous].notna()
                if both.sum():
                    row["Out of order %"] = round(
                        100 * float((hours[column][both] < hours[previous][both]).mean()), 1
                    )
        for other in report.kinds:
            if other == column:
                continue
            both = t.notna() & times[other].notna()
            if both.sum() >= MIN_VALUES and (t[both] == times[other][both]).mean() >= SAME:
                row["Same as"] = other
                break
        rows.append(row)
    rows.sort(key=lambda r: (r["Position"] is None, r["Position"] or 0, r["Parameter"]))

    # The steps between neighbours.
    steps: dict[str, pd.Series] = {}
    for a, b in zip(report.order, report.order[1:]):
        name = f"{a} → {b}"
        report.steps.append((a, b))
        steps[name] = hours[b] - hours[a]
        report.step_hours[name] = [_round(v) for v in steps[name]]
    span = pd.concat([hours[c] for c in report.order], axis=1).max(axis=1) if hours else None
    report.span_of = [_round(v) for v in span] if span is not None else [None] * len(frame)

    # Clusters of batches on their step durations.
    n = len(frame)
    usable = [s for s, v in steps.items() if v.notna().sum() >= MIN_SHARE * n]
    scaled = pd.DataFrame(index=frame.index)
    for step in usable:
        v = steps[step]
        # A deviation counts against the step's typical length: a day late is much in a
        # three-day step and little in a two-month one. Shorter than a day counts as a day.
        if v.nunique() > 1:
            scaled[step] = ((v - v.median()) / max(abs(v.median()), FLOOR_HOURS)).clip(-CLIP, CLIP)
    report.clustered_on = list(scaled.columns)
    report.cluster_of = [""] * n
    report.silhouette_of = [None] * n
    if not report.clustered_on:
        report.unclustered = "no step has a value in half the batches and varies between them"
        labels = None
    else:
        enough = scaled.notna().mean(axis=1) >= MIN_SHARE
        scaled = scaled[enough].fillna(0.0)
        labels, report.silhouette, report.unclustered = _cluster(scaled)
    if labels is not None:
        # Named by their median anchor date, so Timing 1 is the earliest.
        here = pd.Series(labels, index=scaled.index)
        first = {lab: when[here.index[here == lab]].median() for lab in set(labels)}
        named = {
            lab: f"Timing {i + 1}" for i, lab in enumerate(sorted(first, key=lambda x: first[x]))
        }
        widths = silhouette_samples(scaled.to_numpy(), labels)
        where = {idx: i for i, idx in enumerate(frame.index)}
        for idx, lab, width in zip(scaled.index, labels, widths):
            report.cluster_of[where[idx]] = named[lab]
            report.silhouette_of[where[idx]] = round(float(width), 3)
        cluster_of = pd.Series(report.cluster_of, index=frame.index)
        for name in sorted(set(named.values()), key=lambda x: int(x.split()[-1])):
            members = cluster_of.index[cluster_of == name]
            apart = []
            others = cluster_of.index[(cluster_of != name) & (cluster_of != "")]
            for step in report.clustered_on:
                v = steps[step]
                mine, theirs = v[members].median(), v[others].median()
                if pd.isna(mine) or pd.isna(theirs):
                    continue
                distance = (mine - theirs) / max(abs(v.median()), FLOOR_HOURS)
                if abs(distance) >= APART:
                    apart.append(
                        (
                            abs(distance),
                            f"{step} {mine:.1f} h, against {theirs:.1f} h in the other batches",
                        )
                    )
            dated = anchor[members].dropna()
            report.clusters.append(
                Cluster(
                    name=name,
                    batches=len(members),
                    first="" if dated.empty else f"{dated.min():%Y-%m-%d}",
                    last="" if dated.empty else f"{dated.max():%Y-%m-%d}",
                    span=_round(span[members].median()) if span is not None else None,
                    silhouette=round(
                        float(np.nanmean([report.silhouette_of[where[i]] for i in members])), 3
                    ),
                    apart=[text for _, text in sorted(apart, reverse=True)[:3]],
                    profile=[_round(hours[c][members].median()) for c in report.order],
                )
            )
        # Clusters no step sets apart differ by less than matters: one timing, not several.
        if not any(c.apart for c in report.clusters):
            report.unclustered = (
                f"the batches split cleanly (silhouette {report.silhouette:.2f}), but no step "
                f"differs between the groups by {APART:.0%} of its length (at least "
                f"{APART * FLOOR_HOURS:.1f} h): one timing, with small fixed variations"
            )
            report.clusters, report.silhouette = [], None
            report.cluster_of, report.silhouette_of = [""] * n, [None] * n
            labels = None
    if labels is not None:
        cluster_of = pd.Series(report.cluster_of, index=frame.index)
        # Eras or interleaved: runs of one cluster in date order.
        dated = [c for c, d in zip(report.cluster_of, report.dates) if c and d]
        if len(dated) >= MIN_VALUES:
            report.runs, report.runs_expected, report.runs_p = _runs(
                dated, np.random.default_rng(0)
            )
        # The weekday a batch starts on, by cluster.
        start = anchor.dt.weekday
        table = pd.crosstab(cluster_of[cluster_of != ""], start[cluster_of != ""])
        table = table.reindex(columns=range(7), fill_value=0)
        report.weekdays = {name: [int(x) for x in table.loc[name]] for name in table.index}
        observed = table.loc[:, table.sum() > 0]
        if observed.shape[0] > 1 and observed.shape[1] > 1:
            report.weekdays_p = float(stats.chi2_contingency(observed.to_numpy())[1])

    # How the batches sit in calendar time.
    starts = anchor.dropna()
    if len(starts) > 1:
        report.gaps = [
            round(float(g), 2) for g in starts.diff().dropna().dt.total_seconds() / 86400.0
        ]
    if span is not None and not starts.empty:
        end = (anchor + pd.to_timedelta(span.clip(lower=0), unit="h")).dropna()
        first_day, last_day = starts.min().normalize(), end.max().normalize()
        days = pd.date_range(first_day, last_day, freq="D")
        if len(days) <= 5000:
            begin = anchor.dt.normalize().reindex(end.index)
            finish = end.dt.normalize()
            counts = np.zeros(len(days), dtype=int)
            for b, f in zip(begin, finish):
                if pd.notna(b) and pd.notna(f):
                    counts[days.get_loc(b) : days.get_loc(f) + 1] += 1
            report.progress_days = [f"{d:%Y-%m-%d}" for d in days]
            report.progress = counts.tolist()

    report.findings = _findings(report, rows)
    log.success(
        MODULE,
        f"{len(report.kinds):,} date columns, {len(report.order):,} on the timeline; "
        + (
            f"{len(report.clusters)} timing clusters (silhouette {report.silhouette})"
            if report.clusters
            else f"no timing clusters: {report.unclustered or 'none found'}"
        ),
        " → ".join(report.order[:8]) + (" …" if len(report.order) > 8 else ""),
    )
    return rows, report


def _findings(report: DatesReport, rows: list[dict]) -> list[str]:
    """What the dates say, in sentences, most telling first."""
    out: list[str] = []
    count = {kind: sum(k == kind for k in report.kinds.values()) for kind in KINDS}
    out.append(
        f"{len(report.kinds)} date columns: {count[DATE_TIME]} with a time of day, "
        f"{count[DATE_ONLY]} dates only, {count[TIME_ONLY]} times of day with no date. "
        f"{len(report.order)} are placed on each batch's timeline, in hours from "
        + (f"*{report.anchor}*." if report.anchor else "the batch's first timepoint.")
    )
    if report.clusters:
        strength = "weak" if (report.silhouette or 0) < WEAK else "clear"
        out.append(
            f"The batches fall into {len(report.clusters)} timing clusters ({strength}: mean "
            f"silhouette {report.silhouette:.2f}), on {len(report.clustered_on)} step "
            "durations. "
            + " ".join(
                f"**{c.name}** ({c.batches} batch{'' if c.batches == 1 else 'es'}, "
                f"{c.first} to {c.last})"
                + (f": {c.apart[0]}." if c.apart else ": no single step sets it apart.")
                for c in report.clusters
            )
        )
        if report.runs_p is not None:
            if report.runs_p < P_MAX:
                out.append(
                    f"The clusters follow one another in time: {report.runs} runs of one cluster "
                    f"in date order, where shuffles give {report.runs_expected} (p = "
                    f"{report.runs_p:.3f}). The timing changed over the months: a process "
                    "change, a new procedure or a new site practice, not day-to-day variation."
                )
            else:
                out.append(
                    f"The clusters interleave in time ({report.runs} runs of one cluster in date "
                    f"order, shuffles give {report.runs_expected}; p = {report.runs_p:.2f}): "
                    "day-to-day variation rather than a change in the process."
                )
        if report.weekdays_p is not None and report.weekdays_p < P_MAX:
            leaning = []
            for name, counts in report.weekdays.items():
                total = sum(counts)
                if total:
                    day = int(np.argmax(counts))
                    leaning.append(f"{name} {100 * counts[day] / total:.0f}% on {WEEKDAYS[day]}")
            out.append(
                "The weekday a batch starts on goes with its cluster (χ² p = "
                f"{report.weekdays_p:.3f}): " + ", ".join(leaning) + ". A weekend inside the "
                "process lengthens the steps it falls on."
            )
    elif report.unclustered:
        out.append(f"No timing clusters: {report.unclustered}.")
    disordered = [
        r
        for r in rows
        if r["Out of order %"] is not None and r["Out of order %"] >= 100 * OUT_OF_ORDER
    ]
    for r in sorted(disordered, key=lambda r: -r["Out of order %"])[:5]:
        previous = report.order[r["Position"] - 2]
        share = r["Out of order %"]
        out.append(
            f"*{r['Parameter']}* comes before *{previous}* in {share:.0f}% of the batches with "
            "both"
            + (
                ": the two have no fixed order."
                if share >= 100 * NO_ORDER
                else ", though it usually comes after: a recording error, or a step that is not "
                "always done in the same order."
            )
        )
    for r in rows:
        if r["Same as"] and r["Parameter"] < r["Same as"]:
            out.append(f"*{r['Parameter']}* and *{r['Same as']}* hold the same timepoint.")
    for (a, b), (name, values) in zip(report.steps, report.step_hours.items()):
        v = pd.Series(values, dtype=float)
        when = pd.Series(
            [pd.Timestamp(d).toordinal() if d else np.nan for d in report.dates], dtype=float
        )
        rho, p = _rho(when, v)
        if rho is not None and p < P_MAX and abs(rho) >= 0.3:
            out.append(
                f"The step *{name}* {'lengthens' if rho > 0 else 'shortens'} over the months "
                f"(ρ {rho:+.2f} with the date): now a median of "
                f"{v.tail(max(5, len(v) // 5)).median():.1f} h, against "
                f"{v.head(max(5, len(v) // 5)).median():.1f} h at first."
            )
    for column, ages in report.shared.items():
        age = pd.Series(ages, dtype=float) / 24.0
        if age.notna().sum():
            when = pd.Series(
                [pd.Timestamp(d).toordinal() if d else np.nan for d in report.dates], dtype=float
            )
            rho, p = _rho(when, age)
            out.append(
                f"*{column}* is the date of a lot or a material, shared by batches started weeks "
                f"apart: a median of {age.median():.0f} days before the anchor (from "
                f"{age.min():.0f} to {age.max():.0f})"
                + (
                    f", {'older' if rho > 0 else 'younger'} at use over the months (ρ {rho:+.2f})."
                    if rho is not None and p < P_MAX and abs(rho) >= 0.3
                    else "."
                )
                + " It is kept out of the steps and the clusters."
            )
    anchored = next((r for r in rows if r["Parameter"] == report.anchor), None)
    base = anchored["Weekend %"] if anchored and anchored["Weekend %"] is not None else None
    weekend = [
        r
        for r in rows
        if r["Parameter"] in report.order
        and r["Parameter"] != report.anchor
        and (r["Weekend %"] or 0) >= (base + WEEKEND_POINTS if base is not None else 10)
    ]
    if weekend:
        out.append(
            "On a weekend more often than the anchor"
            + (f" ({base:.0f}% of the batches)" if base is not None else "")
            + ": "
            + ", ".join(f"*{r['Parameter']}* {r['Weekend %']:.0f}%" for r in weekend[:8])
            + ". A weekend inside the process is a step done by a weekend crew, or one that waits."
        )
    elif base is not None:
        out.append(
            f"The anchor falls on a weekend in {base:.0f}% of the batches, and no later "
            "timepoint does so more often."
        )
    if report.progress:
        busiest = max(report.progress)
        out.append(
            f"Up to {busiest} batches are in progress at once (anchor to last timepoint)"
            + (
                f"; a median of {np.median(report.gaps):.1f} days between consecutive starts, "
                f"{sum(g < 1 for g in report.gaps)} starts on the same day as the one before."
                if report.gaps
                else "."
            )
        )
    if report.not_placed:
        out.append(
            "Not on the timeline: "
            + ", ".join(f"*{c}* ({why})" for c, why in report.not_placed.items())
            + "."
        )
    return out


def batch_rows(report: DatesReport) -> list[dict]:
    """One row per batch: its anchor date, cluster, silhouette and span; what dates.xlsx
    keeps."""
    return [
        {
            "Batch": batch,
            "Anchor date": date,
            "Cluster": cluster,
            "Silhouette": width,
            "Span hours": span,
            **{f"{step} (h)": report.step_hours[step][i] for step in report.step_hours},
        }
        for i, (batch, date, cluster, width, span) in enumerate(
            zip(
                report.batches,
                report.dates,
                report.cluster_of,
                report.silhouette_of,
                report.span_of,
            )
        )
    ]


def step_rows(report: DatesReport) -> list[dict]:
    """One row per step: its duration's spread, how often it is negative, and its drift."""
    when = pd.Series(
        [pd.Timestamp(d).toordinal() if d else np.nan for d in report.dates], dtype=float
    )
    rows = []
    for (a, b), (name, values) in zip(report.steps, report.step_hours.items()):
        v = pd.Series(values, dtype=float)
        present = v.dropna()
        rho, p = _rho(when, v)
        rows.append(
            {
                "Step": name,
                "From": a,
                "To": b,
                "Batches with both": int(len(present)),
                "Median hours": _round(present.median()) if len(present) else None,
                "Hours P25": _round(present.quantile(0.25)) if len(present) else None,
                "Hours P75": _round(present.quantile(0.75)) if len(present) else None,
                "Shortest": _round(present.min()) if len(present) else None,
                "Longest": _round(present.max()) if len(present) else None,
                "Negative %": round(100 * float((present < 0).mean()), 1) if len(present) else None,
                "Drift ρ": rho,
                "Drift p": None if p is None else float(f"{p:.3g}"),
                "Clustered on": "yes" if name in report.clustered_on else "",
            }
        )
    return rows
