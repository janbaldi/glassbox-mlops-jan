"""Every figure in the report, as plotly JSON.

One function per figure, each returning ``None`` when there is nothing to draw,
so the section it belongs to drops out rather than rendering an empty axis.
This module imports :mod:`ml.blocks` and nothing else from the project.

Colour is spent only where it carries something: everything single-series is
neutral ink. The colours and sizes are the CAR-T manufacturing
dashboard's, so a figure here reads like one there.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .blocks import Block, figure

INK = "#2c3e50"
MUTED = "#9aa5b1"
GRID = "rgba(44,62,80,0.12)"
#: the dashboard's categorical palette: no dark blue, red, pink or orange
CATEGORY_COLOURS = ("#7a8b00", "#35a02c", "#e5b400", "#12a4b6", "#6a3fb5", "#8c6239")
#: correlation, -1 to 1: orange through near-white to blue (no red)
DIVERGING = [[0.0, "#eb6834"], [0.5, "#f4f6f8"], [1.0, "#0047ab"]]


def _fig(fig: go.Figure, caption: str = "") -> Block:
    """The one choke point every figure passes through.

    ``to_json`` rather than ``to_dict`` because it is what encodes numpy and
    pandas values to plain JSON, and the payload has to survive a round trip
    through a script tag. The default template is dropped: it serialises trace
    defaults for every trace type this plotly knows about, and the plotly in the
    browser rejects the ones it has since renamed.
    """
    spec = json.loads(fig.to_json())
    spec.get("layout", {}).pop("template", None)
    spec.setdefault("layout", {}).setdefault("font", {}).setdefault("size", 11)
    return figure(spec, caption)


def _style(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 360) -> go.Figure:
    """Layout shared by every figure: no background, so the page theme shows through.

    The title is pinned to the top edge and a legend, when there is one, gets a
    row of its own between the title and the plot. Placed inside the title's
    margin, the two are drawn over each other.
    """
    legend = bool(fig.layout.showlegend)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13),
            x=0,
            xanchor="left",
            y=1,
            yref="container",
            yanchor="top",
            pad=dict(t=10),
        ),
        xaxis_title=x,
        yaxis_title=y,
        height=height + (28 if legend else 0),
        margin=dict(l=10, r=10, t=72 if legend else 44, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
def cluster_fits(cluster) -> Block | None:
    """How well each member is explained by the parameters it was fitted on.

    Observed against predicted, not the member against the representative: a
    sequential residual is fitted on several parameters at once, so a single
    curve drawn against one of them would suggest a relationship that was never
    fitted.

    Members are drawn in units of their own spread, because a cell count in the
    billions and a slope around one cannot share an axis — on a common scale the
    smaller one collapses onto the origin and reads as a perfect fit. The axes
    say so, and the hover text carries the measured values.
    """
    fits = [fit for fit in cluster.fits if fit.y and fit.fitted]
    if not fits:
        return None

    palette = CATEGORY_COLOURS
    figure_ = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Observed against predicted, on the rows the fit used",
            "Residual against predicted — what the dataset gets",
        ),
    )

    for index, fit in enumerate(fits):
        colour = palette[index % len(palette)]
        short = fit.column if len(fit.column) <= 34 else fit.column[:33] + "…"
        detail = f"{fit.model}; {fit.rows} batches"
        observed = np.asarray(fit.y, dtype="float64")
        predicted = np.asarray(fit.fitted, dtype="float64")
        residual = np.asarray(fit.residual, dtype="float64")
        centre = float(np.mean(observed))
        spread = float(np.std(observed)) or 1.0

        figure_.add_scatter(
            x=(predicted - centre) / spread,
            y=(observed - centre) / spread,
            mode="markers",
            name=short,
            legendgroup=short,
            marker=dict(color=colour, size=6, opacity=0.65),
            customdata=np.column_stack([predicted, observed]),
            hovertemplate=(
                f"<b>{fit.column}</b><br>predicted %{{customdata[0]:.4g}}, "
                f"observed %{{customdata[1]:.4g}}<extra>{detail}</extra>"
            ),
            row=1,
            col=1,
        )
        figure_.add_scatter(
            x=(predicted - centre) / spread,
            y=residual / spread,
            mode="markers",
            name=short,
            legendgroup=short,
            showlegend=False,
            marker=dict(color=colour, size=6, opacity=0.65),
            customdata=np.column_stack([predicted, residual]),
            hovertemplate=(
                f"<b>{fit.column}</b><br>predicted %{{customdata[0]:.4g}}, "
                f"residual %{{customdata[1]:.4g}}<extra>{detail}</extra>"
            ),
            row=1,
            col=2,
        )

    figure_.add_shape(
        type="line",
        x0=-3,
        y0=-3,
        x1=3,
        y1=3,
        line=dict(color=MUTED, width=1, dash="dash"),
        row=1,
        col=1,
    )
    figure_.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=2)
    figure_.update_xaxes(
        title_text="predicted, in standard deviations of the member",
        showgrid=False,
        zeroline=False,
    )
    figure_.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure_.update_yaxes(title_text="observed (standardised)", row=1, col=1)
    figure_.update_yaxes(title_text="residual (standardised)", row=1, col=2)
    figure_.update_layout(
        title=dict(
            text=f"{cluster.name} — members against their fits",
            font=dict(size=13),
            x=0,
            xanchor="left",
        ),
        height=460,
        # Room under the axis titles for the legend, which is a row of parameter
        # names and would otherwise be drawn straight through them.
        margin=dict(l=10, r=10, t=76, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
    )
    return _fig(
        figure_,
        "Each member is standardised by its own spread so they can share an axis; the "
        "measured values are in the hover text. Points on the dashed line are fitted "
        "exactly, and the right-hand cloud is what the dataset receives.",
    )


def cluster_correlation(cluster) -> Block | None:
    """The cluster's correlation matrix before and after, signed.

    Colour is the axis here — the sign of a correlation is half of what it says —
    which is the case where a diverging scale earns its place. A pair with too
    few shared batches is left blank rather than drawn as zero, because those are
    different statements.
    """
    before, after = cluster.corr_matrix_before, cluster.corr_matrix_after
    if not before:
        return None

    figure_ = make_subplots(rows=1, cols=2, subplot_titles=("Before", "After"))
    names = [name if len(name) <= 40 else name[:39] + "…" for name in cluster.members]
    codes_before = [str(i + 1) for i in range(len(before))]
    codes_after = [str(i + 1) for i in range(len(after))]
    overlap = cluster.overlap_matrix or [[0] * len(before) for _ in before]
    hover = [
        [
            f"{names[row]}<br>vs {names[column]}<br>"
            + (
                "no supported estimate"
                if before[row][column] is None
                else f"r = {before[row][column]:+.2f}"
            )
            + f"<br>{overlap[row][column]} shared batches"
            for column in range(len(before))
        ]
        for row in range(len(before))
    ]

    figure_.add_heatmap(
        z=before,
        x=codes_before,
        y=codes_before,
        zmin=-1,
        zmax=1,
        colorscale=DIVERGING,
        showscale=False,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        row=1,
        col=1,
    )
    if after:
        labels_after = [
            name if len(name) <= 40 else name[:39] + "…" for name in cluster.labels_after
        ]
        after_hover = [
            [
                f"{labels_after[row]}<br>vs {labels_after[column]}<br>"
                + (
                    "no supported estimate"
                    if after[row][column] is None
                    else f"r = {after[row][column]:+.2f}"
                )
                for column in range(len(after))
            ]
            for row in range(len(after))
        ]
        figure_.add_heatmap(
            z=after,
            x=codes_after,
            y=codes_after,
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING,
            colorbar=dict(title="r", thickness=12),
            text=after_hover,
            hovertemplate="%{text}<extra></extra>",
            row=1,
            col=2,
        )

    figure_.update_yaxes(autorange="reversed", showgrid=False)
    figure_.update_xaxes(showgrid=False)
    figure_.update_layout(
        title=dict(
            text=f"{cluster.name} — {cluster.action} · correlation within the cluster",
            font=dict(size=13),
            x=0,
            xanchor="left",
        ),
        height=340,
        margin=dict(l=10, r=10, t=76, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return _fig(
        figure_,
        "Blank means too few shared batches to estimate, which is not the same as no "
        "correlation. Axis numbers are the parameters listed below; full names are in the "
        "hover text.",
    )


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------
def target_distribution(values, name: str, unit: str = "", classes=None) -> Block | None:
    """What the thing being predicted actually looks like across the cohort."""
    numbers = [float(v) for v in values if v == v]
    if not numbers:
        return None

    if classes:
        counts = {label: numbers.count(float(code)) for code, label in classes.items()}
        figure_ = go.Figure(
            go.Bar(
                x=list(counts),
                y=list(counts.values()),
                marker_color=INK,
                hovertemplate="%{x}: %{y} batches<extra></extra>",
            )
        )
        return _fig(
            _style(figure_, f"{name} — class balance", y="batches", height=300),
            f"{counts.get(list(counts)[-1], 0)} of {len(numbers)} batches are the positive class.",
        )

    figure_ = go.Figure(
        go.Histogram(x=numbers, marker_color=INK, hovertemplate="%{x}: %{y}<extra></extra>")
    )
    return _fig(
        _style(
            figure_,
            f"{name} — distribution over the cohort",
            x=unit or name,
            y="batches",
            height=300,
        ),
        f"{len(numbers)} batches, from {min(numbers):.4g} to {max(numbers):.4g}.",
    )


def missingness(rows, limit: int = 25) -> Block | None:
    """The columns with the least data, worst first."""
    worst = sorted(rows, key=lambda row: row["missing_pct"], reverse=True)[:limit]
    worst = [row for row in worst if row["missing_pct"] > 0]
    if not worst:
        return None
    labels = [
        row["column"] if len(row["column"]) <= 42 else row["column"][:41] + "…" for row in worst
    ]
    figure_ = go.Figure(
        go.Bar(
            x=[row["missing_pct"] for row in worst],
            y=labels,
            orientation="h",
            marker_color=INK,
            customdata=[[row["column"], row["present"]] for row in worst],
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% missing, "
            "%{customdata[1]} batches with a value<extra></extra>",
        )
    )
    figure_ = _style(
        figure_,
        f"The {len(worst)} columns with the most missing values",
        x="% of batches with no value",
        height=max(300, 22 * len(worst)),
    )
    figure_.update_yaxes(autorange="reversed")
    return _fig(figure_, "Full names are in the hover text.")


def encoder_split(routes) -> Block | None:
    """How many categorical parameters went to each encoder."""
    if not routes:
        return None
    order = ["binary", "one_hot", "target", "hashing", "skip"]
    labels = {
        "binary": "binary",
        "one_hot": "one-hot",
        "target": "target encoding",
        "hashing": "hashing",
        "skip": "left out",
    }
    counts = [sum(1 for r in routes if r.strategy == strategy) for strategy in order]
    if not any(counts):
        return None

    figure_ = go.Figure(
        go.Bar(
            x=[labels[s] for s in order],
            y=counts,
            marker_color=INK,
            hovertemplate="%{x}: %{y} parameters<extra></extra>",
        )
    )
    return _fig(
        _style(figure_, "Categorical parameters per encoder", y="parameters", height=300),
        "Every categorical parameter goes to exactly one of these.",
    )


def missing_pattern(group) -> Block | None:
    """Which batches miss which member of a group: one row per member, one column per batch."""
    if not group.pattern:
        return None
    names = [name if len(name) <= 40 else name[:39] + "…" for name in group.members]
    figure_ = go.Figure(
        go.Heatmap(
            z=group.pattern,
            x=list(range(1, len(group.batch_labels) + 1)),
            y=names,
            colorscale=[[0.0, "#eef1f4"], [1.0, INK]],
            zmin=0,
            zmax=1,
            showscale=False,
            customdata=[group.batch_labels for _ in group.members],
            hovertemplate="%{y}<br>batch %{customdata}<br>missing = %{z}<extra></extra>",
        )
    )
    figure_ = _style(
        figure_,
        f"{group.name} — missing values per batch",
        x="batch (in the order shown in the caption)",
        height=max(260, 26 * len(group.members) + 120),
    )
    figure_.update_yaxes(autorange="reversed")
    return _fig(figure_, "Dark cells are missing values.")


def missing_by_category(group) -> Block | None:
    """How often the group is missing, per category of its best explainer."""
    explainer = group.explainer
    if explainer is None or not explainer.rates:
        return None
    categories = list(explainer.rates)
    figure_ = go.Figure(
        go.Bar(
            x=categories,
            y=[100 * explainer.rates[c] for c in categories],
            marker_color=INK,
            customdata=[explainer.batches.get(c, 0) for c in categories],
            hovertemplate="%{x}: %{y:.0f}% of %{customdata} batches missing<extra></extra>",
        )
    )
    return _fig(
        _style(
            figure_,
            f"{group.name} — how often the group is missing, by {explainer.column}",
            y="% of batches missing at least one member",
            height=320,
        )
    )


# ---------------------------------------------------------------------------
# The dates report
# ---------------------------------------------------------------------------
#: A light band behind every other calendar day.
DAY_BAND = "rgba(44,62,80,0.05)"


def _short(name: str, length: int = 40) -> str:
    return name if len(name) <= length else name[: length - 1] + "…"


def _jitter(count: int, seed: int = 0, width: float = 0.28) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, count)


def _day_bands(figure_: go.Figure, low: float, high: float) -> None:
    """A light band behind every other calendar day, and each day named on the x-axis, in
    the middle of its day."""
    first, last = int(np.floor(low)), int(np.ceil(high))
    step = max(1, (last - first) // 20)
    for day in range(first, last):
        if (day - first) % 2 == 0:
            figure_.add_vrect(x0=day, x1=day + 1, fillcolor=DAY_BAND, line_width=0, layer="below")
    named = list(range(first, last, step))
    figure_.update_xaxes(
        tickvals=[d + 0.5 for d in named], ticktext=[f"day {d}" for d in named], range=[first, last]
    )


def _ticks_by_hour(low: float, high: float) -> tuple[list[float], list[str]]:
    """Clock ticks every three hours over a range that may run past midnight."""
    start, stop = int(np.floor(low / 3) * 3), int(np.ceil(high / 3) * 3)
    values = list(range(start, stop + 1, 3))
    return values, [f"{v % 24:02d}:00" for v in values]


def _clouds(figure_: go.Figure, rows: list[tuple[str, list, list]], seed: int) -> None:
    """Every batch as a grey dot, one WebGL trace per row, so a thousand batches over thirty
    columns draws at once. ``rows`` is (name, value per batch, hover label per batch); the
    hover shows the label and the row's name."""
    for row, (name, values, labels) in enumerate(rows):
        jitter = _jitter(len(values), seed=seed + row)
        present = [i for i, v in enumerate(values) if v is not None]
        figure_.add_trace(
            go.Scattergl(
                x=[round(values[i], 3) for i in present],
                y=[round(row + jitter[i], 2) for i in present],
                mode="markers",
                marker=dict(size=5, color=MUTED, opacity=0.55),
                name="one batch",
                legendgroup="batches",
                showlegend=row == 0,
                text=[labels[i] for i in present],
                hovertemplate="%{text}<extra>" + _short(name) + "</extra>",
            )
        )


def _spreads(figure_: go.Figure, boxes: list[tuple[int, str, list]], median: list[str]) -> None:
    """The 10th–90th percentile as a thin line, the middle half as a bar, the median as a
    ringed dot, for every row at once: three traces, however many rows. ``boxes`` is (row,
    name, [p10, p25, p50, p75, p90]); ``median`` the median of each, as hover text. WebGL
    like the dots: a browser draws WebGL above SVG, so SVG marks would sit under them."""

    def segments(low: int, high: int) -> tuple[list, list]:
        xs, ys = [], []
        for row, _, q in boxes:
            xs += [q[low], q[high], None]
            ys += [row, row, None]
        return xs, ys

    for (low, high), width in (((0, 4), 1.5), ((1, 3), 8)):
        xs, ys = segments(low, high)
        figure_.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=INK, width=width),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure_.add_trace(
        go.Scattergl(
            x=[q[2] for _, _, q in boxes],
            y=[row for row, _, _ in boxes],
            mode="markers",
            marker=dict(size=10, color=INK, line=dict(color="white", width=2)),
            name="median, middle half, 10th–90th percentile",
            text=[f"<b>{name}</b><br>{m}" for (_, name, _), m in zip(boxes, median)],
            hovertemplate="%{text}<extra></extra>",
        )
    )


def date_timeline(report) -> Block | None:
    """Every date column on one timeline: every batch as a dot, the middle half as a bar,
    four batches in five between the whiskers, the median as a ringed dot."""
    if not report.order:
        return None
    names = [_short(c) for c in report.order]
    figure_ = go.Figure()
    _clouds(figure_, [(c, report.days[c], report.batches) for c in report.order], seed=0)
    boxes = [
        (row, column, report.quantiles[column])
        for row, column in enumerate(report.order)
        if report.quantiles[column][2] is not None
    ]
    _spreads(
        figure_,
        boxes,
        [
            f"median day {q[2]:.2f}; middle half {q[1]:.2f} to {q[3]:.2f}; usual day "
            f"{report.usual_day.get(c, '')}"
            for _, c, q in boxes
        ],
    )
    low = min(q[0] for _, _, q in boxes)
    high = max(q[4] for _, _, q in boxes)
    _day_bands(figure_, min(low, 0), high + 0.5)
    figure_.update_layout(showlegend=True)
    figure_ = _style(
        figure_,
        "Every date column on the batch's timeline",
        x="calendar days from the anchor's (day 0)",
        height=max(300, 34 * len(names) + 140),
    )
    figure_.update_yaxes(
        tickvals=list(range(len(names))),
        ticktext=names,
        autorange="reversed",
        showgrid=False,
        zeroline=False,
    )
    figure_.update_xaxes(showgrid=False)
    return _fig(
        figure_,
        "Top to bottom in the order the process runs in. A grey dot is one batch (hover for "
        "it); the bar is the middle half of the batches, the thin line four in five, the "
        "ringed dot the median. The bands are calendar days.",
    )


def _hhmm(hours: float) -> str:
    minutes = int(round(hours * 60)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def date_clock(report) -> Block | None:
    """The time of day of every column with one: shifts, nights, and steps done any time.
    The hours come unwrapped (a step done around midnight runs past 24), as the report keeps
    them."""
    # Earliest median time of day at the top. The hours are unwrapped, so a step done around
    # midnight sorts by its evening start rather than by the small hours.
    medians = {
        c: float(np.median(present))
        for c, values in report.clock.items()
        if (present := [v for v in values if v is not None])
    }
    columns = sorted(medians, key=lambda c: (medians[c], c))
    if not columns:
        return None
    figure_ = go.Figure()
    _clouds(
        figure_,
        [
            (
                c,
                report.clock[c],
                [
                    "" if v is None else f"{b} {_hhmm(v)}"
                    for b, v in zip(report.batches, report.clock[c])
                ],
            )
            for c in columns
        ],
        seed=100,
    )
    boxes = []
    for row, column in enumerate(columns):
        present = [v for v in report.clock[column] if v is not None]
        if present:
            boxes.append(
                (row, column, [float(q) for q in np.quantile(present, [0.1, 0.25, 0.5, 0.75, 0.9])])
            )
    _spreads(figure_, boxes, [f"median {_hhmm(q[2])}" for _, _, q in boxes])
    values_all = [v for c in columns for v in report.clock[c] if v is not None]
    low, high = (min(values_all), max(values_all)) if values_all else (0, 24)
    for start, end in ((low - 1, 6), (20, high + 1), (low - 1 + 24, 6 + 24)):
        if end > start and end > low and start < high:
            figure_.add_vrect(
                x0=max(start, low - 1),
                x1=min(end, high + 1),
                fillcolor=DAY_BAND,
                line_width=0,
                layer="below",
            )
    values, labels = _ticks_by_hour(low, high)
    figure_.update_layout(showlegend=True)
    figure_ = _style(
        figure_, "The time of day of each column", height=max(300, 34 * len(columns) + 140)
    )
    figure_.update_yaxes(
        tickvals=list(range(len(columns))),
        ticktext=[
            _short(c) + (" (time only)" if report.kinds.get(c) == "time of day only" else "")
            for c in columns
        ],
        autorange="reversed",
        showgrid=False,
    )
    figure_.update_xaxes(tickvals=values, ticktext=labels, showgrid=True, gridcolor=GRID)
    return _fig(
        figure_,
        "Clock time, earliest median at the top; the shaded hours are 20:00 to 06:00. A column "
        "done around midnight is drawn across it rather than split at 00:00. *(time only)* marks "
        "a column that holds a time of day with no date.",
    )
