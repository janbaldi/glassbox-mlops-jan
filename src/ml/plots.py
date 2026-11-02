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
#: Timing clusters, in fixed order (Timing 1 is always the first hue). Validated with the
#: dataviz palette checks: lightness band, chroma, colour-blind and normal-vision separation
#: of neighbours. Three fall below 3:1 against a white page, so every figure names its
#: clusters in the legend and the hover, and the cluster table repeats them.
TIMING_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
#: Sequential, light to dark, one hue: a share from 0 to 1.
SEQUENTIAL_BLUE = [[0.0, "#f4f6f8"], [0.5, "#86b6ef"], [1.0, "#104281"]]


def _short(name: str, length: int = 40) -> str:
    return name if len(name) <= length else name[: length - 1] + "…"


def _unit(report) -> tuple[str, float]:
    """Hours, or days when the batches' timelines run beyond four days."""
    spans = [s for s in report.span_of if s is not None]
    return ("days", 24.0) if spans and float(np.median(spans)) > 96 else ("hours", 1.0)


def _colour_of(report) -> dict[str, str]:
    return {c.name: TIMING_COLOURS[i % len(TIMING_COLOURS)] for i, c in enumerate(report.clusters)}


def timing_profiles(report) -> Block | None:
    """Each batch's timeline through the sorted columns, by cluster, with each cluster's
    median in bold: what the clusters look like over the process."""
    if len(report.order) < 2:
        return None
    unit, per = _unit(report)
    names = [_short(c) for c in report.order]
    colours = _colour_of(report)
    groups = [(c.name, colours[c.name]) for c in report.clusters] or [("All batches", INK)]
    figure_ = go.Figure()
    for name, colour in groups:
        xs, ys, text = [], [], []
        for i, batch in enumerate(report.batches):
            if report.clusters and report.cluster_of[i] != name:
                continue
            for column, label in zip(report.order, names):
                value = report.hours[column][i]
                if value is not None:
                    xs.append(value / per)
                    ys.append(label)
                    text.append(f"{batch} · {report.dates[i]}")
            xs.append(None)
            ys.append(None)
            text.append("")
        figure_.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=colour, width=1),
                opacity=0.25,
                legendgroup=name,
                showlegend=False,
                text=text,
                hovertemplate=f"%{{text}}<br>%{{y}}: %{{x:.1f}} {unit}<extra>{name}</extra>",
            )
        )
    medians = (
        [(c.name, colours[c.name], c.profile, c.batches) for c in report.clusters]
        if report.clusters
        else [
            (
                "All batches",
                INK,
                [
                    float(np.median([v for v in report.hours[c] if v is not None]))
                    for c in report.order
                ],
                len(report.batches),
            )
        ]
    )
    for name, colour, profile, count in medians:
        figure_.add_trace(
            go.Scatter(
                x=[None if v is None else v / per for v in profile],
                y=names,
                mode="lines+markers",
                line=dict(color=colour, width=2.5),
                marker=dict(size=8, color=colour, line=dict(color="white", width=1.5)),
                name=f"{name} ({count})",
                legendgroup=name,
                hovertemplate=f"{name} median<br>%{{y}}: %{{x:.1f}} {unit}<extra></extra>",
            )
        )
    figure_.update_layout(showlegend=True)
    figure_ = _style(
        figure_,
        "Each batch's timeline, by timing cluster",
        x=f"{unit} from the anchor",
        height=max(300, 30 * len(names) + 120),
    )
    figure_.update_yaxes(autorange="reversed", showgrid=False)
    figure_.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=MUTED)
    return _fig(
        figure_,
        "One thin line per batch, through its timepoints in the sorted order; the bold line "
        "is the cluster's median. Where the bold lines part, the clusters differ.",
    )


def timing_calendar(report) -> Block | None:
    """When each cluster's batches ran: one lane per cluster, one dot per batch at its
    anchor date."""
    dated = [i for i, d in enumerate(report.dates) if d]
    if len(dated) < 2 or not report.clusters:
        return None
    unit, per = _unit(report)
    colours = _colour_of(report)
    lanes = [(c.name, colours[c.name]) for c in report.clusters]
    if any(not report.cluster_of[i] for i in dated):
        lanes.append(("Not clustered", MUTED))
    figure_ = go.Figure()
    for name, colour in lanes:
        mine = [i for i in dated if (report.cluster_of[i] or "Not clustered") == name]
        if not mine:
            continue
        figure_.add_trace(
            go.Scatter(
                x=[report.dates[i] for i in mine],
                y=[name] * len(mine),
                mode="markers",
                marker=dict(
                    size=10, color=colour, symbol="line-ns", line=dict(color=colour, width=2)
                ),
                name=f"{name} ({len(mine)})",
                text=[report.batches[i] for i in mine],
                customdata=[
                    "" if report.span_of[i] is None else f"{report.span_of[i] / per:.1f} {unit}"
                    for i in mine
                ],
                hovertemplate="%{text} · %{x}<br>anchor to last timepoint %{customdata}"
                f"<extra>{name}</extra>",
            )
        )
    figure_.update_layout(showlegend=True)
    figure_ = _style(
        figure_, "When each timing cluster ran", height=max(200, 46 * len(lanes) + 110)
    )
    figure_.update_yaxes(autorange="reversed", showgrid=False)
    figure_.update_xaxes(showgrid=True, gridcolor=GRID)
    return _fig(
        figure_,
        "One tick per batch at its anchor date, one lane per cluster. Lanes that fill "
        "stretches of the calendar one after the other are eras: the timing changed. Lanes "
        "that run side by side vary from batch to batch.",
    )


def step_calendar(report, limit: int = 12) -> Block | None:
    """Each step's duration over calendar time, by cluster: small multiples."""
    steps = report.clustered_on or list(report.step_hours)
    if not steps:
        return None
    steps = steps[:limit]
    colours = _colour_of(report)
    columns = 3 if len(steps) > 4 else 2 if len(steps) > 1 else 1
    rows = int(np.ceil(len(steps) / columns))
    figure_ = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[_short(s, 48) for s in steps],
        vertical_spacing=min(0.12, 0.6 / rows),
        horizontal_spacing=0.06,
    )
    groups = [(c.name, colours[c.name]) for c in report.clusters] or [("All batches", INK)]
    for n, step in enumerate(steps):
        row, col = n // columns + 1, n % columns + 1
        values = report.step_hours[step]
        for name, colour in groups:
            mine = [
                i
                for i, v in enumerate(values)
                if v is not None
                and report.dates[i]
                and (not report.clusters or report.cluster_of[i] == name)
            ]
            figure_.add_trace(
                go.Scatter(
                    x=[report.dates[i] for i in mine],
                    y=[values[i] for i in mine],
                    mode="markers",
                    marker=dict(size=8, color=colour, line=dict(color="white", width=1)),
                    name=name,
                    legendgroup=name,
                    showlegend=n == 0,
                    text=[report.batches[i] for i in mine],
                    hovertemplate=f"%{{text}} · %{{x}}<br>{_short(step, 60)}: %{{y:.1f}} h"
                    f"<extra>{name}</extra>",
                ),
                row=row,
                col=col,
            )
    figure_.update_layout(showlegend=True)
    figure_ = _style(figure_, "Each step's duration over the months", height=230 * rows + 40)
    # The panel titles sit at the top of the plot area: the legend goes above them, just
    # under the figure's title.
    figure_.update_layout(
        legend=dict(y=1 - 40 / (230 * rows + 68), yref="container", yanchor="top"),
        margin=dict(t=96),
    )
    figure_.update_yaxes(showgrid=True, gridcolor=GRID)
    for row in range(1, rows + 1):
        figure_.update_yaxes(title_text="hours", row=row, col=1)
    figure_.update_annotations(font_size=11)
    return _fig(
        figure_,
        "One panel per step (hours between two columns next to each other in the sorted order), "
        "one dot per batch at its anchor date. A cloud that rises or falls is a step whose "
        "length changed over time.",
    )


def timing_order(report) -> Block | None:
    """How often each column comes before each other: the order the process runs in."""
    if len(report.order) < 2:
        return None
    names = [_short(c, 32) for c in report.order]
    z = [[None if v is None else 100 * v for v in row] for row in report.before]
    figure_ = go.Figure(
        go.Heatmap(
            z=z,
            x=names,
            y=names,
            colorscale=SEQUENTIAL_BLUE,
            zmin=0,
            zmax=100,
            colorbar=dict(title="%", thickness=10),
            text=[["" if v is None else f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}" if len(names) <= 14 else None,
            hovertemplate="%{y} before %{x}: %{z:.0f}% of the batches with both<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    figure_ = _style(
        figure_,
        "How often the row comes before the column",
        height=max(320, 26 * len(names) + 160),
    )
    figure_.update_yaxes(autorange="reversed", showgrid=False)
    figure_.update_xaxes(tickangle=-40)
    return _fig(
        figure_,
        "Columns in the sorted order. Above the diagonal should be dark (100%: the row "
        "always comes first); a light cell there is a pair that swaps places in some batches.",
    )


def in_progress(report) -> Block | None:
    """Batches in progress per day: how the batches overlap in calendar time."""
    if not report.progress:
        return None
    figure_ = go.Figure(
        go.Scatter(
            x=report.progress_days,
            y=report.progress,
            mode="lines",
            line=dict(color=INK, width=2, shape="hv"),
            hovertemplate="%{x}: %{y} batches in progress<extra></extra>",
        )
    )
    figure_ = _style(figure_, "Batches in progress per day", y="batches", height=280)
    figure_.update_yaxes(rangemode="tozero")
    return _fig(
        figure_,
        "A batch is in progress from its anchor date to the day of its last timepoint.",
    )
