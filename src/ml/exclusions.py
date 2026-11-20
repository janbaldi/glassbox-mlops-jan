"""What a new task leaves out of its dataset, and why: one report, on screen and in a file.

``ml task new`` leaves parameters out for several reasons at once: the screening
register, the catalog's Variable Role and Functional Type, the dates and shifts a build
found, and — when its usable inputs are worked out — the task's roles, the target's
lineage and the site-relative features. This module gathers them into groups, one per
reason, and renders them as plain text: the console prints it with colour, and
``task_new.log`` in the task folder keeps it without.

A parameter left out for several reasons is listed in every group that leaves it out,
with the other groups named beside it, so each group is complete and each line says
the whole story.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

#: Run-time stages of :func:`ml.dataset.candidate_predictors`, as groups of the report.
#: "task exclusion" and "screening" are not here: they are the task file's own groups.
STAGES = {
    "role": (
        "Task roles",
        "the target, identifier, grouping and patient columns describe a batch; they are "
        "kept as metadata, not used as predictors",
    ),
    "category exclusion": (
        "Catalog rules (exclude_by_catalog / include_by_catalog)",
        "the task file leaves these catalog categories out by rule",
    ),
    "target lineage": (
        "Computed from the target",
        "derived from the column the task predicts, so they would leak it",
    ),
    "site-relative": (
        "Site-relative features",
        "standardised within their own site's batches, so their value depends on which "
        "batches were processed with them",
    ),
    "include list": (
        "Not in predictors.include",
        "the task file lists the only predictors it uses, and these are not among them",
    ),
}


@dataclass
class Group:
    """One reason for leaving parameters out, and the parameters it leaves out."""

    title: str
    why: str
    #: (parameter, the detail particular to it, or "" when ``why`` says it all)
    rows: list[tuple[str, str]] = field(default_factory=list)
    #: Lines said about the group as a whole (a rule that matched nothing, a conflict).
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    """Every group, and the counts that frame them."""

    task: str
    file: str
    config: str
    site: str
    target: str
    groups: list[Group] = field(default_factory=list)
    #: Catalog parameters, those in the batch table, and those usable as predictors;
    #: None when the usable inputs could not be worked out.
    catalog_parameters: int | None = None
    in_batch_table: int | None = None
    usable: int | None = None
    #: Whatever kept the report from being complete, in words.
    caveats: list[str] = field(default_factory=list)
    made: datetime = field(default_factory=datetime.now)

    def excluded(self) -> list[str]:
        """Every parameter some group leaves out, once each, in the order first met."""
        return list(dict.fromkeys(p for g in self.groups for p, _ in g.rows))


def build(
    *,
    task: str,
    file: str,
    config: str,
    site: str,
    target: str,
    screened: list[tuple[str, str]] = (),
    screened_as_of: str = "",
    conflicts: list[str] = (),
    catalog_groups: list[tuple[str, str | None, list[str]]] | None = None,
    catalog_rules: dict[str, tuple[str, ...]] | None = None,
    dated: list[tuple[str, str]] = (),
    dated_built: str = "",
    keep_dates: bool = False,
    shifting: list[tuple[str, str]] = (),
    shifting_built: str = "",
    exclude_shifting: bool = False,
    result=None,
) -> Report:
    """The report of a new task.

    The first groups are those the task file lists (``predictors.exclude``); ``result``
    is the :class:`ml.dataset.TaskResult` that worked out the task's usable inputs, and
    adds what a run leaves out besides, and what the batch table does not have.
    """
    report = Report(task=task, file=file, config=config, site=site, target=target)
    groups = report.groups

    # ── The screening register ─────────────────────────────────────────────
    group = Group(
        "Screening register",
        "screened out by a decision in the screening register at this site"
        + (f" (last decision {screened_as_of})" if screened_as_of else ""),
        [(p, why) for p, why in screened],
    )
    group.notes += [f"CONFLICT: {line}" for line in conflicts]
    if group.rows or group.notes:
        groups.append(group)

    # ── The catalog's Variable Role and Functional Type ────────────────────
    catalog_why = (
        "the catalog says this {column} cannot explain how a batch turned out, or is known "
        "only at or after release"
    )
    if catalog_groups is None:
        rules = catalog_rules or {}
        if rules:
            report.caveats.append(
                "The catalog could not be read, so the catalog rules are in the task file as "
                "exclude_by_catalog, and their parameters are not listed one by one here"
                + ("" if result is None else "; they are under the catalog rules group below")
            )
    else:
        unmatched = Group(
            "Catalog rules that left nothing out",
            "a default catalog value no catalog parameter has, or a column the catalog lacks: "
            "check the spelling against the catalog",
        )
        for column, value, members in catalog_groups:
            if value is None:
                unmatched.notes.append(f"{column}: the catalog has no such column")
            elif not members:
                unmatched.notes.append(f"{column} = '{value}': no catalog parameter has it")
            else:
                groups.append(
                    Group(
                        f"Catalog {column}: {value}",
                        catalog_why.format(column=column),
                        [(m, "") for m in members],
                    )
                )
        if unmatched.notes:
            groups.append(unmatched)

    # ── Dates and shifts the latest build found ────────────────────────────
    if keep_dates:
        report.caveats.append("--keep-dates: date columns are not left out")
    elif dated:
        groups.append(
            Group(
                "Dates and times",
                "a model fitted on a date learns when a batch was made; the durations derived "
                "from the dates stay usable"
                + (f" (as the build of {dated_built} found them)" if dated_built else ""),
                [(p, why) for p, why in dated],
            )
        )
    if exclude_shifting and shifting:
        groups.append(
            Group(
                "Run in groups over time",
                "run in groups over time at a manufacturing site, so a model would learn time"
                + (f" (as the build of {shifting_built} found them)" if shifting_built else ""),
                [(p, why) for p, why in shifting],
            )
        )

    # ── What a run leaves out besides, as the task's usable inputs found ───
    if result is None:
        report.caveats.append(
            "The task's usable inputs were not worked out, so the target, the roles, the "
            "target's lineage and parameters missing from the batch table are not listed"
        )
        return report

    listed = set(report.excluded())
    by_stage: dict[str, list[tuple[str, str]]] = {}
    for decision in result.decisions:
        if decision.decision != "excluded":
            continue
        stage = decision.stage
        if stage in ("task exclusion", "screening"):
            if decision.parameter in listed:
                continue  # one of the groups above
            stage = "other"
        by_stage.setdefault(stage, []).append((decision.parameter, decision.reason))
    for stage, (title, why) in STAGES.items():
        if by_stage.get(stage):
            groups.append(Group(title, why, by_stage.pop(stage)))
    for stage, rows in by_stage.items():  # anything not foreseen still shows
        title = "Other exclusions" if stage == "other" else f"Excluded at '{stage}'"
        groups.append(Group(title, "left out when the usable inputs were worked out", rows))

    if result.missing_from_batch_table:
        groups.append(
            Group(
                "In the catalog, not in the batch table",
                "the site's batch table has no such column, so there is nothing to use",
                [(p, "") for p in result.missing_from_batch_table],
            )
        )
    available = len(result.candidates) + sum(
        1 for d in result.decisions if d.decision == "excluded"
    )
    report.in_batch_table = available
    report.catalog_parameters = available + len(result.missing_from_batch_table)
    report.usable = len(result.candidates)
    return report


def render(report: Report, paint: Callable[..., str] = lambda text, *styles: text) -> list[str]:
    """The report as lines of text; ``paint(text, *styles)`` colours them on a console."""
    out: list[str] = []
    excluded = report.excluded()
    seen_in: dict[str, list[str]] = {}
    for group in report.groups:
        for parameter, _ in group.rows:
            seen_in.setdefault(parameter, []).append(group.title)

    out.append(paint("Parameters left out of the task's dataset", "blue", "bold"))
    facts = [
        ("Task", report.task),
        ("Task file", report.file),
        ("Config", report.config or "none found"),
        ("Site", report.site or "not set"),
        ("Target", report.target or "not set (--target)"),
        ("Made", report.made.strftime("%Y-%m-%d %H:%M:%S")),
    ]
    if report.catalog_parameters is not None:
        facts += [
            ("Catalog parameters", f"{report.catalog_parameters:,}"),
            ("In the batch table", f"{report.in_batch_table:,}"),
            ("Usable as predictors", f"{report.usable:,}"),
        ]
    facts.append(
        ("Left out", f"{len(excluded):,} catalog parameters, in {len(report.groups)} groups")
    )
    width = max(len(label) for label, _ in facts)
    out += [f"  {paint(f'{label:<{width}}', 'dim')}  {value}" for label, value in facts]
    for caveat in report.caveats:
        out.append(f"  {paint('Note', 'yellow', 'bold')}  {paint(caveat, 'yellow')}")

    if report.groups:
        out.append("")
        out.append(paint("  Overview", "blue", "bold"))
        title_width = max(len(g.title) for g in report.groups)
        for group in report.groups:
            count = f"{len(group.rows):>5,}" if group.rows else "    -"
            out.append(f"    {group.title:<{title_width}}  {count}")
        out.append(
            paint(
                "  A parameter left out for several reasons is listed in each group, "
                "with the others named beside it.",
                "dim",
            )
        )

    for number, group in enumerate(report.groups, 1):
        out.append("")
        head = f"[{number}] {group.title} ({len(group.rows):,} parameter"
        head += ")" if len(group.rows) == 1 else "s)"
        out.append(paint(head, "blue", "bold"))
        out.append(f"    {paint('Why:', 'dim')} {group.why}")
        for note in group.notes:
            out.append(f"    {paint('-', 'yellow')} {note}")
        width = max((len(p) for p, _ in group.rows), default=0)
        for parameter, detail in group.rows:
            others = [t for t in seen_in[parameter] if t != group.title]
            words = [detail] if detail else []
            if others:
                words.append("also: " + ", ".join(others))
            line = f"    - {parameter:<{width}}"
            if words:
                line += "  " + paint(" | ".join(words), "dim")
            out.append(line.rstrip())
    if not report.groups:
        out.append("")
        out.append("  Nothing is left out.")
    return out
