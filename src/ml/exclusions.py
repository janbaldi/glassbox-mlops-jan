"""What a new task leaves out of its dataset, and why: one report, on screen and in a file.

``ml task new`` leaves parameters out for several reasons at once: the screening
register, the catalog's Variable Role and Functional Type, the dates and shifts a build
found, and — when its usable inputs are worked out — the task's roles, the target's
lineage and the site-relative features. This module gathers them into groups, one per
reason, and renders them as plain text: the console prints it with colour, and
``excluded_parameters.log`` in the task folder keeps it without. ``ml task catalog``
writes it again from the task file as it stands, after an edit.

A parameter left out for several reasons is listed in every group that leaves it out,
with the other groups named beside it, so each group is complete and each line says
the whole story.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

CATALOG_WHY = (
    "the catalog says this {column} cannot explain how a batch turned out, or is known only "
    "at or after release"
)
SCREENED_WHY = "screened out by a decision in the screening register at this site"
#: The catalog columns the screening groups are divided by, under each reason, in order.
SCREENING_METADATA = ("Variable Role", "Functional Type")
#: One screening decision as a task names it (:attr:`ml.screening.Decision.label`):
#: ``S-0003: retired (Ghent)``, perhaps with ``, represented by 'X'``.
_DECISION = re.compile(
    r"(?P<id>[^\s;:]+): (?P<reason>[^;()]+?) \((?P<site>[^()]*)\)"
    r"(?:, represented by '(?P<represented>[^']*)')?"
)
DATES_WHY = (
    "a model fitted on a date learns when a batch was made; the durations derived from the "
    "dates stay usable"
)
SHIFTS_WHY = "run in groups over time at a manufacturing site, so a model would learn time"

#: Run-time stages of :func:`ml.dataset.candidate_predictors`, as groups of the report.
#: "task exclusion" is not here: those are the task file's own groups.
STAGES = {
    "screening": (
        "Screening register, not listed in the task file",
        "screened out by a decision the task file does not list (taken after the task was "
        "made, or its line deleted); every run applies the register",
    ),
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
    #: (heading, its parameters) when the rows are shown divided, as the screening groups
    #: are by catalog metadata; empty to show them in one list.
    sections: list[tuple[str, list[str]]] = field(default_factory=list)


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
    #: The command that wrote it.
    command: str = "ml task new"
    #: Parameter → its catalog metadata in words, which divides the screening groups.
    metadata: dict[str, str] = field(default_factory=dict)

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
    catalog=None,
) -> Report:
    """The report of a new task.

    The first groups are those the task file lists (``predictors.exclude``); ``result``
    is the :class:`ml.dataset.TaskResult` that worked out the task's usable inputs, and
    adds what a run leaves out besides, and what the batch table does not have.
    ``catalog`` (the catalog as read) divides the screening groups by its metadata.
    """
    report = Report(task=task, file=file, config=config, site=site, target=target)
    groups = report.groups
    report.metadata = metadata_of(catalog)

    # ── The screening register, a group per reason ─────────────────────────
    as_of = f" (last decision {screened_as_of})" if screened_as_of else ""
    groups += screening_groups(screened, report.metadata, why=SCREENED_WHY + as_of)
    if conflicts:
        groups.append(
            Group(
                "Screening register conflicts",
                "a representative the register excludes too, so its group is left out whole: "
                "pick another representative in the correlation report",
                notes=list(conflicts),
            )
        )

    # ── The catalog's Variable Role and Functional Type ────────────────────
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
                        CATALOG_WHY.format(column=column),
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
                DATES_WHY + (f" (as the build of {dated_built} found them)" if dated_built else ""),
                [(p, why) for p, why in dated],
            )
        )
    if exclude_shifting and shifting:
        groups.append(
            Group(
                "Run in groups over time",
                SHIFTS_WHY
                + (f" (as the build of {shifting_built} found them)" if shifting_built else ""),
                [(p, why) for p, why in shifting],
            )
        )

    return _add_run_groups(report, result)


def metadata_of(catalog, columns: tuple[str, ...] = SCREENING_METADATA) -> dict[str, str]:
    """Parameter → its catalog metadata in words: ``Variable Role: Process parameter ·
    Functional Type: Process``, over the ``columns`` the catalog has. Empty without a
    catalog, or when it has none of them."""
    if catalog is None:
        return {}
    present = [c for c in columns if c in catalog.columns]
    if not present:
        return {}
    out: dict[str, str] = {}
    for _, row in catalog.iterrows():
        words = []
        for column in present:
            value = row[column]
            value = "" if value is None or value != value else str(value).strip()
            words.append(f"{column}: {value or '(empty)'}")
        out.setdefault(str(row["Parameter"]), " · ".join(words))
    return out


def screening_groups(
    rows: list[tuple[str, str]],
    metadata: dict[str, str],
    title: str = "Screening register",
    why: str = SCREENED_WHY,
) -> list[Group]:
    """The screened-out parameters as one group per screening reason, each divided by the
    parameters' catalog metadata.

    ``rows`` is (parameter, ``screened out by S-0001: retired (North); …``): a parameter
    several decisions leave out is under each of their reasons, with the decisions of that
    reason beside it. ``metadata`` is :func:`metadata_of`'s; without it the groups are not
    divided.
    """
    by_reason: dict[str, dict[str, list[str]]] = {}
    for parameter, text in rows:
        found = list(_DECISION.finditer(text))
        if not found:
            by_reason.setdefault("unrecognised reason", {}).setdefault(parameter, []).append(
                text.removeprefix("screened out by ").strip()
            )
        for match in found:
            label = f"{match['id']}, {match['site']}"
            if match["represented"]:
                label += f", represented by '{match['represented']}'"
            decisions = by_reason.setdefault(match["reason"].strip(), {})
            decisions.setdefault(parameter, []).append(label)
    out: list[Group] = []
    for reason in sorted(by_reason, key=_reason_order):
        members = by_reason[reason]
        group = Group(
            f"{title}: {reason}",
            why,
            [(p, "; ".join(labels)) for p, labels in members.items()],
        )
        if metadata:
            sections: dict[str, list[str]] = {}
            for parameter in members:
                heading = metadata.get(parameter, "not in the catalog")
                sections.setdefault(heading, []).append(parameter)
            group.sections = sorted(sections.items(), key=lambda s: (-len(s[1]), s[0]))
        out.append(group)
    return out


def _reason_order(reason: str) -> tuple[int, str]:
    """The register's own order of reasons, then any other alphabetically."""
    from .screening import REASONS

    return (REASONS.index(reason), "") if reason in REASONS else (len(REASONS), reason)


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
        ("Written", f"{report.made:%Y-%m-%d %H:%M:%S} by {report.command}"),
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
        details = dict(group.rows)

        def row(parameter: str, indent: str) -> str:
            others = [t for t in seen_in[parameter] if t != group.title]
            words = [details[parameter]] if details[parameter] else []
            if others:
                words.append("also: " + ", ".join(others))
            line = f"{indent}- {parameter:<{width}}"
            if words:
                line += "  " + paint(" | ".join(words), "dim")
            return line.rstrip()

        if group.sections:
            for heading, members in group.sections:
                out.append(f"    {paint(heading, 'bold')} ({len(members):,})")
                out += [row(p, "      ") for p in members]
        else:
            out += [row(p, "    ") for p, _ in group.rows]
    if not report.groups:
        out.append("")
        out.append("  Nothing is left out.")
    return out


def from_task_file(
    *,
    task: str,
    file: str,
    config: str,
    site: str,
    target: str,
    excludes: list[tuple[str, str]],
    result=None,
    catalog=None,
) -> Report:
    """The report of an existing task, from its task file as it stands now.

    ``excludes`` is the task file's ``predictors.exclude`` entries, one (column, reason)
    each, as written: a parameter listed under several groups comes once per group. Each
    reason is sorted into the group :func:`build` would have put it in, by its wording;
    a reason written by hand goes under "Task file exclusions".
    """
    report = Report(task=task, file=file, config=config, site=site, target=target)
    report.metadata = metadata_of(catalog)
    groups: dict[str, Group] = {}
    screened: list[tuple[str, str]] = []
    by_catalog = re.compile(r"^catalog (?P<column>.+?) is '(?P<value>.*)'$")
    for column, reason in excludes:
        match = by_catalog.match(reason)
        if match:
            title = f"Catalog {match['column']}: {match['value']}"
            why = CATALOG_WHY.format(column=match["column"])
            detail = ""
        elif reason.startswith("screened out by"):
            screened.append((column, reason))
            continue
        elif reason.startswith("a date"):
            title, why, detail = "Dates and times", DATES_WHY, reason
        elif reason.startswith("runs in groups over time"):
            title, why, detail = "Run in groups over time", SHIFTS_WHY, reason
        else:
            title = "Task file exclusions"
            why = "listed in the task file's predictors.exclude, with the reason given there"
            detail = reason
        group = groups.setdefault(title, Group(title, why))
        if all(p != column for p, _ in group.rows):
            group.rows.append((column, detail))
    order = [t for t in groups if t.startswith("Catalog ")]
    order += ["Dates and times", "Run in groups over time", "Task file exclusions"]
    report.groups = screening_groups(screened, report.metadata) + [
        groups[t] for t in order if t in groups
    ]
    return _add_run_groups(report, result)


def _add_run_groups(report: Report, result) -> Report:
    """What a run leaves out besides the task file's groups, as the task's usable inputs
    (``result``, a :class:`ml.dataset.TaskResult`) found, and the counts."""
    groups = report.groups
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
        if stage in ("task exclusion", "screening") and decision.parameter in listed:
            continue  # one of the task file's groups
        if stage == "task exclusion":
            stage = "other"
        by_stage.setdefault(stage, []).append((decision.parameter, decision.reason))
    for stage, (title, why) in STAGES.items():
        if not by_stage.get(stage):
            continue
        if stage == "screening":
            groups += screening_groups(by_stage.pop(stage), report.metadata, title, why)
        else:
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
