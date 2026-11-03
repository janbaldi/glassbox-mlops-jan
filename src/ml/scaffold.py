"""Creating a task workspace.

``ml task new <folder>`` makes a folder with a commented ``task.yaml`` in it.
The folder is the task's workspace: every run of the task is written to its
``runs/`` folder, with the batch table and catalog it read, so the folder holds everything
needed to redo the analysis. The batch table and catalog themselves are read from
SharePoint, where the ``config.yaml`` the task names locates them.
"""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from pathlib import Path

from .config import ConfigError
from .taskconfig import TASK_FILE

#: The catalog columns, and their values, a new task leaves out by default: what a
#: parameter's role or function says cannot be a cause of how a batch turns out, or is known
#: only at or after release. A new task.yaml lists every such parameter on its own line.
EXCLUDED_BY_CATALOG: dict[str, tuple[str, ...]] = {
    "Variable Role": (
        "Identifier",
        "Timestamp",
        "Context / design factor",
        "CQA / release result",
        "Disposition label",
        "Equipment ID",
        "Material lot ID",
        "Post-release logistics",
        "Setpoint / plan",
    ),
    "Functional Type": ("Formulation and dose", "Quality and disposition"),
}


def template(name: str) -> str:
    return (resources.files("ml") / "templates" / name).read_text(encoding="utf-8")


def init(
    folder: str | Path,
    target: str = "",
    question: str = "",
    force: bool = False,
    config: str | Path | None = None,
    screened: list[tuple[str, str]] = (),
    screened_as_of: str = "",
    shifting: list[tuple[str, str]] = (),
    shifting_built: str = "",
    conflicts: list[str] = (),
    site: str = "",
    dated: list[tuple[str, str]] = (),
    dated_built: str = "",
    catalog_groups: list[tuple[str, str | None, list[str]]] | None = None,
) -> Path:
    """Write ``<folder>/task.yaml``, pointing at ``config``. Returns its path.

    ``screened`` is (parameter, reason) for every parameter the screening register
    excludes when the task is made, and ``shifting`` for every parameter the latest
    build (run at ``shifting_built``) found running in groups over time, when asked
    for, and ``dated`` for every date column the latest build (run at
    ``dated_built``) found at the site. They are written out as ``predictors.exclude``
    entries, so the task file itself says what it leaves out and why; ``conflicts``
    (representatives the register excludes too) as comments above them. ``site`` is
    the key of the site the task is about (config.yaml's ``site:``).

    ``catalog_groups`` is (catalog column, value, its parameters) for every rule of
    :data:`EXCLUDED_BY_CATALOG`, the value None when the catalog has no such column: each
    parameter is written on its own ``predictors.exclude`` line, under a comment naming its
    group. Without them (no catalog could be read) the rules are written as
    ``predictors.exclude_by_catalog`` instead, so a run still applies them.

    A parameter is listed in every group that leaves it out, each time with that group's
    reason, so every group shows all its members; a run joins its reasons.
    """
    path = (Path(folder).expanduser() / TASK_FILE).resolve()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists; pass --force to overwrite it")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name).strip("_") or "task"
    link = (
        "config: " + os.path.relpath(Path(config).resolve(), path.parent).replace("\\", "/")
        if config
        else "# config: path/to/config.yaml"
    )
    text = (
        template("task.yaml")
        .replace("__CONFIG__", link)
        .replace("__NAME__", slug)
        .replace("site: __SITE__", f"site: {site}" if site else "# site: a")
        .replace("__QUESTION__", question or f"What goes with {target or 'the target'}?")
        .replace("__TARGET__", target or "<the column to predict>")
        .replace("__TARGET_TYPE__", "numeric")
        .replace(
            "  exclude_by_catalog: {}          # {catalog column: [values left out]}\n",
            _catalog_rules()
            if catalog_groups is None
            else "  exclude_by_catalog: {}          # {catalog column: [values left out]}\n",
        )
        .replace(
            "  exclude: []                     # single columns: [{column: ..., reason: ...}]\n",
            _excludes(
                screened,
                screened_as_of,
                shifting,
                shifting_built,
                conflicts,
                dated,
                dated_built,
                catalog_groups or [],
            ),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _catalog_rules() -> str:
    """The default catalog rules as ``exclude_by_catalog``, when no catalog could be read to
    list their parameters one by one."""
    lines = [
        "  exclude_by_catalog:              # {catalog column: [values left out]}",
        "    # No catalog could be read when this task was made, so its parameters are not",
        "    # listed one by one under exclude: these rules leave them out instead.",
    ]
    for column, values in EXCLUDED_BY_CATALOG.items():
        lines.append(f"    {json.dumps(column)}:")
        lines += [f"      - {json.dumps(v)}" for v in values]
    return "\n".join(lines) + "\n"


def _excludes(
    screened: list[tuple[str, str]],
    as_of: str,
    shifting: list[tuple[str, str]] = (),
    built: str = "",
    conflicts: list[str] = (),
    dated: list[tuple[str, str]] = (),
    dated_built: str = "",
    catalog_groups: list[tuple[str, str | None, list[str]]] = (),
) -> str:
    """The ``predictors.exclude`` lines of a new task: empty, or every parameter left out on
    its own line, in groups. A parameter is listed in every group that leaves it out, each
    time with that group's reason; a run joins its reasons."""
    if not screened and not shifting and not dated and not catalog_groups:
        return "  exclude: []                     # single columns: [{column: ..., reason: ...}]\n"
    lines = ["  exclude:                        # single columns: [{column: ..., reason: ...}]"]

    def entries(pairs) -> list[str]:
        return [_exclude(column, reason) for column, reason in pairs]

    if screened:
        lines += [
            f"    # Screened out by the screening register (last decision {as_of}) at this site.",
            "    # Every run applies the register anyway; these lines make it visible here.",
            "    # A run warns when one of them has since been reinstated: then delete the line.",
        ]
        if conflicts:
            lines += [
                "    # CONFLICT: a representative the register excludes too, so its group is",
                "    # left out whole. Resolve it in the correlation report: pick another",
                "    # representative.",
            ]
            lines += [f"    #   {line}" for line in conflicts]
        lines += entries(screened)
    if catalog_groups:
        lines += [
            "    # By the catalog: parameters whose role or function says they cannot explain",
            "    # how a batch turned out, or are known only at or after release. Delete a line",
            "    # to use the parameter.",
        ]
        for column, value, members in catalog_groups:
            if value is None:
                lines.append(f"    # Catalog {column}: the catalog has no such column")
            elif not members:
                lines.append(f"    # Catalog {column}: {value} — no catalog parameter has it")
            else:
                count = f"{len(members)} parameter{'' if len(members) == 1 else 's'}"
                lines.append(f"    # Catalog {column}: {value} ({count})")
                lines += entries((m, f"catalog {column} is '{value}'") for m in members)
    if dated:
        lines += [
            f"    # Dates and times at this site, as the build of {dated_built} found them",
            "    # (dates_report.html): a model would learn when a batch was made. The durations",
            "    # derived from them stay usable. Delete a line to use the date again.",
        ]
        lines += entries(dated)
    if shifting:
        lines += [
            f"    # Run in groups over time at a manufacturing site, as the build of {built}",
            "    # found (shifts_report.html; `ml task new --exclude-shifting`). A lot does this,",
            "    # and so the model would learn time. Delete a line to use the parameter again.",
        ]
        lines += entries(shifting)
    return "\n".join(lines) + "\n"


def _exclude(column: str, reason: str) -> str:
    return f"    - {{column: {json.dumps(column)}, reason: {json.dumps(reason)}}}"
