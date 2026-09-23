"""Shared machinery for the scripts that build the paper's tables and figures.

Seven scripts read run artifacts and write something a paper includes -- a table and a set of curves
per domain, plus the StuLife far-recall figure and the two combined figures. Everything here is what
they would otherwise each copy: the run manifest, the `--check` staleness gate and its reporting, and
the rounding the tables share.
"""

# The reason there is a module rather than seven copies is the one the AppWorld table's header states
# about itself: the first hand-maintained version of that table had three wrong cells, all in rows a
# human retyped. A generator only removes that class of error while the *comparison* it does is
# trustworthy, and seven separately-drifting copies of a comparison are not.

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import shlex
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

REPO = Path(__file__).resolve().parent.parent

# A rendered artifact's provenance lines: LaTeX comments. Separated from content because a
# difference confined to them means "same numbers, different build", which `--check` reports
# differently -- see `describe_text_difference`.
_COMMENT = "%"


def d1(x: float) -> str:
    """One decimal place, rounded half-up.

    Python's `round` is banker's rounding, so `round(0.45, 1)` is 0.4 -- a table built with it
    disagrees with the same number typed into a calculator, on exactly the ties a reader is most
    likely to spot-check.
    """
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load_manifest(path: Path) -> dict[str, list[str]]:
    """Arm key -> run-directory globs, from a shared manifest.

    Held in a file rather than in a script so a reproducer can point every artifact at their own
    runs without editing code, and so a domain's table and its curves cannot be built from
    different runs.
    """
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read run manifest {path}: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"run manifest {path} is not valid JSON: {exc}") from exc
    arms: Any = cast("dict[str, Any]", raw).get("arms") if isinstance(raw, dict) else None
    if not isinstance(arms, dict):
        raise SystemExit(f"run manifest {path} has no `arms` object")
    # Validate the value shape rather than coercing it. `{"codeact": "runs/x/*"}` -- a plausible
    # typo, a string where a list belongs -- iterates character by character into single-character
    # globs, and `null` raises a bare `TypeError`; both slip past the check above and surface much
    # later as `NotImplementedError` from `Path.glob`. Name the manifest and the key here instead.
    out: dict[str, list[str]] = {}
    for key, value in cast("dict[str, Any]", arms).items():
        if not isinstance(value, list):
            raise SystemExit(
                f"run manifest {path}: arm {key!r} must map to a list of globs, got {type(value).__name__}"
            )
        globs: list[str] = []
        for glob in cast("list[Any]", value):
            if not isinstance(glob, str):
                raise SystemExit(f"run manifest {path}: arm {key!r} has a non-string glob {glob!r}")
            # `Path.glob` refuses an absolute pattern outright (`NotImplementedError: Non-relative
            # patterns are unsupported`), so an absolute glob can never resolve against --runs-root.
            # Point the reader at --runs-root rather than letting them read that traceback.
            if Path(glob).is_absolute():
                raise SystemExit(
                    f"run manifest {path}: arm {key!r} has an absolute glob {glob!r}. Globs are "
                    "resolved against --runs-root, so write them relative and pass the prefix as "
                    "--runs-root instead."
                )
            globs.append(glob)
        out[str(key)] = globs
    return out


def display(path: Path) -> str:
    """`path` as a repo-relative string when it is inside the repo, else as the caller gave it."""
    # Resolved ONCE and used for both halves. It used to test `path.resolve()` and then call
    # `relative_to` on the unresolved `path`, so a relative argument passed the test and raised on
    # the very next call -- `--out dist/x` blew up after a 393 MB archive had been written, which is
    # the worst moment to discover a formatting helper is wrong.
    resolved = path.resolve()
    return str(resolved.relative_to(REPO)) if resolved.is_relative_to(REPO) else str(path)


def runs_root_line(runs_root: Path) -> str:
    """`runs_root` as a generated header names it: the default said in words, anything else as given."""
    # Not abbreviated for privacy. An absolute runs root prints absolutely, username and all, which
    # is the same thing the published trace archive ships out of every run's `provenance.json`; a
    # header that hid it while the archive printed it would only make the two disagree. The default
    # is the one case worth naming, since `display(REPO)` is "." and tells a reader nothing.
    return "the repo root (the default)" if runs_root.resolve() == REPO else display(runs_root)


def rerun_command(script: str, args: argparse.Namespace, defaults: dict[str, Any]) -> str:
    """The command that rebuilds this artifact with the same inputs this invocation used.

    `defaults` maps a flag to the value that means "not passed", so a `--check` run against
    someone else's runs is told to rebuild from THOSE runs rather than from the shipped manifest.
    """
    cmd = f"uv run python scripts/{script}"
    for flag, default in defaults.items():
        attr = flag.lstrip("-").replace("-", "_")
        # No `getattr` default: a flag named here that the script does not define is a typo in the
        # caller's dict, and silently omitting it would print a command that rebuilds something
        # else. Fail where the mistake is.
        value = getattr(args, attr)
        if value != default:
            # `shlex.quote` because a path can contain a space, and the whole point of this string
            # is that it can be pasted. `Path` and `str` compare unequal even for the same path, so
            # callers pass `Path` defaults for `Path`-typed flags.
            cmd += f" {flag} {shlex.quote(str(value))}"
    return cmd


def describe_text_difference(current: str, rendered: str, *, context: int = 2) -> list[str]:
    """Human-readable account of how a committed text artifact differs from a fresh build.

    Returns the lines to print. The distinction it exists to draw is between a difference in the
    NUMBERS and one confined to the provenance header, because those mean opposite things to a
    reader: the first says the committed artifact no longer describes these runs, the second says
    only that it was built somewhere else. Reporting both as "STALE" -- which is all the first
    version of this gate did -- sends someone hunting a data change that did not happen.
    """
    # `splitlines()` drops a trailing-newline difference, so two files that differ ONLY there would
    # otherwise reach the "numbers identical" branch and print an empty diff -- a confidently wrong
    # answer. Name it before splitting anything.
    if current.splitlines() == rendered.splitlines():
        return [
            "  The two differ only in trailing whitespace (a missing or extra final newline);",
            "  every line is identical. Rebuilding will normalise it.",
        ]

    cur_lines = current.splitlines()
    new_lines = rendered.splitlines()
    cur_body = [ln for ln in cur_lines if not ln.startswith(_COMMENT)]
    new_body = [ln for ln in new_lines if not ln.startswith(_COMMENT)]

    out: list[str] = []
    if cur_body == new_body:
        out.append("  Numbers are IDENTICAL. Only the provenance header differs, so the committed")
        out.append("  artifact describes the same results, built from a different tree or manifest:")
        out += _diff_block(
            [ln for ln in cur_lines if ln.startswith(_COMMENT)],
            [ln for ln in new_lines if ln.startswith(_COMMENT)],
            context=context,
        )
        return out

    out.append("  CONTENT differs -- the committed artifact no longer matches these runs:")
    out += _diff_block(cur_body, new_body, context=context)
    header_cur = [ln for ln in cur_lines if ln.startswith(_COMMENT)]
    header_new = [ln for ln in new_lines if ln.startswith(_COMMENT)]
    if header_cur != header_new:
        out.append("")
        out.append("  The provenance header differs too:")
        out += _diff_block(header_cur, header_new, context=context)
    return out


def _diff_block(a: list[str], b: list[str], *, context: int) -> list[str]:
    """A unified diff, indented, `committed` against `rebuilt`, capped so it stays readable."""
    lines = list(difflib.unified_diff(a, b, fromfile="committed", tofile="rebuilt", n=context, lineterm=""))
    # A wholesale change (a renamed arm, a reordered table) diffs as hundreds of lines and buries
    # the answer. Past the cap, say how much was elided rather than truncating silently.
    cap = 60
    shown = lines[:cap]
    out = ["    " + ln for ln in shown]
    if len(lines) > cap:
        out.append(f"    ... {len(lines) - cap} more diff lines elided")
    return out


def aggregate_marks(
    stats: dict[str, list[float]], columns: tuple[str, ...], lower_is_better: frozenset[str]
) -> dict[str, dict[int, str]]:
    """Which arm index gets \\textbf (best) and \\underline (second) in each aggregate column.

    NaN columns are skipped: an arm with no value for a metric (the AppWorld baseline reports no
    meta spend) is not a competitor for "best" in it.
    """
    marks: dict[str, dict[int, str]] = {}
    for col in columns:
        scored = [(i, v) for i, v in enumerate(stats[col]) if not math.isnan(v)]
        if len(scored) < 2:
            marks[col] = {}
            continue
        scored.sort(key=lambda p: p[1], reverse=col not in lower_is_better)
        marks[col] = {scored[0][0]: r"\textbf", scored[1][0]: r"\underline"}
    return marks


def report_pdf_check(built: list[Path], out_dir: Path, rebuild_command: str) -> int:
    """Compare freshly built figures against the committed ones; return an exit status.

    Shared by both plot scripts, which otherwise held byte-identical copies differing only in the
    script name they name in the rebuild line.
    """
    stale = False
    for fresh in built:
        committed = out_dir / fresh.name
        if not committed.is_file():
            print(f"{display(committed)} does not exist", file=sys.stderr)
            stale = True
            continue
        a, b = committed.read_bytes(), fresh.read_bytes()
        if pdf_content(a) == pdf_content(b):
            print(f"{display(committed)} is up to date")
            continue
        stale = True
        print(f"{display(committed)} differs from a fresh build:", file=sys.stderr)
        for line in describe_pdf_difference(a, b):
            print(line, file=sys.stderr)
    if stale:
        print(f"\n  Rebuild with: {rebuild_command}", file=sys.stderr)
    return 1 if stale else 0


# matplotlib stamps a wall-clock `/CreationDate` into every PDF, so two byte-different PDFs are the
# norm even when the figure is identical. Comparing modulo that string is what makes a `--check`
# on a figure mean "the picture changed" rather than "time passed".
_PDF_CREATION_DATE = re.compile(rb"/CreationDate\s*\([^)]*\)")


def pdf_content(data: bytes) -> bytes:
    """A PDF's bytes with the embedded creation timestamp removed."""
    return _PDF_CREATION_DATE.sub(b"", data)


def describe_pdf_difference(current: bytes, rebuilt: bytes) -> list[str]:
    """Human-readable account of how a committed figure differs from a fresh build."""
    if pdf_content(current) == pdf_content(rebuilt):
        return [
            "  Figure content is IDENTICAL; only the embedded creation timestamp differs.",
            "  Nothing to rebuild -- committing this would be a binary diff of a clock reading.",
        ]
    return [
        f"  Figure CONTENT differs ({len(current)} bytes committed, {len(rebuilt)} rebuilt).",
        "  A PDF is not diffable line by line; rebuild and view it to see what moved.",
    ]
