"""Build the StuLife results table from run artifacts.

The StuLife counterpart to `build_appworld_table.py`, and written for the same reason: a table
retyped by a human is a table with wrong cells in it. Every number below is computed from
`attempt-*/results.json`, and `--check` says whether the committed file still matches the runs.

Usage:
    uv run python scripts/build_stulife_table.py            # writes tables/stulife_results.tex
    uv run python scripts/build_stulife_table.py --check    # exits 1 if the committed file is stale

Reproducing with your OWN runs: the arms' run directories live in `scripts/stulife_runs.json`, not
in this file. Copy it, replace each glob with yours, and pass `--runs-manifest yours.json`. Globs
resolve against `--runs-root` (default: the repo root). `--out` chooses where the table is written.
`scripts/plot_stulife_curves.py` reads the same manifest, so the table and the curves cannot be
built from different runs.

A REP IS AN ATTEMPT, not a run directory. The StuLife arms were launched with `--attempts 3`, so
one run directory holds three independent attempts, where the AppWorld arms have one attempt each
across several run directories. This reads every attempt of every matched run, which handles both.

`--check` is a gate you run by hand, and cannot be automated: it reads run directories, which
are gitignored, so a fresh clone has no inputs and the check would fail for the wrong reason.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, NamedTuple, cast

sys.path.append(str(Path(__file__).resolve().parent))
from _artifacts import (
    REPO,
    aggregate_marks,
    d1,
    describe_text_difference,
    display,
    load_manifest,
    rerun_command,
    runs_root_line,
)

DEFAULT_MANIFEST = REPO / "scripts" / "stulife_runs.json"
OUT = REPO / "tables" / "stulife_results.tex"

# `pass_rate` is the fraction of scored tasks fully passed; `avg_score` is the mean partial credit
# over the same tasks. Both exclude trigger rows, which carry no score, so every rate here is out of
# the scored subset (the header reports the counts, read from the runs). The `_far_recall` pair is
# quantities restricted to tasks whose answer was taught far enough back to have left the context
# window, which is the subset the long-horizon claim rests on.
COLUMNS = ("pass", "score", "far_pass", "far_score", "cost")
# Lower is better for money; the aggregate table's bold/underline marks follow this.
LOWER_IS_BETTER = frozenset({"cost"})
COLUMN_HEADS = {
    "pass": "Pass (\\%)",
    "score": "Score (\\%)",
    "far_pass": "Pass (\\%)",
    "far_score": "Score (\\%)",
    "cost": "Cost (\\$)",
}


class Arm(NamedTuple):
    """One method's runs. `key` indexes the run manifest."""

    label: str
    short: str  # plain-text name for the generated provenance block (LaTeX has no place there)
    key: str
    prompt_only: str  # LaTeX for the aggregate table's prompt-only column


ARMS = (
    Arm(
        r"CodeAct\textsubscript{\oursimpl{}} \citep{wang2024codeact} \scriptsize{(per task)}",
        "CodeAct (per task)",
        "codeact_per_task",
        r"\checkmark",
    ),
    Arm(
        r"CodeAct+subagents\textsubscript{\citep{roucher2025smolagents}}",
        "CodeAct+subagents (smolagents)",
        "codeact_subagents_smolagents",
        r"\checkmark",
    ),
    Arm(
        r"CodeAct+subagents\textsubscript{\oursimpl{}}",
        "CodeAct+subagents (ours)",
        "codeact_subagents",
        r"\checkmark",
    ),
    Arm(
        r"Letta Agent \citep{packer2023memgpt}",
        "Letta",
        "letta",
        r"\xmark",
    ),
    Arm(
        r"\oursimpl{} \lstinline|invoke|",
        "invoke",
        "jaz_invoke",
        r"\checkmark",
    ),
)


def _metrics(attempt: Path) -> dict[str, float]:
    """The five reported quantities for one attempt."""
    d = json.loads((attempt / "results.json").read_text())
    e = d["extra"]
    return {
        "pass": 100 * e["pass_rate"],
        "score": 100 * e["avg_score"],
        "far_pass": 100 * e["pass_rate_far_recall"],
        "far_score": 100 * e["avg_score_far_recall"],
        "cost": d["usage"]["cost_usd"],
    }


class Counts(NamedTuple):
    """The episode's shape, read from the runs rather than written down."""

    tasks: int  # every row, triggers included
    scored: int  # rows carrying a score
    far: int  # scored rows whose answer was taught far enough back to have left the window

    @property
    def triggers(self) -> int:
        return self.tasks - self.scored


def _counts(attempt: Path) -> Counts:
    """The episode's shape as this attempt recorded it."""
    e = json.loads((attempt / "results.json").read_text())["extra"]
    return Counts(tasks=e["tasks_total"], scored=e["n_total"], far=e["n_total_far_recall"])


def _provenance(run: Path, attempt: Path) -> dict[str, str]:
    """What a rep records about itself, for the generated provenance block."""
    out = {"run": run.name, "attempt": attempt.name}
    prov = run / "provenance.json"
    if prov.is_file():
        try:
            p: Any = json.loads(prov.read_text())
        except ValueError:
            return out
        # `repos`, not `sources`: that is the key `write_provenance` actually writes
        # (`provenance.py:114`), and reading the wrong one made every row of this block report `?`
        # while looking like it had been derived. `build_appworld_table` reads `repos` too.
        repos = cast("dict[str, Any]", p).get("repos") if isinstance(p, dict) else None
        jaz = cast("dict[str, Any]", repos).get("jaz") if isinstance(repos, dict) else None
        if isinstance(jaz, dict):
            entry = cast("dict[str, Any]", jaz)
            out["jaz"] = str(entry.get("commit", "?"))[:9]
            out["dirty"] = "dirty" if entry.get("dirty") else "clean"
    return out


def collect(
    arm: Arm, manifest: dict[str, list[str]], runs_root: Path
) -> tuple[dict[str, list[float]], list[dict[str, str]], list[Counts]]:
    """Per-rep metrics, provenance and episode shape for one arm, in rep order.

    Every `attempt-*` of every matched run is a rep. Attempts are sorted numerically, not
    lexically, so a tenth attempt does not sort between the first and the second.
    """
    data: dict[str, list[float]] = {c: [] for c in COLUMNS}
    prov: list[dict[str, str]] = []
    counts: list[Counts] = []
    for pattern in manifest.get(arm.key, []):
        for run in sorted(runs_root.glob(pattern)):
            attempts = sorted(
                (a for a in run.glob("attempt-*") if (a / "results.json").is_file()),
                key=lambda a: int(a.name.rsplit("-", 1)[-1]),
            )
            for attempt in attempts:
                m = _metrics(attempt)
                for c in COLUMNS:
                    data[c].append(m[c])
                prov.append(_provenance(run, attempt))
                counts.append(_counts(attempt))
    return data, prov, counts


def _cell(value: float, sem: float, mark: str | None) -> str:
    # `---` for a metric an arm does not report, matching the AppWorld table. `aggregate_marks`
    # already skips NaN when picking best/second, so without this the two disagree: no arm would be
    # marked but every arm would still print `nan $\pm$ nan`.
    if math.isnan(value):
        return "---"
    body = d1(value) if mark is None else f"{mark}{{{d1(value)}}}"
    return f"{body} $\\pm$ {d1(sem)}"


def _aggregate_table(
    labels: list[str],
    prompt_only: list[str],
    stats: dict[str, list[float]],
    sems: dict[str, list[float]],
) -> str:
    marks = aggregate_marks(stats, COLUMNS, LOWER_IS_BETTER)
    rows = [
        "\\begin{tabular}{rcccccc}",
        "\\toprule",
        # The benchmark is cited IN THE TABLE, not only in a `%` comment: a LaTeX comment renders
        # nowhere, and StuLife's authors made their permission to redistribute task descriptions
        # conditional on citation. Every method in this table carries a `\\citep`; the benchmark the
        # whole table measures must too.
        " & & \\multicolumn{2}{c}{\\textsc{StuLife}~\\citep{stulife2025} (all)}",
        " & \\multicolumn{2}{c}{\\textsc{StuLife} (far recall)} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "& \\smash{\\shortstack{prompt-\\\\only?\\,*}} & "
        + " & ".join(COLUMN_HEADS[c] for c in COLUMNS)
        + " \\\\",
        "\\midrule",
    ]
    for i, label in enumerate(labels):
        cells = [_cell(stats[c][i], sems[c][i], marks[c].get(i)) for c in COLUMNS]
        # The last arm is the method under test; a \midrule separates it from the baselines.
        if i == len(labels) - 1:
            rows.append("\\midrule")
        rows.append(f"{label} & {prompt_only[i]} & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


def _per_rep_table(labels: list[str], per_arm: list[dict[str, list[float]]], counts: Counts) -> str:
    rows = [
        "\\begin{tabular}{rlccccc}",
        "\\toprule",
        f" & & \\multicolumn{{5}}{{c}}{{StuLife ({counts.tasks} tasks, {counts.scored} scored)}} \\\\",
        "\\cmidrule(lr){3-7}",
        " & rep & " + " & ".join(COLUMN_HEADS[c] for c in COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for label, data in zip(labels, per_arm, strict=True):
        n = len(data[COLUMNS[0]])
        for r in range(n):
            head = f"\\multirow{{{n}}}{{*}}{{{label}}}" if r == 0 else ""
            rows.append(f"{head} & {r + 1} & " + " & ".join(d1(data[c][r]) for c in COLUMNS) + " \\\\")
        rows.append("\\cmidrule(lr){2-7}")
        # No bold/underline on rep rows: a single rep is not a "best" result, and marking one
        # invites reading across arms at a rep index that means nothing (reps are not paired).
        means = [st.mean(data[c]) for c in COLUMNS]
        sems = [st.stdev(data[c]) / math.sqrt(n) if n > 1 else 0.0 for c in COLUMNS]
        rows.append(
            " & mean & "
            + " & ".join(f"{d1(m)}$^{{\\pm{d1(s)}}}$" for m, s in zip(means, sems, strict=True))
            + " \\\\"
        )
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    rows.append("\\end{tabular}")
    return "\n".join(rows)


def _source_lines(manifest: dict[str, list[str]]) -> str:
    """The `%   <arm>  <glob>` provenance block, rendered from the manifest actually in use."""
    out: list[str] = []
    for arm in ARMS:
        for i, pattern in enumerate(manifest.get(arm.key, [])):
            out.append(f"%   {arm.short if i == 0 else '':<30} {pattern}")
    return "\n".join(out)


def _provenance_block(prov_by_arm: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "% PROVENANCE IS DERIVED, NOT ASSERTED -- every row below is read from the run directory",
        "% and its `provenance.json`, so a claim here cannot drift from the runs the numbers came",
        "% from. A `?` means the run recorded no provenance.",
        "%",
        "% 'jaz commit' is the commit of the jaz submodule (this project's agent framework dependency,",
        "% installed editable) that rep actually ran -- see src/jaz_evals/provenance.py for why the",
        "% commit, not the version pin, is what is recorded. 'tree' is that same checkout's state at",
        "% launch: 'dirty' means it had uncommitted tracked changes and/or untracked files. A tracked",
        "% diff, if there was one, was written to a `jaz.diff` alongside that run's `provenance.json`",
        "% (never the untracked files themselves, and no file at all when the tree was dirty only from",
        "% untracked ones) -- not reproduced in this table.",
        "%",
        f"%   {'arm':<32}{'rep':<11}{'jaz commit':<13}{'tree':<8}run",
    ]
    for short, reps in prov_by_arm.items():
        for i, p in enumerate(reps):
            lines.append(
                f"%   {short if i == 0 else '':<32}{p.get('attempt', '?'):<11}"
                f"{p.get('jaz', '?'):<13}{p.get('dirty', '?'):<8}{p.get('run', '?')}"
            )
    return "\n".join(lines)


def _header(
    prov_by_arm: dict[str, list[dict[str, str]]],
    manifest: dict[str, list[str]],
    runs_root: Path,
    counts: Counts,
) -> str:
    return f"""% StuLife (full episode, {counts.tasks} tasks of which {counts.scored} are scored) results.
%
% BENCHMARK ATTRIBUTION. StuLife is the benchmark of "Building a Self-Evolving Agent via
% Experience-Driven Lifelong Learning: A Framework and Benchmark" (arXiv:2508.19005), and its authors
% ask that work releasing StuLife-derived material cite them. Emitted here rather than left to the
% paper, because this file is itself a released artifact and the obligation travels with it. The
% table header also carries a rendered \\citep{{stulife2025}}, since a % comment reaches no reader of
% the compiled paper -- SO THE BIBLIOGRAPHY MUST DEFINE THAT KEY. An unmatched key renders
% "StuLife [?]", which fails the condition quietly, in a generated file nobody hand-reads.
%
% LaTeX requirements: booktabs (\\toprule/\\midrule/\\cmidrule/\\bottomrule), multirow, natbib or
% biblatex (\\citep), listings (\\lstinline), a \\checkmark and \\xmark source (amssymb + pifont, or
% bbding), and the paper's own \\oursimpl macro. Bibliography keys used: stulife2025,
% wang2024codeact, roucher2025smolagents, packer2023memgpt.
%
% GENERATED FILE -- do not hand-edit. Rebuild with:
%     uv run python scripts/build_stulife_table.py
% and verify it is current with `--check`. Every number below is computed from run artifacts by
% that script.
%
% Two tables: the aggregate (mean +/- SEM over reps) and the per-rep breakdown. In the aggregate
% table, bold marks the best value in a column and underline the second-best, computed across all
% arms including baselines; per-rep rows carry neither (a single rep is not a "best" result). The
% `*` marker in the aggregate table's "prompt-only?" column header references a note that lives in
% the paper, not in this file.
%
% Sources, all globs relative to --runs-root, which for this build was:
%   {runs_root_line(runs_root)}
{_source_lines(manifest)}
%
% Some directory names above carry a same-day build label, e.g. "-post-stdout-fix-" or
% "-subagent-rename-": a one-time note from whoever launched the run about which code change was
% topical that day, not a caveat about the run itself. Like the paths above, none of this survives
% a fresh clone -- only the generator does.
%
% A REP IS AN ATTEMPT. Each arm here is one run directory launched with `--attempts 3`, so its
% three reps are three attempts of that run -- unlike the AppWorld table, where a rep is a separate
% run directory. The per-rep table's `rep` column is the attempt directory.
%
% How each column is computed, all from attempt-*/results.json:
%   Pass       extra.pass_rate -- the fraction of SCORED tasks fully passed. StuLife's {counts.tasks}
%              tasks include {counts.triggers} trigger rows, which carry no score, so every rate is
%              out of {counts.scored}.
%   Score      extra.avg_score -- mean partial credit over the same {counts.scored}.
%   far recall the same two quantities restricted to tasks whose answer was taught far enough back
%              to have left the context window (extra.*_far_recall, n={counts.far} here).
%              This is the subset the long-horizon claim rests on.
%   Cost       usage.cost_usd -- the whole attempt.
%
% +/- is the standard error of the mean over reps, NOT a standard deviation: it describes the
% precision of the reported mean, which is what a reader comparing two arms needs.
%
{_provenance_block(prov_by_arm)}"""


def build(manifest: dict[str, list[str]], runs_root: Path, manifest_path: Path) -> str:
    labels: list[str] = []
    prompt_only: list[str] = []
    per_arm: list[dict[str, list[float]]] = []
    prov_by_arm: dict[str, list[dict[str, str]]] = {}
    all_counts: list[Counts] = []
    for arm in ARMS:
        data, prov, counts = collect(arm, manifest, runs_root)
        prov_by_arm[arm.short] = prov
        all_counts.extend(counts)
        labels.append(arm.label)
        prompt_only.append(arm.prompt_only)
        per_arm.append(data)
    # A checkout without `runs/` (any fresh clone -- it is gitignored) reaches here with empty
    # column lists and would die in the statistics below with an unhelpful IndexError. Name it.
    empty = [arm.label for arm, data in zip(ARMS, per_arm, strict=True) if not data[COLUMNS[0]]]
    if empty:
        raise SystemExit(
            f"no run data found for: {', '.join(empty)}, under --runs-root {runs_root}.\n"
            "Either that tree has no `runs/` (it is gitignored, so a fresh clone has none -- point "
            "--runs-root at the checkout the runs were produced in), or an arm's globs in "
            f"{manifest_path} no longer match any directory because a run was renamed, moved or "
            "archived."
        )
    # One episode shape for the whole table, or the header's counts describe only some of it. Reps
    # that disagree mean the arms ran different task files, which no caption could honestly cover.
    shapes = set(all_counts)
    if len(shapes) > 1:
        raise SystemExit(
            f"the matched runs disagree on the episode's shape: {sorted(shapes)}. Every arm must be "
            "scored over the same task file for one table to describe them."
        )
    counts = all_counts[0]
    stats = {c: [st.mean(d[c]) for d in per_arm] for c in COLUMNS}
    sems = {
        c: [st.stdev(d[c]) / math.sqrt(len(d[c])) if len(d[c]) > 1 else 0.0 for d in per_arm] for c in COLUMNS
    }
    return "\n".join(
        [
            _header(prov_by_arm, manifest, runs_root, counts),
            "",
            "% " + "-" * 73,
            "% Aggregate: mean +/- SEM over reps.",
            "% " + "-" * 73,
            _aggregate_table(labels, prompt_only, stats, sems),
            "",
            "% " + "-" * 73,
            "% Per-rep breakdown.",
            "% " + "-" * 73,
            _per_rep_table(labels, per_arm, counts),
            "",
        ]
    )


def _rerun(args: argparse.Namespace) -> str:
    return rerun_command(
        "build_stulife_table.py",
        args,
        {"--runs-manifest": DEFAULT_MANIFEST, "--runs-root": REPO, "--out": OUT},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed table is stale")
    parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON mapping arm key -> run-directory globs (default: scripts/stulife_runs.json)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO,
        help="directory the manifest's globs are resolved against (default: the repo root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help=f"where to write the table (default: {OUT.relative_to(REPO)})",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.runs_manifest)
    out: Path = args.out
    rendered = build(manifest, args.runs_root.resolve(), args.runs_manifest)
    if args.check:
        if not out.exists():
            print(f"{display(out)} does not exist -- run `{_rerun(args)}`", file=sys.stderr)
            return 1
        current = out.read_text()
        if current == rendered:
            print(f"{display(out)} is up to date")
            return 0
        print(f"{display(out)} differs from a fresh build:", file=sys.stderr)
        for line in describe_text_difference(current, rendered):
            print(line, file=sys.stderr)
        print(f"\n  Rebuild with: {_rerun(args)}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(f"wrote {display(out)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
