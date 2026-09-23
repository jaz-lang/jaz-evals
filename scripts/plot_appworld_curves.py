"""Per-piece performance curves over the AppWorld queue, one line per arm.

Splits the 417-task `test_challenge` queue into 5 roughly equal pieces IN QUEUE ORDER (the seed-42
order the self-improving arms actually experienced), and for each piece plots a summary of each
arm's pass rate across its reps, with a band for spread.

Two summaries, chosen with `--stat`, written to different files so both can be kept:

  median (default) -- an order-statistic band, and NOT the same statistic for every arm, because the
      arms do not have the same number of reps: min/max over a 3-rep baseline, x_(2)/x_(5) over a
      6-rep self-improving method (the 78.13%-coverage interval the results table uses; see
      `build_appworld_table.py`). Distribution-free, and every band edge is an observed value. The
      legend names the band per line, because min/max of 3 and x_(2)/x_(5) of 6 are not comparable
      widths and a reader should not have to infer which one a line carries.
  mean -- mean +/- 1 SEM, the summary the results table quotes, so the two artefacts agree.

They are not interchangeable. +/-1 SEM is a ~68% interval resting on a normality assumption 3 and 6
reps cannot support, and it is NARROWER than the order-statistic band on the same data -- the mean
panel therefore LOOKS more precise than the median panel while resting on more. Prefer the median
panel for reading the data and the mean panel for agreeing with the table.

READ THE CURVES WITH THE BASELINES IN VIEW. A rising line does NOT by itself show a method
improving: the pieces are not equally hard, and the two arms that carry nothing between tasks rise
too (CodeAct's median goes 66.7 -> 74.7 across the queue). Whatever an arm gains has to be read
against that drift, which is what the lower panel does -- each arm minus the CodeAct baseline on the
SAME piece, so flat there means "tracks the difficulty curve and no more". The lower panel is the
one that answers "did it improve"; the upper one mostly shows how hard each piece was.

Piece count is a real analysis knob, not a display one: fewer pieces means more tasks per point and
so a tighter band, at the cost of resolution on when a change happened. 5 pieces of ~83 tasks is the
default because at 10 the per-piece bands were wide enough to swamp the between-arm differences.
`--pieces` changes it.

THE FIGURE DOES NOT NAME ITS OWN BAND -- the paper's caption has to. Whichever `--stat` produced a
figure, say so in the caption along with the band it implies: for `median`, "band: min-max over 3
reps for the non-self-improving baselines, x_(2)-x_(5) over 6 for the self-improving methods"; for
`mean`, "band: +/-1 SEM". An earlier version put that in the title, which cost a line of the plot
area for something a caption states better.

Needs the optional `plots` extra (matplotlib): `uv sync --extra plots`. The import is inside `main`
so the rest of `scripts/` runs without it.

Usage:
    uv run python scripts/plot_appworld_curves.py                # median, tables/appworld_curves_median.pdf
    uv run python scripts/plot_appworld_curves.py --stat mean    # tables/appworld_curves_mean.pdf
    uv run python scripts/plot_appworld_curves.py --pieces 10
    uv run python scripts/plot_appworld_curves.py --print        # also dump the numbers as text

Reproducing with your OWN runs: the arms' run directories live in `scripts/appworld_runs.json`, shared
with `scripts/build_appworld_table.py`. Copy it, replace each glob with yours, and pass
`--runs-manifest yours.json`; globs resolve against `--runs-root` (default: the repo root). Pass
`--out-dir` too -- otherwise the figures built from your runs land on top of this repo's committed ones.
The official ReAct line needs artifacts that live outside this repo, at `--official-root` (default: the
original run's own machine); when they are absent at that path it is dropped from the figure (with a
note on stderr) rather than failing the build.

    uv run python scripts/plot_appworld_curves.py \
        --runs-manifest mine.json --runs-root /path/to/checkout --out-dir myfigs/ \
        --official-root /path/to/official/react/outputs
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

# The run manifest and its loader are shared with `build_appworld_table.py` -- one definition, so the
# table and the curves are always built from the same runs. They previously held separate hardcoded
# copies of these globs, which could drift apart with nothing detecting it.
# `append`, not `insert(0, ...)`: this only needs `scripts/` on the path at all, and prepending
# it would let a file here shadow a stdlib module for the rest of the process.
sys.path.append(str(Path(__file__).resolve().parent))
from _artifacts import (
    REPO,
    display,
    load_manifest,
    report_pdf_check,
    rerun_command,
)
from _curves import (
    JAZ_COLOUR,
    JAZ_LABEL,
    PANEL_SIZE,
    Series,
    apply_y_break,
    assert_legend_visible,
    break_is_warranted,
    draw_panel,
    draw_y_break_mark,
    emphasise_jaz_entry,
    legend_below,
    ordered_handles,
    piece_bounds,
    set_theme,
)
from build_appworld_table import DEFAULT_MANIFEST, _official_evaluation


# PDF only. It is vector (so it scales in the paper), it is what pdflatex includes with no conversion
# step, and it is the smallest of the formats matplotlib writes here -- measured on this figure,
# PDF 26 KB vs SVG 100 KB vs PNG 264 KB. SVG and PNG were both dropped: neither had a consumer the
# PDF did not already serve, and they were 4-10x the bytes for a file nothing referenced.
#
# Four files, because the absolute and difficulty-controlled views are separate figures rather than
# stacked panels: they answer different questions ("how hard was each piece" vs "did the method
# improve"), and a paper will usually want only one of them, at full column width rather than half a
# stacked pair.
def out_path(out_dir: Path, stat: str, relative: bool) -> Path:
    """`<out_dir>/appworld_curves_<stat>[_relative].pdf`."""
    # The directory is a parameter rather than a constant because `--runs-root` made this script
    # readable from another checkout's runs while it still wrote into THIS repo's `tables/` -- so a
    # reproducer plotting their own runs silently overwrote the paper's committed figures.
    suffix = "_relative" if relative else ""
    return out_dir / f"appworld_curves_{stat}{suffix}.pdf"


N_PIECES = 5
N_TASKS = 417

# The official ReAct baseline's artifacts live outside this repo; see build_appworld_table.py. This is
# a default, not a hardcoded requirement -- `--official-root` overrides it, since a reproducer's copy
# of those artifacts is never at this path.
OFFICIAL_ROOT = Path(
    "/scratch/zli11010/jaz/experiments/outputs/simplified_react_code_agent/openai/gpt-5.4-nano-high-reasoning"
)
#: One arm: (label, manifest key, self_improving, colour, linestyle). The manifest key is empty for
#: the official baseline, which is read from its own artifacts rather than from `runs/`.
Arm = tuple[str, str, bool, str, str]

# The arm the lower/relative figure divides out. Named once, and used both as an ARMS label and as
# the lookup key, so a change to the rendered label cannot silently stop matching the lookup -- which
# it would have done when these labels became mathtext.
BASELINE = r"$\mathrm{CodeAct}_{\mathrm{JAZ}}$"

# Legend order, which is NOT the draw order. Arms are drawn baselines-first so the black reference
# lines sit behind the coloured methods; the legend instead leads with the methods under test. With
# `ncol=2` matplotlib fills COLUMN-major, so this sequence puts the three methods down the left
# column and the two carry-nothing baselines down the right. Any label not listed here falls to the
# end, so adding an arm degrades to "appended" rather than to a KeyError.
LEGEND_ORDER: tuple[str, ...] = (
    JAZ_LABEL,
    r"$\mathrm{CodeAct{+}subagents}_{\mathrm{JAZ}}$",
    r"$\mathrm{ACE\ on\ CodeAct}_{\mathrm{JAZ}}$",
    r"$\mathrm{CodeAct}_{\mathrm{JAZ}}$",
    r"$\mathrm{CodeAct}_{\mathrm{AppWorld\ official}}$",
)

# (label, manifest key, self_improving, colour, linestyle). `self_improving` picks the band statistic
# AND is what the reader needs in order to know which lines a rising curve is even interesting for.
#
# The two arms that carry nothing between tasks are BLACK -- solid for CodeAct, dotted for the
# official ReAct baseline -- so the reference lines read as reference lines and colour is reserved for
# the methods under test. Their bands are drawn fainter for the same reason (`_curves.draw_panel`
# keys both treatments off the colour).
#
# The two ablations take "C3"/"C0" rather than hex: those resolve against whatever prop_cycle is
# active, so they track seaborn's theme instead of pinning a colour that would clash if the theme
# changed. JAZ invoke is the one hardcoded hue (`_curves.JAZ_COLOUR`), because it is the method the
# figure is about: it is also drawn thicker than every other line, so the figure has one visual
# subject rather than five equal ones.
# BROKEN Y AXIS, on the two views that earn one -- see `_curves.apply_y_break`. The official
# baseline runs far below every other arm, so an empty band sits between the two groups and a
# continuous axis spends the panel on it. Keyed by (stat, relative), because the four views this
# script writes do not have the same band: the order-statistic bands the median views draw are wide
# enough at n=3/6 to close most of the gap, while +/-1 SEM leaves it open.
#
# MEASURED empty gap, as a share of each view's full y range (band edges, not centre lines):
#
#     mean   absolute    7.14 pts of 40.6   17.6%   broken
#     mean   relative    7.10 pts of 39.0   18.2%   broken
#     median absolute    3.61 pts of 45.8    7.9%   left alone
#     median relative    2.32 pts of 45.6    5.1%   left alone
#
# The two left alone would save less axis than the zigzag and its clearances cost, so they would
# trade a continuous axis for nothing. A break earns its place or it does not get one; `assert_
# break_is_empty` below refuses the build if a band ever creeps into one of the two that do.
#
# `("mean", False)` is byte-for-byte the band `plot_combined_curves.py` uses, on purpose: that
# figure's AppWorld panel IS this view, and a reader moving between them must not find the same
# arm drawn against two different axes.
Y_BREAKS: dict[tuple[str, bool], tuple[float, float, tuple[float, ...]]] = {
    ("mean", False): (56.0, 61.5, (45.0, 50.0, 55.0, 65.0, 70.0, 75.0, 80.0)),
    ("mean", True): (-8.5, -3.5, (-25.0, -20.0, -15.0, -10.0, 0.0, 5.0, 10.0)),
}


ARMS: tuple[Arm, ...] = (
    (r"$\mathrm{CodeAct}_{\mathrm{AppWorld\ official}}$", "", False, "black", ":"),
    (
        BASELINE,
        "codeact",
        False,
        "black",
        "-",
    ),
    (
        r"$\mathrm{CodeAct{+}subagents}_{\mathrm{JAZ}}$",
        "codeact_subagents",
        True,
        "C3",
        "-",
    ),
    (
        r"$\mathrm{ACE\ on\ CodeAct}_{\mathrm{JAZ}}$",
        "ace_on_codeact",
        True,
        # C2 (green), not C0: Letta is C0 in the StuLife panel, and the combined figures put both
        # arms under ONE legend, where two identical blue entries read as one arm drawn twice. The
        # colours are shared across figures on purpose, so the fix belongs at the arm, not in the
        # figure that exposed the clash.
        "C2",
        "-",
    ),
    (
        JAZ_LABEL,
        "jaz_invoke",
        True,
        JAZ_COLOUR,
        "-",
    ),
)


def _queue_order(manifest: dict[str, list[str]], runs_root: Path) -> list[str]:
    """Task ids in seed-42 queue order, read from the first run the manifest matches.

    Needed because the official baseline ran the split in AppWorld's own order, so its per-task
    results have to be re-indexed onto the order our arms saw before the pieces mean the same tasks.
    """
    # Taken from whichever run is available rather than one named path: every arm ran the same seeded
    # queue, so any of them supplies the order, and a reproducer's runs have different directory names.
    for _label, key, *_ in ARMS:
        for pattern in manifest.get(key, []):
            for run in sorted(runs_root.glob(pattern)):
                path = run / "attempt-0" / "task_results.jsonl"
                if not path.is_file():
                    continue
                rows = [json.loads(line) for line in path.open()]
                # A run that stopped early has a SHORT queue, and the official baseline's curve is
                # indexed against `piece_bounds()`, which is computed from N_TASKS regardless -- so a
                # partial order would silently drop tasks off the end of the last piece and divide by
                # the full piece size anyway. Skip it and keep looking for a complete run.
                if len(rows) != N_TASKS:
                    continue
                rows.sort(key=lambda r: r["task_index"])
                return [r["task_id"] for r in rows]
    raise SystemExit(
        f"no complete {N_TASKS}-task run matched the manifest, so the seed-42 queue order cannot be "
        "recovered. Pass --runs-root / --runs-manifest pointing at your runs."
    )


def _our_curves(globs: list[str], bounds: list[tuple[int, int]], runs_root: Path) -> list[list[float]]:
    curves: list[list[float]] = []
    for pattern in globs:
        for run in sorted(runs_root.glob(pattern)):
            rows = [json.loads(line) for line in (run / "attempt-0" / "task_results.jsonl").open()]
            rows.sort(key=lambda r: r["task_index"])
            curves.append([100 * sum(1 for r in rows[a:b] if r["success"]) / (b - a) for a, b in bounds])
    return curves


def _official_available(official_root: Path = OFFICIAL_ROOT) -> bool:
    """Whether the official ReAct baseline's own evaluation JSONs are on this machine."""
    # `build_appworld_table.py` falls back to recorded constants when they are absent, but those are
    # per-arm aggregates and this figure needs per-TASK outcomes, which nothing here records. So the
    # honest degradation is to drop the line and say so, rather than to fail the whole figure -- the
    # four JAZ arms are reproducible from `runs/` alone and that is what a reproducer has.
    return all(_official_evaluation(official_root, rep) is not None for rep in (1, 2, 3))


def _official_curves(
    bounds: list[tuple[int, int]],
    manifest: dict[str, list[str]],
    runs_root: Path,
    official_root: Path = OFFICIAL_ROOT,
) -> list[list[float]]:
    order = _queue_order(manifest, runs_root)
    curves: list[list[float]] = []
    for rep in (1, 2, 3):
        evaluation = _official_evaluation(official_root, rep)
        if evaluation is None:
            continue
        individual = json.loads(evaluation.read_text())["individual"]
        ok = [bool(individual[t]["success"]) for t in order]
        curves.append([100 * sum(ok[a:b]) / (b - a) for a, b in bounds])
    return curves


def band(values: list[float], self_improving: bool) -> tuple[float, float]:
    """Low/high edge of the shaded region for one piece.

    n=6 self-improving arms get x_(2)/x_(5) -- the 78.13%-coverage order-statistic interval. n=3
    baselines get min/max, because at n=3 the only order-statistic interval IS the range, and
    dropping to x_(2)/x_(2) would collapse the band to the median.
    """
    s = sorted(values)
    return (s[1], s[-2]) if self_improving and len(s) >= 6 else (s[0], s[-1])


def centre_and_band(values: list[float], self_improving: bool, stat: str) -> tuple[float, float, float]:
    """`(centre, lo, hi)` for one piece under the chosen summary statistic.

    `median` pairs the median with an order-statistic band (see `band`) -- distribution-free, and the
    band edges are observed values. `mean` pairs the mean with +/- one SEM, which is the same summary
    the results table reports, so the two artefacts agree.

    They are not interchangeable, and the difference is not cosmetic at these n. A +/-1 SEM band is
    a ~68% interval under a normal assumption the 3- and 6-rep samples cannot support, and it is
    NARROWER than the order-statistic band -- so the mean panel will look more precise than the
    median panel on identical data. It is offered because SEM is what the table quotes and a reader
    comparing the two should see the same statistic, not because it is the better estimate here.
    """
    if stat == "mean":
        m = st.mean(values)
        sem = st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return m, m - sem, m + sem
    lo, hi = band(values, self_improving)
    return st.median(values), lo, hi


def available_arms(
    arms: Sequence[Arm] = ARMS, *, quiet: bool = False, official_root: Path = OFFICIAL_ROOT
) -> list[Arm]:
    """`arms` minus the official baseline when its artifacts are not at `official_root`.

    Prints a note to stderr when it drops the arm, unless `quiet`.
    """
    # Every caller has to do this, and the one that did not -- the combined figures -- died with a
    # bare `FileNotFoundError` naming a `/scratch/...` path, which made both of them unbuildable on
    # any machine but the one they were written on. Filtering belongs beside `_official_available`,
    # not in each `main`.
    if _official_available(official_root):
        return list(arms)
    if not quiet:
        print(
            f"note: the official ReAct baseline's artifacts are not at {official_root}, "
            "so that line is omitted from the figures.",
            file=sys.stderr,
        )
    return [a for a in arms if a[1]]


def read_curves(
    bounds: list[tuple[int, int]],
    manifest: dict[str, list[str]],
    runs_root: Path,
    arms: Sequence[Arm] | None = None,
    *,
    manifest_path: Path | None = None,
    official_root: Path = OFFICIAL_ROOT,
) -> dict[str, list[list[float]]]:
    """Per-arm, per-rep, per-piece pass rates, for the arms this machine can actually build.

    `arms` defaults to `available_arms()`. Raises `SystemExit` naming the arms whose globs matched
    nothing; `manifest_path` only sharpens that message.

    Module-level rather than inline in `main` because `plot_combined_curves.py` and
    `plot_combined_cost.py` draw the same data into a shared figure; burying it in a closure is what
    forces those to re-implement the reading and drift.
    """
    arms = available_arms(official_root=official_root) if arms is None else arms
    curves = {
        label: (
            _official_curves(bounds, manifest, runs_root, official_root)
            if not key
            else _our_curves(manifest.get(key, []), bounds, runs_root)
        )
        for label, key, _si, _c, _ls in arms
    }
    # Same failure the table script guards: an arm whose globs match nothing yields an empty rep list
    # and dies in `band()` with `IndexError: list index out of range`, which names neither the arm nor
    # the cause. `runs/` is gitignored, so a fresh clone hits this on every arm at once. It lives here
    # rather than in `main` so the combined figures inherit it -- they used to reach `band()` instead.
    empty = [label for label, key, *_ in arms if key and not curves[label]]
    if empty:
        where = f" in {manifest_path}" if manifest_path else " in the run manifest"
        raise SystemExit(
            f"no run data found for: {', '.join(empty)}, under --runs-root {runs_root}.\n"
            "Either that tree has no `runs/` (it is gitignored, so a fresh clone has none -- point "
            "--runs-root at the checkout the runs were produced in), or an arm's globs"
            f"{where} no longer match any directory because a run was renamed, moved or archived."
        )
    return curves


def absolute_series(
    curves: dict[str, list[list[float]]],
    stat: str,
    arms: Sequence[Arm] | None = None,
) -> list[Series]:
    """One line per arm: the centre and band of its per-piece pass rate.

    `arms` defaults to the arms `curves` actually holds, in `ARMS` order.
    """
    # Which arms exist is decided by `read_curves`, so the two cannot disagree: an arm dropped
    # for want of artifacts is absent from `curves` and therefore absent from the lines.
    arms = [a for a in ARMS if a[0] in curves] if arms is None else arms
    out: list[Series] = []
    for label, _key, si, colour, ls in arms:
        reps = curves.get(label) or []
        if not reps:
            continue
        cb = [centre_and_band([c[i] for c in reps], si, stat) for i in range(len(reps[0]))]
        out.append(
            Series(
                label=label,
                colour=colour,
                linestyle=ls,
                centre=[x[0] for x in cb],
                lo=[x[1] for x in cb],
                hi=[x[2] for x in cb],
                n_runs=len(reps),
            )
        )
    return out


def relative_series(
    curves: dict[str, list[list[float]]],
    stat: str,
    arms: Sequence[Arm] | None = None,
) -> list[Series]:
    """Each arm minus the baseline's centre on the same piece, so difficulty divides out.

    `arms` defaults to the arms `curves` actually holds, in `ARMS` order.
    """
    # Which arms exist is decided by `read_curves`, so the two cannot disagree: an arm dropped
    # for want of artifacts is absent from `curves` and therefore absent from the lines.
    arms = [a for a in ARMS if a[0] in curves] if arms is None else arms
    reps0 = curves.get(BASELINE) or []
    n = len(reps0[0]) if reps0 else 0
    base = [centre_and_band([c[i] for c in reps0], False, stat)[0] for i in range(n)]
    return [
        s._replace(
            centre=[v - base[i] for i, v in enumerate(s.centre)],
            lo=[v - base[i] for i, v in enumerate(s.lo)],
            hi=[v - base[i] for i, v in enumerate(s.hi)],
        )
        for s in absolute_series(curves, stat, arms)
        if s.label != BASELINE
    ]


def _rebuild_command(args: argparse.Namespace) -> str:
    return rerun_command(
        "plot_appworld_curves.py",
        args,
        {
            "--runs-manifest": DEFAULT_MANIFEST,
            "--runs-root": REPO,
            "--out-dir": REPO / "tables",
            "--stat": "median",
            "--pieces": N_PIECES,
            "--official-root": OFFICIAL_ROOT,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="dump", help="also print the numbers")
    parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON mapping arm key -> run-directory globs (default: scripts/appworld_runs.json)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO,
        help="directory the manifest's globs are resolved against (default: the repo root)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "tables",
        help="directory to write the PDFs into (default: tables/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed figures are stale; writes nothing",
    )
    parser.add_argument(
        "--pieces", type=int, default=N_PIECES, help=f"number of queue pieces (default {N_PIECES})"
    )
    parser.add_argument(
        "--stat",
        choices=("median", "mean"),
        default="median",
        help="median with an order-statistic band (default), or mean with +/- 1 SEM",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=OFFICIAL_ROOT,
        help="AppWorld's experiment-output root for the official ReAct baseline "
        "(default: the path these runs were produced at; that arm is dropped, with a note, "
        "when it is absent)",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.runs_manifest)
    runs_root: Path = args.runs_root.resolve()

    # Imported here, not at module scope: matplotlib is the optional `plots` extra, and the rest of
    # scripts/ must stay runnable without it.
    plt = set_theme()

    bounds = piece_bounds(N_TASKS, args.pieces)
    arms = available_arms(official_root=args.official_root)
    curves = read_curves(
        bounds, manifest, runs_root, arms, manifest_path=args.runs_manifest, official_root=args.official_root
    )
    if BASELINE not in curves:
        raise SystemExit(f"the {BASELINE} arm is missing, so the relative panel has no baseline.")

    check_dir = tempfile.TemporaryDirectory() if args.check else None

    def figures_for(stat: str) -> tuple[list[Path], list[Series], list[Series]]:
        """Build both panels for one summary statistic; returns the paths and the two series."""
        abs_series = absolute_series(curves, stat, arms)
        # Relative panel: subtract the CodeAct baseline's per-piece MEDIAN -- the arm that carries
        # nothing between tasks, so its curve is the piece's difficulty and nothing else. Subtracting
        # a median rather than pairing rep-to-rep is deliberate: reps are not paired across arms
        # (different keys, different jaz commits), so a rep-to-rep difference would invent a pairing
        # the design lacks. `relative_series` subtracts the SAME statistic that centres the lines --
        # median from median, mean from mean -- so a single number never mixes two summaries.
        rel_series = relative_series(curves, stat, arms)

        # One figure per view, each one text column wide (`_curves.PANEL_SIZE` carries the
        # derivation). Separate files rather than stacked panels: the two answer different questions,
        # a paper will usually want one of them, and at this width a single panel gets the full
        # height instead of half of it.
        def render(series: list[Series], title: str, ylabel: str, relative: bool) -> Path:
            fig, ax = plt.subplots(figsize=PANEL_SIZE)
            draw_panel(ax, series, bounds, title, ylabel)
            break_spec = Y_BREAKS.get((stat, relative))
            if break_spec is not None and not break_is_warranted(
                ((s.label, min(s.lo), max(s.hi)) for s in series),
                break_spec[0],
                break_spec[1],
                f"the {stat} {'relative' if relative else 'absolute'} view",
            ):
                break_spec = None
            if break_spec is not None:
                apply_y_break(ax, *break_spec)
            if relative:
                ax.axhline(0, color="#555", linewidth=0.9, linestyle="--", zorder=1)
            # `fontsize=6.8, y=-0.32`, matching the two StuLife scripts rather than taking
            # `legend_below`'s defaults. Both halves matter. The `y`: at -0.22 the legend's top ink sits
            # INSIDE the x-label on this 3.125in panel -- measured -0.054in on the absolute view and
            # -0.021in on the relative one -- and `assert_legend_visible` cannot see it, because it
            # checks the figure's edges, not the axis furniture. Measured clearance at -0.32 is +0.047in
            # and +0.094in. The `fontsize`: these four figures and the StuLife four are printed at the
            # same size in the same paper, and the legend is the one element a reader compares directly
            # across them, so they take one size rather than 7.4 here and 6.8 there.
            legend = legend_below(ax, *ordered_handles([ax], LEGEND_ORDER), ncol=2, fontsize=6.8, y=-0.32)
            emphasise_jaz_entry(legend)
            fig.tight_layout(pad=0.5)
            assert_legend_visible(legend)
            # Last, so the mark is placed against the final limits.
            if break_spec is not None:
                draw_y_break_mark(ax, break_spec[0], break_spec[1])
            # Under --check the figure is built into a scratch directory and compared, never written
            # over the committed one: a gate that has to clobber the artifact to tell you it is stale
            # is not a gate.
            target_dir = Path(check_dir.name) if check_dir is not None else args.out_dir
            path = out_path(target_dir, stat, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            plt.close(fig)
            return path

        built = [
            render(abs_series, "AppWorld self-improvement", "pass rate (%)", relative=False),
            render(
                rel_series,
                f"AppWorld self-improvement, minus {BASELINE}",
                "pass rate difference (points)",
                relative=True,
            ),
        ]
        return built, abs_series, rel_series

    # `--check` builds BOTH statistics, because all four PDFs are committed and a gate that only
    # looked at the selected one could pass while the other two were stale.
    stats = ("median", "mean") if args.check else (args.stat,)
    written: list[Path] = []
    abs_series: list[Series] = []
    rel_series: list[Series] = []
    for stat in stats:
        paths, abs_series, rel_series = figures_for(stat)
        written += paths

    if args.dump:
        head = "  " + "arm".ljust(21) + "".join(f"{f'{a + 1}-{b}':>10}" for a, b in bounds)
        print(f"\n  {args.stat} pass rate per piece")
        print(head)
        for s in abs_series:
            print(f"  {s.label:<21}" + "".join(f"{v:>10.1f}" for v in s.centre))
        print(f"\n  {args.stat} minus CodeAct on the same piece")
        print(head)
        for s in rel_series:
            print(f"  {s.label:<21}" + "".join(f"{v:>+10.1f}" for v in s.centre))

    # `--print` is honoured under `--check` too: the numbers are what a reader wants when the gate
    # says a figure moved.
    if check_dir is not None:
        with check_dir:
            # `with`, so the scratch figures are removed even on an early return.
            return report_pdf_check(written, args.out_dir, _rebuild_command(args))

    print("wrote " + ", ".join(display(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
