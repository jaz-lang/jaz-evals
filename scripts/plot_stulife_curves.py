"""Per-piece performance curves over the StuLife episode, one line per arm.

The StuLife counterpart to `plot_appworld_curves.py`. Splits the 939 SCORED tasks into 5 roughly
equal pieces IN EPISODE ORDER and plots each arm's pass rate per piece, with a band for spread.

WHAT THE X-AXIS MEANS HERE, AND WHY IT IS NOT THE APPWORLD FIGURE. On AppWorld the queue is the
axis a self-improving method could improve along, so a rising line is the question. StuLife's
episode is a *simulated life*: later tasks are not harder versions of earlier ones, they are later
in time, and the thing that changes across the episode is how far back the answer was taught. So a
falling line is the interesting direction -- it is memory decaying, not the method failing to
improve -- and the arms that carry nothing between tasks fall hardest. Read it with the lower panel,
which divides out the per-piece difficulty the same way the AppWorld figure does.

Trigger tasks are excluded. StuLife's 1284 rows include 345 triggers that carry no score, so the
pieces are cut over the 939 scored tasks; including them would put ~27% of the axis on rows that
can never pass and make every arm look worse in the same places.

Two summaries, chosen with `--stat`, written to different files so both can be kept:

  median (default) -- an order-statistic band: min/max over the 3 reps each arm has here. Every
      band edge is an observed value, and it assumes nothing about the distribution.
  mean -- mean +/- 1 SEM, the summary the results table quotes, so the two artefacts agree.

They are not interchangeable. +/-1 SEM is a ~68% interval resting on a normality assumption 3 reps
cannot support, and it is NARROWER than the order-statistic band on the same data -- so the mean
panel LOOKS more precise while resting on more.

Needs the optional `plots` extra (matplotlib): `uv sync --extra plots`. The import is inside `main`
so the rest of `scripts/` runs without it.

Usage:
    uv run python scripts/plot_stulife_curves.py               # median, tables/stulife_curves_median.pdf
    uv run python scripts/plot_stulife_curves.py --stat mean
    uv run python scripts/plot_stulife_curves.py --check       # exits 1 if the committed figures are stale
    uv run python scripts/plot_stulife_curves.py --print       # also dump the numbers as text

Reproducing with your OWN runs: the arms' run directories live in `scripts/stulife_runs.json`,
shared with `scripts/build_stulife_table.py`. Copy it, replace each glob with yours, and pass
`--runs-manifest yours.json`; globs resolve against `--runs-root`. Pass `--out-dir` too, or the
figures built from your runs land on top of this repo's committed ones.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import tempfile
from pathlib import Path

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
    assert_legend_visible,
    draw_panel,
    emphasise_jaz_entry,
    legend_below,
    ordered_handles,
    piece_bounds,
    set_theme,
)
from build_stulife_table import DEFAULT_MANIFEST

N_PIECES = 5
BASELINE = r"$\mathrm{CodeAct}_{\mathrm{JAZ}}$"
# Named for the same reason `BASELINE` is, and now load-bearing: labels carry cross-figure meaning,
# and `ordered_handles` dedupes by label STRING -- so a one-character drift between two copies of a
# spelling yields a duplicate legend entry (or an arm sorted to the end) rather than an error.
SMOLAGENTS = r"$\mathrm{CodeAct{+}subagents}_{\mathrm{smolagents}}$"

# (label, manifest key, colour, linestyle). Ordered baselines-first so the black reference lines are
# drawn behind the coloured methods under test.
#
# COLOUR IS THE METHOD, LINESTYLE IS THE IMPLEMENTATION, matching the AppWorld figure: there,
# `CodeAct_{AppWorld official}` is black dotted against `CodeAct_{JAZ}` black solid -- one method, two
# implementations. So the smolagents arm takes C3 dotted against `CodeAct+subagents_{JAZ}` C3 solid.
# It was black dotted, which read as a third per-task baseline rather than as the other implementation
# of the arm beside it.
ARMS: tuple[tuple[str, str, str, str], ...] = (
    (BASELINE, "codeact_per_task", "black", "-"),
    (SMOLAGENTS, "codeact_subagents_smolagents", "C3", ":"),
    (r"$\mathrm{CodeAct{+}subagents}_{\mathrm{JAZ}}$", "codeact_subagents", "C3", "-"),
    (r"$\mathrm{Letta}$", "letta", "C0", "-"),
    (JAZ_LABEL, "jaz_invoke", JAZ_COLOUR, "-"),
)

# Legend order, which is NOT the draw order above: arms draw baselines-first so the black reference
# line sits behind the coloured methods, while the legend leads with the methods under test. With
# `ncol=2` matplotlib fills COLUMN-major and five entries split 3-then-2, so this puts JAZ invoke over
# the two `CodeAct+subagents` implementations down the left column and Letta over the per-task CodeAct
# baseline down the right.
#
# THE SMOLAGENTS ARM SITS UNDER ITS JAZ TWIN rather than sorting last, and that buys the second column
# back: they share a colour and differ only in implementation, so placed apart a reader has to
# reconstruct the pairing from the swatches -- and with the two longest labels in separate columns the
# legend did not fit two at all and `legend_below` dropped to one, costing the panel the rows. An
# unlisted label sorts to the end rather than raising.
LEGEND_ORDER: tuple[str, ...] = (
    JAZ_LABEL,
    r"$\mathrm{CodeAct{+}subagents}_{\mathrm{JAZ}}$",
    SMOLAGENTS,
    r"$\mathrm{Letta}$",
    BASELINE,
)


def out_path(out_dir: Path, stat: str, relative: bool) -> Path:
    """`<out_dir>/stulife_curves_<stat>[_relative].pdf`."""
    suffix = "_relative" if relative else ""
    return out_dir / f"stulife_curves_{stat}{suffix}.pdf"


def _scored_rows(attempt: Path) -> list[dict[str, object]]:
    """The attempt's non-trigger task rows, in episode order."""
    rows = [json.loads(line) for line in (attempt / "task_results.jsonl").open()]
    scored = [r for r in rows if not r.get("is_trigger")]
    scored.sort(key=lambda r: int(r["task_idx"]))
    return scored


def _rep_attempts(run: Path) -> list[Path]:
    """The attempts of `run` that count as reps, in rep order.

    Keyed on `results.json`, the SAME predicate `build_stulife_table` uses, so the two consumers of
    the shared manifest cannot disagree about which attempts an arm has.
    """
    # Filtering here on `task_results.jsonl` instead let a partially-written attempt be a rep for the
    # curves and not for the table -- reachable today, since one archived Letta run has one graded
    # attempt alongside three partial `task_results.jsonl`.
    return sorted(
        (a for a in run.glob("attempt-*") if (a / "results.json").is_file()),
        key=lambda a: int(a.name.rsplit("-", 1)[-1]),
    )


def _curves(globs: list[str], k: int, runs_root: Path, scored: dict[int, Path]) -> list[list[float]]:
    """Per-rep per-piece pass rates for one arm.

    `scored` accumulates `{scored-task count: first attempt seen with it}` ACROSS arms, so the check
    below spans the whole figure.
    """
    # Keeping `scored` local to one arm -- which it was -- let two arms be cut over different task
    # counts, silently making each line's pieces a different slice of the episode while the x-axis
    # claims one.
    curves: list[list[float]] = []
    for pattern in globs:
        for run in sorted(runs_root.glob(pattern)):
            for attempt in _rep_attempts(run):
                if not (attempt / "task_results.jsonl").is_file():
                    raise SystemExit(
                        f"{display(attempt)} is a rep (it has results.json) but has no "
                        "task_results.jsonl, so it would be in the table and not in this figure. "
                        "Drop the run from the manifest, or the attempt from the run."
                    )
                rows = _scored_rows(attempt)
                scored.setdefault(len(rows), attempt)
                if len(scored) > 1:
                    (n_a, a_a), (n_b, a_b) = sorted(scored.items())[:2]
                    raise SystemExit(
                        f"attempts disagree on the scored-task count: {display(a_a)} has {n_a} and "
                        f"{display(a_b)} has {n_b}. The pieces would cover different tasks per arm, "
                        "so the lines would not be comparable. Drop the short run from the manifest."
                    )
                bounds = piece_bounds(len(rows), k)
                curves.append(
                    [100 * sum(1 for r in rows[a:b] if r.get("success")) / (b - a) for a, b in bounds]
                )
    return curves


def centre_and_band(values: list[float], stat: str) -> tuple[float, float, float]:
    """`(centre, lo, hi)` for one piece under the chosen summary statistic."""
    if stat == "mean":
        m = st.mean(values)
        sem = st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return m, m - sem, m + sem
    s = sorted(values)
    return st.median(values), s[0], s[-1]


def _rebuild_command(args: argparse.Namespace) -> str:
    return rerun_command(
        "plot_stulife_curves.py",
        args,
        {
            "--runs-manifest": DEFAULT_MANIFEST,
            "--runs-root": REPO,
            "--out-dir": REPO / "tables",
            "--stat": "median",
            "--pieces": N_PIECES,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="dump", help="also print the numbers")
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
        "--pieces", type=int, default=N_PIECES, help=f"number of episode pieces (default {N_PIECES})"
    )
    parser.add_argument(
        "--stat",
        choices=("median", "mean"),
        default="median",
        help="median with a min/max band (default), or mean with +/- 1 SEM",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.runs_manifest)
    runs_root: Path = args.runs_root.resolve()

    # Imported here, not at module scope: matplotlib is the optional `plots` extra, and the rest of
    # scripts/ must stay runnable without it.
    plt = set_theme()

    curves: dict[str, list[list[float]]] = {}
    scored: dict[int, Path] = {}
    for label, key, _c, _ls in ARMS:
        curves[label] = _curves(manifest.get(key, []), args.pieces, runs_root, scored)
    empty = [label for label, key, *_ in ARMS if not curves[label]]
    if empty:
        raise SystemExit(
            f"no run data found for: {', '.join(empty)}, under --runs-root {runs_root}.\n"
            "Either that tree has no `runs/` (it is gitignored, so a fresh clone has none -- point "
            "--runs-root at the checkout the runs were produced in), or an arm's globs in "
            f"{args.runs_manifest} no longer match any directory."
        )
    n_scored = next(iter(scored))
    bounds = piece_bounds(n_scored, args.pieces)
    n = len(bounds)

    check_dir = tempfile.TemporaryDirectory() if args.check else None

    def figures_for(stat: str) -> tuple[list[Path], list[Series], list[Series]]:
        """Build both panels for one summary statistic; returns the paths and the two series."""
        abs_series: list[Series] = []
        for label, _key, colour, ls in ARMS:
            cb = [centre_and_band([c[i] for c in curves[label]], stat) for i in range(n)]
            abs_series.append(
                Series(
                    label=label,
                    colour=colour,
                    linestyle=ls,
                    centre=[x[0] for x in cb],
                    lo=[x[1] for x in cb],
                    hi=[x[2] for x in cb],
                    n_runs=len(curves[label]),
                )
            )

        # Relative panel: subtract the per-piece centre of the arm that carries nothing between tasks,
        # so what is left is the arm's advantage over the piece's own difficulty. Subtract the SAME
        # statistic that centres the lines -- median from median, mean from mean -- so a single number
        # never mixes two summaries. Reps are not paired across arms, so this is not a paired difference.
        base = [centre_and_band([c[i] for c in curves[BASELINE]], stat)[0] for i in range(n)]
        rel_series: list[Series] = []
        for label, _key, colour, ls in ARMS:
            if label == BASELINE:
                continue
            cb = [centre_and_band([c[i] for c in curves[label]], stat) for i in range(n)]
            rel_series.append(
                Series(
                    label=label,
                    colour=colour,
                    linestyle=ls,
                    centre=[x[0] - base[i] for i, x in enumerate(cb)],
                    lo=[x[1] - base[i] for i, x in enumerate(cb)],
                    hi=[x[2] - base[i] for i, x in enumerate(cb)],
                    n_runs=len(curves[label]),
                )
            )

        def render(series: list[Series], title: str, ylabel: str, relative: bool) -> Path:
            fig, ax = plt.subplots(figsize=PANEL_SIZE)
            draw_panel(ax, series, bounds, title, ylabel, xlabel="scored-task range (episode order)")
            if relative:
                ax.axhline(0, color="#555", linewidth=0.9, linestyle="--", zorder=1)
            # 6.8, not `legend_below`'s 7.4 default: the five labels in two columns are 3.00in of a
            # 3.125in panel at 7.4, which survives the absolute figure and overflows the relative one --
            # whose longer title ("... minus CodeAct_JAZ") widens the axes this legend is anchored to.
            # Sized once for both so the pair matches; `assert_legend_visible` catches a regression.
            # `y=-0.32`, below `legend_below`'s -0.22 default: at the default the legend's top ink sits
            # ~0.04in INTO the x-axis label. That was true before the type grew (the gap measured -0.013in
            # at the old sizes) -- `assert_legend_visible` only checks the figure edge, so a legend can
            # collide with the axis furniture and still pass. Measured clearance at -0.32 is +0.048in.
            legend = legend_below(ax, *ordered_handles([ax], LEGEND_ORDER), ncol=2, fontsize=6.8, y=-0.32)
            emphasise_jaz_entry(legend)
            fig.tight_layout(pad=0.5)
            assert_legend_visible(legend)
            # Under --check the figure is built into a scratch directory and compared, never written
            # over the committed one.
            target_dir = Path(check_dir.name) if check_dir is not None else args.out_dir
            path = out_path(target_dir, stat, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            plt.close(fig)
            return path

        written = [
            render(abs_series, "StuLife across the episode", "pass rate (%)", relative=False),
            render(
                rel_series,
                f"StuLife across the episode, minus {BASELINE}",
                "pass rate difference (points)",
                relative=True,
            ),
        ]
        return written, abs_series, rel_series

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
        head = "  " + "arm".ljust(38) + "".join(f"{f'{a + 1}-{b}':>10}" for a, b in bounds)
        print(f"\n  {args.stat} pass rate per piece ({n_scored} scored tasks)")
        print(head)
        for s in abs_series:
            print(f"  {s.label:<38}" + "".join(f"{v:>10.1f}" for v in s.centre))
        print(f"\n  {args.stat} minus {BASELINE} on the same piece")
        print(head)
        for s in rel_series:
            print(f"  {s.label:<38}" + "".join(f"{v:>+10.1f}" for v in s.centre))

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
