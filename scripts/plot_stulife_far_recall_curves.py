"""Per-piece far-recall performance over the StuLife year, one line per arm.

Takes the 207 FAR-RECALL tasks (recall gap > 50; see `_episode_recall_analysis` in `envs/stulife.py`),
splits them into 5 roughly equal pieces IN QUEUE ORDER, and plots each arm's pass rate per piece with
a shaded +/- 1 SEM band across its attempts.

Formatting deliberately matches `plot_appworld_curves.py` -- same figure size, fonts, legend placement
and colour assignments -- because the two figures appear side by side and any difference between them
should be a difference in the data, not in how they were drawn.

WHY FAR-RECALL ONLY, AND WHY IN PIECES. These are the tasks the benchmark exists to measure: a task
whose answer was delivered >50 tasks earlier cannot be done from recent context. They are also very
unevenly placed -- 80% fall in deciles 6 and 10 of the queue (the midterm and final blocks) -- so a
per-piece view over far-recall tasks is NOT a view over "the year" at even spacing. Piece 3 in
particular is dominated by the midterm.

READ THE CURVES WITH THAT UNEVENNESS IN VIEW. The pieces are neither equally hard nor equally spaced
in wall-clock task numbers, so a rising line does not by itself show an arm improving as the year goes
on. What the shape does support is comparing ARMS against each other within a piece.

THE FIGURE DOES NOT NAME ITS OWN BAND -- the paper's caption has to: "band: +/-1 SEM over 3 attempts".
That interval is a ~68% one resting on a normality assumption 3 attempts cannot support; it is drawn
because it is the summary the results tables use, not because it is well founded at this n. Prefer
between-arm gaps that exceed both arms' bands.

RUN DATA LIVES OUTSIDE THIS REPO. `runs/` is gitignored, and the StuLife runs were made in a sibling
checkout, so `--runs-root` is how you point this at the tree that has them. The figure it writes IS
committed, for the same reason the AppWorld table is: nobody can regenerate it from a fresh clone.

Drawing is shared with the AppWorld and combined figures -- see `_curves.py`, which owns every choice
about how a panel LOOKS so the three figures cannot drift apart.

Needs the optional `plots` extra (matplotlib, seaborn): `uv sync --extra plots`. The imports are
inside `main` so the rest of `scripts/` runs without it.

Usage:
    uv run python scripts/plot_stulife_far_recall_curves.py
    uv run python scripts/plot_stulife_far_recall_curves.py --pieces 10 --print
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

sys.path.append(str(Path(__file__).resolve().parent))
from _artifacts import (
    display,
    load_manifest,
    report_pdf_check,
    rerun_command,
)
from _curves import (
    JAZ_COLOUR,
    JAZ_LABEL,
    PANEL_SIZE,
    REPO,
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

# PDF only, and vector: see the same note in plot_appworld_curves.py. This figure is committed because
# the runs it reads are not in any clone.
OUT = REPO / "tables" / "stulife_far_recall_curves.pdf"

# Arms come from `scripts/stulife_runs.json`, shared with `build_stulife_table.py` and
# `plot_stulife_curves.py`, so all three describe the same runs. The manifest is what makes that
# possible for someone else's runs: a hardcoded list of directories, or an absolute path into one
# checkout, is usable only on the machine that wrote it.

N_PIECES = 5
# The env's own threshold (`envs/stulife.py:_FAR_RECALL_GAP`). Duplicated rather than imported so this
# script runs without constructing an env; the assertion in `main` pins them together.
FAR_RECALL_GAP = 50
EXPECTED_FAR = 207

# (label, manifest key, colour, linestyle). Colours match the AppWorld figure arm-for-arm where an
# arm appears in both: JAZ invoke keeps its hue, CodeAct+subagents keeps C3, and the per-task CodeAct
# baseline -- the arm that carries nothing between tasks -- is BLACK, so a reference line reads as one.
# Letta has no AppWorld counterpart and takes C0, the remaining cycle colour.
# FIVE arms, matching `tables/stulife_results.tex` and both combined figures.
#
SMOLAGENTS = r"$\mathrm{CodeAct{+}subagents}_{\mathrm{smolagents}}$"

ARMS: tuple[tuple[str, str, str, str], ...] = (
    (JAZ_LABEL, "jaz_invoke", JAZ_COLOUR, "-"),
    (r"$\mathrm{CodeAct{+}subagents}_{\mathrm{JAZ}}$", "codeact_subagents", "C3", "-"),
    (r"$\mathrm{CodeAct}_{\mathrm{JAZ}}$", "codeact_per_task", "black", "-"),
    # C3 dotted, not black dotted: colour is the METHOD and linestyle the IMPLEMENTATION, so this
    # pairs with `CodeAct+subagents_{JAZ}` above exactly as the AppWorld figure pairs its official
    # CodeAct (black dotted) with `CodeAct_{JAZ}` (black solid).
    (SMOLAGENTS, "codeact_subagents_smolagents", "C3", ":"),
    (r"$\mathrm{Letta}$", "letta", "C0", "-"),
)

# There was a `COMBINED_ARMS` here -- this minus the smolagents baseline -- because seven legend
# entries do not fit the combined figures' one row (7.37in against 6.5in of width). Every figure now
# draws all five and those legends wrap to two rows instead: a figure that silently omits an arm the
# table reports is the worse trade, and this file's own figure had already been bitten by it
# (`stulife_far_recall_curves.pdf` shipped four arms against the table's five, the omitted one being
# the weakest baseline) back when the two figures shared a tuple.

# Legend order, which is NOT the draw order: arms draw baselines-first so the black reference line
# sits behind the coloured methods, while the legend leads with the methods under test. With `ncol=2`
# matplotlib fills COLUMN-major and five entries split 3-then-2, so this puts JAZ invoke over the two
# `CodeAct+subagents` implementations down the left column and Letta over the per-task CodeAct
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
    r"$\mathrm{CodeAct}_{\mathrm{JAZ}}$",
)


def resolve_run(key: str, manifest: dict[str, list[str]], runs_root: Path) -> Path | None:
    """The first run directory the manifest's globs for `key` match, or None."""
    for pattern in manifest.get(key, []):
        for run in sorted(runs_root.glob(pattern)):
            if run.is_dir():
                return run
    return None


def far_recall_rows(run: Path) -> list[list[dict[str, object]]]:
    """Each attempt's far-recall rows, in queue order."""
    # `gap > FAR_RECALL_GAP` reproduces the env's own `n_total_far_recall` exactly (207) on the shipped
    # data, which is what the assert in `main` checks. It is the PAIRED-task rule; the exam rule is
    # stricter (correct option far AND a wrong option far), so this could over-count exams -- it does
    # not here, and the assert is what would catch it if the data changed.
    out: list[list[dict[str, object]]] = []
    for path in sorted(run.glob("attempt-*/task_results.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        rows.sort(key=lambda r: r["task_idx"])
        far = [r for r in rows if not r.get("is_trigger") and (r.get("gap") or 0) > FAR_RECALL_GAP]
        if far:
            out.append(far)
    return out


def series_for(
    label: str,
    key: str,
    colour: str,
    linestyle: str,
    k: int,
    manifest: dict[str, list[str]],
    runs_root: Path,
) -> Series | None:
    run = resolve_run(key, manifest, runs_root)
    if run is None:
        return None
    attempts = far_recall_rows(run)
    if not attempts:
        return None
    # Not `min(len(a) for a in attempts)`: the pieces are cut from the REFERENCE arm's 207 tasks and
    # the x-axis labels say so, so an arm with a short attempt would have its pieces cover a
    # different slice of the queue while the axis claimed a single one. Truncating silently is
    # exactly the failure `plot_stulife_curves` raises on, and the queue is fixed, so a disagreement
    # is a broken or half-written run rather than a shape the figure should absorb.
    short = [len(a) for a in attempts if len(a) != EXPECTED_FAR]
    if short:
        raise SystemExit(
            f"{label}: attempt(s) with {short} far-recall rows, expected {EXPECTED_FAR} "
            f"(run {run}). Every arm is cut into the same pieces, so a short attempt would put a "
            "different slice of the queue under the same x-axis label."
        )
    n = EXPECTED_FAR
    means: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for a, b in piece_bounds(n, k):
        # PASS RATE, not the graded `score` field. StuLife grades some tasks partially (an exam
        # answered 3/5 scores 0.6), so the two differ -- by +1 to +5 points depending on arm, and
        # unevenly, since arms earn partial credit at different rates (5% of Letta's far-recall
        # tasks vs 9% of JAZ invoke's). Pass rate is plotted because the AppWorld figure beside it
        # is TGC, which is inherently all-or-nothing: matching the metric keeps the two panels
        # readable as one comparison. On these rows `success` is exactly `score == 1.0`.
        per_attempt = [
            100 * st.mean([1.0 if r.get("success") else 0.0 for r in att[a:b]]) for att in attempts
        ]
        m = st.mean(per_attempt)
        sem = st.stdev(per_attempt) / math.sqrt(len(per_attempt)) if len(per_attempt) > 1 else 0.0
        means.append(m)
        lo.append(m - sem)
        hi.append(m + sem)
    return Series(
        label=label,
        colour=colour,
        linestyle=linestyle,
        centre=means,
        lo=lo,
        hi=hi,
        n_runs=len(attempts),
    )


def reference_rows(manifest: dict[str, list[str]], runs_root: Path) -> list[dict[str, object]] | None:
    """The first arm's far-recall rows, or None when no run data is reachable.

    One arm's task list stands for every arm's: the queue is fixed, so the pieces and their task
    numbers are a property of the benchmark rather than of whichever run supplied them.
    """
    first = resolve_run(ARMS[0][1], manifest, runs_root)
    if first is None:
        return None
    attempts = far_recall_rows(first)
    # A directory that matched the globs but holds no readable attempt -- a run still being written,
    # or a `.superseded` sibling that kept its name -- is "no run data", not a crash: returning None
    # routes it into `main`'s message instead of an `IndexError` from this line.
    if not attempts:
        return None
    ref = attempts[0]
    # What would catch the task list changing underneath a figure that hard-codes 207.
    assert len(ref) == EXPECTED_FAR, f"far-recall count changed: {len(ref)} != {EXPECTED_FAR}"
    return ref


def far_recall_bounds(
    k: int, manifest: dict[str, list[str]], runs_root: Path
) -> list[tuple[int, int]] | None:
    """Piece bounds over the far-recall task list, or None when no run data is reachable."""
    ref = reference_rows(manifest, runs_root)
    return None if ref is None else piece_bounds(len(ref), k)


def all_series(
    k: int,
    manifest: dict[str, list[str]],
    runs_root: Path,
    arms: Sequence[tuple[str, str, str, str]] = ARMS,
) -> list[Series]:
    """One line per arm, skipping any whose run directory is not reachable.

    `arms` defaults to all five, which is what every figure now draws.
    """
    return [s for s in (series_for(*a, k, manifest, runs_root) for a in arms) if s is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", type=int, default=N_PIECES)
    parser.add_argument("--print", action="store_true", dest="dump")
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
    parser.add_argument("--out", type=Path, default=OUT, help=f"where to write (default: {OUT.name})")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the committed figure is stale; writes nothing"
    )
    args = parser.parse_args()
    manifest = load_manifest(args.runs_manifest)
    runs_root: Path = args.runs_root.resolve()

    # Imported here, not at module scope: these are the optional `plots` extra, and the rest of
    # scripts/ must stay runnable without them.
    plt = set_theme()

    series = all_series(args.pieces, manifest, runs_root)
    bounds = far_recall_bounds(args.pieces, manifest, runs_root)
    if not series or bounds is None:
        raise SystemExit(
            f"no StuLife runs found under --runs-root {runs_root}.\n"
            "`runs/` is gitignored, so a fresh clone has none -- point --runs-root at the checkout "
            f"the runs were produced in, or edit the globs in {args.runs_manifest}."
        )

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    # One parenthetical, not two: "(far recall) (%)" reads as a stutter at this size.
    draw_panel(ax, series, bounds, "StuLife long-horizon", "pass rate (far recall, %)")
    # 6.8, matching `plot_stulife_curves.py`: five labels in two columns are 3.05in of a 3.125in panel
    # at `legend_below`'s 7.4 default, and the axes this legend is anchored to widened when the axis
    # labels grew -- so 7.4 now overflows by hundredths. The two single-panel StuLife figures carry the
    # same legend and are sized together so the pair matches.
    # `y=-0.32`, below `legend_below`'s -0.22 default: at the default the legend's top ink sits
    # ~0.04in INTO the x-axis label. That was true before the type grew (the gap measured -0.013in
    # at the old sizes) -- `assert_legend_visible` only checks the figure edge, so a legend can
    # collide with the axis furniture and still pass. Measured clearance at -0.32 is +0.048in.
    legend = legend_below(ax, *ordered_handles([ax], LEGEND_ORDER), ncol=2, fontsize=6.8, y=-0.32)
    emphasise_jaz_entry(legend)
    fig.tight_layout(pad=0.5)
    assert_legend_visible(legend)

    # Printed before the figure is written, so `--check --print` reports the numbers too: when the
    # gate says a figure moved, the numbers behind it are what a reader wants next, and the check
    # branch returns without reaching anything below it.
    if args.dump:
        ref = reference_rows(manifest, runs_root) or []
        spans = [f"task {int(ref[a]['task_idx']) + 1}-{int(ref[b - 1]['task_idx']) + 1}" for a, b in bounds]
        print(
            "  piece spans: "
            + "; ".join(f"{a + 1}-{b} = {s}" for (a, b), s in zip(bounds, spans, strict=True))
        )
        for s in series:
            cells = " ".join(f"{m:5.1f}+/-{h - m:4.1f}" for m, h in zip(s.centre, s.hi, strict=True))
            print(f"  {s.label:<48} n={s.n_runs}  {cells}")

    # Under --check the figure is built into a scratch directory and compared, never written over
    # the committed one.
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / args.out.name if args.check else args.out
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
        plt.close(fig)
        if args.check:
            cmd = rerun_command(
                "plot_stulife_far_recall_curves.py",
                args,
                {
                    "--runs-manifest": DEFAULT_MANIFEST,
                    "--runs-root": REPO,
                    "--out": OUT,
                    "--pieces": N_PIECES,
                },
            )
            return report_pdf_check([target], args.out.parent, cmd)
    plt.close(fig)
    print(f"wrote {display(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
