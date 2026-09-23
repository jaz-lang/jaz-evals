"""StuLife and AppWorld per-piece curves as one two-panel figure, under a single shared legend.

StuLife on the left, AppWorld on the right, one row of legend entries beneath both. The panels are
the same two figures `plot_stulife_far_recall_curves.py` and `plot_appworld_curves.py` write on their
own -- this script imports THEIR data layers and `_curves.draw_panel`, and adds only the layout, so a
number can never differ between the standalone figure and its panel here.

WHAT THE SHARED LEGEND COSTS. Seven arms across the two benchmarks, but only three run on both, so
most entries do not apply to both panels: Letta and the smolagents implementation of CodeAct+subagents
are StuLife-only, and ACE and the official AppWorld baseline are AppWorld-only. A shared legend cannot
say so, and a reader who assumes every entry appears in every panel will look for lines that were
never drawn. The caption has to name the three shared arms
(JAZ invoke, CodeAct+subagents, CodeAct_JAZ) as the ones that support a cross-benchmark comparison.

BOTH PANELS ARE MEAN +/- 1 SEM, and this figure offers no way to make them anything else. AppWorld's
own script defaults to a median with an order-statistic band and can draw either; StuLife computes
only the mean. Two panels side by side under ONE legend, with no room to label a band per panel, is
exactly where a reader will assume the two are the same statistic -- so the combined figure fixes
them to the one both benchmarks can express, rather than exposing a `--stat` flag whose median
setting would silently pair a median against a mean. Use `plot_appworld_curves.py --stat median` for
the order-statistic view; it is the better summary at n=3 and n=6, and the standalone panel has the
room to say so.

+/-1 SEM is a ~68% interval resting on a normality assumption 3 and 6 runs cannot support, and it is
NARROWER than the order-statistic band on the same data -- so this figure looks more precise than the
standalone AppWorld median panel while resting on more. It is the summary the results tables quote,
which is why it is the one both panels share.

THE TWO PANELS DO NOT SHARE A Y-AXIS, and they are not on the same scale -- band extents run
9.5-87% on StuLife and 43-84% on AppWorld. That is deliberate: a shared axis would flatten AppWorld
into a band too narrow to read. It also means the panels' SLOPES are not visually comparable, only
their orderings.

THE APPWORLD PANEL'S Y AXIS IS BROKEN between 56% and 61.5%, marked by a zigzag on the spine. The
official baseline runs ~10 points below every other arm even at its best piece, and on a continuous
axis a quarter of the panel is the empty band between the two groups. No arm's BAND enters the break
-- the figure refuses to build if one does -- so nothing is hidden by it, but vertical distances in
that panel are not to a single scale, and the gap to the official baseline reads smaller than it is.

Both panels are pass rate, which was not free: StuLife grades some tasks partially, and its own
`score` field would have made the left panel a different metric from the right one. See the comment
in `series_for` there.

Needs the optional `plots` extra (matplotlib, seaborn): `uv sync --extra plots`.

The figure does not name its own band -- the paper's caption has to: "band: +/-1 SEM over 3 StuLife
attempts, and over 3 or 6 AppWorld reps depending on arm".

Usage:
    uv run python scripts/plot_combined_curves.py
    uv run python scripts/plot_combined_curves.py --pieces 10 --print
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

# `append`, not `insert(0, ...)`: this only needs `scripts/` on the path at all, and prepending it
# would let a file here shadow a stdlib module for the rest of the process. Python already puts a
# script's own directory on `sys.path`, but only when it is run AS a script -- without this line the
# module cannot be imported (by a test, or by `runpy`), which is a gap the sibling scripts do not have.
sys.path.append(str(Path(__file__).resolve().parent))

import plot_appworld_curves as aw
import plot_stulife_far_recall_curves as sl
from _artifacts import (
    display,
    load_manifest,
    report_pdf_check,
    rerun_command,
)
from _curves import (
    JAZ_LABEL,
    PANEL_SIZE,
    REPO,
    apply_y_break,
    assert_legend_columns,
    assert_legend_visible,
    break_is_warranted,
    draw_panel,
    draw_y_break_mark,
    emphasise_jaz_entry,
    legend_below,
    ordered_handles,
    set_theme,
)
from build_appworld_table import DEFAULT_MANIFEST as AW_MANIFEST
from build_stulife_table import DEFAULT_MANIFEST as SL_MANIFEST

if TYPE_CHECKING:  # matplotlib is the optional `plots` extra; this module must import without it
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

OUT = REPO / "tables" / "combined_curves.pdf"

# The AppWorld panel's broken y axis, taken from the standalone script rather than restated. This
# panel IS `plot_appworld_curves`'s mean-absolute view -- same arms, same pieces, same statistic -- so
# a reader moving between the two figures must not find one arm drawn against two different axes. The
# bounds, the ticks and the reasoning for all of them live at that definition.
AW_Y_BREAK = aw.Y_BREAKS[("mean", False)]

# Union of both panels' arms over TWO rows, grouped by what an arm carries between tasks:
#
#     row 1   JAZ invoke        Letta                     ACE on CodeAct
#     row 2   CodeAct+subagents CodeAct+subagents         CodeAct         CodeAct
#             (JAZ)             (smolagents)              (JAZ)           (AppWorld official)
#
# Row 1 is the methods that accumulate something; row 2 is the four that do not, paired so each
# method sits beside its other implementation. Two rows rather than one is what buys back the 7.4pt
# font every other figure uses -- seven entries in one row need 7.37in of 6.5in even at 6.4pt.
#
# WRITTEN AS THE ROWS THEMSELVES, not as the flat column-major sequence matplotlib actually wants
# (it fills a multi-column legend down each column before moving right, so entry i lands at row
# i % 2, column i // 2). `legend_grid` filters each row to the arms the panels DREW and transposes,
# which is what keeps the grouping true when an arm is missing -- and one can be: `sl.all_series`
# skips an arm whose run directory is not reachable, and `aw.available_arms()` drops the official
# baseline without `--official-root`. A flat sequence with a blank at a fixed index instead pulls
# every entry after the gap one place forward, i.e. into the other row, which
# `assert_legend_columns` cannot see because the column count is unchanged.
_BLANK = ""
LEGEND_ROWS: tuple[tuple[str, ...], ...] = (
    # Accumulates something across tasks.
    (
        JAZ_LABEL,
        r"$\mathrm{Letta}$",
        r"$\mathrm{ACE\ on\ CodeAct}_{\mathrm{JAZ}}$",
    ),
    # Carries nothing, paired so each method sits beside its other implementation.
    (
        r"$\mathrm{CodeAct{+}subagents}_{\mathrm{JAZ}}$",
        sl.SMOLAGENTS,
        r"$\mathrm{CodeAct}_{\mathrm{JAZ}}$",
        r"$\mathrm{CodeAct}_{\mathrm{AppWorld\ official}}$",
    ),
)

#: Every arm the legend places, flat. For `ordered_handles`' ranking only -- `legend_grid` decides
#: the cells. A label in neither row sorts to the end there and is appended to the last row here, so
#: adding an arm degrades to "appended" rather than raising.
LEGEND_ORDER: tuple[str, ...] = tuple(label for row in LEGEND_ROWS for label in row)

# The FULL 6.5in text width, not two panel widths: this figure spans both columns, so it gets the
# gutter back rather than leaving a 0.25in gap down its middle. Height is one panel's, not two --
# the legend gets its own strip below via `bbox_to_anchor`.
TEXT_WIDTH = 6.5
FIGSIZE = (TEXT_WIDTH, PANEL_SIZE[1])

# Sized to the width this layout leaves, which is what bounds every legend in these figures. Measured
# in Futura at 4 columns over 6.5in: 8.0 -> 6.15in, 8.4 -> 6.45in, 8.8 -> overflows. 8.0 takes the
# comfortable value rather than the largest that fits, because the labels are mathtext and a font
# change moves these numbers -- 8.4 would leave 0.05in of margin for that to eat.
#
# It was 6.4 when this legend was one row, which is why an arm was dropped from the figure rather than
# the legend rewrapped: seven entries need 7.37in of 6.5in at that size. The two-row grid above is what
# made both the arm and the larger type possible.
LEGEND_FONTSIZE = 8.0

# The bottom strip reserved for the legend, as a fraction of figure height. ONE constant for both
# uses -- `tight_layout(rect=...)` keeps the panels out of it, and the legend anchors its top to it
# -- because the two were separate numbers and disagreed: the anchor was 0.0, which with
# `loc="upper center"` puts the legend's TOP on the bottom edge of the canvas, i.e. the whole legend
# below it. Both committed combined figures shipped with an empty reserved strip and no legend at
# all, and neither `--check` (which compares to an equally legend-less file) nor the caller's
# width-only assert could see it. `assert_legend_visible` is what catches this class now.
# Two rows of legend, so twice the strip a one-row figure reserves. `assert_legend_visible` catches a
# strip too small for the ink it has to hold.
LEGEND_STRIP = 0.16


def legend_grid(axes: list[Axes]) -> tuple[list[Artist], list[str], int]:
    """Handles, labels and column count for the two-row legend, invisible artists in the empty cells.

    The rows are `LEGEND_ROWS` narrowed to the arms the panels actually drew, so an arm that was not
    drawn costs its own cell and leaves every other arm where it was. `ordered_handles` cannot supply
    a blank: it works from what the panels drew, and no arm drew one.
    """
    from matplotlib.lines import Line2D

    handles, labels = ordered_handles(axes, LEGEND_ORDER)
    drawn = dict(zip(labels, handles, strict=True))
    rows = [[label for label in row if label in drawn] for row in LEGEND_ROWS]
    # An arm no row names lands in the last cells, matching `ordered_handles`' own "sorts to the end".
    rows[-1] += [label for label in labels if label not in LEGEND_ORDER]
    columns = max(len(row) for row in rows)
    grid_handles: list[Artist] = []
    grid_labels: list[str] = []
    for column in range(columns):
        for row in rows:
            label = row[column] if column < len(row) else _BLANK
            # `color="none"` rather than a zero-length line: the cell still has to occupy a slot so
            # the entries after it keep their positions, it just must not draw. Its column is sized
            # by the label below it, so the blank costs no width.
            grid_handles.append(drawn[label] if label else Line2D([], [], color="none"))
            grid_labels.append(label)
    return grid_handles, grid_labels, columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", type=int, default=aw.N_PIECES)
    parser.add_argument("--print", action="store_true", dest="dump")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO,
        help="directory the StuLife manifest's globs resolve against (default: the repo root)",
    )
    parser.add_argument(
        "--appworld-runs-root",
        type=Path,
        default=REPO,
        help="same, for the AppWorld manifest (the two domains' runs may live in different trees)",
    )
    # The two domains' manifests are separate files with separate globs, so they get separate flags
    # -- the same pair every single-domain script takes, named for the domain rather than defaulted
    # into the script the way they were when only this repo's own tree could build the figure.
    parser.add_argument(
        "--stulife-manifest",
        type=Path,
        default=SL_MANIFEST,
        help=f"StuLife run manifest (default: {SL_MANIFEST.name})",
    )
    parser.add_argument(
        "--appworld-manifest",
        type=Path,
        default=AW_MANIFEST,
        help=f"AppWorld run manifest (default: {AW_MANIFEST.name})",
    )
    parser.add_argument("--out", type=Path, default=OUT, help=f"where to write (default: {OUT.name})")
    parser.add_argument(
        "--official-root",
        type=Path,
        default=aw.OFFICIAL_ROOT,
        help="AppWorld's experiment-output root for the official ReAct baseline "
        "(default: the path these runs were produced at; that arm is dropped, with a note, "
        "when it is absent)",
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the committed figure is stale; writes nothing"
    )
    args = parser.parse_args()
    sl_manifest = load_manifest(args.stulife_manifest)
    aw_manifest = load_manifest(args.appworld_manifest)
    runs_root: Path = args.runs_root.resolve()
    aw_runs_root: Path = args.appworld_runs_root.resolve()

    plt = set_theme()

    # All five StuLife arms, matching `tables/stulife_results.tex` and the standalone figures. This
    # used to draw `COMBINED_ARMS` (everything but the smolagents baseline) because seven entries do
    # not fit one row of legend; the legend now wraps to two rows instead, since a figure that omits
    # an arm the table reports is the worse trade.
    sl_series = sl.all_series(args.pieces, sl_manifest, runs_root, sl.ARMS)
    sl_bounds = sl.far_recall_bounds(args.pieces, sl_manifest, runs_root)
    if not sl_series or sl_bounds is None:
        raise SystemExit(
            f"no StuLife runs found under --runs-root {runs_root}.\n"
            "`runs/` is gitignored, so a fresh clone has none -- point --runs-root at the checkout "
            f"the runs were produced in, or edit the globs in {args.stulife_manifest}."
        )

    aw_bounds = aw.piece_bounds(aw.N_TASKS, args.pieces)
    # "mean" is not a default here, it is the only option -- see the band note in the module
    # docstring. StuLife's panel is mean +/- 1 SEM and cannot be anything else.
    # `available_arms()` rather than `aw.ARMS`: the official baseline is read from artifacts that
    # live outside any run tree, and without this the figure dies with a bare `FileNotFoundError` on
    # every machine but the one those artifacts are on.
    aw_arms = aw.available_arms(official_root=args.official_root)
    aw_curves = aw.read_curves(
        aw_bounds,
        aw_manifest,
        aw_runs_root,
        aw_arms,
        manifest_path=args.appworld_manifest,
        official_root=args.official_root,
    )
    aw_series = aw.absolute_series(aw_curves, "mean", aw_arms)

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGSIZE)
    draw_panel(left, sl_series, sl_bounds, "StuLife long-horizon", "pass rate (far recall, %)")
    draw_panel(right, aw_series, aw_bounds, "AppWorld self-improvement", "pass rate (%)")
    lo, hi, ticks = AW_Y_BREAK
    if break_is_warranted(((s.label, min(s.lo), max(s.hi)) for s in aw_series), lo, hi, "the AppWorld panel"):
        apply_y_break(right, lo, hi, ticks)
        draw_y_break_mark(right, lo, hi)

    # Handles from both panels, deduped: the three shared arms contribute one entry each.
    handles, labels, columns = legend_grid([left, right])
    # Laid out under the FIGURE, not under either panel, so it is centred on the pair.
    # `columns`, not `len(labels)`: the grid `legend_grid` built is positional, so a legend that fell
    # back to a different column count would scramble it rather than merely rewrap.
    legend = legend_below(fig, handles, labels, ncol=columns, y=LEGEND_STRIP, fontsize=LEGEND_FONTSIZE)
    assert_legend_columns(legend, columns)
    emphasise_jaz_entry(legend)
    # `rect` reserves the bottom strip for the legend; `tight_layout` is otherwise blind to an artist
    # anchored outside the axes and would let the panels expand over it.
    fig.tight_layout(rect=(0, LEGEND_STRIP, 1, 1), pad=0.5)

    width = assert_legend_visible(legend)

    # Printed before the figure is written, so `--check --print` reports the numbers too: when the
    # gate says a figure moved, the numbers behind it are what a reader wants next, and the check
    # branch returns without reaching anything below it.
    if args.dump:
        for name, series, bounds in (
            ("StuLife", sl_series, sl_bounds),
            ("AppWorld", aw_series, aw_bounds),
        ):
            print(f"\n  {name}: " + " ".join(f"{a + 1}-{b}" for a, b in bounds))
            for s in series:
                print(f"  {s.label:<48} " + " ".join(f"{v:5.1f}" for v in s.centre))

    # Under --check the figure is built into a scratch directory and compared, never written over the
    # committed one: a gate that has to clobber the artifact to tell you it is stale is not a gate.
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / args.out.name if args.check else args.out
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
        plt.close(fig)
        if args.check:
            cmd = rerun_command(
                "plot_combined_curves.py",
                args,
                {
                    "--stulife-manifest": SL_MANIFEST,
                    "--appworld-manifest": AW_MANIFEST,
                    "--runs-root": REPO,
                    "--appworld-runs-root": REPO,
                    "--out": OUT,
                    "--pieces": aw.N_PIECES,
                    "--official-root": aw.OFFICIAL_ROOT,
                },
            )
            return report_pdf_check([target], args.out.parent, cmd)
    print(f"wrote {display(args.out)}  (legend {width:.2f}in of {FIGSIZE[0]}in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
