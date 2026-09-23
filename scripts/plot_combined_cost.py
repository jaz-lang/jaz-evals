"""Quality against spend: pass rate vs mean cost per run, StuLife and AppWorld side by side.

One point per arm, at its mean cost and mean pass rate across runs, with +/- 1 SEM bars on BOTH
axes. StuLife on the left over its 207 far-recall tasks, AppWorld on the right over the full
417-task queue, under a single shared legend.

Both quantities come from the SAME code paths as the curve figures rather than being recomputed
here: pass rate is `plot_*_curves`'s own per-piece machinery asked for ONE piece, so a point on
this figure is by construction the whole-queue value of the line on that one, and cost is the run
record's `usage.cost_usd`. Nothing about an arm is defined twice.

WHAT "COST" IS. The WHOLE run's spend for one rep -- solver, meta, and every sub-agent -- as the
harness recorded it, not a per-task or per-success figure. That makes the x-axis comparable across
arms only insofar as the runs are: each is one pass over the same fixed queue, so it is spend per
queue, and an arm that ran the queue once at twice the price sits twice as far right. It is NOT a
cost-per-point-of-quality; the diagonal a reader might draw between two arms is a ratio of two
means, and both have real spread.

THE X AXES ARE NOT SHARED, and neither are the y axes. StuLife spans ~$4-45 and AppWorld ~$10-31,
and far-recall pass rate spans a far wider range than AppWorld's TGC. A shared axis would flatten
one panel to make the other legible. Positions are comparable WITHIN a panel only.

THE APPWORLD PANEL'S Y AXIS IS BROKEN between 51% and 65%, marked by a zigzag on the spine.
The official baseline scores ~19 points below every other arm, and on a continuous axis the four
arms this figure compares are crushed into the top fifth of the panel. No arm falls in the broken
band, and no error bar crosses it, so nothing is hidden by the squeeze -- but vertical DISTANCES
in that panel are not to a single scale, and the gap to the official baseline reads smaller than
it is.

THE OFFICIAL APPWORLD BASELINE'S COST IS NOT FROM A RUN RECORD -- it has no `results.json`, so it
comes from `build_appworld_table`'s recovered figures, which are read from that harness's own logs
and fall back to recorded constants when the sibling checkout is absent. Imported rather than
re-derived so this figure and the results table can never disagree about that row.

Needs the optional `plots` extra (matplotlib, seaborn): `uv sync --extra plots`.

Usage:
    uv run python scripts/plot_combined_cost.py
    uv run python scripts/plot_combined_cost.py --print
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# `append`, not `insert(0, ...)`: this only needs `scripts/` on the path at all, and prepending it
# would let a file here shadow a stdlib module for the rest of the process. Python already puts a
# script's own directory on `sys.path`, but only when it is run AS a script -- without this line the
# module cannot be imported (by a test, or by `runpy`), which is a gap the sibling scripts do not have.
sys.path.append(str(Path(__file__).resolve().parent))

import build_appworld_table as bt
import plot_appworld_curves as aw
import plot_stulife_far_recall_curves as sl
from _artifacts import (
    display,
    load_manifest,
    report_pdf_check,
    rerun_command,
)
from _curves import (
    JAZ_COLOUR,
    JAZ_LABEL,
    REPO,
    TICK_SIZE,
    Series,
    apply_y_break,
    assert_legend_columns,
    assert_legend_visible,
    break_is_warranted,
    draw_y_break_mark,
    emphasise_jaz_entry,
    legend_below,
    mean_sem,
    set_theme,
    style_axes,
)
from build_appworld_table import DEFAULT_MANIFEST as AW_MANIFEST
from build_stulife_table import DEFAULT_MANIFEST as SL_MANIFEST
from plot_combined_curves import (
    FIGSIZE,
    LEGEND_FONTSIZE,
    LEGEND_STRIP,
    legend_grid,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

OUT = REPO / "tables" / "combined_cost.pdf"

# The official AppWorld baseline's cost uncertainty. Imported rather than restated for the same reason
# its mean is: one definition, so this figure and the results table cannot disagree about that row.
# `build_appworld_table.OFFICIAL_COST_SEM` carries why it is not the SEM over that arm's three reps.
OFFICIAL_COST_SEM = bt.OFFICIAL_COST_SEM


class Point:
    """One arm in one panel: mean cost and mean pass rate, each with its SEM.

    `cost_sem` overrides the SEM computed over `costs`, for an arm whose reps do not carry it.
    """

    def __init__(
        self,
        label: str,
        colour: str,
        costs: list[float],
        score: float,
        score_sem: float,
        cost_sem: float | None = None,
        linestyle: str = "-",
    ) -> None:
        self.label = label
        self.colour = colour
        # Carried only to pick the marker: a dashed arm in the curve figures gets a hollow point
        # here, so the same two arms read as a pair in both. Nothing draws a line in this figure.
        self.linestyle = linestyle
        # Cost is the only raw sample here; the score arrives already summarised, because it is
        # taken from the curve figures' own series rather than recomputed.
        self.cost, computed = mean_sem(costs)
        self.cost_sem = computed if cost_sem is None else cost_sem
        self.score = score
        self.score_sem = score_sem
        self.n = len(costs)


def _run_cost(run: Path) -> float | None:
    """One run's total spend, from the harness's own record."""
    # `usage.cost_usd` on the attempt's `results.json`, the same field `build_appworld_table`
    # reads for its cost column. Never the human-readable trace render.
    results = run / "results.json"
    if not results.exists():
        return None
    return float(json.loads(results.read_text())["usage"]["cost_usd"])


def stulife_points(
    pieces_series: list[Series],
    sl_manifest: dict[str, list[str]],
    runs_root: Path,
    arms: Sequence[tuple[str, str, str, str]] = sl.ARMS,
) -> list[Point]:
    costs: dict[str, list[float]] = {}
    for label, key, _colour, _ls in arms:
        run = sl.resolve_run(key, sl_manifest, runs_root)
        if run is None:
            continue
        costs[label] = [c for a in sorted(run.glob("attempt-*")) if (c := _run_cost(a)) is not None]
    points: list[Point] = []
    for s in pieces_series:
        if costs.get(s.label):
            # `centre[0]` is the whole-task-list mean because the series was built with ONE
            # piece; `hi[0] - centre[0]` recovers the SEM that series already computed.
            points.append(
                Point(
                    s.label,
                    s.colour,
                    costs[s.label],
                    s.centre[0],
                    s.hi[0] - s.centre[0],
                    linestyle=s.linestyle,
                )
            )
    return points


def appworld_points(
    series: list[Series],
    aw_manifest: dict[str, list[str]],
    aw_runs_root: Path,
    arms: Sequence[aw.Arm] | None = None,
    official_root: Path = bt.OFFICIAL_ROOT,
    official_logs: Path = bt.OFFICIAL_LOGS,
) -> list[Point]:
    costs: dict[str, list[float]] = {}
    official: set[str] = set()
    for label, key, _si, _colour, _ls in aw.ARMS if arms is None else arms:
        if not key:
            # The official baseline: recovered by the table generator, not a run record. Both roots
            # are threaded in rather than left to default, so `--official-root` redirects this arm's
            # COST as well as whether its point is drawn. Defaulting here silently paired a
            # reproducer's own pass rate with this repo's recorded fallback cost -- a point that
            # looks measured and is half borrowed.
            costs[label] = bt._official(official_root, official_logs)[0]["cost"]
            official.add(label)
            continue
        runs = [r for pattern in aw_manifest.get(key, []) for r in sorted(aw_runs_root.glob(pattern))]
        costs[label] = [c for r in runs if (c := _run_cost(r / "attempt-0")) is not None]
    points: list[Point] = []
    for s in series:
        if costs.get(s.label):
            # `centre[0]` is the whole-task-list mean because the series was built with ONE
            # piece; `hi[0] - centre[0]` recovers the SEM that series already computed.
            points.append(
                Point(
                    s.label,
                    s.colour,
                    costs[s.label],
                    s.centre[0],
                    s.hi[0] - s.centre[0],
                    cost_sem=OFFICIAL_COST_SEM if s.label in official else None,
                    linestyle=s.linestyle,
                )
            )
    return points


def _draw(ax: Axes, points: list[Point], title: str, ylabel: str) -> None:
    for p in points:
        # Same emphasis rule as the curve figures, keyed off the colour rather than the label so a
        # rename cannot silently drop it: the method the figures are about is drawn larger.
        lead = p.colour == JAZ_COLOUR
        # Hollow point for a dashed arm, matching `draw_panel`: this figure has no lines to carry a
        # linestyle, so without it the two same-coloured arms are two identical dots and the only way
        # to tell which is which is to read them off the other figure.
        dashed = p.linestyle != "-"
        ax.errorbar(
            p.cost,
            p.score,
            xerr=p.cost_sem,
            yerr=p.score_sem,
            color=p.colour,
            marker="o",
            markersize=7.0 if lead else 5.0,
            markerfacecolor="white" if dashed else p.colour,
            markeredgecolor=p.colour,
            markeredgewidth=1.1 if dashed else 0.0,
            linestyle="none",
            elinewidth=1.4 if lead else 1.0,
            # Dashed error bars too, so the pairing survives where the bars are longer than the
            # marker -- which on this figure is most of the ink. `elinestyle` needs matplotlib
            # >=3.10, which is why the `plots` extra floors there: a capability probe instead would
            # make the COMMITTED PDF depend on the installed matplotlib, and `--check` -- the one
            # gate keeping these figures honest -- would report it stale on a conforming environment
            # with nothing to say the library was the cause.
            **({"elinestyle": p.linestyle} if dashed else {}),
            capsize=2.0,
            zorder=3 if lead else 2,
            label=p.label,
        )
    break_spec = _Y_BREAKS.get(title)
    if break_spec is not None and not break_is_warranted(
        ((p.label, p.score - p.score_sem, p.score + p.score_sem) for p in points),
        break_spec[0],
        break_spec[1],
        f"the {title} panel",
    ):
        break_spec = None
    if break_spec is not None:
        apply_y_break(ax, *break_spec)
    # Headroom before the annotations go on: the `JAZ invoke` label sits above the highest point in
    # both panels, and at the default margins that lands on the title.
    ax.margins(y=0.16)
    style_axes(ax, title, "cost per run (USD)", ylabel)
    _annotate_panel(ax, points, title)
    _draw_orientation_key(ax, title)
    if break_spec is not None:
        draw_y_break_mark(ax, break_spec[0], break_spec[1])


# What each panel annotates, keyed by panel title so a caller cannot pair the wrong note with the wrong
# axes. Per panel: the arrow's (from, to) arm labels, and the note that hangs off each of those two.
#
# The arrows and the orientation key say the same thing two ways, deliberately. The key states which
# corner is good (up is better, left is cheaper); each arrow then shows one comparison already pointing
# that way, so a reader who has not worked out the axes still sees the claim. Both arrows here run
# up-and-left because that is what the data does -- if an arrow ever points elsewhere, the pairing below
# is wrong, not the drawing.
_PANEL_NOTES: dict[str, dict[str, Any]] = {
    "StuLife long-horizon": {
        "arrow": (r"$\mathrm{Letta}$", JAZ_LABEL),
        "notes": {
            # (text, offset in points, horizontal alignment). Offsets are hand-placed per panel: the
            # points sit differently in each, and a rule that worked for one put a label on a title or
            # on another arm's error bar in the other.
            JAZ_LABEL: ("no memory system", (0, 14), "center"),
            r"$\mathrm{Letta}$": ("has memory system", (-6, -14), "right"),
        },
    },
    "AppWorld self-improvement": {
        "arrow": (r"$\mathrm{ACE\ on\ CodeAct}_{\mathrm{JAZ}}$", JAZ_LABEL),
        "notes": {
            JAZ_LABEL: ("no meta-harness", (0, 16), "center"),
            r"$\mathrm{ACE\ on\ CodeAct}_{\mathrm{JAZ}}$": ("has meta-harness", (-8, -14), "right"),
        },
    },
}

# Where the up/left orientation key sits, in axes fractions: the corner the two arms meet at, and how
# long each arm is. Bottom-right because that is the one corner no arm occupies in either panel -- the
# data runs bottom-left to top-right, and "expensive and bad" is empty by construction.
#
# AppWorld sits lower than the shared default because its y break moved its arms down the panel: the
# `w/ meta-harness` note and the key's `better` label ended up stacked at the same x, close enough to
# read as one block of grey text.
_KEY_CORNER = (0.88, 0.16)
_KEY_CORNERS: dict[str, tuple[float, float]] = {"AppWorld self-improvement": (0.88, 0.10)}
_KEY_ARM = 0.17


# Per panel: the empty band's (lo, hi) and the y ticks -- see `apply_y_break`. AppWorld's official
# baseline sits ~19 points below every other arm, so without this the four arms the figure is ABOUT
# land in the top fifth with their error bars overlapping into one smear. The bounds sit just outside
# that arm's upper error bar (49.8) and just below the lowest other arm's lower one (66.3), so no bar
# crosses the squeeze.
_Y_BREAKS: dict[str, tuple[float, float, list[float]]] = {
    "AppWorld self-improvement": (51.0, 65.0, [45.0, 50.0, 65.0, 70.0, 75.0]),
}


def _annotate_panel(ax: Axes, points: list[Point], title: str) -> None:
    """Draw the panel's comparison arrow and its two point labels, if it has any."""
    # Every lookup here is by LABEL, and a label is a display string that a rename changes. Missing one
    # is legitimate -- `--official-root` elsewhere drops an arm, and a panel need not be annotated at
    # all -- so this notes rather than raises. It does not stay silent: `draw_panel` keys its own
    # emphasis off the COLOUR precisely "so a rename cannot silently drop" it, and the only gate that
    # would catch a silently unannotated figure is `--check`, which a person has to remember to run.
    spec = _PANEL_NOTES.get(title)
    if spec is None:
        print(f"note: no annotations for panel {title!r}", file=sys.stderr)
        return
    by_label = {p.label: p for p in points}

    start, end = spec["arrow"]
    if start not in by_label or end not in by_label:
        missing = [label for label in (start, end) if label not in by_label]
        print(
            f"note: no comparison arrow on {title!r} -- {', '.join(missing)} not drawn",
            file=sys.stderr,
        )
    if start in by_label and end in by_label:
        tail, head = by_label[start], by_label[end]
        # `shrinkA/B` keep the arrow clear of both markers and their error bars, so it reads as a
        # relation between the points rather than something attached to them.
        ax.annotate(
            "",
            xy=(head.cost, head.score),
            xytext=(tail.cost, tail.score),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#555",
                "linewidth": 1.0,
                "shrinkA": 9,
                "shrinkB": 11,
            },
            zorder=1,
        )

    for label, (text, offset, align) in spec["notes"].items():
        point = by_label.get(label)
        if point is None:
            print(f"note: no point label for {label} on {title!r} -- not drawn", file=sys.stderr)
            continue
        # Coloured to match its dot: at this size a leader line would be more ink than the label, so
        # colour is what ties the two together.
        ax.annotate(
            text,
            xy=(point.cost, point.score),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=TICK_SIZE,
            color=point.colour,
            zorder=4,
        )


def _draw_orientation_key(ax: Axes, title: str) -> None:
    """A two-armed key in the bottom-right corner: up is better, left is cheaper."""
    # Axes fractions, not data coordinates: the two panels have different x ranges, and the key is a
    # property of the axes' orientation rather than of anything plotted.
    corner_x, corner_y = _KEY_CORNERS.get(title, _KEY_CORNER)
    common = {
        "xycoords": "axes fraction",
        "textcoords": "axes fraction",
        # `shrinkA`/`shrinkB` default to 2 points, which pulls each arm off the corner along its own
        # direction -- one up, one left -- so the two tails miss each other by ~3pt and the key reads
        # as two strays rather than one pair of axes. Zero makes them meet.
        "arrowprops": {
            "arrowstyle": "-|>",
            "color": "#555",
            "linewidth": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        "annotation_clip": False,
    }
    ax.annotate("", xy=(corner_x, corner_y + _KEY_ARM), xytext=(corner_x, corner_y), **common)
    ax.annotate("", xy=(corner_x - _KEY_ARM, corner_y), xytext=(corner_x, corner_y), **common)
    # Each label sits just BEYOND its arm's tip, not alongside the shaft: at this size a label beside
    # the shaft overlaps the arrowhead, which is the one part of the key that carries the meaning.
    label_style = {
        "transform": ax.transAxes,
        "fontsize": TICK_SIZE,
        "color": "#555",
        "clip_on": False,
    }
    # `better` is centred on its arm; `cheaper` is right-aligned onto the end of its own, which is the
    # same thing for a horizontal arm. The key sits near the right edge, so both are checked against
    # the axes bounds rather than assumed to fit.
    ax.text(corner_x, corner_y + _KEY_ARM + 0.012, "better", ha="center", va="bottom", **label_style)
    ax.text(corner_x - _KEY_ARM - 0.012, corner_y, "cheaper", ha="right", va="center", **label_style)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "(default: the path these runs were produced at; with --official-logs it also decides "
        "where that arm's cost comes from)",
    )
    parser.add_argument(
        "--official-logs",
        type=Path,
        default=bt.OFFICIAL_LOGS,
        help="directory of that baseline's per-rep run logs, which carry its cost",
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

    # ONE piece: the per-piece machinery over the whole task list at once, so these points are the
    # curve figures' lines collapsed to their totals rather than a second definition of "pass rate".
    # All five StuLife arms, for the same reason as the curve figure: see the note at its call site.
    sl_series = sl.all_series(1, sl_manifest, runs_root, sl.ARMS)
    if not sl_series:
        raise SystemExit(
            f"no StuLife runs found under --runs-root {runs_root}.\n"
            "`runs/` is gitignored, so a fresh clone has none -- point --runs-root at the checkout "
            f"the runs were produced in, or edit the globs in {args.stulife_manifest}."
        )
    # See the note in `plot_combined_curves.py`: without the filter this dies on any machine that
    # does not have the official baseline's artifacts.
    aw_arms = aw.available_arms(official_root=args.official_root)
    aw_series = aw.absolute_series(
        aw.read_curves(
            aw.piece_bounds(aw.N_TASKS, 1),
            aw_manifest,
            aw_runs_root,
            aw_arms,
            manifest_path=args.appworld_manifest,
            official_root=args.official_root,
        ),
        "mean",
        aw_arms,
    )

    left_points = stulife_points(sl_series, sl_manifest, runs_root)
    right_points = appworld_points(
        aw_series, aw_manifest, aw_runs_root, aw_arms, args.official_root, args.official_logs
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGSIZE)
    _draw(left, left_points, "StuLife long-horizon", "pass rate (far recall, %)")
    _draw(right, right_points, "AppWorld self-improvement", "pass rate (%)")

    # The curve figure's grid, reused whole: the two figures share a legend layout so a reader moving
    # between them finds each arm in the same cell.
    handles, labels, columns = legend_grid([left, right])
    legend = legend_below(fig, handles, labels, ncol=columns, y=LEGEND_STRIP, fontsize=LEGEND_FONTSIZE)
    assert_legend_columns(legend, columns)
    emphasise_jaz_entry(legend)
    fig.tight_layout(rect=(0, LEGEND_STRIP, 1, 1), pad=0.5)

    width = assert_legend_visible(legend)

    # Printed before the figure is written, so `--check --print` reports the numbers too -- see the
    # note on the same block in `plot_combined_curves.py`.
    if args.dump:
        for name, points in (("StuLife", left_points), ("AppWorld", right_points)):
            print(f"\n  {name}")
            for p in points:
                print(
                    f"  {p.label:<48} n={p.n}  ${p.cost:6.2f}+/-{p.cost_sem:5.2f}"
                    f"   {p.score:5.1f}+/-{p.score_sem:4.1f}%"
                )

    # Under --check the figure is built into a scratch directory and compared, never written over the
    # committed one: a gate that has to clobber the artifact to tell you it is stale is not a gate.
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / args.out.name if args.check else args.out
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
        plt.close(fig)
        if args.check:
            cmd = rerun_command(
                "plot_combined_cost.py",
                args,
                {
                    "--stulife-manifest": SL_MANIFEST,
                    "--appworld-manifest": AW_MANIFEST,
                    "--runs-root": REPO,
                    "--appworld-runs-root": REPO,
                    "--out": OUT,
                    "--official-root": aw.OFFICIAL_ROOT,
                    "--official-logs": bt.OFFICIAL_LOGS,
                },
            )
            return report_pdf_check([target], args.out.parent, cmd)
    print(f"wrote {display(args.out)}  (legend {width:.2f}in of {FIGSIZE[0]}in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
