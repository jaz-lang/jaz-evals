"""Shared drawing machinery for the per-piece curve figures.

`plot_appworld_curves.py`, `plot_stulife_curves.py`, `plot_stulife_far_recall_curves.py` and
`plot_combined_curves.py` all draw the same picture -- one centre line per arm with a shaded band,
over a queue split into contiguous pieces -- and differ only in where the numbers come from and what
the band means. `plot_combined_cost.py` draws no curve at all, but takes the colours, legend and axis
styling from here so its arms read as the same arms. Everything that decides how the figure LOOKS
lives here, so the panels stay identical by construction rather than by two authors remembering to
make the same edit twice.

A private module beside the scripts, not a package under `src/jaz_evals/`: this is presentation code
for those five scripts, not suite API, and nothing importable by the harness should depend on
matplotlib. The leading underscore says "not runnable". Python puts a script's own directory on
`sys.path`, so `from _curves import ...` resolves when the sibling scripts are run the documented
way.

matplotlib and seaborn are the optional `plots` extra, so every import of them is INSIDE a function.
Importing this module must stay free for a tree that has not installed the extra.
"""

# That sharing is not a hypothetical: the two scripts were separately written and had already drifted
# (band alphas, lead-line emphasis, and the legend block were duplicated verbatim, and the y-label
# wording had to be fixed in one and then ported to the other). The combined figure makes drift
# visible -- two panels side by side in one PDF -- so it is now cheaper to share than to sync.

from __future__ import annotations

import math
import statistics as st
import sys
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

# Re-exported, not redefined: `_artifacts` owns it because `display()` there is what actually uses
# it, and three copies is how the two halves of this split start disagreeing about the repo root.
from _artifacts import REPO as REPO

# The method the figures are about: its own hue, and the one line drawn thicker than the rest. One
# definition for every figure, so the subject keeps its identity across panels a reader sees together.
JAZ_COLOUR = "#ff00a6"

#: The official AppWorld baseline. Grey rather than a second black: it and the per-task CodeAct
#: baseline are both reference lines, and drawing them in the same ink made the panel read as
#: having two subjects of equal weight. Grey keeps it legible as a reference while letting the
#: black CodeAct line -- the one every other arm is measured against -- stay the darker of the two.
OFFICIAL_COLOUR = "#8c8c8c"

#: Colours that mark an arm as a REFERENCE rather than a method under test. Their bands draw
#: fainter (see ``draw_panel``, which draws a dashed arm's band fainter too); membership is by
#: colour so an arm renamed keeps its treatment.
BASELINE_COLOURS = ("black", OFFICIAL_COLOUR)

# One column of a 6.5in text width (article/letterpaper, 1in margins) split in two with the usual
# 0.25in gutter: (6.5 - 0.25) / 2. Sized so a standalone figure is included at its NATURAL size --
# a figure wider than the column gets scaled down by \includegraphics, which shrinks every font
# below the sizes named here (at the previous 3.4in that was a ~4% reduction, so the 9.5pt title
# rendered near 9.1pt). The combined figure spans the full text width instead, and sets its own.
PANEL_SIZE = (3.125, 3.125)

# Every AXES font size in these figures, in points, in one place -- title, axis label, tick label. They
# were five literals spread across three files (`style_axes`, `draw_panel`'s tick labels,
# `legend_below`'s default, and a per-figure override in each combined script), which is how the legend
# ended up at 6.4 in one figure and 7.4 in its sibling with nothing saying why.
#
# Scaled up together from (9.5, 9, 8) after the figures were read at print size: at 3.125in a panel is
# about a third of a column, so type that looks comfortable on screen is marginal on paper. The ratios
# between them are kept -- title > label > tick -- so the hierarchy survives the change.
#
# LEGENDS ARE NOT IN THIS LADDER, and cannot be: each legend's size is bounded by the width its own
# layout leaves it, which differs per figure (7 entries over 6.5in versus 5 over 3.125in). Each is named
# at the call site that owns it -- 6.8 on every single-panel figure, 8.0 on the combined pair -- and
# `legend_below` drops columns until the ink fits. Its own 7.4 default is now a floor nothing uses.
TITLE_SIZE = 11.0
LABEL_SIZE = 10.0
TICK_SIZE = 9.0


class Series(NamedTuple):
    """One arm's plotted line: a centre value per piece, its band edges, and how to draw it.

    `centre`/`lo`/`hi` are deliberately unnamed as to statistic. AppWorld fills them with a median
    and an order-statistic band (or a mean and +/-1 SEM); StuLife fills them with a mean and +/-1
    SEM. The drawing code never needs to know which, and the figure does not name its own band --
    the paper's caption has to.
    """

    label: str
    colour: str
    linestyle: str
    centre: list[float]
    lo: list[float]
    hi: list[float]
    # How many runs the centre and band were computed from. Optional because it is reporting-only --
    # `--print` names it so a reader can tell a 3-rep band from a 6-rep one -- and the arms with a
    # per-arm rep count already carry it in their own tables.
    n_runs: int | None = None


def piece_bounds(n: int, k: int) -> list[tuple[int, int]]:
    """`k` contiguous pieces covering `n` items, sizes differing by at most one.

    Contiguous and in queue order on purpose: the x-axis means "how far into the queue", which is the
    only axis on which a self-improving arm could show improvement at all.
    """
    q, r = divmod(n, k)
    out: list[tuple[int, int]] = []
    i = 0
    for j in range(k):
        size = q + (1 if j < r else 0)
        out.append((i, i + size))
        i += size
    return out


def mean_sem(values: list[float]) -> tuple[float, float]:
    """Mean and its standard error. SEM is 0.0 for a single value rather than undefined."""
    # The summary every figure in this set uses, defined once so a panel cannot quietly switch to
    # a population sd or a different n. It is a ~68% interval resting on a normality assumption 3
    # and 6 runs cannot support; it is used because it is what the results tables quote.
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0
    return st.mean(values), st.stdev(values) / math.sqrt(len(values))


# BREAKING A Y AXIS. When one arm sits far below the rest, a continuous axis spends most of the panel
# on the empty band between them and crushes the arms the figure is about into a sliver at the top.
# These two squeeze that band out and say so with a zigzag across the spine. The caller supplies the
# band: it must contain no data and no band edge, because a break drawn through a band would shorten
# it on the page without shortening what it claims.
#
# A squeezed scale rather than two stacked axes (the `brokenaxes` shape): both combined figures lay
# their panels out with one `plt.subplots(1, 2)` and build a shared legend by walking the panels'
# handles, so splitting one panel into two axes would change what the legend code iterates over for
# the sake of one panel's y axis.

# What one data unit inside the band is worth outside it. Small enough to read as "removed", large
# enough that the zigzag has somewhere to sit.
BREAK_SQUEEZE = 0.1

# Axes fractions: the zigzag's half-width, and how far past the band it runs at each end.
BREAK_MARK = 0.013


def apply_y_break(ax: Axes, lo: float, hi: float, ticks: Sequence[float]) -> None:
    """Squeeze the empty band `[lo, hi]` out of `ax`'s y axis and pin the ticks around it.

    `ticks` must avoid the band: an automatic locator puts one inside it, where the axis does not
    mean what it says. Raises `SystemExit` if one does.
    """
    # Inside the function, like every other plotting import here: this module is importable without
    # the `plots` extra, and numpy arrives with matplotlib rather than on its own.
    import numpy as np

    # A `function` scale, not hand-placed data: every artist -- lines, bands, points, error bars,
    # annotation anchors -- goes through the axis transform, so all of them follow the squeeze with
    # nothing to keep in sync. `np.where` because matplotlib calls these with arrays.
    kept = (hi - lo) * BREAK_SQUEEZE

    def forward(y: Any) -> Any:
        y = np.asarray(y, dtype=float)
        return np.where(y <= lo, y, np.where(y >= hi, y - (hi - lo) + kept, lo + (y - lo) * BREAK_SQUEEZE))

    def inverse(t: Any) -> Any:
        t = np.asarray(t, dtype=float)
        return np.where(
            t <= lo,
            t,
            np.where(t >= lo + kept, t + (hi - lo) - kept, lo + (t - lo) / BREAK_SQUEEZE),
        )

    # Enforced, not merely documented, for the same reason `assert_break_is_empty` is: the squeezed band
    # is about 1% of the panel's height, so a tick inside it is drawn within a hair of its neighbours and
    # labels a position the axis does not mean. Cheap to check, and silent to miss.
    inside = [t for t in ticks if lo < t < hi]
    if inside:
        raise SystemExit(
            f"y-axis break ({lo}, {hi}) has ticks inside the squeezed band: "
            f"{', '.join(str(t) for t in inside)}. The band is ~1% of the panel's height, so those "
            "labels would sit on top of each other and name positions the axis does not mean."
        )

    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_yticks(list(ticks))


def break_is_warranted(spans: Iterable[tuple[str, float, float]], lo: float, hi: float, where: str) -> bool:
    """Whether a y-axis break over `[lo, hi]` is both safe and useful for these arms.

    `spans` is `(label, low, high)` per arm, covering the BAND or error bar rather than the centre
    value -- the edges are what the squeeze must not cut through. Declines, with a note on stderr,
    when the band is not empty or when nothing lies on one side of it.
    """
    # SAFE: no arm reaches into the band. A band shortened on the page without being shortened in
    # fact is the one way this device can lie, and it would look like an ordinary figure.
    # USEFUL: arms on BOTH sides. A break with nothing beyond it is a zigzag pointing at empty axis,
    # which is worse than no break -- the reader is told something was removed, and it was not.
    #
    # Declining rather than raising is the whole point of returning a bool. The bounds are hand-placed
    # against one dataset at one `--pieces`, and both conditions fail for ordinary, supported reasons:
    # `--pieces 10` widens every band until they meet, and `--official-root` pointed elsewhere removes
    # the arm the band was opened beneath. Those are analysis knobs, so they must degrade to a
    # continuous axis -- which distorts nothing -- rather than refuse to draw the figure at all.
    # A break that silently stops applying to a COMMITTED figure still turns `--check` red, because the
    # PDF changes; that is where the loud signal belongs.
    spans = list(spans)
    intruding = [label for label, low, high in spans if low < hi and high > lo]
    if intruding:
        print(
            f"note: no y-axis break on {where} -- the band ({lo}, {hi}) is not empty: "
            f"{', '.join(intruding)} reaches into it. Drawing a continuous axis.",
            file=sys.stderr,
        )
        return False
    below = [label for label, _, high in spans if high <= lo]
    above = [label for label, low, _ in spans if low >= hi]
    if not (below and above):
        empty = "below" if not below else "above"
        print(
            f"note: no y-axis break on {where} -- nothing lies {empty} the band ({lo}, {hi}), so the "
            "mark would point at empty axis. Drawing a continuous axis.",
            file=sys.stderr,
        )
        return False
    return True


def draw_y_break_mark(ax: Axes, lo: float, hi: float) -> None:
    """Draw the zigzag that says the y axis is broken, across the left spine.

    Call it after everything that can move the y limits -- the mark is placed against them.
    """
    # Autoscaling is otherwise deferred to draw time, which is after this.
    ax.autoscale_view()
    to_axes = ax.transData + ax.transAxes.inverted()
    low = float(to_axes.transform((0.0, lo))[1]) - BREAK_MARK
    high = float(to_axes.transform((0.0, hi))[1]) + BREAK_MARK
    spine = ax.spines["left"]
    step = (high - low) / 3.0
    # White first, one step wider than the spine, so the zigzag replaces that stretch of axis rather
    # than being drawn on top of a straight line it is meant to interrupt.
    ax.plot(
        [0.0, 0.0],
        [low, high],
        transform=ax.transAxes,
        color="white",
        linewidth=spine.get_linewidth() + 1.4,
        clip_on=False,
        zorder=5,
    )
    ax.plot(
        [0.0, BREAK_MARK, -BREAK_MARK, 0.0],
        [low, low + step, high - step, high],
        transform=ax.transAxes,
        color=spine.get_edgecolor(),
        linewidth=spine.get_linewidth(),
        solid_capstyle="round",
        clip_on=False,
        zorder=6,
    )


def style_axes(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Titles, label sizes, and the frame every panel in this set shares."""
    # Split out of ``draw_panel`` so a panel that is not a per-piece line chart -- the cost/quality
    # scatter -- gets the identical frame without copying five numbers that would then drift.
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    # Two spines, no grid: a full box plus gridlines is two frames of chrome competing with a
    # handful of thin lines at this size. Plain matplotlib rather than ``sns.despine`` so the
    # drawing code needs one library's API, as everything else here does.
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def set_theme() -> Any:
    """Apply the shared theme and return `matplotlib.pyplot`.

    Callers do their drawing with the returned pyplot; seaborn is used for its theme only, so
    changing a figure needs one library's API rather than two.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # `ticks`, not `whitegrid`: a full box plus gridlines is two frames of chrome competing with five
    # thin lines at this size. `draw_panel` then drops the top and right spines, leaving one x-axis
    # along the bottom and one y-axis down the left. `paper` context scales fonts and line widths for
    # figures this small.
    sns.set_theme(style="ticks", context="paper")

    # PREFERRED, not required. Seaborn's own stack is `["Arial", "DejaVu Sans", ...]` and takes whichever
    # is installed, so the same script produced Arial figures on a machine with msttcorefonts and DejaVu
    # figures without it -- two people rebuilding a committed PDF got different bytes, `--check` called it
    # stale, and nothing in the diff said why. Naming the font is what stops that.
    #
    # FUTURA IS LICENSED AND NOT IN THIS REPO (executive call, user, 2026-09-22). The committed PDFs are
    # built on the one machine that has it, installed per-user from a licensed copy; `fonts-liberation`
    # is the documented fallback everywhere else and is metric-compatible with Arial. So a rebuild
    # elsewhere produces a readable, correctly laid-out figure in a different face -- which is why the
    # check below WARNS rather than raising. Raising would block anyone without the licence from
    # rebuilding at all, and the figures are an artifact of the machine that built them.
    #
    # Consequence to know before trusting `--check`: on a machine without Futura it reports every figure
    # stale, and the difference is the typeface, not the numbers. Rebuild only where Futura is installed,
    # or the commit will churn every PDF.
    #
    # The mathtext block is NOT optional: the arm labels are `$\mathrm{...}$` and mathtext ignores
    # `font.sans-serif` entirely, so without it the axes and titles change font while every legend label
    # silently does not -- a half-converted figure, harder to notice than an unconverted one. It is also
    # why the family is resolved before any key is set, rather than per key: see below.
    #
    # `Futura.ttc` carries Medium (500), Bold (700) and ExtraBold (800) but no Book/Regular, so matplotlib
    # warns once about weight `normal` and uses Medium. That is the face the figures are meant to use;
    # the warning is cosmetic and left unsuppressed rather than hidden behind a filter.
    font = "Futura"
    fallback = "Liberation Sans"

    # RESOLVE THE FAMILY ONCE, then pin every key to the SAME answer. `font.sans-serif` takes a fallback
    # list, but `mathtext.rm`/`it`/`bf` each take ONE family and fall back to matplotlib's own DejaVu, not
    # to the next name in that list. Pinning `font` to the mathtext keys unconditionally therefore drew
    # Liberation for the axes and DejaVu for every arm label on a machine without Futura -- precisely the
    # half-converted figure the block above says the mathtext pin exists to prevent. DejaVu being the
    # wider face, the combined legend then overflowed and `assert_legend_columns` RAISED, so a rebuild
    # without the licence failed outright rather than producing a usable figure in another face.
    from matplotlib.font_manager import FontProperties, findfont

    try:
        findfont(FontProperties(family=font), fallback_to_default=False)
        family = font
    except ValueError:
        # Warn, never raise: a rebuild without the licensed font should still produce a usable figure
        # (see the note above). `findfont` raising IS the probe -- an earlier version also compared the
        # resolved FILE name against the family, which adds nothing (the raise already covers absence)
        # and misfires on a legitimate install whose file is named for its PostScript name.
        family = fallback
        warnings.warn(
            f"{font} is not installed, so this run draws in {fallback} throughout -- text and mathtext "
            f"alike -- and its figures will differ from the committed ones by typeface. `--check` will "
            f"report them stale for that reason alone. Install {font} per-user (a licensed copy in "
            "$XDG_DATA_HOME/fonts, then `fc-cache -f` and clear ~/.cache/matplotlib) or rebuild nothing.",
            RuntimeWarning,
            stacklevel=2,
        )

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [family, fallback]
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = family
    matplotlib.rcParams["mathtext.it"] = f"{family}:italic"
    matplotlib.rcParams["mathtext.bf"] = f"{family}:bold"
    return plt


def draw_panel(
    ax: Axes,
    series: list[Series],
    bounds: list[tuple[int, int]],
    title: str,
    ylabel: str,
    *,
    xlabel: str = "task range",
) -> None:
    """Draw one panel: a centre line per arm with its band, over the piece axis."""
    for s in series:
        # Baseline bands fainter than the methods': they are context, and at this panel width two
        # overlapping reference bands would otherwise dominate the plot area.
        #
        # A DASHED ARM TAKES THE FAINT BAND WHATEVER ITS COLOUR, because linestyle means "the other
        # implementation of the arm it shares a colour with" (see the marker note below) -- so its
        # band is always drawn over a same-hued one, and two bands of one hue read as one arm's. The
        # smolagents arm is where this bites: its 3-rep spread covers ~14-57% at the widest StuLife
        # piece, which at 0.16 swallows `CodeAct+subagents_JAZ`'s band whole and leaves that arm's
        # line sitting on what a reader takes for its own band's edge.
        alpha = 0.10 if s.colour in BASELINE_COLOURS or s.linestyle != "-" else 0.16
        ax.fill_between(range(len(s.centre)), s.lo, s.hi, color=s.colour, alpha=alpha, linewidth=0)
    for s in series:
        # The subject of the figure is drawn heavier than the arms it is compared against, so a
        # reader finds it without consulting the legend. Keyed off the colour rather than the label
        # so a rename cannot silently drop the emphasis.
        lead = s.colour == JAZ_COLOUR
        # A DASHED ARM GETS HOLLOW MARKERS, so its points are distinguishable from the solid arm it
        # shares a colour with -- the two differ by implementation, not by method, so they are meant
        # to sit together (`CodeAct+subagents` JAZ vs smolagents, `CodeAct` JAZ vs AppWorld official).
        # Linestyle alone separates them only between markers; at five pieces the markers are most of
        # the ink, and two filled dots of one colour read as one arm sampled twice.
        dashed = s.linestyle != "-"
        ax.plot(
            range(len(s.centre)),
            s.centre,
            color=s.colour,
            linestyle=s.linestyle,
            linewidth=2.6 if lead else 1.5,
            marker="o",
            markersize=4.2 if lead else 3.2,
            markerfacecolor="white" if dashed else s.colour,
            markeredgecolor=s.colour,
            # The ring needs its own width: at the default the white centre of a 3.2pt marker is a
            # dot rather than a hole. 0.0 on a SOLID arm is a choice, not a no-op --
            # `lines.markeredgewidth` defaults to 1.0, so leaving it unset rings every solid marker
            # in its own colour and draws it a touch larger than `markersize` says.
            markeredgewidth=0.9 if dashed else 0.0,
            zorder=3 if lead else 2,
            label=s.label,
        )
    style_axes(ax, title, xlabel, ylabel)
    ax.set_xticks(range(len(bounds)))
    ax.set_xticklabels([f"{a + 1}-{b}" for a, b in bounds], fontsize=TICK_SIZE)


def ordered_handles(axes: list[Axes], order: tuple[str, ...]) -> tuple[list[Artist], list[str]]:
    """Legend handles across `axes`, deduped by label and sorted into `order`.

    Deduping is what lets one legend serve several panels: an arm that appears in both contributes
    one entry, taken from the first panel that drew it. A label missing from `order` sorts to the
    end rather than raising, so adding an arm degrades to "appended".
    """
    seen: dict[str, Artist] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            seen.setdefault(label, handle)
    rank = {label: i for i, label in enumerate(order)}
    pairs = sorted(seen.items(), key=lambda kv: rank.get(kv[0], len(rank)))
    return [handle for _, handle in pairs], [label for label, _ in pairs]


def legend_below(
    target: Any,
    handles: list[Artist],
    labels: list[str],
    *,
    ncol: int,
    y: float = -0.22,
    fontsize: float = 7.4,
) -> Any:
    """Place the legend under `target` (an `Axes` or a `Figure`), never inside the plot area.

    `ncol` is the MOST columns to use, not the exact number: the legend is measured against the
    figure width and columns are dropped until it fits. Raises `ValueError` if even one column is
    too wide.

    At this width an in-axes legend covered whichever corner the lines happened to occupy, so it
    goes below at the cost of vertical space.
    """
    # Fitting by dropping columns rather than by shrinking type: the legend font is the one thing a
    # reader compares across figures printed at the same size, so it stays fixed and the layout
    # gives way. The cost is real -- both StuLife panels' legends fell back to one column, and a
    # ~17% shorter plot area, until their entry order was rearranged to fit two -- and it is the
    # cheaper of the two, since `tight_layout` absorbs the height while over-wide text is simply cut
    # off at the page edge.
    #
    # This is a guard, not a convenience. It was written after the panel width dropped 3.4in ->
    # 3.125in and silently clipped the trailing character of the longest StuLife label in all four
    # committed PDFs: nothing raised, and the defect survived review into the paper's artifacts.
    # The combined figures had a caller-side `assert` that would have caught it; the single-panel
    # ones did not, which is why the check now lives here where no caller can forget it.
    kwargs: dict[str, Any] = {}
    # A Figure anchors in figure coordinates; an Axes defaults to its own. Without this a
    # figure-level legend would place itself relative to whatever axes matplotlib picked.
    figure = target if hasattr(target, "transFigure") else target.get_figure()
    if hasattr(target, "transFigure"):
        kwargs["bbox_transform"] = target.transFigure
    limit = figure.get_figwidth()
    for columns in range(ncol, 0, -1):
        legend = target.legend(
            handles,
            labels,
            fontsize=fontsize,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=columns,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.2,
            handletextpad=0.5,
            **kwargs,
        )
        # Width is only knowable once the text has been laid out, so the legend has to be built
        # before it can be measured -- hence build-measure-discard rather than a prediction.
        figure.canvas.draw()
        width = legend.get_window_extent().width / figure.dpi
        if width <= limit:
            return legend
        legend.remove()
    raise ValueError(
        f"legend does not fit: {width:.2f}in wide in one column, figure is {limit:.2f}in. "
        "Shorten a label or widen the figure."
    )


# The one legend label that is not a plain string: `JAZ invoke` is the method every figure is about, so
# it is drawn bold with `invoke` in the line's own colour -- the same emphasis `draw_panel` gives that
# arm's line, carried into the legend so the subject is findable without reading the labels.
JAZ_LABEL = r"$\mathrm{JAZ\ invoke}$"
_JAZ_PARTS = (r"$\mathbf{JAZ}$", r"$\mathbf{\ invoke}$")


def emphasise_jaz_entry(legend: Any) -> None:
    """Redraw the `JAZ invoke` legend entry bold, with `invoke` in `JAZ_COLOUR`.

    A no-op for a legend that has no such entry, so every figure can call it unconditionally.
    """
    # Matplotlib has no per-word colour: one `Text` carries one colour, and mathtext has no `\color`.
    # So the entry's `TextArea` is swapped for an `HPacker` of two of them. That reaches into the
    # legend's internal box (`_legend_handle_box` -> columns -> rows -> [DrawingArea, TextArea]), which
    # is private API and the reason this is one guarded helper rather than four call sites: if a future
    # matplotlib changes that layout, this fails in one place, and the `try` below turns it back into an
    # unemphasised-but-correct legend rather than a traceback.
    #
    # `set_figure` on the new packer is not optional -- an offsetbox built outside the legend has no
    # figure, and drawing one reaches for `self.get_figure().dpi` and raises `AttributeError: 'NoneType'`.
    from matplotlib.offsetbox import HPacker, TextArea

    try:
        for column in legend._legend_handle_box.get_children():  # pyright: ignore[reportPrivateUsage]
            for row in column.get_children():
                children = list(row.get_children())
                for index, child in enumerate(children):
                    if not isinstance(child, TextArea):
                        continue
                    if child.get_children()[0].get_text() != JAZ_LABEL:
                        continue
                    props = child.get_children()[0].get_fontproperties().copy()
                    props.set_weight("bold")
                    head, tail = _JAZ_PARTS
                    packed = HPacker(
                        children=[
                            TextArea(head, textprops={"color": "black", "fontproperties": props}),
                            TextArea(tail, textprops={"color": JAZ_COLOUR, "fontproperties": props}),
                        ],
                        pad=0,
                        sep=0,
                        align="baseline",
                    )
                    packed.set_figure(legend.get_figure())
                    # `legend.texts` is a SEPARATE list from the box tree, and both have to move
                    # together: the replaced `Text` left in it is never laid out, so it reports a stale
                    # position that `assert_legend_visible` folds into the legend's ink bbox -- a correct
                    # legend measured 1.46in tall instead of 0.50in and failed the check, with nothing
                    # visibly wrong in the figure. So locate it BEFORE touching `row._children`: a
                    # failure between the two mutations would leave exactly that state, and the `except`
                    # below would report it as the unemphasised-but-correct legend it is not.
                    texts = legend.texts
                    position = texts.index(child.get_children()[0])
                    replacements = [area.get_children()[0] for area in (packed.get_children())]
                    children[index] = packed
                    row._children = children  # pyright: ignore[reportPrivateUsage]
                    legend.texts = texts[:position] + replacements + texts[position + 1 :]
    # `ValueError` is `list.index`'s, and belongs here for the same reason as the other two: it is how
    # a drift between the box tree and `legend.texts` surfaces, and it must degrade rather than traceback.
    except (AttributeError, IndexError, ValueError):
        # The legend still reads correctly without the emphasis; a figure is worth more than a flourish.
        return


def assert_legend_columns(legend: Any, expected: int) -> None:
    """Raise unless the legend used `expected` columns.

    For a legend whose grouping is positional: `legend_below` drops columns until the ink fits, which
    rewraps a flat legend harmlessly but scrambles one whose entries were placed by index.
    """
    # `_ncols` is private, and read here rather than tracked by the caller because what matters is
    # what the legend DID, not what it was asked for -- the fallback happens inside `legend_below`.
    # `None` rather than `expected` as the default: defaulting to the value that makes the assert
    # pass would turn a matplotlib rename into a silently disabled gate, on the one check standing
    # between a positional legend and shipping it mis-grouped.
    actual = getattr(legend, "_ncols", None)
    if actual is None:
        raise AssertionError(
            "cannot read the legend's column count: matplotlib no longer exposes `_ncols`. Find "
            "what replaced it -- the positional grouping is unguarded until this reads again."
        )
    if actual != expected:
        raise AssertionError(
            f"legend fell back to {actual} columns from {expected}: its grouping is positional, so "
            "the rows are now wrong. Shorten a label or widen the figure."
        )


def assert_legend_visible(legend: Any) -> float:
    """Raise `AssertionError` unless every mark the legend draws lies inside its figure.

    Returns the width of that ink in inches. Call this AFTER `tight_layout`, which is what decides
    how much room the legend actually gets.
    """
    # It measures the legend's TEXTS AND HANDLES, not `legend.get_window_extent()`. The legend's own
    # box carries padding that routinely hangs a few hundredths of an inch past the canvas with
    # nothing drawn in it, so the padded box gives false positives; ink is what a reader loses.
    #
    # Two real defects motivate this, and each is invisible to the other's check. The StuLife panel
    # clipped the trailing character of its longest label when the panel narrowed to 3.125in -- ink
    # over the edge. Both combined figures anchored their legend's TOP to the bottom edge of the
    # canvas, so the whole legend fell off and they shipped with an empty reserved strip and no
    # legend at all -- ink nowhere near an edge, and a width-only assert (which those two scripts
    # had) passes happily. `--check` cannot see either one: it compares a figure against an equally
    # broken committed copy.
    figure = legend.get_figure()
    figure.canvas.draw()
    boxes = [text.get_window_extent() for text in legend.get_texts()]
    boxes += [h.get_window_extent() for h in legend.legend_handles if hasattr(h, "get_window_extent")]
    assert boxes, "legend draws nothing"
    dpi = figure.dpi
    x0 = min(b.x0 for b in boxes) / dpi
    x1 = max(b.x1 for b in boxes) / dpi
    y0 = min(b.y0 for b in boxes) / dpi
    y1 = max(b.y1 for b in boxes) / dpi
    width, height = figure.get_size_inches()
    # A hair of tolerance: the extents are computed in pixels and compared in inches, so an exactly
    # flush edge lands a rounding step either side of zero.
    tol = 0.005
    over = [
        name
        for name, ok in (
            ("left", x0 >= -tol),
            ("right", x1 <= width + tol),
            ("bottom", y0 >= -tol),
            ("top", y1 <= height + tol),
        )
        if not ok
    ]
    assert not over, (
        f"legend runs off the {', '.join(over)} of the figure: its marks span "
        f"({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f})in on a {width:.2f} x {height:.2f}in figure. "
        "Shorten a label, or give the legend more room."
    )
    return x1 - x0
