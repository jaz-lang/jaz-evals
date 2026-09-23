"""Stream an ATIF-shaped trajectory to a browsable directory *as the run happens*.

ATIF is the Agent Trajectory Interchange Format, JAZ's structured trace.

This is the incremental counterpart to `jaz_evals.trace_to_directory`: instead of expanding a finished
`agent.atif.json` in one pass at the end, `StreamingTraceWriter` is fed one event at a time by the
`StreamingTraceDir` hook (`jaz_evals.harnesses.streaming_trace`) and writes each message the moment it is
seen, so a long run can be watched (`tail -f .../trace.md`) while it is still going.

**Why this is a separate module from the hook that drives it**, given the two are useless apart and
their names read as duplicates (this file holds `StreamingTraceWriter`; the hook class is
`StreamingTraceDir`, over in `harnesses/streaming_trace.py` -- the names are crossed, which is the
usual reason someone opens the wrong one). The split is on the jaz boundary: this module imports no
jaz, so the branching trace-writing logic can be exercised without one. Fold it into the hook -- a
`jaz.hooks.Hook` subclass -- and that stops being true; what is left in
`harnesses/streaming_trace.py` is event plumbing.

Nothing else consumes this writer -- the hook is its only production caller (smolagents writes its own
`smolagents_trace.jsonl`/`.md` and never touches it). So the split buys testability, not reuse.

The on-disk *directory* layout is the same **flat, sibling-based** one the batch converter
(`jaz_evals.trace_to_directory`) produces, so the tree is navigable the same way. The per-message content
format differs deliberately: each message is a fenced block preceded by a provenance header, not the batch
converter's `## Role` sections.

    <node>/trace.md         -- THIS agent's conversation only (sub-agents excluded), one fenced block per
                               message, appended live. Each block is preceded by the message's provenance
                               (role, ATIF `kind`, stable id, iteration, buffer index, persistence).
    <node>/overview.md      -- metadata + per-iteration REPL snippets + links to the sub-agents each
                               iteration launched; rewritten (it is small) whenever the node advances.
    <node>/iter<N>_sub<M>/  -- the sub-agent launched in REPL iteration N, position M -- a SIBLING dir,
                               itself the same structure recursively.

Buffer messages are rendered once, as each first appears (`append_message`) -- except hook-*added* ones
(`kind=ADDED`), which the enter walk skips. Hook *edits* -- messages a hook ADDED this turn (a compaction
summary, a `ContextWindowWarning` delegation prompt) or DROPPED -- are rendered separately, per turn, by
`append_edits`. That separate path is necessary because a *transient* add (shown for one query, never kept
in the buffer) never appears in a buffer snapshot; drops leave no buffer trace; and a *persistent* add,
though it does sit in the buffer, is skipped there so it is not rendered twice. So the streaming view shows
every edit the batch converter's `_render_hook_edits` shows -- persistent and transient adds, and drops --
each exactly once, matching batch (which likewise renders adds only from `extra.edits`, never as a message).

This module is a pure function of the values the hook extracts (no `jaz` import), so it type-checks under
strict pyright and is exercisable without jaz: the hook does the event-to-argument translation, the
writer does all formatting and I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


class AddedEdit(TypedDict):
    """One message a hook added this turn (persistent, or transient like a ContextWindowWarning)."""

    role: str
    persistent: bool
    id: str | None
    index: int | None
    content: str


class DroppedEdit(TypedDict):
    """One message a hook dropped this turn (a compaction, a warning it retracted)."""

    role: str
    persistent: bool
    id: str | None
    index: int | None
    content: str


# "First couple of lines" of REPL code/output shown per iteration in overview.md.
_SNIPPET_LINES = 3
# Heavy single-line cap on each `inputs`/`scope` value's repr in overview.md.
_OVERVIEW_VALUE_LEN = 200
# Single-line cap on a dropped message's content snippet in a hook-edits block: enough head to tell which
# message left (its own block above carries the whole thing), not the whole message.
_DROPPED_SNIPPET_LEN = 200
# Fence width for a message block. Three backticks is the default (the standard Markdown fence); a message
# whose content itself contains a run of >=3 backticks escalates to one longer than that run, so the fence a
# reader relies on to delimit the block can never be closed early by the content.
_MIN_FENCE = 3

# Message role -> Markdown heading label. `agent`/`assistant` both mean the model's turn.
_ROLE_LABEL = {"system": "System", "user": "User", "agent": "Assistant", "assistant": "Assistant"}


def _role_label(role: str) -> str:
    return _ROLE_LABEL.get(role, role.capitalize() or "?")


def _is_agent(role: str) -> bool:
    """Whether `role` is the model's own turn -- the only role that gets a `py`-tagged fence."""
    return role in ("agent", "assistant")


# TODO: these helpers, the snippet/length constants, the role-label map, the overview.md layout in
# `_write_overview`, and the subtree-cost roll-up (`_subtree_cost` here / `_subtree_cost_usd` there) are
# duplicated from `trace_to_directory.py`; consolidate the shared format in one place.
def _one_line(text: str, max_len: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= max_len else collapsed[:max_len] + "…"


def _first_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    head = "\n".join(lines[:n])
    return head + ("\n…" if len(lines) > n else "")


def _short_id(msg_id: str) -> str:
    """First 8 hex chars of a provenance id -- enough to Ctrl-F within one trace. Callers omit the id
    field entirely when it is absent, so this is only reached with a real id."""
    return msg_id[:8]


def _fence(content: str) -> str:
    """The backtick fence for a message: `_MIN_FENCE` normally, widened past any backtick run in the
    content so the block cannot be closed early."""
    longest_run = max((len(m) for m in re.findall(r"`+", content)), default=0)
    return "`" * max(_MIN_FENCE, longest_run + 1)


def _provenance_line(
    role: str,
    kind: str | None,
    msg_id: str | None,
    iteration: int | None,
    index: int | None,
    persistent: bool | None,
) -> str:
    """The `### Role — kind=… · id=… · iter=… · index=… · persistent=…` header shown before a message,
    dropping whichever coordinates the trace does not carry."""
    meta: list[str] = []
    if kind is not None:
        meta.append(f"kind={kind}")
    if msg_id:
        # Omitted (not shown as `id=?`) when absent -- only the final assistant, which no enter follows
        # and so is never stamped; its content is complete, just un-ided.
        meta.append(f"id={_short_id(msg_id)}")
    if iteration is not None:
        meta.append(f"iter={iteration}")
    if index is not None:
        meta.append(f"index={index}")
    if persistent is not None:
        meta.append(f"persistent={str(persistent).lower()}")
    return f"### {_role_label(role)} — " + " · ".join(meta)


def _message_block(
    role: str,
    kind: str | None,
    msg_id: str | None,
    iteration: int | None,
    index: int | None,
    persistent: bool | None,
    content: str,
) -> str:
    """One message rendered for trace.md: provenance header, then the content in a fenced block
    (```py`` for the model's turn, a bare fence otherwise)."""
    fence = _fence(content)
    open_fence = f"{fence}py" if _is_agent(role) else fence
    return (
        "\n".join(
            [
                _provenance_line(role, kind, msg_id, iteration, index, persistent),
                open_fence,
                content,
                fence,
                "",
            ]
        )
        + "\n"
    )


def _edit_coords(msg_id: str | None, index: int | None) -> str:
    """`id=… · index=…` for a hook edit, dropping whichever it does not carry (a transient add has no id)."""
    parts: list[str] = []
    if msg_id:
        parts.append(f"id={_short_id(msg_id)}")
    if index is not None:
        parts.append(f"index={index}")
    return " · ".join(parts)


def _edits_block(iteration: int | None, added: list[AddedEdit], dropped: list[DroppedEdit]) -> str:
    """A turn's hook edits rendered for trace.md: each ADDED message in a fenced block (labelled
    persistent/transient), then a compact list of DROPPED messages. Transient adds never enter the buffer,
    so this is the only place they -- and drops -- appear, matching the batch converter's hook-edits view."""
    it = "" if iteration is None else f" — iter {iteration}"
    lines: list[str] = [f"### Hook edits{it}", ""]
    for a in added:
        persistence = "persistent" if a["persistent"] else "transient"
        header = f"#### Added ({persistence}, {_role_label(a['role'])})"
        coords = _edit_coords(a["id"], a["index"])
        if coords:
            header += f" — {coords}"
        fence = _fence(a["content"])
        open_fence = f"{fence}py" if _is_agent(a["role"]) else fence
        lines.extend([header, open_fence, a["content"], fence, ""])
    if dropped:
        lines.extend(["#### Dropped", ""])
        for d in dropped:
            persistence = "persistent" if d["persistent"] else "transient"
            coords = _edit_coords(d["id"], d["index"])
            tags = f"{coords} · {persistence}" if coords else persistence
            snippet = _one_line(d["content"], _DROPPED_SNIPPET_LEN)
            lines.append(f"- {_role_label(d['role'])} ({tags}): {snippet}")
        lines.append("")
    return "\n".join(lines) + "\n"


@dataclass
class _Turn:
    """One REPL iteration's data, accumulated for overview.md."""

    code: str
    output: str | None = None
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class _Node:
    """Per-invoke writer state: where its files live, what it has already emitted, and the running
    material overview.md is rebuilt from."""

    node_dir: Path
    depth: int
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    task_name: str = "main"
    # `{name: (type_name, repr)}` for the invoke's committed inputs / resolved scope, shown in overview.
    inputs: dict[str, tuple[str, str]] = field(default_factory=dict[str, tuple[str, str]])
    scope: dict[str, tuple[str, str]] = field(default_factory=dict[str, tuple[str, str]])
    emitted_ids: set[str] = field(default_factory=set[str])
    turns: dict[int, _Turn] = field(default_factory=dict[int, "_Turn"])
    turn_order: list[int] = field(default_factory=list[int])
    # iteration -> [(sibling dir name, sub-agent task label)] for the overview's per-iteration sub links.
    subs_by_iter: dict[int, list[tuple[str, str]]] = field(default_factory=dict[int, list[tuple[str, str]]])
    sub_counts: dict[int, int] = field(default_factory=dict[int, int])
    result_str: str = ""
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_iterations: int = 0
    # Child invoke ids, for rolling this node's `total_cost` up over the whole delegation subtree.
    children: list[str] = field(default_factory=list[str])

    @property
    def trace_path(self) -> Path:
        return self.node_dir / "trace.md"

    @property
    def overview_path(self) -> Path:
        return self.node_dir / "overview.md"


class StreamingTraceWriter:
    """Fed one event's worth of data at a time, writes the browsable trace directory incrementally.

    All methods are keyed on the invoke id the hook supplies. `open_node` must precede any other call for
    an invoke; `close_node` finalizes it. Unknown ids are ignored (a defensive no-op), so an event for an
    invoke that opened before this writer was attached cannot crash the run it is only observing.
    """

    def __init__(self, root: Path):
        self._root = root
        self._nodes: dict[str, _Node] = {}
        self._root_count = 0

    # -- lifecycle ---------------------------------------------------------------------------------

    def open_node(
        self, invoke_id: str, parent_invoke_id: str | None, parent_repl_iteration: int | None, depth: int
    ) -> None:
        """Create the directory for a newly-entered invoke and start its (empty) trace.md."""
        parent = self._nodes.get(parent_invoke_id) if parent_invoke_id is not None else None
        if parent is not None and parent_repl_iteration is not None:
            # A sub-agent: a sibling dir under the parent, named by the launching iteration and its index
            # among that iteration's sub-agents (an iteration can launch more than one).
            m = parent.sub_counts.get(parent_repl_iteration, 0)
            parent.sub_counts[parent_repl_iteration] = m + 1
            dirname = f"iter{parent_repl_iteration}_sub{m}"
            node_dir = parent.node_dir / dirname
            parent.subs_by_iter.setdefault(parent_repl_iteration, []).append((dirname, "main"))
        elif self._root_count == 0:
            # The first top-level invoke expands directly into the output dir, which is the layout the
            # batch converter writes for a SINGLE root (`trace_to_directory.py:438`). Correct for every
            # harness that attaches this writer today: `JazHarness` makes exactly one root per attempt.
            node_dir = self._root
            self._root_count = 1
        else:
            # Subsequent roots use `task<n>`, the same vocabulary the batch converter uses for a
            # multi-root trace (`trace_to_directory.py:454`). It was `root<n>` here, under a comment
            # claiming to match the batch converter -- which it did not, and which mattered because
            # `JazPerTaskHarness` genuinely produces MANY roots (one `jaz.invoke` per task). It does not
            # attach this writer today, so the mismatch was unreachable rather than absent; wiring live
            # tracing into the per-task harness is a natural thing to want, and would have produced
            # `root1..root99` beside a batch layout of `task00..task99`.
            #
            # AGREEMENT IS PARTIAL, and cannot be total: the batch converter knows the root count before
            # it names anything, so with N > 1 it puts even the first root in `task00`. A streaming
            # writer must commit the first root's location before knowing whether a second will arrive,
            # and defaulting it to `task00` would break the single-root layout for every run that has
            # only one. So the residual difference is the FIRST root's location; the rest align.
            #
            # Width 2 matches the batch converter's `max(2, ...)` floor for up to 100 roots, which is
            # every shipped queue length; beyond that the batch converter widens and this does not.
            self._root_count += 1
            node_dir = self._root / f"task{self._root_count - 1:02d}"

        node_dir.mkdir(parents=True, exist_ok=True)
        node = _Node(
            node_dir=node_dir,
            depth=depth,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
        )
        self._nodes[invoke_id] = node
        if parent is not None:
            parent.children.append(invoke_id)  # record the edge for subtree-cost roll-up
        node.trace_path.write_text("", encoding="utf-8")
        self._write_overview(node)
        if parent is not None:
            self._write_overview(parent)  # surface the new (as-yet-unnamed) sub link right away

    def set_meta(
        self,
        invoke_id: str,
        task_name: str,
        inputs: dict[str, tuple[str, str]],
        scope: dict[str, tuple[str, str]],
    ) -> None:
        """Record the invoke's task label and committed inputs/scope (known at send, not enter)."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        node.task_name = task_name
        node.inputs = inputs
        node.scope = scope
        self._write_overview(node)
        # Update the parent's sub link so it shows this sub-agent's task, not the "main" placeholder.
        self._relabel_in_parent(node)

    def close_node(
        self,
        invoke_id: str,
        result_str: str,
        total_cost: float,
        total_prompt_tokens: int,
        total_iterations: int,
    ) -> None:
        """Finalize the invoke: record its result and totals and write the last overview."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        node.result_str = result_str
        node.total_cost = total_cost
        node.total_prompt_tokens = total_prompt_tokens
        node.total_iterations = total_iterations
        self._write_overview(node)

    # -- content -----------------------------------------------------------------------------------

    def append_message(
        self,
        invoke_id: str,
        role: str,
        kind: str | None,
        msg_id: str | None,
        iteration: int | None,
        index: int | None,
        persistent: bool | None,
        content: str,
    ) -> None:
        """Append one message to the node's trace.md, deduped by stable id (a buffer message reappears
        every later turn). A message with no id -- only the final assistant, unstamped because no enter
        follows it -- is always appended, since it is emitted exactly once from invoke-exit."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        if msg_id is not None:
            if msg_id in node.emitted_ids:
                return
            node.emitted_ids.add(msg_id)
        with node.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(_message_block(role, kind, msg_id, iteration, index, persistent, content))

    def append_edits(
        self, invoke_id: str, iteration: int | None, added: list[AddedEdit], dropped: list[DroppedEdit]
    ) -> None:
        """Append a turn's hook edits to the node's trace.md: the messages a hook ADDED (persistent or
        transient) and the ones it DROPPED. Transient adds -- e.g. a `ContextWindowWarning` delegation
        prompt -- never enter the message buffer, so they are invisible to `append_message` (which renders
        the buffer) and surface only here. This is the streaming analogue of the batch converter's
        `_render_hook_edits`, so both show every edit."""
        if not (added or dropped):
            return
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        with node.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(_edits_block(iteration, added, dropped))

    def record_turn(
        self,
        invoke_id: str,
        iteration: int,
        code: str,
        cost_usd: float | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Record an iteration's REPL code + per-call metrics for overview.md (its output arrives later,
        via `set_turn_output`)."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        if iteration not in node.turns:
            node.turn_order.append(iteration)
        node.turns[iteration] = _Turn(
            code=code, cost_usd=cost_usd, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        self._write_overview(node)

    def set_turn_output(self, invoke_id: str, iteration: int, output: str) -> None:
        """Fill in an iteration's REPL output (its observation, seen at the following enter)."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        turn = node.turns.get(iteration)
        if turn is not None:
            turn.output = output
            self._write_overview(node)

    # -- overview rendering ------------------------------------------------------------------------

    def _relabel_in_parent(self, node: _Node) -> None:
        parent = self._nodes.get(node.parent_invoke_id) if node.parent_invoke_id is not None else None
        if parent is None or node.parent_repl_iteration is None:
            return
        subs = parent.subs_by_iter.get(node.parent_repl_iteration)
        if not subs:
            return
        expected = node.node_dir.name
        parent.subs_by_iter[node.parent_repl_iteration] = [
            (d, node.task_name if d == expected else label) for d, label in subs
        ]
        self._write_overview(parent)

    def _subtree_cost(self, node: _Node) -> float:
        """This node's own cost plus every descendant's -- the whole delegation subtree's spend.

        Correct to read at any node's `close_node`: jaz delegation is synchronous, so every descendant
        invoke has already closed (its `total_cost` set) before an ancestor closes. Read mid-flight (an
        intermediate overview write) it undercounts still-open children, but the file is rewritten at
        close, so the final overview is exact.

        One residual undercount survives to the final write: a descendant whose `close_node` never fires
        -- an interrupted run (budget/context/crash), or a sub-agent that dies without an invoke-exit --
        keeps `total_cost=0.0` and contributes nothing. That is the same downward direction as the
        mid-flight case (never an overcount), and early stopping is an ordinary outcome here.
        """
        return node.total_cost + sum(
            self._subtree_cost(child) for cid in node.children if (child := self._nodes.get(cid)) is not None
        )

    def _write_overview(self, node: _Node) -> None:
        iters = node.total_iterations or len(node.turn_order)
        lines: list[str] = [
            f"# {node.task_name}",
            "",
            f"**Depth:** {node.depth}  |  **Iterations:** {iters}  |  "
            f"**Total cost:** ${node.total_cost:.4f}  |  **Prompt tokens:** {node.total_prompt_tokens}",
        ]
        # `total_cost` above mirrors ATIF's per-trajectory `total_cost_usd` (self-cost only). When this
        # node delegated, also show the subtree roll-up so the figure agrees with the run's authoritative
        # cost; omitted for a leaf, where it would just repeat the self-cost.
        # Gated on delegation, not on subtree != self, by decision: a node that delegated to ~zero-cost
        # children would show a line equal to its self-cost, but that case is negligible and the presence
        # check is simpler than comparing rounded costs.
        if node.children:
            lines.append(f"**Subtree cost (incl. subagents):** ${self._subtree_cost(node):.4f}")
        lines.extend([f"**Result:** {node.result_str or '(in progress)'}", ""])
        for section, namespace in (("Inputs", node.inputs), ("Scope", node.scope)):
            if not namespace:
                continue
            lines.extend([f"## {section}", ""])
            lines.extend(
                f"- `{name}` ({typ}): {_one_line(rep, _OVERVIEW_VALUE_LEN)}"
                for name, (typ, rep) in namespace.items()
            )
            lines.append("")

        lines.extend(["## Iterations", ""])
        for iteration in node.turn_order:
            turn = node.turns[iteration]
            lines.extend([f"### Iteration {iteration}", ""])
            if turn.cost_usd is not None:
                toks = f"{turn.prompt_tokens or 0}->{turn.completion_tokens or 0}"
                lines.extend([f"*cost ${turn.cost_usd:.4f} | tokens {toks}*", ""])
            lines.extend(["REPL code:", "```", _first_lines(turn.code, _SNIPPET_LINES), "```"])
            if turn.output:
                lines.extend(["REPL output:", "```", _first_lines(turn.output, _SNIPPET_LINES), "```"])
            for dirname, label in node.subs_by_iter.get(iteration, []):
                lines.append(f"Sub-agent: [`{dirname}/`]({dirname}/overview.md) — {label}")
            lines.append("")

        node.overview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
