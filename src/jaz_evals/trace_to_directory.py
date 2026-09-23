"""Expand an ATIF trace (`agent.atif.json`) into a browsable directory.

ATIF is the Agent Trajectory Interchange Format, JAZ's structured trace.

This is jaz-evals' own converter, used by `JazHarness._expand_trace` in place of
`jaz.utils.trace_to_directory`. It is a pure function of the ATIF JSON (no jaz import), so it runs
without jaz and can be re-applied to any archived `agent.atif.json`.

Each invoke trajectory (the agent's own session, one per invoke node) becomes a directory with a
**flat, sibling-based** layout:

    <node>/overview.md      -- heavily-truncated reprs of the invoke's `inputs`/`scope` values, then
                               one snippet per iteration (the first few lines of REPL code and
                               output) with links to the sub-agents that iteration launched.
    <node>/trace.md         -- the FULL conversation of THIS agent only (sub-agents excluded),
                               rendered like `jaz.utils.log_to_markdown` (`## System`/`## User`/
                               `## Assistant` headers followed by each message whole).
    <node>/iter<N>_sub<M>/  -- the sub-agent launched in REPL iteration N, position M (0-based) --
                               a SIBLING of overview.md/trace.md, itself the same structure
                               recursively.
"""

# The flat, sibling layout (rather than nesting sub-agents under `iter_N/subagent_K/`) keeps each
# agent's own transcript (`trace.md`) whole and un-interleaved, and encodes *where* a sub-agent was
# launched in its folder name -- an iteration can launch more than one, hence both N and M.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

# "First couple of lines" of REPL code/output shown per iteration in overview.md.
_SNIPPET_LINES = 3
# Heavy single-line cap on each `inputs`/`scope` value's repr in overview.md.
_OVERVIEW_VALUE_LEN = 200
# Single-line cap on a dropped message's content snippet in trace.md -- enough of the head to tell
# *which* message left (the reader can open its own step for the whole thing), not the whole message.
_DROPPED_SNIPPET_LEN = 200

# ATIF `source` -> Markdown role header, matching `log_to_markdown`'s System/User/Assistant.
_ROLE = {"system": "System", "user": "User", "agent": "Assistant"}


def _as_dict(value: Any) -> dict[str, Any]:
    """`value` as a str-keyed dict when it is a mapping, else `{}`."""
    # One cast site: the ATIF trace is `json.loads`ed to `Any`, and `isinstance(x, dict)` alone
    # narrows to `dict[Unknown, Unknown]`; routing every nested lookup through here keeps the module
    # clean under strict type-checking.
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _one_line(text: str, max_len: int) -> str:
    """Collapse whitespace to a single line and hard-truncate with an ellipsis."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= max_len else collapsed[:max_len] + "…"


def _first_lines(text: str, n: int) -> str:
    """The first `n` lines of `text`, with a trailing `…` line when more were dropped."""
    lines = text.splitlines()
    head = "\n".join(lines[:n])
    return head + ("\n…" if len(lines) > n else "")


def _steps(node: dict[str, Any]) -> list[dict[str, Any]]:
    """The node's step list as typed dicts (non-dict entries skipped)."""
    return [_as_dict(s) for s in cast("list[Any]", node.get("steps") or []) if isinstance(s, dict)]


def _agent_steps_with_outputs(
    steps: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str | None]]:
    """Pair each `agent` step with its REPL observation output.

    The observation is the first `user` step after the agent step and before the next agent step,
    preferring the one stamped `extra.provenance.kind == "observation"` (so a hook-added
    `<return_type>` reminder is skipped), else the first user step in that window. A terminal agent
    step has no observation, so its pair carries `None`.
    """
    pairs: list[tuple[dict[str, Any], str | None]] = []
    n = len(steps)
    for i, step in enumerate(steps):
        if step.get("source") != "agent":
            continue
        observation: str | None = None
        first_user: str | None = None
        for j in range(i + 1, n):
            src = steps[j].get("source")
            if src == "agent":
                break
            if src != "user":
                continue
            msg = steps[j].get("message")
            if first_user is None and isinstance(msg, str):
                first_user = msg
            prov = _as_dict(_as_dict(steps[j].get("extra")).get("provenance"))
            if prov.get("kind") == "observation" and isinstance(msg, str):
                observation = msg
                break
        pairs.append((step, observation if observation is not None else first_user))
    return pairs


def _task_name(node: dict[str, Any]) -> str:
    """Human label for a trajectory: `extra.inputs.task`'s repr when present, else `"main"`."""
    task = _as_dict(_as_dict(node.get("extra")).get("inputs")).get("task")
    if isinstance(task, dict):
        rep = _as_dict(task).get("repr_prefix_10000")
        if isinstance(rep, str) and rep:
            return _one_line(rep, 70)
    return "main"


def _format_cost(cost: float | None) -> str:
    return "N/A" if cost is None else f"${cost:.4f}"


def _subtree_cost_usd(node: dict[str, Any]) -> float:
    """This trajectory's `final_metrics.total_cost_usd` plus every nested subagent's, recursively.

    ATIF's per-trajectory `total_cost_usd` is self-cost only (nested invokes live under
    `subagent_trajectories`, each with its own metrics), so a delegating run's root figure excludes
    the subagents that did most of the spending. This rolls the whole subtree up to the run total.
    """
    metrics = _as_dict(node.get("final_metrics"))
    # A missing `total_cost_usd` contributes 0.0 to the roll-up -- its only sensible value -- even though
    # the node's own `Total cost` line renders that same missing value as `N/A`. The asymmetry is deliberate.
    total = cast("float | None", metrics.get("total_cost_usd")) or 0.0
    for child in cast("list[Any]", node.get("subagent_trajectories") or []):
        if isinstance(child, dict):
            total += _subtree_cost_usd(cast("dict[str, Any]", child))
    return total


def _result_str(extra: dict[str, Any]) -> str:
    """The invoke's outcome as `type: value` (or `type: exception`), heavily truncated."""
    # The return value / exception live in `extra`, not in any step, so surfacing them here is the
    # only place the expanded tree carries them -- and they are the first thing a diagnostic reaches for.
    result_type = str(extra.get("result_type", "unknown"))
    value = extra.get("result_value")
    exception = extra.get("result_exception")
    if value is not None:
        return _one_line(f"{result_type}: {value}", _OVERVIEW_VALUE_LEN)
    if exception is not None:
        return _one_line(f"{result_type}: {exception}", _OVERVIEW_VALUE_LEN)
    return result_type


def _short_id(msg_id: Any) -> str:
    """The first 8 hex chars of a provenance id -- enough to be unique within one trace and to Ctrl-F
    between a message's header and any add/drop that references it (the full id is in the structured
    trace). `?` when a message was never stamped (a hand-authored or pre-provenance trace)."""
    return msg_id[:8] if isinstance(msg_id, str) else "?"


def _role_label(source: Any) -> str:
    """A message's display role (`System`/`User`/`Assistant`), so a message and its later drop/add
    read the same way -- the raw wire role (`user`/`agent`) is never shown on its own."""
    src = cast("str", source or "")
    return _ROLE.get(src, src.capitalize() or "?")


def _step_meta(step: dict[str, Any]) -> tuple[str | None, Any, Any]:
    """A step's authoritative `(id, iteration, index)` from the trace: id + iteration from
    `extra.provenance` (iteration falling back to `extra.iteration`, since an agent step is stamped
    only at the next enter), and the buffer slot from `extra.index`. Any that the trace does not carry
    (an older trace, predating these fields) comes back `None`."""
    extra = _as_dict(step.get("extra"))
    prov = _as_dict(extra.get("provenance"))
    iteration = prov.get("iteration")
    if iteration is None:
        iteration = extra.get("iteration")
    msg_id = prov.get("id")
    return (msg_id if isinstance(msg_id, str) else None, iteration, extra.get("index"))


def _coords(msg_id: Any, iteration: Any, index: Any) -> str:
    """The identity/coordinate tags shared by a message header and any edit that names it:
    `id=… -- iter N -- index I`, dropping whichever the trace does not carry. The id is the stable
    identity; iteration is stable too; the index is a buffer slot and drifts across turns."""
    parts: list[str] = []
    if isinstance(msg_id, str):
        parts.append(f"id={_short_id(msg_id)}")
    if iteration is not None:
        parts.append(f"iter {iteration}")
    if index is not None:
        parts.append(f"index {index}")
    return " -- ".join(parts)


def _render_trace_md(node: dict[str, Any]) -> str:
    """This agent's full conversation (its own steps only -- sub-agents excluded, since they are
    separate trajectories), rendered `log_to_markdown`-style: a `## Role` header then each message
    whole. A step's hook message edits (`extra.edits`) are surfaced alongside it, so a hook that
    injected a warning or dropped context is visible here and not only in the structured trace."""
    by_id = _message_index(node)
    lines: list[str] = []
    for step in _steps(node):
        message = step.get("message")
        if isinstance(message, str):
            lines.extend([_step_header(step), "", message, ""])
        lines.extend(_render_hook_edits(step, by_id))
    return "\n".join(lines)


def _step_header(step: dict[str, Any]) -> str:
    """`## Role` plus the message's authoritative coordinates from the trace -- stable id, iteration,
    and buffer `index` (`extra.index`) -- each omitted when the trace does not carry it. The index is a
    buffer position and drifts as later turns add/drop around it, so the id is the stable identity an
    edit keys on."""
    parts = [f"## {_role_label(step.get('source'))}"]
    coords = _coords(*_step_meta(step))
    if coords:
        parts.append(coords)
    return " -- ".join(parts)


def _message_index(node: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map each stamped message's provenance `id` to its `{role, content}`.

    A drop records only the dropped message's id + slot; its role and content live on the message's own
    record, so this resolves them for the drop's snippet. Indexes both conversation steps and stamped
    hook-added messages: JAZ records a stamped (persistent) add only inside `extra.edits.added` (never
    as a step of its own), yet it is a buffer message that a later turn can drop.
    """
    by_id: dict[str, dict[str, str]] = {}
    for step in _steps(node):
        msg_id = _step_meta(step)[0]
        message = step.get("message")
        if msg_id is not None and isinstance(message, str):
            by_id[msg_id] = {"role": cast("str", step.get("source") or "?"), "content": message}
        edits = _as_dict(_as_dict(step.get("extra")).get("edits"))
        for entry in cast("list[Any]", edits.get("added") or []):
            for msg in cast("list[Any]", _as_dict(entry).get("messages") or []):
                m = _as_dict(msg)
                add_id = _as_dict(m.get("provenance")).get("id")
                content = m.get("content")
                if isinstance(add_id, str) and isinstance(content, str):
                    by_id[add_id] = {"role": str(m.get("role", "?")), "content": content}
    return by_id


def _render_hook_edits(step: dict[str, Any], by_id: dict[str, dict[str, str]]) -> list[str]:
    """Markdown for a step's `extra.edits`: the messages a hook added and the ones it dropped. These
    are not ordinary conversation steps -- a context/iteration warning, a compaction drop -- so
    surfacing them keeps the render faithful to what the agent actually saw next.

    The block is headed by the query iteration it was composed for (`extra.edits.iteration`), which is
    the turn the edits happened -- distinct from the anchor step's own iteration, since the edits hang
    on the previous turn's last message. Each add is shown inline with its content (JAZ records both
    persistent and transient adds only here, never as a step) with its id (stamped adds only) and its
    landing index. Each drop names the removed message by id + the slot it sat in when dropped + role +
    a content snippet resolved from `by_id` (its content lives on its own record).
    """
    edits = _as_dict(_as_dict(step.get("extra")).get("edits"))
    added = cast("list[Any]", edits.get("added") or [])
    dropped = cast("list[Any]", edits.get("dropped") or [])
    if not (added or dropped):
        return []
    query = edits.get("iteration")
    out: list[str] = [f"## Hook edits{f' (query {query})' if query is not None else ''}", ""]
    for entry in added:
        entry_dict = _as_dict(entry)
        persistence = "persistent" if entry_dict.get("persistent") else "transient"
        for msg in cast("list[Any]", entry_dict.get("messages") or []):
            msg_dict = _as_dict(msg)
            content = msg_dict.get("content")
            if not isinstance(content, str):
                continue
            # id from provenance (stamped/persistent adds only); index is the add's landing slot. The
            # iteration is the block's query above, so it is not repeated per add.
            add_id = _as_dict(msg_dict.get("provenance")).get("id")
            coords = _coords(add_id, None, msg_dict.get("index"))
            header = f"### Added ({persistence}, {_role_label(msg_dict.get('role'))})"
            if coords:
                header += f" -- {coords}"
            out.extend([header, "", content, ""])
    if dropped:
        out.extend(["### Dropped", ""])
        for record in dropped:
            rec = _as_dict(record)
            drop_id = rec.get("id")
            persistence = "persistent" if rec.get("persistent") else "transient"
            coords = _coords(drop_id, None, rec.get("index"))  # id + slot-when-dropped
            tags = f"{coords}, {persistence}" if coords else persistence
            looked = by_id.get(drop_id) if isinstance(drop_id, str) else None
            if looked is not None:
                snippet = _one_line(looked["content"], _DROPPED_SNIPPET_LEN)
                out.append(f"- {_role_label(looked['role'])} ({tags}): {snippet}")
            else:
                # No record carried this id's content (a hand-authored or pre-provenance trace).
                out.append(f"- ({tags}); content not in trace")
        out.append("")
    return out


def _scope_inputs_lines(extra: dict[str, Any]) -> list[str]:
    """Heavily-truncated one-line reprs of the invoke `inputs` and `scope` values for overview.md."""
    out: list[str] = []
    for section in ("inputs", "scope"):
        namespace = _as_dict(extra.get(section))
        if not namespace:
            continue
        out.extend([f"## {section.capitalize()}", ""])
        for name, value in namespace.items():
            # TrajectoryRecorder serializes each namespace value as {"type", "repr_prefix_10000"}; fall back
            # to a live repr for a hand-authored trace that stored the raw object.
            if isinstance(value, dict):
                value_dict = _as_dict(value)
                typ = value_dict.get("type", "?")
                rep = value_dict.get("repr_prefix_10000", "")
            else:
                typ = type(value).__name__
                rep = repr(value)
            out.append(f"- `{name}` ({typ}): {_one_line(str(rep), _OVERVIEW_VALUE_LEN)}")
        out.append("")
    return out


def _subagents_by_iteration(node: dict[str, Any]) -> dict[Any, list[dict[str, Any]]]:
    """Group `subagent_trajectories` by the iteration that launched them
    (`extra.parent_repl_iteration`), preserving order; the key is `None` when unstamped."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for raw in cast("list[Any]", node.get("subagent_trajectories") or []):
        if not isinstance(raw, dict):
            continue
        child = _as_dict(raw)
        it = _as_dict(child.get("extra")).get("parent_repl_iteration")
        groups.setdefault(it, []).append(child)
    return groups


def _sub_dirname(iter_key: Any, sub_idx: int) -> str:
    """Sibling folder name encoding the launching iteration and the sub-agent's index within it."""
    return f"iter{'NA' if iter_key is None else iter_key}_sub{sub_idx}"


def _write_node(node_dir: Path, node: dict[str, Any]) -> None:
    """Recursively write one ATIF trajectory as `overview.md` + `trace.md` + sibling sub-agent
    directories (`iter<N>_sub<M>/`)."""
    node_dir.mkdir(parents=True, exist_ok=True)

    steps = _steps(node)
    extra = _as_dict(node.get("extra"))
    final_metrics = _as_dict(node.get("final_metrics"))
    pairs = _agent_steps_with_outputs(steps)

    # trace.md -- the full conversation of this agent; sub-agents live in their own dirs.
    (node_dir / "trace.md").write_text(_render_trace_md(node) + "\n", encoding="utf-8")

    # Write each sub-agent into its sibling dir, remembering the dir name per launching iteration.
    subdirs_by_iter: dict[Any, list[tuple[str, dict[str, Any]]]] = {}
    for iter_key, children in _subagents_by_iteration(node).items():
        for sub_idx, child in enumerate(children):
            dirname = _sub_dirname(iter_key, sub_idx)
            _write_node(node_dir / dirname, child)
            subdirs_by_iter.setdefault(iter_key, []).append((dirname, child))

    total_cost = cast("float | None", final_metrics.get("total_cost_usd")) or 0.0
    total_prompt_tokens = cast("int | None", final_metrics.get("total_prompt_tokens")) or 0
    ov: list[str] = [
        f"# {_task_name(node)}",
        "",
        f"**Depth:** {extra.get('depth', 1)}  |  **Iterations:** {len(pairs)}  |  "
        f"**Total cost:** {_format_cost(total_cost)}  |  **Prompt tokens:** {total_prompt_tokens}",
    ]
    # `total_cost` above is ATIF's self-cost; when this node delegated, also show the subtree roll-up so
    # the figure agrees with the run's authoritative cost. Omitted for a leaf (it equals the self-cost).
    # Gated on delegation, not on subtree != self, by decision: the redundant-line case (a delegating node
    # whose children cost ~0) is negligible and the presence check is simpler than comparing rounded costs.
    if node.get("subagent_trajectories"):
        ov.append(f"**Subtree cost (incl. subagents):** {_format_cost(_subtree_cost_usd(node))}")
    ov.extend([f"**Result:** {_result_str(extra)}", ""])
    ov.extend(_scope_inputs_lines(extra))

    ov.extend(["## Iterations", ""])
    referenced: set[Any] = set()
    for pos, (agent_step, output) in enumerate(pairs):
        # 0-based iterations (fallback to positional order, which already matches when unstamped).
        iter_num = _as_dict(agent_step.get("extra")).get("iteration", pos)
        repl_code = cast("str", agent_step.get("message") or "")
        ov.extend([f"### Iteration {iter_num}", ""])
        metrics = _as_dict(agent_step.get("metrics"))
        if metrics:
            cost = _format_cost(cast("float | None", metrics.get("cost_usd")))
            ov.extend(
                [
                    f"*cost {cost} | tokens {metrics.get('prompt_tokens', 0)}"
                    f"->{metrics.get('completion_tokens', 0)}*",
                    "",
                ]
            )
        ov.extend(["REPL code:", "```", _first_lines(repl_code, _SNIPPET_LINES), "```"])
        if output:
            ov.extend(["REPL output:", "```", _first_lines(output, _SNIPPET_LINES), "```"])
        subs = subdirs_by_iter.get(iter_num, [])
        if subs:
            referenced.add(iter_num)
            ov.append("Sub-agents:")
            ov.extend(f"- [`{d}/`]({d}/overview.md) -- {_task_name(c)}" for d, c in subs)
        ov.append("")

    # Sub-agents whose launching iteration matched no agent step (e.g. unstamped) still get listed,
    # so every directory this function wrote is reachable from the overview.
    leftover = [(k, v) for k, v in subdirs_by_iter.items() if k not in referenced]
    if leftover:
        ov.extend(["## Sub-agents (unmatched iteration)", ""])
        for _key, subs in leftover:
            ov.extend(f"- [`{d}/`]({d}/overview.md) -- {_task_name(c)}" for d, c in subs)
        ov.append("")

    (node_dir / "overview.md").write_text("\n".join(ov) + "\n", encoding="utf-8")


def trace_to_directory(trace_path: str | Path, output_dir: str | Path, *, all_nodes: bool = False) -> None:
    """Convert an ATIF trace JSON file to a browsable directory (see the module docstring).

    `all_nodes` writes every top-level invoke trajectory (into zero-padded `task<i>/` subdirs, in
    queue order); the default writes only the first. Raises `ValueError` if the file is not an ATIF
    object or non-empty list.
    """
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)
    data: Any = json.loads(trace_path.read_text(encoding="utf-8"))

    # TrajectoryRecorder writes a single root trajectory as an object, or a list when the traced scope had
    # multiple top-level invokes. Normalize both to a list of trajectory dicts.
    if isinstance(data, dict):
        nodes: list[dict[str, Any]] = [_as_dict(data)]
    elif isinstance(data, list):
        nodes = [_as_dict(n) for n in cast("list[Any]", data) if isinstance(n, dict)]
    else:
        raise ValueError(f"expected an ATIF object or non-empty list in {trace_path}")
    if not nodes:
        raise ValueError(f"expected a non-empty ATIF trace in {trace_path}")

    nodes = nodes if all_nodes else nodes[:1]
    if len(nodes) == 1:
        _write_node(output_dir, nodes[0])
    else:
        # Named by index alone, zero-padded so the dirs list in queue order rather than
        # `0, 1, 10, 11, 2`. The task text is deliberately NOT in the name: it is a whole task
        # statement, so the name carried personal details from the task (names, emails, phone
        # numbers) and truncated mid-word at an arbitrary width, which made the dirs unreadable,
        # unsearchable, and awkward to share. The text is still in the node's own `overview.md`,
        # which is where a reader looks for it.
        #
        # An id would be better than an ordinal, but no env exposes one to this layer: a trajectory
        # node carries only `extra.inputs` (`instructions`, `task`), and the task id lives in the
        # env's `task_results.jsonl`, keyed by the same index. Joining on the index is what a reader
        # (or `analysis.py`) does, so the ordinal is the join key rather than a placeholder for one.
        width = max(2, len(str(len(nodes) - 1)))
        for i, node in enumerate(nodes):
            _write_node(output_dir / f"task{i:0{width}d}", node)
    print(f"Wrote {len(nodes)} invoke node(s) to {output_dir}/")


def main(argv: list[str] | None = None) -> int:
    """CLI: `python -m jaz_evals.trace_to_directory trace.atif.json output_dir/ [--all]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json", type=Path, help="Path to an ATIF trace JSON file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--all", action="store_true", help="Write all top-level invoke nodes (default: first only)"
    )
    args = parser.parse_args(argv)
    if not args.trace_json.is_file():
        print(f"no such file: {args.trace_json}")
        return 1
    try:
        trace_to_directory(args.trace_json, args.output_dir, all_nodes=args.all)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
