# pyright: basic
# Walks untyped JSONL artifacts: every record is a `dict[str, Any]` whose shape varies by message type,
# so strict mode reports each `.get` on a narrowed value as unknown. Checked at basic, like
# `letta_log_viewer.py`; it is jaz- and SDK-free, so it needs no import suppression.
"""Post-run analysis for a Letta attempt: how it used memory, search, and batching.

The generic outcome breakdowns (`analysis.outcome_report`) already transfer to any method, since the env
streams `task_results.jsonl` however the agent ran. What has no JAZ analogue is *how a Letta agent
worked*: whether it wrote to core memory, whether `conversation_search` found anything, and whether the
model batched tool calls. `letta_stats` reads that off the artifacts a run leaves behind.

Read `letta_stats(attempt_dir)` for the numbers and `format_letta_stats` for the text.
"""

# Why these four metrics and not others: each one caused a wrong conclusion during the harness's first
# pilots, so each is here to make that failure visible next time rather than re-derivable only by hand.
#
# - memory writes: the sole discriminator between a good and a bad attempt on far-recall tasks (one
#   attempt captured 42 named protocols, another 34, and the missing ones mapped exactly onto its
#   failed exams). Core memory is the recall path with no retrieval risk, so how much reaches it matters.
# - search hits: a search that returns "No results found" is indistinguishable from one never issued if
#   you only count calls. Without Turbopuffer every search missed, and the run still looked busy.
# - search `limit`: Letta's own docstring advertises `limit` and its only example doubles the default of
#   5. An attempt that escalated to 20/50 retrieved 4.8x the text, triggered 3.4x the compaction, and
#   cost 84% more for a *lower* score. Cheap to watch, invisible otherwise.
# Coverage against the JAZ metric set (`analysis.py`), so a Letta run is diagnosable in the same terms:
#
#   JAZ metric                        Letta equivalent here
#   --------------------------------  ---------------------------------------------------------------
#   exception rate / by type / top    `errors`, `errors_by_tool`, `top_error_messages` (tool returns
#                                     the server marked failed), `error_rate` per round
#
# Same SHAPE, different UNIT -- do not compare the error counts across frameworks directly. JAZ finds an
# exception by text-scanning REPL output for `Traceback` markers, so one turn can contribute several and
# an exception the agent catches and swallows is invisible. Letta reads the server's structured
# `status == "error"` on a tool return, so it sees at most one per tool call but never misses one. The
# denominators differ too: JAZ's rate is per *turn* (a code block that may make many calls), Letta's is
# per round and per tool call. Use them to compare a Letta attempt against another Letta attempt, and
# read the JAZ column as "the same kind of signal", not as the same number.
#   tool_calls_by_name                `tool_calls_by_name`
#   tool_calls_per_turn_dist          `calls_per_round`
#   output_char_dist                  `return_chars`
#   history_search_rate               `search_rate` = `searches`/`rounds` (JAZ's is per parseable
#                                     REPL input; + hits/misses, no JAZ analogue)
#   history_before_tool /             `search_batched_with_other_tool` -- a batch containing a search
#     search_with_tool                and something else, i.e. acting before results exist
#
# Genuinely N/A, and why -- omitted rather than faked: `try_except`, `assign_to_tool`, `lines_per_code`,
# `parseable_inputs`, `imports_by_module` and syntax errors all describe *code the agent writes*, and a
# function-calling agent writes none. `loop_contains_finish`, `get_task_with_other_tool` and
# `finish_then_get_task` and `get_task_then_finish` (the drain) all describe misuse of a fetch tool the
# agent does not hold under task delivery -- it cannot fetch-then-finish when it cannot fetch.
#
# `tasks_completed_without_tool_use` is NOT the drain analogue and must not be read as one: quiz and INFO
# tasks legitimately need no campus tool, so it runs ~68% here against JAZ's 0.1% drain rate. It is kept
# as a Letta-native signal -- a jump in it means the agent stopped acting on tasks that need action.
# Delegation adherence and `next_task_openers` describe sub-agents and a pull queue Letta has neither of.
#
# - batching: only detectable from the container's `PARALLEL_BATCH` lines, which die with the container.
#   The harness persists them to `letta_batches.jsonl` for exactly this reason; counting consecutive
#   `tool_call_message`s instead reports zero however much batching happened (Letta interleaves a batch
#   as call, return, call, return), which is a trap that has already produced confident wrong answers.

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz_evals.harnesses.letta_log_viewer import flatten_content

MESSAGES_FILE = "letta_messages.jsonl"
BATCHES_FILE = "letta_batches.jsonl"
ROUNDS_FILE = "letta_rounds.jsonl"


@dataclass
class LettaStats:
    """How one Letta attempt used memory, search and batching. Counts are 0 when the artifact is absent."""

    messages: int = 0
    rounds: int = 0
    nudge_rounds: int = 0
    memory_writes: int = 0
    memory_chars: int = 0
    memory_errors: int = 0
    searches: int = 0
    search_hits: int = 0
    search_misses: int = 0
    search_limits: dict[str, int] = field(default_factory=dict)
    search_result_chars: int = 0
    batches: int = 0
    batched_calls: int = 0
    batch_sizes: dict[str, int] = field(default_factory=dict)
    tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    # JAZ counts REPL exceptions per turn; the Letta analogue is a tool return the server marked failed.
    errors: int = 0
    errors_by_tool: dict[str, int] = field(default_factory=dict)
    # {tool: {"calling_turns": n, "erroring_turns": m}} -- JAZ's `tool_error_rate` shape, so a tool that
    # errors 10 times out of 12 calls is distinguishable from one that errors 10 times out of 300.
    tool_error_rate: dict[str, dict[str, int]] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)
    top_error_messages: list[tuple[str, int]] = field(default_factory=list)
    calls_per_round: list[int] = field(default_factory=list)
    return_chars: list[int] = field(default_factory=list)
    # Hygiene analogues of JAZ's REPL checks (see the mapping note above).
    search_batched_with_other_tool: int = 0
    tasks_completed_without_tool_use: int = 0

    @property
    def search_hit_rate(self) -> float | None:
        return self.search_hits / self.searches if self.searches else None

    @property
    def return_char_dist(self) -> dict[str, float]:
        """JAZ's `output_char_dist` analogue: how big a tool return typically is, and its tail."""
        from jaz_evals.analysis import dist

        return dist(self.return_chars)

    @property
    def calls_per_round_dist(self) -> dict[str, float]:
        """JAZ's `tool_calls_per_turn_dist` analogue: how many tools a round drives."""
        from jaz_evals.analysis import dist

        return dist(self.calls_per_round)

    @property
    def error_rate(self) -> float | None:
        """Failed tool returns per round -- the analogue of JAZ's exceptions-per-turn."""
        return self.errors / self.rounds if self.rounds else None

    @property
    def search_rate(self) -> float | None:
        """Searches per round, the analogue of JAZ's `history_search_rate` over REPL inputs."""
        # Rendered by `format_letta_stats`, which is the point: the JAZ side prints its counterpart as
        # a percentage (`analysis.py:653`), and this module exists so a Letta run is diagnosable in the
        # same terms. It briefly had no reader and was deleted as dead; the deletion was the wrong
        # direction -- nothing read it because the report had forgotten it, not because it was unwanted.
        #
        # The DENOMINATORS DIFFER and the two rates must not be compared directly: JAZ's is per
        # parseable REPL input (`history_searches / parseable`), this one is per round.
        return self.searches / self.rounds if self.rounds else None

    @property
    def default_limit_share(self) -> float | None:
        """Share of searches that left `limit` unset. Escalating it is the cost risk, so watch the inverse."""
        total = sum(self.search_limits.values())
        return self.search_limits.get("default", 0) / total if total else None


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated final line is what a killed run leaves; skip it rather than fail


def _tool_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def letta_stats(attempt_dir: Path) -> LettaStats | None:
    """Letta-specific stats for one attempt; None if it left no Letta artifacts."""
    messages = list(_read_jsonl(attempt_dir / MESSAGES_FILE))
    batches = list(_read_jsonl(attempt_dir / BATCHES_FILE))
    rounds = list(_read_jsonl(attempt_dir / ROUNDS_FILE))
    if not messages and not batches and not rounds:
        return None

    stats = LettaStats(messages=len(messages), rounds=len(rounds))
    stats.nudge_rounds = sum(1 for r in rounds if not r.get("delivered_task"))

    # Per-round bookkeeping for the hygiene analogues: a user message opens a round, and what the agent
    # does before its `complete_task` is what the JAZ drain check looks at.
    calls_this_round = 0
    worked_this_round = False
    error_messages: dict[str, int] = {}
    in_round = False

    for i, msg in enumerate(messages):
        kind = msg.get("message_type")
        if kind == "user_message":
            # Zero-call rounds are recorded as 0, not dropped. Dropping them made `n` disagree with
            # `rounds` and biased the mean up -- and under delivery the dropped rounds are precisely the
            # no-progress nudge rounds, i.e. the ones worth seeing. JAZ's `tool_calls_per_turn_dist`
            # analogue counts its idle turns, so dropping them here also broke the comparison.
            # `in_round` suppresses only the spurious entry before the first task is delivered.
            if in_round:
                stats.calls_per_round.append(calls_this_round)
            in_round = True
            calls_this_round = 0
            worked_this_round = False
        if kind == "tool_call_message":
            call = msg.get("tool_call") or {}
            name = call.get("name")
            stats.tool_calls += 1
            calls_this_round += 1
            stats.tool_calls_by_name[str(name)] = stats.tool_calls_by_name.get(str(name), 0) + 1
            if name == "complete_task" and not worked_this_round:
                # Recorded a task without using a campus tool. NOT JAZ's drain check (that needs a fetch
                # tool the agent does not have here): quiz and INFO tasks legitimately need no tool, so
                # this is a level to compare across attempts, never a violation count.
                stats.tasks_completed_without_tool_use += 1
            elif name not in ("complete_task", "conversation_search", "memory_insert", "memory_replace"):
                worked_this_round = True
            if name == "memory_insert":
                stats.memory_writes += 1
                args = _tool_args(call)
                stats.memory_chars += len(str(args.get("new_string") or args.get("new_str") or ""))
                # Attribute the failure to *this* call's own return. Scanning every tool return for the
                # error text instead over-counts badly, because `conversation_search` results quote
                # earlier messages and so echo past errors back into the transcript.
                for nxt in messages[i + 1 : i + 3]:
                    if nxt.get("message_type") == "tool_return_message":
                        body = nxt.get("tool_return")
                        body = body if isinstance(body, str) else json.dumps(body, default=str)
                        if "does not exist (available sections" in body:
                            # `memory_insert` cannot create blocks, so an invented label silently
                            # loses the write.
                            stats.memory_errors += 1
                        break
            elif name == "conversation_search":
                stats.searches += 1
                limit = _tool_args(call).get("limit")
                key = "default" if limit is None else str(limit)
                stats.search_limits[key] = stats.search_limits.get(key, 0) + 1
                # The result is the next tool_return; "No results found" is a miss even though the call
                # succeeded, which is the distinction that matters and that a call count hides.
                for nxt in messages[i + 1 : i + 3]:
                    if nxt.get("message_type") == "tool_return_message":
                        body = nxt.get("tool_return")
                        body = body if isinstance(body, str) else json.dumps(body, default=str)
                        stats.search_result_chars += len(body)
                        if "No results found" in body:
                            stats.search_misses += 1
                        else:
                            stats.search_hits += 1
                        break
    if in_round:
        stats.calls_per_round.append(calls_this_round)

    # Failed tool returns, attributed to the call they answer -- the analogue of JAZ's exception tallies.
    for i, msg in enumerate(messages):
        if msg.get("message_type") != "tool_return_message":
            continue
        body = msg.get("tool_return")
        body = body if isinstance(body, str) else json.dumps(body, default=str)
        stats.return_chars.append(len(body))
        if msg.get("status") != "error":
            continue
        stats.errors += 1
        caller = "?"
        for prev in reversed(messages[max(0, i - 3) : i]):
            if prev.get("message_type") == "tool_call_message":
                caller = str((prev.get("tool_call") or {}).get("name", "?"))
                break
        stats.errors_by_tool[caller] = stats.errors_by_tool.get(caller, 0) + 1
        # The exception class the env raised, mirroring JAZ's `exceptions_by_type`: the bridge surfaces
        # tool errors as "Error executing function <tool>: <ExcType>: <message>".
        m = re.search(r":\s*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b", body)
        if m:
            stats.errors_by_type[m.group(1)] = stats.errors_by_type.get(m.group(1), 0) + 1
        key = " ".join(body.split())[:120]
        error_messages[key] = error_messages.get(key, 0) + 1
    stats.top_error_messages = sorted(error_messages.items(), key=lambda kv: -kv[1])[:5]
    for tool, calls in stats.tool_calls_by_name.items():
        errs = stats.errors_by_tool.get(tool, 0)
        if errs:
            stats.tool_error_rate[tool] = {"calling_turns": calls, "erroring_turns": errs}

    for record in batches:
        n = int(record.get("n") or 0)
        if n > 1:
            stats.batches += 1
            stats.batched_calls += n
            stats.batch_sizes[str(n)] = stats.batch_sizes.get(str(n), 0) + 1
        tools = [str(t) for t in (record.get("tools") or [])]
        if "conversation_search" in tools and any(t != "conversation_search" for t in tools):
            # JAZ's `history_before_tool` / `search_with_tool`: acting in the same breath as searching, so
            # the action was chosen before its results existed. `letta.md` forbids exactly this.
            stats.search_batched_with_other_tool += 1
    return stats


def format_letta_stats(stats: LettaStats) -> str:
    """Render `letta_stats` as indented text for the analyze CLI."""
    lines = [
        f"Letta behaviour: {stats.messages} messages, {stats.rounds} rounds "
        f"({stats.nudge_rounds} no-progress), {stats.tool_calls} tool calls"
    ]
    lines.append(
        f"  core memory: {stats.memory_writes} writes, {stats.memory_chars:,} chars"
        + (f", {stats.memory_errors} rejected (no such block)" if stats.memory_errors else "")
    )
    if stats.searches:
        # No `hit is not None` guard: `search_hit_rate` is None only when `searches == 0`, which this
        # branch has already excluded. The ternary was dead, and the `if line` filter on the join below
        # existed only to absorb the empty string it could never produce.
        hit = stats.search_hit_rate
        assert hit is not None  # narrowing for the type checker; unreachable when searches > 0
        # `search_rate` alongside the hit rate: the JAZ report prints its counterpart as a rate, so
        # without this a reader comparing the two does the division on one side only.
        rate = stats.search_rate
        assert rate is not None  # same narrowing: None only when rounds == 0, and searches > 0 here
        lines.append(
            f"  conversation_search: {stats.searches} calls, {stats.search_hits} hit / "
            f"{stats.search_misses} miss ({hit:.0%} hit), {rate:.2f}/round"
        )
        limits = ", ".join(f"{k}={v}" for k, v in sorted(stats.search_limits.items()))
        share = stats.default_limit_share
        lines.append(
            f"    limit used: {limits}"
            + (f"  ({share:.0%} left at the default of 5)" if share is not None else "")
        )
        lines.append(f"    retrieved {stats.search_result_chars / 1e6:.2f}M chars total")
    else:
        lines.append("  conversation_search: never called")
    if stats.errors:
        by_tool = ", ".join(
            f"{k}={v}" for k, v in sorted(stats.errors_by_tool.items(), key=lambda kv: -kv[1])
        )
        rate = stats.error_rate
        lines.append(
            f"  tool errors: {stats.errors}"
            + (f" ({rate:.2f}/round)" if rate is not None else "")
            + f"  [{by_tool}]"
        )
        if stats.errors_by_type:
            by_type = ", ".join(
                f"{k}={v}" for k, v in sorted(stats.errors_by_type.items(), key=lambda kv: -kv[1])
            )
            lines.append(f"    by type: {by_type}")
        rates = ", ".join(
            f"{t} {v['erroring_turns']}/{v['calling_turns']}"
            for t, v in sorted(stats.tool_error_rate.items())
        )
        lines.append(f"    error rate per tool (errored/called): {rates}")
        for msg, n in stats.top_error_messages[:3]:
            lines.append(f"    x{n}: {msg}")
    else:
        lines.append("  tool errors: none")
    if stats.calls_per_round:
        d = stats.calls_per_round_dist
        lines.append(
            f"  tool calls per round: n={d['n']:.0f} mean={d['mean']:.1f} p50={d['p50']:.0f} "
            f"p90={d['p90']:.0f} max={d['max']:.0f}"
        )
    if stats.return_chars:
        d = stats.return_char_dist
        lines.append(
            f"  tool return size (chars): n={d['n']:.0f} mean={d['mean']:.0f} p50={d['p50']:.0f} "
            f"p90={d['p90']:.0f} max={d['max']:.0f}"
        )
    if stats.tasks_completed_without_tool_use:
        lines.append(
            f"  tasks recorded with no tool use: {stats.tasks_completed_without_tool_use}"
            " (quiz/INFO tasks legitimately need none -- watch for a jump, not the level)"
        )
    if stats.search_batched_with_other_tool:
        lines.append(
            f"  searches batched with another tool: {stats.search_batched_with_other_tool}"
            " (acted before results existed)"
        )
    if stats.batches:
        sizes = ", ".join(
            f"n={k}: {v}" for k, v in sorted(stats.batch_sizes.items(), key=lambda kv: int(kv[0]))
        )
        lines.append(
            f"  parallel tool calls: {stats.batches} batches covering {stats.batched_calls} calls  [{sizes}]"
        )
    else:
        lines.append("  parallel tool calls: none recorded (no letta_batches.jsonl, or none occurred)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Letta equivalents of the JAZ per-task metrics that feed `analysis.outcome_report`
# ---------------------------------------------------------------------------


def _episodes(attempt_dir: Path) -> list[list[dict[str, Any]]]:
    """Split the message trace into one list of messages per delivered task, in task order.

    Returns `[]` when the trace or the rounds log is missing.
    """
    # Task boundaries come from `letta_rounds.jsonl` rather than being guessed from the trace: only the
    # harness knows which user messages were task deliveries and which were no-progress nudges, and the
    # k-th delivery is task_idx k, which is what `task_results.jsonl` keys on.
    messages = list(_read_jsonl(attempt_dir / MESSAGES_FILE))
    delivered = [
        r.get("message") or "" for r in _read_jsonl(attempt_dir / ROUNDS_FILE) if r.get("delivered_task")
    ]
    if not messages or not delivered:
        return []
    starts: list[int] = []
    di = 0
    for i, m in enumerate(messages):
        if m.get("message_type") != "user_message" or di >= len(delivered):
            continue
        # Flatten with the viewer's own helper rather than `json.dumps`: the SDK may deliver `content`
        # as a list of typed parts, and dumping that yields `[{"type": "text", "text": "Task 1/253 ...`
        # -- whose first 60 chars never match the delivered text, so EVERY episode boundary is missed,
        # `starts` comes back empty, and every per-task metric silently reads zero. A zero that means
        # "not measured" is the precise failure this module exists to avoid.
        content = flatten_content(m.get("content"))
        # Prefix-match: the delivered text is sent verbatim, but a trace message may carry extra framing.
        if content[:60] == delivered[di][:60]:
            starts.append(i)
            di += 1
    return [
        messages[s : starts[k + 1] if k + 1 < len(starts) else len(messages)] for k, s in enumerate(starts)
    ]


def task_activity(attempt_dir: Path) -> dict[int, tuple[int, int]]:
    """Map 0-based task index -> (tool calls made, tool errors hit) while working that task.

    The Letta analogue of `analysis.per_task_activity`, which counts REPL turns and tracebacks.
    """
    # "Turns" maps to *tool calls*, not rounds: a JAZ turn is one REPL block whose output may carry a
    # traceback, so the JAZ error rate is errors per unit of agent action. Under task delivery a Letta
    # round is ~one task by construction, which would make every task cost ~1 "turn" and the error rate
    # meaningless; tool calls are the comparable unit of action, and a failed tool return is the
    # comparable failure.
    out: dict[int, tuple[int, int]] = {}
    for idx, episode in enumerate(_episodes(attempt_dir)):
        calls = sum(1 for m in episode if m.get("message_type") == "tool_call_message")
        errors = sum(
            1
            for m in episode
            if m.get("message_type") == "tool_return_message" and m.get("status") == "error"
        )
        out[idx] = (calls, errors)
    return out


def search_by_task(attempt_dir: Path) -> dict[int, bool]:
    """Map 0-based task index -> whether the agent searched its history while working that task.

    The Letta analogue of `analysis.history_search_by_task`, which detects `prev_history` REPL searches.
    """
    # `conversation_search` is Letta's whole recall mechanism, so without this the search-vs-correct join
    # reported `searched: 0/0` on a run that searched hundreds of times -- indistinguishable from an agent
    # that never tried to recall anything.
    out: dict[int, bool] = {}
    for idx, episode in enumerate(_episodes(attempt_dir)):
        out[idx] = any(
            m.get("message_type") == "tool_call_message"
            and (m.get("tool_call") or {}).get("name") == "conversation_search"
            for m in episode
        )
    return out


def outcome_for_attempt(attempt_dir: Path) -> Any:
    """`analysis.outcome_report` for a Letta attempt, or None if this is not one.

    Supplies the Letta per-task search flags and activity, so the search-vs-correct join and the
    effort/error metrics are populated instead of silently reading zero.
    """
    # A JAZ-shaped zero is worse than an absent metric: `error_rate_by_outcome: {passed: 0.0}` reads as
    # flawless execution when it actually meant "this metric counts REPL tracebacks and Letta has none".
    from jaz_evals.analysis import outcome_report, read_task_results

    records = read_task_results(attempt_dir)
    if not records or not (attempt_dir / MESSAGES_FILE).exists():
        return None
    return outcome_report(records, search_by_task(attempt_dir), task_activity(attempt_dir))
