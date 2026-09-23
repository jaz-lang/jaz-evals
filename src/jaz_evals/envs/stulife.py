# pyright: basic, reportMissingImports=false
# `stulife` (the ELL-StuLife campus-life benchmark) is vendored as the `third_party/ELL-StuLife`
# submodule rather than installed from an index, so it is absent from any checkout that has not synced
# that group -- and strict mode would then report every symbol as unknown. This module is checked at
# basic, exactly like `harnesses/jaz_harness.py`: the seams that cross into stulife are narrow and
# stulife is imported lazily, so the parts that do not need it work without it.
"""StuLifeEnv -- the ELL-StuLife campus-life benchmark, wrapped in this package's `Env` interface.

ELL-StuLife simulates a full academic year at a virtual university: the agent works a long sequence of
tasks (email, navigation, reservations, course selection, quizzes) against persistent campus subsystems
whose state carries across tasks. It is a *long-horizon* environment: what it measures is whether
information from early tasks is still usable much later, with no external memory store.

The environment logic here -- the per-task state machine (`_SequentialCampus`), the scoring adapters,
the campus-tool wiring, and the episode aggregation -- is copied from the existing suite's
`evals/slife/stulife_env.py` (its `SequentialCampusEnv`, `_evaluate_task`, `_check_course_selection_delta`,
`_build_campus_library`, `load_tasks`, and the tail of `run_episode`), so a run here presents the same
world, the same tools, and the same scoring as a run there. The world simulation and every `_check_*`
scorer live in the `stulife` package itself; this module is a thin adapter over `CampusTask` /
`CampusEnvironment`, just as `stulife_env.py` was.
"""

# What differs from the old harness-function contract (`load_tasks`/`run_episode`/`jaz.scope`/
# `jaz.Library`/hooks), and why -- rationale kept out of the docstring, which is public API:
#
# - One attempt is one episode over the whole task sequence. `run_episode` drove that with a single
#   `jaz.invoke`; here it is `setup()` building the sequence once and the harness running the agent over
#   it once. The old module kept a *module-global* `_campus_task` because its harness called `run_sample`
#   once per task across many calls in one process, and the lifelong world had to persist between them.
#   Here a full episode is one `setup()`, so each attempt builds its own fresh `CampusTask` -- which is
#   also what "every attempt faces the same setup" requires, and removes the process-global entirely.
#
# - Tools are the env's public surface, reached through the harness's binding, not two `jaz.Library`
#   namespaces (`campus.`/`task_queue.`). Every method binds them as bare names (`send_email`,
#   `get_next_task`, ...), which is how `get_instructions` names them.
#   They are surfaced dynamically (see `tools`/`__getattr__`) so the agent sees the actual `raw_*`
#   docstrings and signatures, exactly as it did when JAZ injected them from the Library.
#
# Two behaviours of the old episode runner are deliberately NOT carried over, because they are the
# calling method's concern rather than the environment's, and the JAZ-harness path already omits
# them: the search/act-separation REPL-input validator
# (`_stulife_repl_input_validator`, a bespoke `ValidateREPLCode` guard) and the memory/delegate/
# sliding-window hooks. They belong behind a method config, not in the environment; adding the validator
# needs a harness seam that does not exist yet. The world and the scoring are unaffected.

from __future__ import annotations

import functools
import inspect
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jaz_evals.analysis import (
    analyze_attempt,
    answer_spread_for_attempt,
    count_return_rejections,
    delegation_adherence,
    delegation_shape_for_attempt,
    history_upkeep_for_attempt,
    next_task_openers,
    outcome_for_attempt,
    transcript_stats_for_attempt,
)
from jaz_evals.env import Env, Grade, ToolSpec

# The ELL-StuLife benchmark data lives inside this repo, vendored as the `third_party/ELL-StuLife` git
# submodule (pinned in `.gitmodules`); `[tool.uv.sources]` installs the `stulife` package from the same
# submodule. `task_data/` holds `tasks.json` and the `background/` world data `CampusTask` loads. A
# config may point `data_dir` elsewhere. `parents[3]` is the repo root (this file is at
# src/jaz_evals/envs/stulife.py), so the submodule's task_data is a child of it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "third_party" / "ELL-StuLife" / "task_data"

# Filename, in the attempt's artifacts dir, of the append-only per-task results stream (one JSON object
# per line, written as each task is scored). The per-task breakdown lives here rather than in the final
# `results.json` (which keeps only the aggregate), so it exists even for a run killed before grading.
_TASK_RESULTS_FILE = "task_results.jsonl"

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

# The narrative half of the agent-facing instructions, adapted from the existing suite's
# `prompts/stulife_sequential_hint_3.jinja2` (the template its shipped configs render). The old
# template's two namespaces (`task_queue.` / `campus.`) collapse to bare names here, since every method
# binds each tool as its own bare name. These instructions deliberately do NOT enumerate the tools: every
# harness already surfaces each bound callable's signature and docstring itself (JAZ renders each into its
# prompt, smolagents lists them as Tools, Letta compiles them to tool schemas), so a catalog here would
# only duplicate that -- more weakly, without the docstrings. An env is not required to name its own tools;
# that was an old invariant, dropped once every harness was confirmed to document them.
#
# The loop is phrased in method-agnostic "turns"/"steps", not "REPL iterations": what a turn *is* (one
# jaz REPL code, whether to search history in it, how to batch tool calls in it) is the calling
# method's concern and lives in the domain-method prompt (`prompts/long_horizon/jaz.md`). The env
# states the task-loop shape (one task per turn, step by step); the prompt
# maps a turn onto the jaz REPL and adds the REPL-code mechanics.
# The one sentence in `_SequentialCampus`'s agent-facing text that names the fetch tool. Hoisted to a
# constant because the push interface has to rewrite it in two places -- the returned `next_step` and the
# raised double-submit error -- and a copy that drifted from the original would silently stop matching.
_PULL_NEXT_STEP = "Your next step is to call get_next_task() and read its output."
_PUSH_NEXT_STEP = "Stop and wait: the next task will be delivered to you as a message."


# The task-interface sections of the two instruction variants -- the ONLY text that differs between
# them. Everything else lives once, in `_INSTRUCTIONS_SKELETON` below.
_WORKFLOW_PULL = """\

1. Call get_next_task() to print the current task so that you can read it.
2. Use the available tools to complete the task, step by step, over as many turns as it takes.
3. Call complete_task() to record your result and advance. For a quiz_question task, \
pass your answer letter, e.g. complete_task(answer="B"). For every other task, \
including INFO tasks, call complete_task() with no arguments.
4. If complete_task() reports a positive tasks_remaining, go back to Step 1 for the next task.
   Otherwise, you are done.
"""

_WORKFLOW_PUSH = """\

1. Read the current task: it is given to you in the most recent message.
2. Use the available tools to complete the task, step by step, over as many turns as it takes.
3. Call complete_task() to record your result. For a quiz_question task, \
pass your answer letter, e.g. complete_task(answer="B"). For every other task, \
including INFO tasks, call complete_task() with no arguments.
4. After complete_task(), stop and wait: the next task arrives as the next message. When no tasks
   remain, you are done.
"""

# The shared body of both variants, with the workflow section left as a placeholder.
#
# Assembled rather than written out twice. The two variants were previously whole separate templates
# that happened to be identical for 86% of their length (3231 of ~3760 chars), which meant every edit
# to the campus overview, the recall guidance, or the fictional-world note had to be made in both
# places and silently applied to only one interface if it wasn't. The workflow section genuinely must
# differ -- under `task_delivery` the agent has no `get_next_task()` to call -- so that is the one part
# kept as a pair. `str.replace` on a sentinel, not `.format`, so the `{num_tasks}` field survives for
# `get_instructions` to fill in later.
# The single-task counterpart of the two workflow sections above. It is NOT a queue loop -- there is one
# task and one session -- so all it has to carry is how the session ENDS.
#
# It has to carry that, though, and this is the one thing the campus overview alone does not supply. Both
# other framings state the answer protocol on `complete_task`'s own tool card ("For a `quiz_question`
# task, pass your answer letter"), and a per-task session does not hold that tool: the driver does. Ship
# the overview alone and nothing ever tells the session that a quiz wants the bare letter `B`, so it
# returns the prose a chat model returns by default, `_score_quiz` scores 0.0, and the arm reports a
# memory result that is really a protocol result. Same reason AppWorldEnv's `_SINGLE_TASK_INSTRUCTIONS`
# open by saying what a valid answer is, and the same sentence its per-task arms rely on.
#
# It is not technique and not guidance: it describes the submission channel, which is the environment's
# to describe (`Env.get_instructions`), and it teaches nothing about how to solve anything.
_WORKFLOW_SINGLE = """\

Use the available tools to complete it, step by step, over as many turns as it takes, then end your \
session by returning your answer. For a quiz_question task, return the answer letter alone, e.g. \
"B". For every other task, including INFO tasks, return None -- the work you did with the tools is \
what is scored, so never return a report of what you did.
"""

_INSTRUCTIONS_SKELETON = """\
You are a student at Lifelong Agent University, working {num_tasks} tasks sequentially across a full \
academic year.

## Campus overview

The campus has 155 buildings. Each building has floors with named amenities (e.g. "Periodicals \
Reading Room", "Weight Room", "Parking").

- Navigation: get_current_location() then find_building_id(name) then \
find_optimal_path(source_id, target_id) then walk_to(path).
- Finding a library: find_library(subject) finds the library building for an academic subject \
(e.g. "engineering", "music", "psychology").
- Booking: always call query_availability(building_id, date, time_slot, features) first to discover bookable \
amenities and seats. Use the amenity names from the result as the `amenity` argument to \
make_booking(). Some amenities have individual seats with features (window_seat, power_outlet, \
quiet_zone, ...) -- pick one matching the task and pass its `seat_id`.
- People: 101 student clubs and 1000 faculty advisors. Use list_by_category() to browse and \
query_by_identifier() for details. Advisors have an `email` field for send_email(); club \
emails are in the `recruitment_info` field.
- Course selection: browse_courses() shows the catalog and the pass rules; build a draft with \
add_course()/remove_course()/assign_pass(), check it with view_draft(), and finalize with submit_draft().

## Your workflow
%%WORKFLOW%%
## IMPORTANT: Search your memory or history for tasks involving recall

Before issuing your next action, determine whether or not the task involves recall according to the
following guidelines and state your reasoning in your plan:
- quiz_question tasks, including midterm and final exams, typically test recall of information from earlier
  lecture tasks (INFO tasks or earlier quiz_question tasks), such as rules, protocols, and definitions.
  Search your memory or history to recall the lecture content in order to answer the question correctly.
- Some course_selection tasks sometimes require recalling advice given in an earlier task, such as which
  courses to prioritize, and applying it; recall that earlier advice when making your course selection.
- If a walking_simple or multi_system task arrives with only the current time/location information
  and *no* instructions, then it was scheduled by an earlier task; recall that earlier task to know what
  to do at the current time/location.

For tasks described above, state in your plan how you will recall the required information by searching
your memory or history.

## NOTE: Lifelong Agent University is a fictional world -- real world knowledge does not apply!

In this environment, lectures teach knowledge about a fictional world, which often intentionally contradicts
facts about the real world to test your ability to *recall* information delivered by the environment.
Quizzes and exams test knowledge about the *fictional world*, not the real world. As a result, if you
use your existing knowledge about the real world when answering a quiz or exam question, you will get the
question wrong! To answer the question correctly, you MUST recall the actual content of the relevant lecture
delivered to you in a previous task. If you do not recall the lecture, you must search your memory or history
to find it!
"""

_INSTRUCTIONS = _INSTRUCTIONS_SKELETON.replace("%%WORKFLOW%%", _WORKFLOW_PULL)

# The single-task framing: the opening sentence rewritten, the campus overview kept, and nothing else.
#
# DERIVED from the same skeleton rather than written out again, so the campus description -- 155
# buildings, the booking flow, the course-selection flow -- cannot drift between the two framings. A
# duplicated copy is exactly how the whole-queue text and its restatements drifted before
# `single_task_instructions` existed.
#
# WHY EACH DROPPED SECTION IS DROPPED, since removing agent-facing text needs a reason per section:
#   - `## Your workflow` teaches the `get_next_task()`/`complete_task()` loop. A per-task session must
#     NOT drive the queue -- the harness does -- so this would instruct the agent to fight it.
#   - `## IMPORTANT: Search your memory or history ...` tells the agent to recall earlier tasks. A
#     per-task session is fresh and has no earlier tasks, so the instruction is unfollowable.
#   - `## NOTE: ... fictional world ...` says quiz answers must come from recalling the lecture
#     delivered in a previous task. Same problem: there is no previous task to recall from.
#
# The last two are the arm's POINT, not an oversight. This baseline exists to measure what per-task
# framing costs on a lifelong-memory benchmark, and a session that cannot reach earlier tasks is
# expected to lose the recall-dependent ones. Keeping instructions it cannot act on would measure a
# confused agent instead of a memoryless one.
_SINGLE_TASK_OPENING = (
    "You are a student at Lifelong Agent University. You have been given exactly one task to complete."
)
_SINGLE_TASK_INSTRUCTIONS = (
    _SINGLE_TASK_OPENING
    + _WORKFLOW_SINGLE
    + "\n"
    + _INSTRUCTIONS_SKELETON[: _INSTRUCTIONS_SKELETON.index("## Your workflow")].split("\n", 1)[1].strip()
    + "\n"
)
# The skeleton's opening line is the only place `{num_tasks}` appears above the workflow section, and it
# is the line replaced above -- so this text needs no `.format()` and must contain no placeholder. A
# stray one would reach the agent as a literal `{num_tasks}`.
assert "{num_tasks}" not in _SINGLE_TASK_INSTRUCTIONS
assert "## Campus overview" in _SINGLE_TASK_INSTRUCTIONS
assert "## Your workflow" not in _SINGLE_TASK_INSTRUCTIONS
# The two tools the driver owns must not be named at a session that cannot call either -- the failure
# this catches is a future edit to the skeleton's campus overview, which is spliced in above verbatim.
assert "get_next_task" not in _SINGLE_TASK_INSTRUCTIONS
assert "complete_task" not in _SINGLE_TASK_INSTRUCTIONS

_INSTRUCTIONS_DELIVERED = _INSTRUCTIONS_SKELETON.replace("%%WORKFLOW%%", _WORKFLOW_PUSH)


# ---------------------------------------------------------------------------
# Campus tools (copied from stulife_env._build_campus_library, minus the jaz.Library wrapper)
# ---------------------------------------------------------------------------


def _campus_tool_bindings(
    env: Any,
    *,
    allow_calendar_modifications: bool,
    on_call: Any,
    action_history: list[Any],
) -> list[tuple[str, Any]]:
    """Return `(tool_name, callable)` pairs exposing the campus subsystems as agent tools.

    System -> method mapping mirrors `_build_campus_library` in `evals/slife/stulife_env.py`, which
    itself mirrors `ActionExecutor._build_action_mapping()` in ELL-StuLife. Episode mode exposes the
    whole campus (the old code passed `available_systems=None`), so there is no per-task filtering.

    Names carry no `campus.` prefix: the harness binds each tool as a bare REPL name, so the agent
    reaches them as `send_email(...)` etc.; the campus docstrings' code examples are bare too -- the stale
    `campus.` receiver was stripped at the source (the ELL-StuLife submodule). Each tool is wrapped to (1)
    mark that a campus tool has been used
    since the last `get_next_task` -- `complete_task` rejects a non-trigger (a "trigger" is a
    read-and-remember INFO task, not the graded task its own `task_type` names; see
    `_SequentialCampus.get_next_task`), non-quiz task finished with no tool call, and in driver mode
    scores it 0.0 rather than raising -- and (2) append to `action_history` for the sequence-based
    `_score_multi_system` scorer. The `system_type` mapping copies
    `_extract_system_type_from_action` in ELL-StuLife's `task.py`; the key actions it checks are
    {send_email, make_booking, add_event}.
    """
    calendar_tools: list[tuple[str, Any]] = []
    if allow_calendar_modifications:
        calendar_tools = [("add_event", env.raw_add_event)]

    groups: dict[str, list[tuple[str, Any]]] = {
        "email": [("send_email", env.raw_send_email)],
        "calendar": calendar_tools,
        # `find_library` (subject -> library building) is an ELL-StuLife *fork* addition, not one of the
        # paper's tools. We keep it as a stand-in for the paper's laborious subject->library discovery
        # path: to pick the right library for a study subject an agent would enumerate buildings with
        # `query_buildings_by_property` (which returns no aliases and has no "library" building type) and
        # call `get_building_details` on each to read the department alias (e.g. "Art History Department")
        # that names the subject. `find_library` returns that mapping directly, so those two tools -- and
        # `list_valid_query_properties`, which exists only to drive `query_buildings_by_property` -- have no
        # remaining use here. The paper's other map-inspection tools, `get_building_complex_info` and
        # `find_room_location`, are omitted on independent grounds: nothing in our task set needs
        # building-complex membership or room-level lookup -- a claim scoped to the `task_type_filter`
        # the env is built with (see the constructor; None = every task type), so a wider set means
        # re-checking it.
        "map": [
            ("find_building_id", env.raw_find_building_id),
            ("find_library", env.raw_find_library),
            ("find_optimal_path", env.raw_find_optimal_path),
        ],
        "geography": [
            ("walk_to", env.raw_walk_to),
            ("get_current_location", env.raw_get_current_location),
        ],
        "reservation": [
            ("query_availability", env.raw_query_availability),
            ("make_booking", env.raw_make_booking),
        ],
        "data_system": [
            ("list_by_category", env.raw_list_by_category),
            ("query_by_identifier", env.raw_query_by_identifier),
        ],
        "course_selection": [("browse_courses", env.raw_browse_courses)],
        "draft": [
            ("add_course", env.raw_add_course),
            ("remove_course", env.raw_remove_course),
            ("assign_pass", env.raw_assign_pass),
            ("view_draft", env.raw_view_draft),
        ],
        "registration": [("submit_draft", env.raw_submit_draft)],
    }

    functions: list[tuple[str, Any]] = []
    for tools in groups.values():
        functions.extend(tools)

    wrapped: list[tuple[str, Any]] = []
    for name, fn in functions:

        @functools.wraps(fn)
        def _wrapped(*args: Any, _orig: Any = fn, _name: str = name, **kwargs: Any) -> Any:
            on_call()
            result = _orig(*args, **kwargs)
            if "send_email" in _name:
                sys_type = "email"
            elif "make_booking" in _name or "query_availability" in _name:
                sys_type = "reservation"
            elif "add_event" in _name or "update_event" in _name or "view_schedule" in _name:
                sys_type = "calendar"
            elif "walk_to" in _name or "find_optimal_path" in _name:
                sys_type = "geography"
            elif _name in {"browse_courses", "add_course", "remove_course", "assign_pass", "view_draft"}:
                sys_type = "course_selection"
            else:
                sys_type = "unknown"
            action_history.append(
                {
                    "timestamp": time.time(),
                    "system_type": sys_type,
                    # Format so ELL-StuLife's `\.(\w+)\(` action regex matches.
                    "action_content": f"{sys_type}.{_name}(...)",
                    "success": True,
                    "message": str(result)[:200] if result is not None else "",
                }
            )
            return result

        wrapped.append((name, _wrapped))
    return wrapped


# ---------------------------------------------------------------------------
# Evaluation -- thin adapter over CampusTask._check_* methods (copied from stulife_env.py)
# ---------------------------------------------------------------------------


def _make_dataset_item(task_data: dict[str, Any]) -> Any:
    """Build a `CampusDatasetItem` from a raw task_data dict.

    Some `tasks.json` entries store `require_time`/`require_place` as `false` instead of `null`; coerce
    to `None` to match the `Optional[str]` field type, mirroring ELL-StuLife's own loader.
    """
    from stulife.tasks.instance.campus_life_bench.task import CampusDatasetItem

    if task_data.get("require_time") is False:
        task_data = {**task_data, "require_time": None}
    if task_data.get("require_place") is False:
        task_data = {**task_data, "require_place": None}
    return CampusDatasetItem.model_validate(task_data)


def _get_draft_snapshot(campus_task: Any) -> dict[str, str]:
    """Return {course_code: assigned_pass} for the current draft schedule."""
    draft = campus_task.campus_environment.course_selection_system.get_draft_schedule_for_evaluation()
    return {s.course_code: s.assigned_pass or "" for s in draft.selected_sections}


def _check_course_selection_delta(
    task_data: dict[str, Any],
    campus_task: Any,
    prev_ground_truth: dict[str, str] | None,
    pre_task_snapshot: dict[str, str],
) -> tuple[float, str]:
    """Evaluate course selection by checking the agent made the correct *changes*.

    Computes the expected delta (adds, removes, pass changes) between the previous and current ground
    truth and checks the agent applied exactly those, with partial credit `correct / (expected + extra)`;
    falls back to an absolute check for the first task.
    """
    # DIFF from ELL-StuLife: delta-based instead of absolute. The upstream `_check_course_selection`
    # compares the full draft against the expected schedule, so once a cumulative course-selection task
    # fails, every later one fails too even when the agent's own changes are right -- which the delta
    # check avoids.
    expected_sections = task_data["ground_truth"]["expected_schedule_outcome"]["selected_sections"]
    current_gt = {s["course_code"]: s["assigned_pass"] for s in expected_sections}

    # For the very first course_selection task, absolute comparison with partial credit.
    if prev_ground_truth is None:
        agent_entries = _get_draft_snapshot(campus_task)
        gt_entries = current_gt
        all_courses = set(agent_entries) | set(gt_entries)
        if not all_courses:
            return 1.0, "no courses expected or selected"
        n_correct = sum(
            1
            for c in all_courses
            if c in agent_entries and c in gt_entries and agent_entries[c] == gt_entries[c]
        )
        score = n_correct / len(all_courses)
        if score == 1.0:
            return 1.0, "all expected courses in draft"
        missing = {c: gt_entries[c] for c in gt_entries if c not in agent_entries}
        extra = {c: agent_entries[c] for c in agent_entries if c not in gt_entries}
        wrong_pass = {
            c: (agent_entries[c], gt_entries[c])
            for c in agent_entries
            if c in gt_entries and agent_entries[c] != gt_entries[c]
        }
        errors = []
        if missing:
            errors.append(f"missing: {missing}")
        if extra:
            errors.append(f"extra: {extra}")
        if wrong_pass:
            errors.append(f"wrong pass: {wrong_pass}")
        return score, "; ".join(errors)

    # Expected delta: what should have changed from prev_ground_truth to current_gt.
    expected_adds = {c: p for c, p in current_gt.items() if c not in prev_ground_truth}
    expected_removes = {c for c in prev_ground_truth if c not in current_gt}
    expected_pass_changes = {
        c: current_gt[c]
        for c in current_gt
        if c in prev_ground_truth and prev_ground_truth[c] != current_gt[c]
    }

    # Actual delta: what the agent actually changed.
    post_snapshot = _get_draft_snapshot(campus_task)
    actual_adds = {c: p for c, p in post_snapshot.items() if c not in pre_task_snapshot}
    actual_removes = {c for c in pre_task_snapshot if c not in post_snapshot}
    actual_pass_changes = {
        c: post_snapshot[c]
        for c in post_snapshot
        if c in pre_task_snapshot and pre_task_snapshot[c] != post_snapshot[c]
    }

    errors: list[str] = []

    missing_adds = {c: p for c, p in expected_adds.items() if c not in actual_adds}
    wrong_adds = {c: p for c, p in actual_adds.items() if c not in expected_adds}
    if missing_adds:
        errors.append(f"missing course additions: {missing_adds}")
    if wrong_adds:
        errors.append(f"unexpected course additions: {wrong_adds}")

    missing_removes = expected_removes - actual_removes
    wrong_removes = actual_removes - expected_removes
    if missing_removes:
        errors.append(f"courses should have been removed but weren't: {missing_removes}")
    if wrong_removes:
        errors.append(f"courses removed unexpectedly: {wrong_removes}")

    missing_pass_changes = {
        c: (pre_task_snapshot.get(c, "?"), expected_pass_changes[c])
        for c in expected_pass_changes
        if c not in actual_pass_changes or actual_pass_changes[c] != expected_pass_changes[c]
    }
    wrong_pass_changes = {
        c: (pre_task_snapshot.get(c, "?"), actual_pass_changes[c])
        for c in actual_pass_changes
        if c not in expected_pass_changes
    }
    if missing_pass_changes:
        errors.append(f"missing pass changes: {missing_pass_changes}")
    if wrong_pass_changes:
        errors.append(f"unexpected pass changes: {wrong_pass_changes}")

    # Partial score correct / (expected + extra): penalises missing and spurious operations alike.
    n_expected = len(expected_adds) + len(expected_removes) + len(expected_pass_changes)
    n_extra = len(wrong_adds) + len(wrong_removes) + len(wrong_pass_changes)
    if n_expected == 0 and n_extra == 0:
        return 1.0, "correctly made no changes"

    n_correct = (
        (len(expected_adds) - len(missing_adds))
        + (len(expected_removes) - len(missing_removes))
        + (len(expected_pass_changes) - len(missing_pass_changes))
    )
    score = n_correct / (n_expected + n_extra) if (n_expected + n_extra) > 0 else 0.0
    if errors:
        return score, "; ".join(errors)
    return 1.0, "correct course selection changes applied"


def _evaluate_task(
    task_data: dict[str, Any],
    campus_task: Any,
    *,
    prev_course_gt: dict[str, str] | None = None,
    pre_task_draft_snapshot: dict[str, str] | None = None,
) -> tuple[float, str]:
    """Evaluate a finished task and return `(score, reason)`, score in [0.0, 1.0].

    Non-multi-system tasks are binary; multi_system tasks get partial credit for the fraction of
    ground-truth components satisfied. `quiz_question` is handled by `complete_task` before this is
    called. Copied from `stulife_env._evaluate_task`.
    """
    task_type = task_data["task_type"]
    if task_data.get("is_trigger", False):
        return 1.0, "trigger task (skipped)"

    task_item = _make_dataset_item(task_data)

    try:
        if task_type == "walking_simple":
            score, reason = campus_task._score_walking_simple(task_item)
            if score is None:
                score = 0.0
        elif task_type == "course_selection":
            score, reason = _check_course_selection_delta(
                task_data, campus_task, prev_course_gt, pre_task_draft_snapshot or {}
            )
        elif task_type == "quiz_question":
            raise AssertionError(
                "quiz_question should be handled by complete_task() before _evaluate_task() is called"
            )
        elif task_type in ("multi_system", "single_system"):
            score, reason = campus_task._score_multi_system(task_item)
            if score is None:
                score = 0.0
        else:
            return 0.0, f"unsupported task type: {task_type!r}"
    except Exception as e:
        return 0.0, f"evaluation error: {e}"

    if score == 1.0 and task_item.require_precheck and campus_task.precheck_failed:
        # State matches, but a precheck failure means the condition was already satisfied before the
        # task began -- count as incorrect (false positive).
        return (
            0.0,
            "incorrect: the required actions were already performed before the current task began -- "
            "the previous task requiring these actions required that they be performed now, not before.",
        )
    return score, reason


# ---------------------------------------------------------------------------
# Sequential episode state machine (copied from stulife_env.SequentialCampusEnv)
# ---------------------------------------------------------------------------


class _SequentialCampus:
    """Walks `CampusTask` through its tasks one at a time within a single agent episode.

    `get_next_task()` prepares and reveals the current task; the agent works it with campus tools;
    `complete_task()` scores it against ground truth, records the result, and advances -- until
    `tasks_remaining == 0`. This is `SequentialCampusEnv` from `evals/slife/stulife_env.py`, unchanged
    except that it no longer builds jaz libraries (its two agent methods are surfaced as tools by
    `StuLifeEnv`) and it never writes an incremental results file (grading reads `results` in-process).
    """

    def __init__(
        self,
        all_tasks: list[dict[str, Any]],
        campus_task: Any,
        *,
        initial_course_gt: dict[str, str] | None,
        result_sink: Path | None = None,
        driver_mode: bool = False,
    ) -> None:
        self._all_tasks = all_tasks
        # Recall gap per non-trigger paired/exam task_id (computed once over the whole sequence), stamped
        # onto each streamed result so downstream analysis can bin accuracy by memory distance without
        # re-loading the task file. None for standalone/trigger tasks.
        self._gaps = _recall_gaps(all_tasks)
        self._campus_task = campus_task
        self._env = campus_task.campus_environment
        self._idx = 0
        self._results: list[dict[str, Any]] = []
        # Append-only JSONL: one line per task, written the moment `complete_task` scores it, so a run
        # killed before `grade()` still leaves its results-so-far on disk. `None` disables it (unit
        # tests, or an env not wired with an artifacts dir). The in-memory `_results` is authoritative
        # for grading; this file is the durable/streamable mirror of the same records.
        self._result_sink = result_sink
        self._sim_day: str | None = None
        self._task_read = False  # must call get_next_task() before complete_task()
        self._campus_tools_called = False  # any campus tool called since get_next_task()
        # Course-selection delta tracking. `initial_course_gt` seeds the draft when a filtered subset
        # omits the earlier course_selection tasks that would have built it.
        self._prev_course_gt = initial_course_gt
        self._pre_task_draft_snapshot: dict[str, str] | None = None
        self._task_prepared = False
        # Who calls `complete_task`: the agent (False) or a harness loop driving the queue (True).
        #
        # It changes what a REJECTED submission does, and only that. Three of `complete_task`'s guards
        # are written to be *read by the agent and retried* -- no campus tool called, a quiz with no
        # answer, a quiz with an unparseable answer -- and raising them at an agent is correct: it gets
        # the message and tries again. Raising them at a driver is not. `JazPerTaskHarness` calls
        # `complete_task` OUTSIDE its per-session guard (deliberately: an ungraded task would stall the
        # cursor and spin the loop forever), so a raise there escapes to the harness's outer handler,
        # which ends the RUN -- one session that returned nothing usable would discard the rest of the
        # queue. In driver mode each of the three records a 0.0 for that task instead and the queue
        # goes on, which is also the honest score: a task the session neither acted on nor answered
        # earned nothing. The double-submit guard is NOT relaxed -- the driver submits exactly once per
        # task, so it firing would mean the harness itself is broken, and that must stay loud.
        self._driver_mode = driver_mode

    def note_campus_tool_called(self) -> None:
        """Mark that the agent has used a campus tool since the last `get_next_task()`."""
        self._campus_tools_called = True

    def _prepare_task(self, idx: int) -> None:
        """Apply world-state changes and location reset for the task at `idx`."""
        from stulife.tasks.instance.campus_life_bench.systems.course_selection import DraftScheduleEntry

        if idx >= len(self._all_tasks):
            return
        task_data = self._all_tasks[idx]["task_data"]

        # Clear cross-task accumulation so each task scores from a fresh slate. (DIFF from ELL-StuLife:
        # walk_history/sent-emails/reservations/calendar are cleared per task; upstream only clears some
        # of these on daily_reset, which corrupts path/email/booking scoring across same-day tasks.)
        self._campus_task.action_history.clear()
        self._env.geography_system._state.walk_history.clear()
        self._env.email_system._sent_emails_log.clear()
        self._env.reservation_system._global_reservations.clear()
        for cal in self._env.calendar_system._global_calendars.values():
            cal.clear()

        self._campus_task.precheck_failed = False
        self._campus_task.precheck_failure_details.clear()

        changes = task_data.get("world_state_change") or []
        if changes:
            self._env.apply_world_state_changes(changes)

        require_time = task_data.get("require_time")
        if require_time:
            m = re.match(r"(Week \d+, \w+)", str(require_time))
            if m:
                day = m.group(1)
                if day != self._sim_day:
                    self._env.daily_reset(day)
                    self._sim_day = day

        source = task_data.get("source_building_id")
        if source:
            self._env.set_initial_location(source)

        # DIFF from ELL-StuLife: teleport to require_place instead of a walk-to loop (upstream's walk
        # prepends extra nodes to walk_history and breaks path scoring, which starts at require_place).
        require_place = task_data.get("require_place")
        if require_place:
            if not re.match(r"^B\d{3}$", str(require_place)):
                require_place = self._env.geography_system.map_lookup_system.find_building_id(require_place)[
                    "id"
                ]
            self._env.set_initial_location(require_place)

        if task_data.get("task_type") == "course_selection" and not task_data.get("is_trigger", False):
            initial_draft = task_data.get("initial_draft")
            cs = self._env.course_selection_system
            if initial_draft is not None:
                cs._draft_schedule.selected_sections = [
                    DraftScheduleEntry(course_code=s["course_code"], assigned_pass=s["assigned_pass"])
                    for s in initial_draft
                ]
                self._prev_course_gt = {s["course_code"]: s["assigned_pass"] for s in initial_draft}
            elif self._prev_course_gt is not None:
                expected_sections = task_data["ground_truth"]["expected_schedule_outcome"][
                    "selected_sections"
                ]
                current_gt_codes = {s["course_code"] for s in expected_sections}
                if current_gt_codes.isdisjoint(self._prev_course_gt):
                    # Semester boundary: no overlap with the previous ground truth, so the draft starts
                    # empty and delta eval falls back to the absolute check for this task.
                    self._prev_course_gt = None
                    cs._draft_schedule.selected_sections = []
                else:
                    cs._draft_schedule.selected_sections = [
                        DraftScheduleEntry(course_code=code, assigned_pass=pass_type)
                        for code, pass_type in self._prev_course_gt.items()
                    ]

        # Reservation task context: without it booking_task_id is "unknown" and evaluation finds no
        # reservations. Mirrors ELL-StuLife's CampusTask.
        if hasattr(self._env.reservation_system, "set_task_context"):
            entry = self._all_tasks[idx]
            self._env.reservation_system.set_task_context(
                {
                    "task_id": task_data.get("task_id", entry["task_key"]),
                    "task_type": task_data.get("task_type"),
                    "details": task_data.get("details"),
                    "ground_truth": task_data.get("ground_truth"),
                    "target_date": require_time,
                }
            )

        if task_data.get("task_type") == "course_selection" and not task_data.get("is_trigger", False):
            self._pre_task_draft_snapshot = _get_draft_snapshot(self._campus_task)

        if not task_data.get("is_trigger", False):
            self._campus_task._perform_precheck(_make_dataset_item(task_data))

    def get_next_task(self) -> str:
        """Return the next task's description (number, type, and instruction),
        or a completion notice when the sequence is finished.

        After calling ``get_next_task()``, one must complete the task and call ``complete_task()``
        before being allowed to call ``get_next_task()`` again.
        """
        if self._idx >= len(self._all_tasks):
            return "All tasks complete. End your session."

        if self._task_prepared:
            raise ValueError(
                "You have not called complete_task() for the current task yet. "
                "Complete the current task and call complete_task() before calling get_next_task() again."
            )
        self._prepare_task(self._idx)
        self._task_prepared = True

        entry = self._all_tasks[self._idx]
        task_data = entry["task_data"]
        task_type = task_data.get("task_type", "unknown")
        is_trigger = task_data.get("is_trigger", False)
        total = len(self._all_tasks)
        instruction = task_data.get("instruction", "")

        self._task_read = True
        self._campus_tools_called = False
        # A trigger is a read-and-remember INFO task, not the graded task its `task_type` names. Showing
        # that type (e.g. `quiz_question` on a lecture trigger) contradicts the `[INFO TASK]` note below and
        # makes the agent hunt for a question/answer that isn't there -- a one-off observation from an
        # internal run (not reproducible from anything shipped here): gpt-5.4-nano history-searched a
        # 38k-char lecture for the "missing" quiz, then fabricated an answer to read-only content.
        # Show `info` for any trigger so the header agrees with the note. Grading is unaffected: it reads
        # the task's own `task_type`, never this header string.
        display_type = "INFO" if is_trigger else task_type
        header = f"Task {self._idx + 1}/{total} (type={display_type})"

        context_parts: list[str] = []
        require_time = task_data.get("require_time")
        require_place = task_data.get("require_place")
        if require_time:
            context_parts.append(f"Current time: {require_time}")
        if require_place:
            # DIFF from ELL-StuLife: show require_place as the current location, not a walk-to target
            # (we teleport there in _prepare_task, so telling the agent to walk wastes iterations and
            # can corrupt walk_history).
            if not re.match(r"^B\d{3}$", str(require_place)):
                require_place = self._env.geography_system.map_lookup_system.find_building_id(require_place)[
                    "id"
                ]
            current_id = self._env.geography_system._state.current_location_id
            assert current_id == require_place, (
                f"Expected agent at {require_place} after _prepare_task, but found at {current_id}"
            )
            loc_name = self._env.geography_system._state.current_location_name
            context_parts.append(f"Current location: {loc_name} ({require_place})")
        context = "\n".join(context_parts)

        if is_trigger:
            # The closing instruction names the tool that ends the task, and under a per-task driver the
            # session does not have one -- `complete_task` is the driver's there. Same fix as
            # `_delivered_complete_task` makes for the push interface: tell the agent to do the thing it
            # can actually do, rather than leave it hunting for a tool it was not given.
            note = (
                "This is an INFO task. Read and remember the information above, then end your session."
                if self._driver_mode
                else "This is an INFO task. Read and remember the information above, "
                "then call complete_task()."
            )
            return (
                f"{header}\n{context}\n\n{instruction}\n\n{note}"
                if context
                else f"{header}\n\n{instruction}\n\n{note}"
            )

        # No per-task-type suffix, and no recall hint on the empty-instruction (paired trigger) tasks
        # either: what a task requires is left to its own description and its graded ground truth.
        # Nudging the agent toward the METHOD of answering is the calling method's concern. That
        # applies most sharply to retrieval -- a paired trigger needs something an earlier INFO TASK
        # set up, and the env deliberately does not name the channel to recover it through (the REPL
        # history, a memory store, ...), because naming one would hand a method its own technique.
        return f"{header}\n{context}\n\n{instruction}" if context else f"{header}\n\n{instruction}"

    def _append_result_line(self, record: dict[str, Any]) -> None:
        """Append one task result to the JSONL sink, if configured. Opened and closed per line so a
        killed process loses at most the line it was mid-write on -- the point is durability, and a
        run is one task every few seconds, so the per-line open cost is nothing.

        Durability is against process death (SIGKILL / budget / context exit), the actual threat model:
        the close flushes each line to the OS. It is not fsync'd, so a machine crash or power loss can
        still lose recently-written lines -- not worth the per-line fsync cost for a diagnostic stream."""
        if self._result_sink is None:
            return
        with self._result_sink.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _score_quiz(self, task_data: dict[str, Any], answer: str | None) -> tuple[float, str]:
        """Score one `quiz_question` submission as `(score, reason)`.

        Raises `ValueError` for a missing or unparseable answer, so the agent can read the message and
        resubmit -- except in driver mode, where the same two cases score 0.0 instead.
        """
        # Split out of `complete_task` when driver mode arrived: the two rejections need a value on one
        # path and a raise on the other, and inlining that doubled the branching of an already long
        # method. The order of the checks is unchanged, so an agent sees exactly the messages it did.
        correct = str(task_data.get("ground_truth", "")).strip().upper()
        # A driver submits whatever the session RETURNED, which nothing forces to be a string -- a
        # session that raised returns `None`, and one that ends with a dict or an int returns that. In
        # agent mode this check is skipped so a non-string still fails as it always has, on `.strip()`.
        if self._driver_mode and not isinstance(answer, str):
            return 0.0, f"no answer letter submitted (session returned {type(answer).__name__})"
        if answer is None:
            raise ValueError(
                "quiz_question requires an answer -- pass answer='A'/'B'/'C'/'D' to complete_task()"
            )
        submitted = answer.strip().upper()
        if submitted not in {"A", "B", "C", "D"}:
            # Not salvaged by pulling a letter out of a longer string, on purpose: a session that
            # returned prose did not answer, and guessing which letter it "meant" would score a
            # different thing than the agent submitted. The single-task instructions state the format.
            if self._driver_mode:
                return 0.0, f"invalid answer {answer!r} -- must be one of 'A', 'B', 'C', 'D'"
            raise ValueError(f"invalid answer {answer!r} -- must be one of 'A', 'B', 'C', 'D'")
        if submitted == correct:
            return 1.0, f"correct answer: {correct}"
        return 0.0, f"wrong answer: submitted {submitted!r}, correct {correct!r}"

    def complete_task(self, answer: str | None = None) -> dict[str, Any]:
        """Marks the current task as complete and grades it.

        For a `quiz_question` task, pass your answer letter, e.g. `complete_task(answer="B")`;
        for every other task (including INFO tasks) pass no arguments.
        Returns a dict with `tasks_remaining` and `next_step` fields.

        After calling ``complete_task()``, call ``get_next_task()`` to fetch the next task and complete
        it before calling ``complete_task()`` again -- unless no tasks remain (``tasks_remaining`` is 0),
        in which case end your session.
        """
        if self._idx >= len(self._all_tasks):
            return {"error": "No more tasks", "tasks_remaining": 0}

        if not self._task_read:
            raise ValueError(
                "You have already called complete_task() on the current task. "
                "Your submission was final, and you are no longer allowed to work on the current task. "
                "You must now move on to the next task. " + _PULL_NEXT_STEP
            )

        entry = self._all_tasks[self._idx]
        task_data = entry["task_data"]
        task_id = task_data.get("task_id", entry["task_key"])
        task_type = task_data.get("task_type", "unknown")
        is_trigger = task_data.get("is_trigger", False)

        # Reject -- every time, not just once -- if no campus tools were used for a non-trigger, non-quiz
        # task: completing an action task (walking/booking/course-selection) without having acted did
        # nothing, so there is nothing to score. The agent must call a tool before it can complete.
        unacted = not is_trigger and task_type != "quiz_question" and not self._campus_tools_called
        if unacted and not self._driver_mode:
            raise ValueError(
                "You haven't completed the current task yet. "
                "You must complete the current task with tools before calling complete_task()."
            )

        if unacted:
            # Driver mode only (the branch above raised otherwise). Scored 0.0 rather than graded: the
            # session called no campus tool, so any credit `_evaluate_task` returned would come from
            # world state an EARLIER task left behind, which is the unearned score this guard exists to
            # refuse. Stating it as its own reason also keeps these rows greppable in `task_results.jsonl`
            # -- a rising count of them is the tell that sessions are dying before they act.
            score, reason = 0.0, "no campus tool called -- the session acted on nothing"
        elif is_trigger:
            score, reason = 1.0, "trigger task (noted)"
        elif task_type == "quiz_question":
            score, reason = self._score_quiz(task_data, answer)
        else:
            score, reason = _evaluate_task(
                task_data,
                self._campus_task,
                prev_course_gt=self._prev_course_gt,
                pre_task_draft_snapshot=self._pre_task_draft_snapshot,
            )

        if task_type == "course_selection" and not is_trigger:
            expected_sections = task_data["ground_truth"]["expected_schedule_outcome"]["selected_sections"]
            self._prev_course_gt = {s["course_code"]: s["assigned_pass"] for s in expected_sections}

        success = score == 1.0
        record = {
            "task_idx": self._idx,
            "task_id": task_id,
            "task_type": task_type,
            "is_trigger": is_trigger,
            "success": success,
            "score": score,
            # The CORRECT option's lecture gap for an exam, the trigger distance for a paired task;
            # None for standalone/trigger tasks. NOT a reconstruction of far-recall membership: since
            # the refined exam rule also requires a far distractor, `gap > _FAR_RECALL_GAP` can be true
            # for a task the grade's `*_far_recall` metrics exclude. Downstream binning on this field
            # (`analysis.py`'s `_GAP_BINS`) therefore measures recall *distance*, which is a different
            # question from the grade's far-recall subset -- the two agree on every shipped task file
            # today, but they are not the same predicate. See `_episode_recall_analysis`.
            "gap": self._gaps.get(task_id),
            "reason": reason,
        }
        self._results.append(record)
        self._append_result_line(record)

        self._task_read = False
        self._idx += 1
        self._task_prepared = False
        tasks_remaining = len(self._all_tasks) - self._idx

        # `success` is recorded on `record` above (for grading) but deliberately NOT returned to the
        # agent: echoing per-task correctness back is a ground-truth feedback leak -- it would let the
        # agent learn it got a quiz wrong and retry/steer, which the sequential episode must not allow.
        response: dict[str, Any] = {"tasks_remaining": tasks_remaining}
        if tasks_remaining > 0:
            response["next_step"] = (
                "Your submission was final — you are no longer allowed to work on the current task. "
                "You must now move on to the next task. " + _PULL_NEXT_STEP
            )
        else:
            response["next_step"] = "All tasks complete. End your session."
        return response

    @property
    def tasks_remaining(self) -> int:
        """Number of tasks not yet finished."""
        return len(self._all_tasks) - self._idx

    @property
    def results(self) -> list[dict[str, Any]]:
        """Per-task result dicts accumulated so far."""
        return self._results


# ---------------------------------------------------------------------------
# Task loading and episode aggregation (copied from stulife_env.load_tasks / run_episode tail)
# ---------------------------------------------------------------------------


def _resolve_data_dir(data_dir: str) -> Path:
    """Absolute path for a configured `data_dir`, a relative value taken from the repo root."""
    # Relative-to-the-REPO, matching `AppWorldEnv`'s `root` and this module's own `_DEFAULT_DATA_DIR`,
    # which is computed from `__file__` and is therefore correct from any working directory. A bare
    # `Path(data_dir)` kept a relative value relative, so it opened against the CWD -- the default was
    # launch-independent and an explicit value silently was not. No shipped config sets `data_dir`, so
    # this fixes a latent inconsistency rather than an observed failure.
    # An absolute value is resolved too, not merely passed through, so it matches `AppWorldEnv`'s
    # `_resolve_root` exactly: two spellings of one directory -- a symlink and its target -- otherwise
    # compare unequal here while comparing equal there.
    path = Path(data_dir)
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _load_task_entries(
    data_dir: Path,
    tasks_file: str,
    task_type_filter: list[str] | None,
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    """Load task entries from `tasks_file`, sorted by their numeric key prefix, filtered and capped."""
    with open(data_dir / tasks_file) as f:
        raw = json.load(f)
    items = [(k, v) for k, v in raw.items() if k != "metadata"]
    items.sort(key=lambda x: int(x[0].split("_")[0]))
    if task_type_filter:
        items = [(k, v) for k, v in items if v.get("task_type") in task_type_filter]
    entries = [{"task_key": k, "task_data": v} for k, v in items]
    if max_tasks is not None:
        entries = entries[:max_tasks]
    return entries


def _compute_initial_course_gt(
    data_dir: Path,
    tasks_file: str,
    tasks_for_episode: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Seed the course-selection draft when running a filtered subset.

    Finds the last course_selection task in the *full* `tasks.json` that comes before the first task in
    the subset, so the draft starts correct even though earlier course_selection tasks are absent. Only
    meaningful for a filtered `tasks_file`; returns None for the full set. Copied from `run_episode`.
    """
    if not tasks_for_episode or tasks_file == "tasks.json":
        return None
    first_idx = int(tasks_for_episode[0]["task_key"].split("_")[0])
    with open(data_dir / "tasks.json") as f:
        full_tasks = json.load(f)
    for _k, _v in sorted(full_tasks.items(), key=lambda x: int(x[0].split("_")[0]), reverse=True):
        if _k == "metadata" or int(_k.split("_")[0]) >= first_idx:
            continue
        if _v.get("task_type") == "course_selection" and not _v.get("is_trigger", False):
            sections = (
                _v.get("ground_truth", {}).get("expected_schedule_outcome", {}).get("selected_sections", [])
            )
            if sections:
                return {s["course_code"]: s["assigned_pass"] for s in sections}
            break
    return None


# Exam-course token (in `midterm_exam_<name>`/`final_exam_<name>`) -> lecture-course token (in
# `class_<name>_task`). The two naming schemes differ, so the 8 courses that have an exam are mapped
# explicitly. If StuLife renames a course, this mapping is what has to follow it.
_EXAM_COURSE_TO_LECTURE = {
    "InnovationandEntrepreneurship": "Innovation",
    "IntroductiontoComputerScience": "IntroductionCS",
    "LinearAlgebra": "LinearAlgebra",
    "MathematicalAnalysisI": "MathematicalAnalysis",
    "MentalHealthAndDevelopmentofCollegeStudents": "MentalHealth",
    "MilitaryTheory": "MilitaryTheory",
    "Programming": "Programming",
    "ProgrammingforEveryone": "Programming4Everyone",
}

# Recall gap above which a task counts toward the "far recall" subscore -- the hard long-horizon cases,
# where the information an answer needs was delivered more than this many tasks earlier. Matches the
# `> 50` bin the upstream `analyze_results.py` reports.
_FAR_RECALL_GAP = 50


def _norm(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy protocol-name matching (ports the upstream `normalize`)."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _option_protocol(instruction: str, letter: str) -> str | None:
    """The protocol name quoted in the exam option labelled `letter`, or None if it quotes none.

    Called only for exam tasks (`midterm_exam_*`/`final_exam_*`), not paired or standalone quizzes.
    For the *correct* option the None result never fires on shipped data -- measured, a protocol is
    extracted for all 80 midterm-subset exams and all 160 full-episode ones -- so there it is a guard
    for a malformed/renamed exam, handled by the caller's far-recall fallback. Distractors do reach it
    (4 of the 640 full-episode options quote no `'... Protocol'`/`'... Directive'`).
    """
    # This is the matcher `generate_exam_protocol_gaps` applies to the correct option, unchanged and
    # merely generalized to any letter -- so the correct option's protocol, and hence its reported gap,
    # is still extracted exactly as upstream does it.
    m = re.search(rf"{re.escape(letter)}:\s*(.+?)(?:\n[A-Z]:|$)", instruction, re.DOTALL)
    if not m:
        return None
    protos = re.findall(r"'([^']*(?:Protocol|Directive)[^']*)'", m.group(1))
    return protos[0] if protos else None


def _option_protocols(instruction: str) -> dict[str, str | None]:
    """The protocol name quoted in each lettered option (A/B/C/D/...), keyed by option letter.

    Applies `_option_protocol` to *every* option, not just the correct one -- the refined far-recall
    rule needs each distractor's defining lecture too. The value is None for an option quoting no
    `'... Protocol'`/`'... Directive'` (4 of the 640 full-episode options), whose callers treat a
    missing lecture as maximally far.
    """
    # Option letters sit at line start ("A:", "B:", ...); dedupe while preserving order. Every letter
    # goes through the same matcher, so the correct option's gap is computed identically to the
    # correct-option-only rule (measured: all 160 exams carry exactly options A-D and the ground-truth
    # letter is always among them).
    letters = dict.fromkeys(re.findall(r"(?m)^\s*([A-Z]):", instruction))
    return {letter: _option_protocol(instruction, letter) for letter in letters}


def _episode_recall_analysis(tasks_for_episode: list[dict[str, Any]]) -> tuple[dict[str, int], set[str]]:
    """Compute `(recall_gaps, far_recall_ids)` for the episode in one pass over its task ordering.

    `recall_gaps` maps each non-trigger *paired or exam* task_id to its recall gap: how many tasks earlier
    the information it needs was delivered -- for a paired task its matching `_trigger`; for a
    midterm/final exam the same-course lecture that defines the protocol its *correct* answer applies.
    Standalone (non-recall) tasks are absent. This is the reported per-task `gap`.

    `far_recall_ids` is the subset of those task_ids that count as *far* recall (the hard long-horizon
    cases): a paired task with gap > `_FAR_RECALL_GAP`; an exam whose correct option's lecture gap
    > `_FAR_RECALL_GAP` AND at least one WRONG option's lecture gap > `_FAR_RECALL_GAP`.

    Returns `({}, set())` rather than raising on any malformed input -- this feeds `grade()`, which must
    not break on a data quirk.

    The exam->lecture match ports `generate_exam_protocol_gaps.py`: search lectures of the exam's OWN
    course at ANY `is_trigger` (StuLife teaches ~half its protocols in non-trigger lecture-quizzes, and
    reuses protocol names across courses with different rules, so the search must be course-scoped). An
    option whose defining lecture is absent from the episode falls back to a sentinel gap larger than any
    real gap and past the far threshold, so an absent-lecture option always reads as far. For the correct
    option this fallback never fires on the shipped data (every exam's correct option matches a lecture:
    160/160 full episode, 80/80 midterm subset); for a wrong option it fires for a handful (8 of the 640
    full-episode options: the 4 that quote no protocol at all, plus 4 quoting one that no prior lecture
    of their course teaches) and defensively counts an untaught distractor as far -- one can't be
    eliminated by recent knowledge.
    """
    # The exam far-recall rule is a THIRD predicate, distinct from both upstream and this env's own prior
    # one. Upstream `generate_exam_protocol_gaps.py` already reads all four options and emits
    # `longest_gap`, the MAX across them, which `analyze_results.py` bins at `> 50` -- an OR over options.
    # (Correct-option-only was this env's earlier choice, not upstream's: the reported `gap` equals
    # upstream's `longest_gap` for just 41 of the 160 full-episode exams.) The three read:
    #   upstream:  max over all options > _FAR_RECALL_GAP
    #   previously: correct option > _FAR_RECALL_GAP
    #   here:      correct > _FAR_RECALL_GAP AND at least one wrong option > _FAR_RECALL_GAP
    # The added conjunct is the point: if the correct option is far but every distractor's lecture is
    # recent, the agent can answer by ELIMINATING the recent distractors without recalling anything far,
    # so it isn't a genuine far-recall test. Requiring >=1 far distractor captures "can't be
    # short-circuited by elimination", and can only shrink the far-recall set (and thus the
    # `n_*_far_recall` counts / `_far_recall` subscores) against the correct-option-only rule, never grow
    # it. Measured, though, it shrinks it by ZERO on the shipped data: every far-by-correct-option exam
    # already has a far distractor (160/160 full episode, 156/156 in `tasks_paired_quiz_exam.json`, 72/72
    # midterm subset), and the far-recall set is identical under both rules on all ten shipped task files.
    # Nor is that a near miss -- every full-episode exam has ALL THREE distractors far, and the smallest
    # largest-distractor gap across them is 102, over twice `_FAR_RECALL_GAP`. Exams sit at the end of a
    # semester and their options are protocols from that course's lectures, so nothing is recent. The
    # refinement therefore only bites on hypothetical data with an exam whose distractors are all recent.
    # This applies ONLY to exams; paired tasks (single trigger, no options) are unchanged. The reported
    # per-task `gap` also stays the CORRECT option's gap -- the meaningful "how far is the answer's
    # lecture" scalar -- so only far-recall SET membership can change here.
    try:
        ordered = [td["task_data"] for td in tasks_for_episode]
        trig_idx = {
            td.get("task_id", "")[: -len("_trigger")]: i
            for i, td in enumerate(ordered)
            if td.get("is_trigger") and td.get("task_id", "").endswith("_trigger")
        }
        lectures_by_course: dict[str, list[tuple[int, str]]] = {}
        for i, td in enumerate(ordered):
            m = re.match(r"class_(.+?)_task\d+", td.get("task_id", "") or "")
            if m:
                lectures_by_course.setdefault(m.group(1), []).append((i, _norm(td.get("instruction") or "")))

        def _lecture_gap(proto: str | None, course: str | None, exam_idx: int) -> int:
            """Course-scoped gap from `exam_idx` back to the EARLIEST prior lecture whose text contains
            `proto` -- the largest such gap, not the smallest. Absent (unknown course, no protocol, or no
            matching lecture) -> a maximally-far sentinel: bigger than any real gap and past the far
            threshold, so an absent lecture reads as far rather than as the exam's small absolute index."""
            # `lectures_by_course` is built in ascending index order and this returns on the first hit,
            # so a protocol taught in several of its course's lectures is measured from the FIRST one.
            # That is deliberate, not just parity with upstream `generate_exam_protocol_gaps.py`: a
            # protocol's definition ACCRETES CLAUSES across its course's lectures, and an exam question
            # names the attribute that selects which clause applies. Measured on the full episode, 81
            # course+protocol pairs are defined in two or more lectures, and of the 125 definition pairs
            # none contradict -- they have disjoint triggers, or share a trigger while setting different
            # attributes (`'Startup Celestial Alignment Protocol'`: a celestial startup name sets
            # Retention Rate 'High', viral coefficient > 1, Gross Margin 80%, and market opportunity
            # 'Cosmic Scale', taught across four lectures). Applying the protocol therefore needs every
            # clause taught so far, so the earliest is the OLDEST piece the agent must still hold -- the
            # recall distance this metric exists to measure. The same holds for a distractor: ruling one
            # out needs its protocol's relevant clause, which reaches back to that bundle's first one.
            # Note the reuse is WITHIN a course too, not only across courses as the search-scoping note
            # above describes. Measured, the pick changes no far-recall membership on any shipped file.
            if course and proto:
                needle = _norm(proto)
                for lec_idx, lec_norm in lectures_by_course.get(course, []):
                    if lec_idx < exam_idx and needle in lec_norm:
                        return exam_idx - lec_idx
            return max(len(ordered), _FAR_RECALL_GAP + 1)

        gaps: dict[str, int] = {}
        far: set[str] = set()
        for i, td in enumerate(ordered):
            if td.get("is_trigger"):
                continue
            tid = td.get("task_id", "") or ""
            if tid in trig_idx:
                gap = i - trig_idx[tid]
                gaps[tid] = gap
                if gap > _FAR_RECALL_GAP:
                    far.add(tid)
                continue
            m = re.match(r"(?:midterm|final)_exam_(.+?)_\d+$", tid)
            if not m:
                continue  # standalone: no recall gap
            course = _EXAM_COURSE_TO_LECTURE.get(m.group(1))
            gt = td.get("ground_truth")
            option_protos = _option_protocols(td.get("instruction") or "")
            # Reported gap = the correct option's lecture gap (unchanged from the correct-option-only rule).
            correct_gap = _lecture_gap(option_protos.get(gt) if isinstance(gt, str) else None, course, i)
            gaps[tid] = correct_gap
            # A wrong option whose lecture is absent is treated as far (sentinel via `_lecture_gap`): an
            # untaught distractor can't be eliminated by recent knowledge, so it never makes an exam look
            # near. Defensive -- never fires on shipped data, where every option matches a lecture.
            wrong_far = any(
                _lecture_gap(proto, course, i) > _FAR_RECALL_GAP
                for letter, proto in option_protos.items()
                if letter != gt
            )
            if correct_gap > _FAR_RECALL_GAP and wrong_far:
                far.add(tid)
        return gaps, far
    except Exception:
        # Gap metrics are diagnostic; a data quirk must never take down grading (see grade()).
        return {}, set()


def _recall_gaps(tasks_for_episode: list[dict[str, Any]]) -> dict[str, int]:
    """Map each non-trigger paired/exam task_id to its reported recall gap (the correct option's, for an
    exam). Thin wrapper over `_episode_recall_analysis`; see it for the gap definition and the caveats."""
    return _episode_recall_analysis(tasks_for_episode)[0]


def _aggregate(results: list[dict[str, Any]], tasks_for_episode: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-task results to the episode's score and breakdown. Copied from `run_episode`'s tail.

    The primary score is `total_score / n_total`, where `n_total` counts every non-trigger task in the
    sequence (not just the attempted ones) -- so stopping early is not flattered by its own
    incompleteness. Triggers (INFO tasks) are excluded from all scores. The paired/standalone split
    separates tasks that require recall from earlier tasks (paired triggers + midterm/final exams) from
    the rest.
    """
    scored = [r for r in results if not r["is_trigger"]]
    total_score = sum(r["score"] for r in scored)
    n_success = sum(1 for r in scored if r["success"])
    n_total = sum(1 for td in tasks_for_episode if not td["task_data"].get("is_trigger", False))

    trigger_base_ids = {
        td["task_data"]["task_id"][: -len("_trigger")]
        for td in tasks_for_episode
        if td["task_data"].get("is_trigger", False)
        and td["task_data"].get("task_id", "").endswith("_trigger")
    }

    def _is_paired_or_exam(task_id: str) -> bool:
        return task_id in trigger_base_ids or "midterm" in task_id.lower() or "final" in task_id.lower()

    paired = [r for r in scored if _is_paired_or_exam(r["task_id"])]
    standalone = [r for r in scored if not _is_paired_or_exam(r["task_id"])]
    n_paired_total = sum(
        1
        for td in tasks_for_episode
        if not td["task_data"].get("is_trigger", False)
        and _is_paired_or_exam(td["task_data"].get("task_id", ""))
    )
    n_standalone_total = sum(
        1
        for td in tasks_for_episode
        if not td["task_data"].get("is_trigger", False)
        and not _is_paired_or_exam(td["task_data"].get("task_id", ""))
    )
    total = len(tasks_for_episode)

    extra: dict[str, Any] = {
        "n_success": n_success,
        "n_total": n_total,
        "tasks_submitted": len(results),
        "tasks_total": total,
        "fraction_submitted": len(results) / total if total > 0 else 0.0,
        "paired_submitted": len(paired),
        "standalone_submitted": len(standalone),
        "fraction_paired_submitted": len(paired) / n_paired_total if n_paired_total > 0 else 0.0,
        "fraction_standalone_submitted": (
            len(standalone) / n_standalone_total if n_standalone_total > 0 else 0.0
        ),
        "complete": len(scored) == n_total,
    }
    # The per-task breakdown is deliberately NOT duplicated here: it is streamed to `task_results.jsonl`
    # in the attempt's artifacts as each task is scored (see `_SequentialCampus._append_result_line`).
    # `results.json` keeps only this aggregate; the two never drift because both derive from the same
    # `_results`, and the JSONL -- unlike this grade -- survives a run killed before `grade()`.
    if paired:
        extra["score_paired_tasks"] = (
            sum(r["score"] for r in paired) / n_paired_total if n_paired_total else 0.0
        )
    if standalone:
        extra["score_standalone_tasks"] = (
            sum(r["score"] for r in standalone) / n_standalone_total if n_standalone_total else 0.0
        )

    # Explicit average-score (with partial credit) and pass-rate (binary success, partials floored to 0)
    # over (a) every non-trigger task and (b) the FAR-RECALL subset -- the hard long-horizon cases.
    # Far recall is membership in `_episode_recall_analysis`'s set: a paired task whose trigger was
    # more than `_FAR_RECALL_GAP` tasks earlier, or an exam whose correct AND >=1 wrong option both
    # have a defining-lecture gap past `_FAR_RECALL_GAP` (so it can't be answered by eliminating recent
    # distractors). Membership comes from that set and NOT from the per-task `gap`, which reports only
    # the correct option's distance and so no longer decides it. Denominators count every such task in
    # the sequence (not just the submitted ones), so stopping early is not flattered by its own
    # incompleteness, matching the primary score.
    _gaps, far_ids = _episode_recall_analysis(tasks_for_episode)
    far = [r for r in scored if r["task_id"] in far_ids]
    n_far_total = sum(
        1
        for td in tasks_for_episode
        if not td["task_data"].get("is_trigger", False) and td["task_data"].get("task_id", "") in far_ids
    )
    # `avg_score` mirrors the top-level Grade `score` (the same `total_score / n_total`) by design: it is
    # the whole-set member of the uniform metric family, named to sit alongside the `_far_recall`
    # variants. The two are intentionally identical, not two figures that can diverge.
    extra["avg_score"] = total_score / n_total if n_total > 0 else 0.0
    extra["pass_rate"] = n_success / n_total if n_total > 0 else 0.0
    extra["n_total_far_recall"] = n_far_total
    extra["n_success_far_recall"] = sum(1 for r in far if r["success"])
    extra["avg_score_far_recall"] = sum(r["score"] for r in far) / n_far_total if n_far_total else 0.0
    extra["pass_rate_far_recall"] = sum(1 for r in far if r["success"]) / n_far_total if n_far_total else 0.0

    # Per-task-type breakdown, mirroring the whole-set metrics: for each task type present in the
    # sequence, its mean score (partials) and pass rate over EVERY non-trigger task of that type (not just
    # the submitted ones), so an incomplete run is not flattered. Flat `<metric>_<task_type>` keys so they
    # aggregate across attempts like the other scalars.
    # The sequence side reads `task_type` off each task directly; `type_by_id` exists only to classify the
    # *result* records, since a result dict may omit its type. Per-type keys share the flat
    # `<metric>_<task_type>` namespace with the `*_far_recall` subset metrics above, so a task type named
    # literally `far_recall` would collide -- no real StuLife type does.
    type_by_id = {
        td["task_data"].get("task_id"): td["task_data"].get("task_type") for td in tasks_for_episode
    }
    non_trigger = [td for td in tasks_for_episode if not td["task_data"].get("is_trigger", False)]
    for task_type in sorted(
        {td["task_data"].get("task_type") for td in non_trigger if td["task_data"].get("task_type")}
    ):
        n_type = sum(1 for td in non_trigger if td["task_data"].get("task_type") == task_type)
        type_results = [r for r in scored if type_by_id.get(r["task_id"]) == task_type]
        extra[f"n_total_{task_type}"] = n_type
        extra[f"avg_score_{task_type}"] = sum(r["score"] for r in type_results) / n_type if n_type else 0.0
        extra[f"pass_rate_{task_type}"] = (
            sum(1 for r in type_results if r["success"]) / n_type if n_type else 0.0
        )

    score = total_score / n_total if n_total > 0 else 0.0
    return {"score": score, "extra": extra}


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def _tasks_remaining_tool(seq: Any) -> Any:
    """A `tasks_remaining()` callable over `seq`, for a harness that drives the queue itself."""
    # A function rather than the inner class's property: the tool surface is built from callables, and a
    # property read at build time would bind an int frozen at setup.

    def tasks_remaining() -> int:
        """How many tasks of this episode are still unfinished."""
        return seq.tasks_remaining

    return tasks_remaining


def _delivered_complete_task(seq: _SequentialCampus) -> Callable[..., Any]:
    """`complete_task` for the push interface: same grading, wording that fits a delivered queue."""

    # Both the docstring (which reaches the agent as the tool's description) and the returned
    # `next_step` string are written for the pull interface and name `get_next_task()`. Under delivery
    # that tool is the driver's, so leaving either in place hands the agent an instruction it cannot
    # follow -- the exact contradiction that a harness-side "ignore the above" override was papering over.
    def complete_task(answer: str | None = None) -> dict[str, Any]:
        """Marks the current task as complete and grades it.

        For a `quiz_question` task, pass your answer letter, e.g. `complete_task(answer="B")`;
        for every other task (including INFO tasks) pass no arguments.
        Returns a dict with `tasks_remaining` and `next_step` fields.

        After calling ``complete_task()``, stop and wait: the next task will be delivered to you as
        the next user message.

        Args:
            answer: Your answer letter for a `quiz_question` task; omit for every other task.
        """
        try:
            result = seq.complete_task(answer=answer)
        except ValueError as exc:
            # The double-submit guard *raises* rather than returns, so rewriting only the returned dict
            # would still leave the agent told to call a tool it does not have -- on the error path, which
            # is exactly when it is already confused. The bridge surfaces `str(exc)` alone, so the agent
            # never sees the original wording.
            raise ValueError(str(exc).replace(_PULL_NEXT_STEP, _PUSH_NEXT_STEP)) from exc
        # Rewrite in place rather than overwrite: the success path's `next_step` opens with the finality
        # guard ("Your submission was final -- you are no longer allowed to work on the current task"),
        # and assigning the push text wholesale dropped it, leaving delivery-mode agents without the one
        # sentence that stops them reopening a submitted task. `str.replace` is a no-op on the terminal
        # "All tasks complete" message, so no `tasks_remaining` guard is needed to leave that alone.
        if isinstance(result, dict) and isinstance(result.get("next_step"), str):
            result["next_step"] = result["next_step"].replace(_PULL_NEXT_STEP, _PUSH_NEXT_STEP)
        return result

    return complete_task


class StuLifeEnv(Env):
    """ELL-StuLife campus-life benchmark: a full academic year of sequential campus tasks.

    The agent reads each task with `get_next_task()`, completes it with the campus tools (email,
    navigation, reservations, course selection, ...), and records it with `complete_task()`, repeating
    until no tasks remain. Persistent subsystems carry state across tasks, so information from early
    tasks must still be available late in the run.
    """

    # `allow_calendar_modifications` defaults to True (exposes `add_event`), matching the raw ELL-StuLife
    # benchmark's all-systems default. It is safe here even though `stulife_env.py` (the JAZ suite this env
    # was copied from) defaulted it False: that False guarded upstream's *persistent* calendar, where a
    # lingering agent event could contaminate a later same-calendar task's grading. This env clears every
    # calendar at the start of each task (`_SequentialCampus._prepare_task`) and grades at the end of the
    # same task, so writes never leak -- and the multi_system scorer credits the best-matching event per
    # criterion, so extra events are never penalized. With it False, the 59 non-trigger multi_system tasks
    # whose ground truth includes a `calendar_event` are capped below full credit, since only `add_event`
    # can create the required event. The 59 is a static count over the shipped task file; club_task_066
    # is one worked example (2/3 without `add_event`, 1.0 with it), reproducible against the pinned
    # ELL-StuLife submodule since the scorer is deterministic.
    def __init__(
        self,
        *,
        data_dir: str | None = None,
        tasks_file: str = "tasks.json",
        task_type_filter: list[str] | None = None,
        max_tasks: int | None = None,
        allow_calendar_modifications: bool = True,
        task_delivery: bool = False,
        per_task_driver: bool = False,
    ) -> None:
        self._data_dir = _resolve_data_dir(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self._tasks_file = tasks_file
        self._task_type_filter = task_type_filter
        self._max_tasks = max_tasks
        self._allow_calendar_modifications = allow_calendar_modifications
        # Two interfaces onto the same sequence. Default (False) is the *pull* interface: the agent calls
        # `get_next_task()` itself. True is the *push* interface, for a method whose driver delivers each
        # task to the agent (Letta's `deliver_task_tool`): `get_next_task` becomes `@root_only` -- the
        # driver's, not the agent's -- and the instructions and `complete_task` describe waiting for the
        # next task rather than fetching it. The env owns this because the env owns its own instructions:
        # a harness rewriting them would leave the agent holding two contradictory sets of rules.
        self._task_delivery = task_delivery
        # Set by a pairing whose HARNESS drives the queue one task per session (`jaz_per_task`).
        # It does two things, both required together: the instructions switch to the single-task
        # framing, and `tasks_remaining` joins the tool surface the driver needs.
        #
        # OFF BY DEFAULT, and that default is load-bearing rather than merely conservative.
        # `JazHarness` binds `single_task_instructions` as an EXTRA invoke input whenever it differs
        # from `get_instructions()` (`jaz_harness.py`), so an env that always returned a distinct
        # single-task text would add a prompt block to EVERY StuLife JAZ arm -- silently changing the
        # prompt of every recorded run's successor and breaking byte-comparability with them.
        # Off, `get_single_task_instructions()` returns the whole-queue text verbatim, the harness sees
        # no difference, and nothing is bound. The same applies to `tasks_remaining`: see `tools()`.
        # Only a per-task pairing turns it on.
        self._per_task_driver = per_task_driver

        # Built in setup(); None until then. `stulife` is imported lazily there, so constructing this
        # env -- and importing the package -- never requires stulife installed.
        self._seq: _SequentialCampus | None = None
        self._tasks: list[dict[str, Any]] = []
        self._tools: dict[str, Any] = {}

    # --- framework API -----------------------------------------------------------------

    def setup(self) -> None:
        """Build a fresh campus and task sequence for this attempt."""
        # A fresh `CampusTask` each time (rather than the old module-global) is what makes every attempt
        # face the same lifelong world from its start, and is correct because one attempt is one full
        # episode over the whole sequence.
        from stulife.tasks.instance.campus_life_bench.task import CampusTask

        campus_task = CampusTask(task_name=None, chat_history_item_factory=None, data_dir=self._data_dir)
        self._tasks = _load_task_entries(
            self._data_dir, self._tasks_file, self._task_type_filter, self._max_tasks
        )
        initial_course_gt = _compute_initial_course_gt(self._data_dir, self._tasks_file, self._tasks)
        # Stream per-task results to `<artifacts>/task_results.jsonl` when the harness gave us an
        # artifacts dir (`set_artifacts_dir`, before setup). None in unit tests -> the sink is disabled.
        result_sink = self._artifacts_dir / _TASK_RESULTS_FILE if self._artifacts_dir is not None else None
        # setup() is re-callable (each attempt gets a fresh episode), and the sink is opened append-only,
        # so start it empty: a re-setup on the same artifacts dir must not leave the previous play's lines
        # behind while the fresh `_results` grades only this play. (The harness gives each attempt its own
        # dir, so this only bites a re-setup within one dir -- e.g. a test.)
        if result_sink is not None:
            result_sink.unlink(missing_ok=True)
        seq = _SequentialCampus(
            self._tasks,
            campus_task,
            initial_course_gt=initial_course_gt,
            result_sink=result_sink,
            driver_mode=self._per_task_driver,
        )
        self._seq = seq

        # The agent surface: the two queue tools plus the campus tools, under one flat namespace. The
        # campus tools are the live wrapped `raw_*` bound methods, so the agent sees their real
        # signatures and docstrings.
        campus_tools = _campus_tool_bindings(
            campus_task.campus_environment,
            allow_calendar_modifications=self._allow_calendar_modifications,
            on_call=seq.note_campus_tool_called,
            action_history=campus_task.action_history,
        )
        self._tools = {
            "get_next_task": seq.get_next_task,
            # Under the push interface the agent still needs a completion tool -- only *fetching* moves to
            # the driver -- but `complete_task`'s own text points at `get_next_task()`, which the agent no
            # longer has. Swap in a variant whose docstring and `next_step` say to wait instead, so the
            # tool the agent holds describes the interface it is actually on.
            "complete_task": _delivered_complete_task(seq) if self._task_delivery else seq.complete_task,
            **dict(campus_tools),
        }
        # `tasks_remaining` is added ONLY for a per-task driver, and the omission is load-bearing rather
        # than tidiness. `JazPerTaskHarness` requires it as a precondition on the env (its `_QUEUE_TOOLS`),
        # but `JazHarness` binds every `@root_only` tool as an invoke input -- so adding it
        # unconditionally would put a new tool card in the prompt of EVERY existing StuLife JAZ arm and
        # break byte-comparability with every recorded run. Declared here, with the tools, because
        # this env builds its surface dynamically rather than from public methods.
        if self._per_task_driver:
            self._tools["tasks_remaining"] = _tasks_remaining_tool(seq)

    def grade(self) -> Grade:
        """Score the attempt: the mean per-task score over every non-trigger task in the sequence.

        Reducing the whole sequence to one number is the env's job. Called even when the harness raised,
        so it reports on whatever `complete_task` recorded; a run that stopped early is scored on the tasks
        it completed, with the rest counting as unearned (`n_total` is the full non-trigger count).
        """
        if self._seq is None:
            # setup() never completed (e.g. stulife missing or data unreadable): nothing was run.
            return Grade(score=0.0, extra={"complete": False, "tasks_submitted": 0})
        aggregate = _aggregate(self._seq.results, self._tasks)
        return Grade(score=aggregate["score"], extra=aggregate["extra"])

    def is_complete(self) -> bool:
        """True once every task in the sequence has been finished (tasks_remaining == 0)."""
        # The episode is one agent session over the whole sequence, so the calling method uses this to
        # reject an agent's `return` while tasks remain (the JAZ harness turns it into a ValidateReturn
        # guard). Without it the agent ends after the first complete_task(), leaving the rest of the
        # sequence unrun and scored as unearned -- the early stop upstream `run_episode` guarded with
        # `assert seq.tasks_remaining == 0`. Also restores the base `Env.is_complete` docstring's claim
        # that this env is the overriding case.
        return self._seq is not None and self._seq.tasks_remaining == 0

    def analyze_run(self, artifacts: Path) -> dict[str, Any] | None:
        """Diagnostics about how the agent behaved during this run. Never part of the score.

        Returns a dict of named sub-reports -- REPL-input hygiene, delegation behaviour, task outcomes,
        transcript statistics, and other run-quality signals -- built from whatever log and trace files
        this attempt happened to produce, or None if none of them found anything to report on. See the
        comment below for what each key means, and `jaz_evals.analysis` for how each is computed. This
        runs on every StuLife attempt (see `eval_harness.run_attempt`), so it is a standing diagnostic
        rather than something a caller opts into.

        The runtime per-tool call count (`actual_tool_calls`) is *not* added here: it is tallied
        generically on the `Env` and folded into `analysis.json` by the eval harness for every env.
        """
        # What each key in the returned dict means. ATIF (Agent Trajectory Interchange Format) is
        # JAZ's structured trace format; where it says "ATIF-only" below, the signal needs that trace
        # and is absent for a harness that writes only the plainer `agent.log`.
        #
        # The top level (`report.to_dict()`) is REPL-code hygiene: how often the agent did each of the
        # things the ELL-StuLife REPL-input validator rejects -- mixing history search with tool calls,
        # `try`/`except`, looping over `complete_task`, combining `get_next_task` with other tool calls,
        # or shadowing a tool's name -- plus how often it searched history.
        #
        # `delegation`: per-session delegation adherence for a run where the agent delegates to itself
        # (an ATIF trace, or the FileLogger log) -- how far each session's own `__history__` grew and
        # whether it handed off past the threshold.
        #
        # `next_task_openers`: how many invokes (the root session plus any delegated sub-invokes) opened
        # by calling `get_next_task` -- a sub-invoke doing so *may* mean a subagent restarted the queue
        # instead of continuing the parent's delegated task, but is correct if the parent delegated at a
        # task boundary (see `NextTaskOpenerReport` for that caveat). ATIF-only, like `delegation`.
        #
        # `answer_spread`: the submitted-vs-correct quiz answer-letter distributions -- a run-VALIDITY
        # signal rather than a behaviour one, since an attempt that stops reading the question and emits
        # a constant still reports `status: completed` and a plausible score.
        #
        # `delegation_shape`: the smolagents hand-off tree -- cue-driven hand-offs (delegation the prompt
        # told the agent to do) against ad-hoc ones, the return guard's refusals split across those two
        # populations and by depth, the longest consecutive refusal streak, and any single step that
        # consumed a contiguous run of queue tasks. `delegation` above is the JAZ/ATIF counterpart --
        # exactly one of the two populates on a given run, since each reads a different trace format.
        #
        # `outcome`: per-task accuracy joined with recall behaviour -- pass rate by task type and
        # recall-distance bin, whether searching history predicted a correct answer, and error-vs-outcome
        # effort.
        #
        # `transcript`: the exception rate and breakdown by type, the *static* per-tool call counts, and
        # the distribution of tool calls per REPL turn. `outcome` and `transcript` need a streamed
        # results file / ATIF trace respectively, so either can be absent on a bare `agent.log`-only run.
        #
        # `return_rejections`: how many times the return guard rejected an early finish, read from
        # whatever log exists -- the one signal available for the smolagents baseline too, since it
        # produces no ATIF trace.
        #
        # `history_upkeep`: how well the agent kept its own `output_history` and searched `prev_history`
        # -- non-string appends, history-wiping re-inits, search-output pollution, the polarity of the
        # pollution guard, task-text capture, and erosion of the prompt handed to sub-invokes. Absent for
        # an arm that keeps no self-managed history.
        #
        # Prefers the untruncated `agent.atif.json` (TrajectoryRecorder) and falls back to `agent.log`; a
        # method harness that writes neither yields None (no REPL code to analyse), which is why this
        # lives here rather than being forced on every env. The checks key on bare tool calls, so they
        # need this env's tool-name set (the harness binds each tool as a bare REPL name). The set is
        # recorded in the result too, so `jaz-evals-analyze` can re-analyse an archived run without an
        # env to ask. Delegation/outcome/transcript are each added only when their source is present
        # (session tree, streamed results, ATIF turns), rather than folded into the hygiene report.
        tool_names = frozenset(self._tools)
        report = analyze_attempt(artifacts, tool_names=tool_names)
        delegation = delegation_adherence(artifacts)
        outcome = outcome_for_attempt(artifacts)
        stats = transcript_stats_for_attempt(artifacts, tool_names=tool_names)
        # `next_task_openers`: how many invokes (root + delegated sub-invokes) opened by calling
        # `get_next_task`. A sub-invoke doing so is an upper bound on "subagent restarted the queue" --
        # correct if the parent delegated at a task boundary, a restart only if mid-task (see
        # `NextTaskOpenerReport`). ATIF-only, like delegation.
        openers = next_task_openers(artifacts)
        # `return_rejections` reads whatever log exists (the JAZ trace/log, `smolagents_trace.jsonl`,
        # or `smolagents.log`), so it is
        # the one signal available for the smolagents baseline too, whose run produces no ATIF trace.
        return_rejections = count_return_rejections(artifacts)
        # `history_upkeep`: CodeAct-arm self-managed-history discipline. Recorded, never enforced -- the
        # agent's imperfect upkeep IS the effect the hook-removal ablation exists to measure, so the
        # harness must not silently repair it (see `HistoryUpkeepReport`). None for an arm that never
        # mentions `output_history`, so the section is absent rather than reporting a perfect or zero
        # score for a discipline that arm does not practise.
        upkeep = history_upkeep_for_attempt(artifacts)
        # `answer_spread`: submitted-vs-correct quiz letters, a run-VALIDITY signal rather than a
        # behaviour one. An attempt that stops reading the question and emits a constant still reports
        # `status: completed` and a plausible score, so the score alone cannot distinguish it from a
        # weak-but-real run; the two distributions side by side can. Reads `task_results.jsonl`, so it
        # works for every method, not only the ones that write a REPL trace.
        answers = answer_spread_for_attempt(artifacts)
        # `delegation_shape`: the smolagents hand-off tree -- cue-driven hand-offs vs ad-hoc
        # delegations, how the guard's refusals split between them, and any single step that
        # consumed many queue tasks at once. `delegation` above is the JAZ/ATIF counterpart; this
        # one reads `smolagents_trace.jsonl`, so exactly one of the two populates on a given run.
        shape = delegation_shape_for_attempt(artifacts)
        if (
            report is None
            and delegation is None
            and outcome is None
            and stats is None
            and openers is None
            and return_rejections is None
            and upkeep is None
            and answers is None
            and shape is None
        ):
            return None
        result: dict[str, Any] = report.to_dict() if report is not None else {}
        result["tool_names"] = sorted(tool_names)
        if delegation is not None:
            result["delegation"] = delegation.to_dict()
        if openers is not None:
            result["next_task_openers"] = openers.to_dict()
        if outcome is not None:
            result["outcome"] = outcome.to_dict()
        if stats is not None:
            result["transcript"] = stats.to_dict()
        if return_rejections is not None:
            result["return_rejections"] = return_rejections
        if upkeep is not None:
            result["history_upkeep"] = upkeep.to_dict()
        if answers is not None:
            result["answer_spread"] = answers.to_dict()
        if shape is not None:
            result["delegation_shape"] = shape.to_dict()
        return result

    def get_instructions(self) -> str:
        """The agent-facing instructions: the ELL-StuLife sequential narrative. Available after `setup()`.

        Tools are named bare (`get_next_task()`, not `env.get_next_task()`). The instructions do not
        enumerate the tool set -- every harness surfaces each bound callable's own signature and docstring
        (see the comment on `_INSTRUCTIONS`).
        """
        template = _INSTRUCTIONS_DELIVERED if self._task_delivery else _INSTRUCTIONS
        return template.format(num_tasks=len(self._tasks))

    def get_single_task_instructions(self) -> str | None:
        """The instructions for a session handed exactly one task, when `per_task_driver` is set.

        `None` otherwise, since without that flag this env frames every session as the whole queue.
        """
        # The `None` is what keeps every existing pairing's prompt byte-identical: `JazHarness` binds
        # this as an extra input only when it is not `None`, so an arm that never sets the flag gets
        # no new prompt block.
        if not self._per_task_driver:
            return None
        return _SINGLE_TASK_INSTRUCTIONS

    def root_only_tool_names(self) -> set[str]:
        """Under task delivery, `get_next_task` is the driver's tool, not the agent's.

        The driver calls it to obtain each task's text and delivers that text itself; the agent must not
        also be able to pull, or the two race for the same queue.
        """
        # Union, never replace: the base contract is that a `@root_only` tool cannot be un-marked by an
        # override. Returning only this set happened to be equivalent (the base set is empty today) and
        # would silently start dropping a tool the day one is added.
        base = super().root_only_tool_names()
        # Under a per-task driver ALL THREE queue tools are the driver's, for the same reason: it runs
        # the loop, and a session is handed one task's text and returns one answer.
        #
        # `get_next_task` and `complete_task` are the load-bearing two, and withholding them is not
        # tidiness. `JazPerTaskHarness` scopes `shared_tool_bindings()` into every session, so a queue
        # tool left out of this set is bound in every one of them -- and `complete_task()` called by the
        # session grades and advances ITS OWN task, after which the loop's own `complete_task()` finds no
        # open task, raises `ValueError` outside the per-session guard, and ends the whole run. That is
        # the exact failure `root_only`'s docstring records from AppWorld (`env.py`), reached here by the
        # opposite route: not a sub-agent inheriting a scoped tool, but the solver holding the driver's.
        # The harness assumes this -- `_inputs` says "the queue tools are withheld from the session's
        # scope" -- and nothing on that side re-checks it, so the assumption is only true if it is stated
        # here.
        if self._per_task_driver:
            base = base | {"tasks_remaining", "get_next_task", "complete_task"}
        return base | {"get_next_task"} if self._task_delivery else base

    def delivered_task_tool(self) -> str | None:
        """`get_next_task` under `task_delivery`, which withholds it -- see `Env.delivered_task_tool`."""
        # Must name the same tool `root_tool_bindings()` withholds above, or a delivering harness looks
        # it up and finds nothing -- so the two are written from the one condition, not kept in step by
        # hand.
        return "get_next_task" if self._task_delivery else None

    def tools(self) -> list[ToolSpec]:
        """The agent-facing tools, built from the live callables set up in `setup()`.

        Overrides the base `Env.tools()` (which reflects public methods): this env's tools are dynamic
        -- the campus `raw_*` methods and the two queue methods -- so they are described from the bound
        callables directly, which also carries their real signatures and docstrings to the agent.

        Ordered as `setup()` built them: the two queue tools first, then the campus tools grouped by
        subsystem.
        """
        # In `_tools` order rather than sorted, because that order is deliberate and reaches the agent:
        # `get_next_task`/`complete_task` bracket every task, so they belong at the front, and the
        # campus tools read as their subsystems (`_campus_tool_bindings`'s groups) rather than as an
        # alphabetical interleaving of them.
        specs: list[ToolSpec] = []
        for name, fn in self._tools.items():
            specs.append(
                ToolSpec(
                    name=name,
                    signature=str(inspect.signature(fn)),
                    description=inspect.cleandoc(fn.__doc__ or ""),
                )
            )
        return specs

    def __getattr__(self, name: str) -> Any:
        # Resolve dynamic tool names (campus tools + queue tools) set up in setup(). Only reached when
        # normal attribute lookup fails, so it never shadows real attributes/methods. `_tools` is read
        # via the instance dict to avoid recursing through __getattr__ before setup() runs.
        tools = self.__dict__.get("_tools", {})
        if name in tools:
            return tools[name]
        raise AttributeError(name)
