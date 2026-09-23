"""The ACE playbook: its on-disk format, and the curator operations that grow it.

A *playbook* is the context ACE engineers. It is a Markdown-ish document of fixed sections, each
holding numbered bullets the curator appends over the course of a run::

    ## Problem Solving Heuristics and Workflows
    problem solving heuristics; task decomposition and workflow
    [psw-00001] helpful=0 harmful=0 :: Read the task statement twice before calling anything.

This module is the whole format: parsing it, rendering it for a model, and applying the curator's
`ADD` operations. It is pure -- no LLM, no JAZ, no environment -- so it is the part of ACE that can
be tested exhaustively and cheaply, and `ace.py` holds everything that talks to a model.
"""

# Ported from two places, and it is worth keeping them apart. Both are private sibling checkouts -- the
# predecessor suite and the `jaz-lang/ace`/`ace-appworld` repos it carries as submodules -- not present in or
# reachable from this repository; the paths and commits named below are a provenance record of what this port
# was checked against, not something a reader here can independently verify.
#
# - The parsers and the bullet format are **ACE's own** (`playbook_utils.py` and `utils.py` in the ACE
#   repo, reached via the predecessor suite's `jaz-lang/ace` submodule). `split_bullet_blocks` is the
# exception:
#   it comes from the fork's local commits (`8aa5cdc` multi-line bullet parsing, `43f351a` its tests),
#   which ACE does not have. That divergence is a DECLINED BUG, not a gratuitous one: upstream tracks a
#   bullet by its anchor line alone (`bullet_line_mapping[i] = line_idx`), so a multi-line bullet's
#   continuation lines are unrecognised and hit the "not a bullet line" branch of its reconstruction
#   loop, which emits them unconditionally. Reproduced by driving that loop verbatim: when the
#   multi-line bullet is ABSORBED by a merge its anchor is dropped and its continuation lines survive
#   under the section header owning nothing -- no id, no counts, no section. It is NOT re-parsed as a
#   bullet on later rounds (checked against upstream's own `parse_playbook_line`, which returns None
#   for such a line even when it contains `::`), which is worse rather than better: the text is
#   permanently invisible to the counters, to dedup and to the curator, while still being read by
#   every model the playbook is shown to. When the bullet is REPLACED by a merged one instead, the
#   pre-merge fragments trail the new text.
#   `extract_playbook_bullets` truncates the same bullets to their first line independently. Both are
#   the one missing abstraction -- a bullet occupies a line RANGE -- which is what `BulletBlock` adds.
#   Fidelity to upstream's design does not extend to its defects.
#   `_SLUG_MAP` no longer comes from that repo's top-level `utils.py` -- whose
#   fin/calc/ctx map this port dropped -- but from the nested `ace-appworld` repo; the fork touched
#   neither. Versions this port was written against, since both are pinned copies rather than a live
#   upstream: `ace-appworld` at `9f3e921` (2025-11-18). Re-checked against `ace-agent/ace-appworld`
#   main `928e868` (2026-07-24): the whitelist, the slug map and both prompt files are byte-identical,
#   so the nine-month gap does not touch anything this module relies on.
# - `strip_counts` is **the predecessor suite's** (`evals/sweb/ace_core.py`), not ACE's. The section
#   slots were that suite's too, and are now upstream ACE-AppWorld's bar one rename; see
#   `SECTION_SPECS` for what changed and
#   what that cost.
#
# Rewritten rather than vendored because ACE's copy is untyped, prints to stdout as a side effect of
# parsing, and reaches into that sibling `utils.py` for one slug helper -- none of which survives
# `pyright --strict`. The *behaviour* is kept bug-compatible where the format depends on it; each place
# ACE does something surprising is marked below, because a playbook written by one implementation has
# to stay readable by the other for a run to be comparable.

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The bullet anchor. `helpful`/`harmful` are permanently zero in this port -- nothing increments them
# -- but they stay in the stored format because upstream's parsers, its dedup analyzer, and any
# playbook carried between the two implementations all key on this exact anchor. See `strip_counts`
# for why they are nonetheless hidden from every model that reads the playbook.
_BULLET_ANCHOR = re.compile(r"^\s*\[[^\]]+\]\s*helpful=\d+\s*harmful=\d+\s*::")
_BULLET_PARSE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)", re.DOTALL)
_BULLET_COUNTS = re.compile(r"^(\s*\[[^\]]+\])\s*helpful=\d+\s+harmful=\d+\s*::\s*(.*)$")
_TRAILING_NUMBER = re.compile(r"-(\d+)$")


@dataclass(frozen=True)
class Bullet:
    """One playbook bullet: its id, its (always-zero) counters, and its content."""

    id: str
    helpful: int
    harmful: int
    content: str


# The section slots the curator may write into, as `(key, header, description)`.
#
# `key` is what the curator names in an ADD operation and what a `## header` normalizes to; the
# assertion below pins them together, because the two arrive from different places (the model's JSON
# and the playbook text) and are matched against each other.
#
# These eight ARE upstream ACE-AppWorld's taxonomy, taken from its authoritative whitelist
# (`ace-appworld/experiments/code/ace/adaptation_react.py:324`), with exactly one rename -- see
# `how_to_find_information` below.
#
# This reverses an earlier decision, and the trade is worth stating because the reversal is not
# free. The predecessor suite had deliberately replaced ACE's taxonomy to de-confound a comparison against the
# TTSI meta-agent: ACE's taxonomy names no REPL-usage slot, so ACE and the meta were being asked to
# fill different-shaped context and the method comparison was partly a taxonomy comparison. Those
# slots mirrored the meta's own "Prompting" bullets one-for-one.
#
# Reversed on the author's instruction, prioritising upstream fidelity over that de-confound. The cost is the
# original concern, unchanged: ACE now has no REPL-usage slot (it held 4-9 bullets per recorded run), so an
# ACE-vs-meta score gap again mixes method with taxonomy. Read that comparison with the difference in mind, or
# restore the predecessor suite's slots -- the replaced set was: repl_usage_discipline, workflow_and_strategy,
# important_facts_about_the_environment, useful_code_examples_and_templates, common_mistakes_and_pitfalls,
# verification_checklist, hard_rules_and_constraints.
#
# Where the dropped slots' content goes now: `important_facts_about_the_environment` is closest to
# `how_to_find_information`, and `repl_usage_discipline` has no successor -- its bullets will land in
# `troubleshooting_and_pitfalls` or, failing that, `others`. Neither is lost, but both are less
# findable than a named slot, which is the cost the paragraph above prices.
SECTION_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "strategies_and_hard_rules",
        "Strategies and Hard Rules",
        "high-level strategy, and rules that must always hold",
    ),
    (
        # Upstream ACE-AppWorld calls this `apis_to_use_for_specific_information`. Renamed because
        # that name presupposes the environment exposes APIs, which is true of AppWorld and of few
        # other benchmarks -- a shell-driven one is bash, a gridworld is moves. The harness is
        # deliberately env-agnostic (it pairs with any queue env), and the slug is model-facing too:
        # bullet ids render into the playbook the subagent reads, so `[api-00002]` would leak the
        # assumption even where the header did not. This is the ONLY section that diverges from
        # upstream's set, and it diverges because domain-agnosticism outranks upstream fidelity.
        #
        # It is also the WEAKEST of the eight on that axis, recorded here rather than left for an
        # auditor to rediscover: "find information" still frames a task as retrieval, which fits both
        # envs here, and fits an env whose tasks are about acting on state rather than locating it
        # barely or not at all. Kept because a section must be ABOUT
        # something and this is the most neutral framing of what upstream's section covers. If a
        # future env makes it read wrong, the fix is this header plus its slug.
        "how_to_find_information",
        "How to Find Information",
        "where a particular piece of information lives, and how to obtain it",
    ),
    (
        "useful_code_snippets_and_templates",
        "Useful Code Snippets and Templates",
        "reusable code snippets and templates",
    ),
    (
        "common_mistakes_and_correct_strategies",
        "Common Mistakes and Correct Strategies",
        "a mistake that was made, paired with the strategy that is correct instead",
    ),
    (
        "problem_solving_heuristics_and_workflows",
        "Problem Solving Heuristics and Workflows",
        "problem solving heuristics; task decomposition and workflow",
    ),
    ("verification_checklist", "Verification Checklist", "checks to run before completing the task"),
    (
        "troubleshooting_and_pitfalls",
        "Troubleshooting and Pitfalls",
        "pitfalls to avoid, and how to diagnose and recover from them",
    ),
    ("others", "Others", "anything that does not belong in a section above"),
)

ALLOWED_SECTIONS: tuple[str, ...] = tuple(key for key, _header, _desc in SECTION_SPECS)

# The curator names a section in prose and the playbook names it in a header; both are normalized through
# `normalize_section` and compared. If a key ever stopped matching its own header, every ADD op for that
# section would silently fall through to the "no matching section" path and land at the end of the playbook. A
# raise, not an `assert`: `python -O` strips asserts, and the failure this guards is silent misfiling of every
# bullet in the affected section -- exactly the kind that must not depend on whether the interpreter was run
# optimized. Inlined rather than calling `normalize_section`, which is defined further down: this has to run
# at import, before the playbook is ever built.
for _key, _header, _desc in SECTION_SPECS:
    if _key != _header.lower().replace(" ", "_").replace("&", "and"):
        raise ValueError(f"SECTION_SPECS key {_key!r} must equal the normalized header {_header!r}")

# Upstream ACE-AppWorld's own slug table (`ace-appworld/experiments/code/ace/utils.py:42-57`), which
# is a superset of that repo's whitelist: all eight sections plus six legacy aliases, the aliases
# omitted here because this port accepts only the eight.
#
# What it is actually for: the shared seven keep upstream's exact spellings, so ids written here and
# ids written by upstream are the same scheme. NOT for seeding from an upstream playbook under a
# section this port dropped -- that path is unreachable, because `section_slug`'s only production
# caller sits behind `validate_operations`' whitelist and the `others` rewrite, so no bullet can be
# minted under a section that is not one of the eight.
#
# `info` is the exception, and the only slug that is not upstream's: it belongs to the renamed section
# below, where upstream has `api`. Slugs are model-facing -- bullet ids render into the playbook the
# subagent reads -- so keeping `api` would have leaked the assumption the rename exists to remove.
_SLUG_MAP = {
    "strategies_and_hard_rules": "shr",
    "how_to_find_information": "info",
    "useful_code_snippets_and_templates": "code",
    "common_mistakes_and_correct_strategies": "cms",
    "problem_solving_heuristics_and_workflows": "psw",
    "verification_checklist": "vc",
    "troubleshooting_and_pitfalls": "ts",
    "others": "misc",
}

# The same class of guard as the header/key loop above, and raised for the same reason. A section
# added to `SECTION_SPECS` without a slug entry does not fail: it falls to `section_slug`'s initials
# rule and mints ids under a prefix that is nobody's, which is quieter than misfiling and therefore
# easier to ship. Cheap to pin here, since the two tables are one-for-one by construction.
for _key in ALLOWED_SECTIONS:
    if _key not in _SLUG_MAP:
        raise ValueError(f"SECTION_SPECS key {_key!r} has no _SLUG_MAP entry")


def normalize_section(name: str) -> str:
    """The canonical key for a section named either as a header or by the curator."""
    return name.strip().lower().replace(" ", "_").replace("&", "and").rstrip(":")


def section_slug(section: str) -> str:
    """The 1-5 character id prefix for bullets in `section`."""
    clean = normalize_section(section)
    if clean in _SLUG_MAP:
        return _SLUG_MAP[clean]
    # Empty parts dropped: a name with a leading, trailing or doubled underscore splits to an empty
    # string, and upstream indexes `w[0]` unguarded -- an IndexError it never hits only because every
    # section name it passes is a well-formed identifier.
    words = [w for w in clean.split("_") if w]
    if not words:
        return "misc"
    # Upstream's rule otherwise: a single-word section takes its first four characters, a multi-word
    # one takes the initials of its first five words. It is not a great scheme -- `workflow_and_strategy`
    # becomes `was` -- but ids appear in stored playbooks, so changing it would break resume against a
    # playbook written by the other implementation.
    return words[0][:4] if len(words) == 1 else "".join(w[0] for w in words[:5])


def empty_playbook() -> str:
    """A fresh playbook: the section headers, each with its description, and no bullets."""
    # The description is rendered as a plain line rather than a bullet, so it introduces the section
    # to the model without being parsed as content the curator can dedup against or count.
    return "\n\n".join(f"## {header}\n{desc}" for _key, header, desc in SECTION_SPECS)


def is_bullet(line: str) -> bool:
    """True if `line` starts a bullet."""
    return bool(_BULLET_ANCHOR.match(line))


@dataclass(frozen=True)
class BulletBlock:
    """One bullet with the line range it occupies, so a multi-line bullet can be replaced whole."""

    bullet: Bullet
    line_start: int
    line_end: int  # inclusive


def split_bullet_blocks(playbook: str) -> tuple[list[str], list[BulletBlock]]:
    """Split `playbook` into its lines and the bullet blocks within them.

    A bullet absorbs the lines after its anchor until the next anchor, a section header, a blank
    line, or the end -- so a bullet whose content spans lines (a code snippet, a template) stays one
    block.
    """
    # Line ranges rather than text alone, because the dedup pass replaces a merged bullet in place:
    # without the range it would rewrite the anchor and leave the continuation lines behind as
    # orphans belonging to a bullet that no longer exists.
    lines = playbook.strip().split("\n")
    blocks: list[BulletBlock] = []
    index = 0
    while index < len(lines):
        if not is_bullet(lines[index]):
            index += 1
            continue
        end = index + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if not stripped or stripped.startswith("#") or is_bullet(lines[end]):
                break
            end += 1
        parsed = parse_bullet("\n".join(lines[index:end]))
        if parsed is not None:
            blocks.append(BulletBlock(parsed, index, end - 1))
        index = end
    return lines, blocks


def parse_bullet(text: str) -> Bullet | None:
    """Parse one bullet (its anchor line plus any continuation lines), or None if `text` is not one."""
    match = _BULLET_PARSE.match(text.strip())
    if match is None:
        return None
    return Bullet(match.group(1), int(match.group(2)), int(match.group(3)), match.group(4))


def format_bullet(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Render one bullet in the stored (counted) format."""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def next_global_id(playbook: str) -> int:
    """The next free bullet number: one past the highest already used, across all sections."""
    # Numbering is global rather than per-section, so an id is unique in the document and the curator
    # can refer to one without naming its section.
    highest = 0
    for line in playbook.strip().split("\n"):
        bullet = parse_bullet(line)
        if bullet is None:
            continue
        match = _TRAILING_NUMBER.search(bullet.id)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def strip_counts(playbook: str) -> str:
    """Render `playbook` for a model: `[id] content`, with the counters removed.

    Section headers and every other line pass through unchanged.
    """
    # Every model-facing copy goes through this -- subagent, reflector and curator alike. The counters
    # survive only in the stored playbook, and showing them would be worse than useless: nothing in
    # any prompt explains what `helpful=0 harmful=0` means, and since nothing increments them they
    # would be a column of zeroes inviting the model to read significance into it. Upstream's AppWorld
    # generator strips them the same way; general ACE displays them, which is the form this rejects.
    out: list[str] = []
    for line in playbook.split("\n"):
        match = _BULLET_COUNTS.match(line)
        out.append(f"{match.group(1)} {match.group(2)}" if match else line)
    return "\n".join(out)


def playbook_stats(playbook: str) -> dict[str, Any]:
    """Bullet counts for `playbook`, in total and per section."""
    by_section: dict[str, int] = {}
    total = 0
    current = "general"
    for line in playbook.strip().split("\n"):
        if line.strip().startswith("##"):
            current = line.strip()[2:].strip()
            continue
        if parse_bullet(line) is not None:
            total += 1
            by_section[current] = by_section.get(current, 0) + 1
    return {"total_bullets": total, "by_section": by_section}


def apply_add_operations(playbook: str, operations: list[dict[str, Any]], next_id: int) -> tuple[str, int]:
    """Append the curator's `ADD` bullets to their sections. Returns `(playbook, next_id)`.

    Each operation needs a `section` (matched against the playbook's headers) and `content`. A bullet
    naming a section this particular playbook lacks is filed under `others`, and an `## Others` header
    is appended for it if the playbook has none.
    """
    # "a section this playbook lacks", not "a section outside the schema": `validate_operations` has
    # already dropped the latter, so the only way to reach the fallback is an `initial_playbook` seed
    # that omits one of `SECTION_SPECS`. The docstring used to say the broader thing and read as a
    # promise that out-of-schema bullets are never lost, which contradicted the drop.
    # ADD is the only operation ACE implements. Upstream's file lists UPDATE / MERGE / DELETE /
    # CREATE_META as commented-out future work, and its curator prompt offers only ADD -- so the
    # playbook is append-only, and the "refine" half of the paper's grow-and-refine is entirely the
    # dedup pass in `ace.py`. Anything other than ADD is rejected before it reaches here.
    lines = playbook.strip().split("\n")
    known = {normalize_section(line.strip()[2:]) for line in lines if line.strip().startswith("##")}

    pending: list[tuple[str, str]] = []
    for op in operations:
        section = normalize_section(str(op.get("section", "general")))
        if section not in known:
            # Upstream's fallback name, so a playbook that grows an `## Others` section keeps
            # collecting misfiled bullets there rather than at the end of the file.
            section = "others"
        bullet_id = f"{section_slug(section)}-{next_id:05d}"
        next_id += 1
        pending.append((section, format_bullet(bullet_id, 0, 0, str(op.get("content", "")))))

    # Walk the document and flush each section's new bullets at the end of its own section rather than
    # at the end of the file. "End of the section" means before the blank line that separates it from
    # the next header, not immediately before the header: flushing there put the new bullet *below*
    # the separator and welded it to the following `##`, which is genuinely ambiguous to the three
    # models that read this text -- the solver, the reflector, and the curator asked to file bullets by
    # reading exactly these headers.
    out: list[str] = []
    current: str | None = None
    for line in lines:
        if line.strip().startswith("##"):
            if current is not None:
                pending = _flush_section(out, pending, current)
            current = normalize_section(line.strip()[2:])
        out.append(line)
    if current is not None:
        pending = _flush_section(out, pending, current)

    if pending:
        # Only `others` bullets can be left, and only when the playbook has no `## Others`. Give them
        # a header: appended bare they sat under whatever section happened to come last, and
        # `playbook_stats` then counted them there -- a silent reattribution.
        out.extend(["", "## Others", ""])
        out.extend(text for _section, text in pending)
    return "\n".join(out), next_id


def _flush_section(out: list[str], pending: list[tuple[str, str]], section: str) -> list[tuple[str, str]]:
    """Insert `section`'s pending bullets at the end of its body in `out`. Returns what is still pending."""
    texts = [text for name, text in pending if name == section]
    if not texts:
        return pending
    blanks = 0
    while blanks < len(out) and not out[len(out) - 1 - blanks].strip():
        blanks += 1
    out[len(out) - blanks : len(out) - blanks] = texts
    return [(name, text) for name, text in pending if name != section]


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of one JSON object from a model's reply, or None.

    Tries the whole reply, then a ```json fence, then the first brace-balanced object in the text.
    """
    # Three strategies because the reflector and curator are asked for bare JSON and all three
    # failures are observed in practice: a model that complies, one that fences the object, and one
    # that writes a sentence before it. Returning None on total failure is deliberate -- the caller
    # keeps the previous playbook and continues, since one unparseable curator reply should cost one
    # task's learning rather than the run.
    stripped = text.strip()
    try:
        parsed: Any = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
    except json.JSONDecodeError:
        pass

    for fenced in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            parsed = json.loads(fenced.strip())
            if isinstance(parsed, dict):
                return parsed  # pyright: ignore[reportUnknownVariableType]
        except json.JSONDecodeError:
            continue

    for candidate in _balanced_objects(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed  # pyright: ignore[reportUnknownVariableType]
        except json.JSONDecodeError:
            continue
    return None


def _balanced_objects(text: str) -> list[str]:
    """Every brace-balanced `{...}` span in `text`, outermost first."""
    # String-aware: a brace inside a JSON string literal must not change the depth, or a reply
    # containing `"content": "use {} for an empty dict"` would cut the object short.
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, start = 1, i
        i += 1
        while i < len(text) and depth > 0:
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == '"':
                i += 1
                while i < len(text) and text[i] != '"':
                    i += 2 if text[i] == "\\" else 1
            i += 1
        if depth == 0:
            out.append(text[start:i])
    return out


def validate_operations(
    payload: dict[str, Any], *, on_unknown_section: Callable[[str], None] | None = None
) -> list[dict[str, Any]]:
    """The `ADD` operations in a curator reply, dropping any that name a section outside the schema.

    `on_unknown_section` is called with the section name of each dropped operation.

    Raises `ValueError` if the reply is not shaped like a curator reply at all -- a missing
    `reasoning` or `operations` field, or an operation that is not an `ADD`.
    """
    # The callback exists because dropping was the one curator failure mode with no record. Every other
    # one lands in `curator_failures.jsonl`; a curator that drifts to upstream ACE's section taxonomy
    # has every operation dropped, and without this the run is indistinguishable from a curator with
    # nothing to add -- exactly the confusion that file was added to prevent. A callback rather than a
    # second return value so the existing callers and their tests read unchanged.
    # The split between raising and dropping is upstream's and worth keeping: a malformed *reply* is a
    # failed curator call (the caller logs it and keeps the playbook), while a well-formed reply naming
    # an unknown section is one bad bullet and costs only that bullet. Without the whitelist the
    # curator invents sections, and an invented section is a slot no subagent prompt ever renders.
    if "reasoning" not in payload:
        raise ValueError("curator reply is missing the 'reasoning' field")
    if "operations" not in payload:
        raise ValueError("curator reply is missing the 'operations' field")
    if not isinstance(payload["reasoning"], str):
        raise ValueError("curator 'reasoning' must be a string")
    raw_operations: object = payload["operations"]
    if not isinstance(raw_operations, list):
        raise ValueError("curator 'operations' must be a list")
    operations: list[object] = raw_operations  # pyright: ignore[reportUnknownVariableType]

    kept: list[dict[str, Any]] = []
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ValueError(f"curator operation {index} is not an object")
        typed: dict[str, Any] = op  # pyright: ignore[reportUnknownVariableType]
        if "type" not in typed:
            raise ValueError(f"curator operation {index} is missing 'type'")
        if typed["type"] != "ADD":
            raise ValueError(f"curator operation {index} has type {typed['type']!r}; only 'ADD' is supported")
        missing = {"type", "section", "content"} - set(typed)
        if missing:
            raise ValueError(f"curator ADD operation {index} is missing {sorted(missing)}")
        section = str(typed["section"])
        if normalize_section(section) not in ALLOWED_SECTIONS:
            if on_unknown_section is not None:
                on_unknown_section(section)
            continue
        kept.append(typed)
    return kept
