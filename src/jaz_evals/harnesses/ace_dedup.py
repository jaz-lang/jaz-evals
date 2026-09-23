# pyright: basic, reportMissingImports=false
# Two optional dependencies live here -- `numpy` and `sentence_transformers` -- neither in a default
# install, so strict mode would report every symbol from them as unknown. Same opt-down as
# `jaz_harness.py` takes for jaz, and for the same reason. The playbook *format* stays strict: it is in
# `ace_playbook.py`, which imports neither.
"""ACE's redundancy pass: the "refine" half of *Agentic Context Engineering*'s (arXiv:2510.04618; see
`ace.py`'s module docstring) grow-and-refine.

The curator only ever appends, so without this the playbook grows monotonically and accumulates
near-duplicate advice. This pass embeds every bullet, groups the ones above a cosine-similarity
threshold, and asks a model to merge each group into a single bullet.

The two expensive halves -- embedding and merging -- are injected as callables, so the algorithm is
testable without an embedding model and the harness decides what backs them (`build_deduplicator`).
"""

# Ported from ACE's own `BulletpointAnalyzer` (`ace/core/bulletpoint_analyzer.py`), reached via the
# predecessor suite, which carries it as a submodule of the `jaz-lang/ace` fork. Both are private sibling
# checkouts, not present in or reachable from this repository -- the commits named below are a provenance
# record of what this port was checked against, not something a reader here can independently verify. The
# distinction matters for what follows: everything called "ACE's" below was checked against the commit *below*
# the fork's one local patch, so it is the published method rather than a downstream choice. The fork's patch
# (`8aa5cdc`, "Fix multi-line bullet parsing: group bullets by anchor, not by newline") touches only parsing
# and reconstruction; it IS ported, and it is why bullets carry line ranges here -- ACE replaced a merged
# bullet by its anchor line alone, orphaning the continuation lines of any bullet whose content spanned
# several (a code snippet, a template).
#
# Two of ACE's dependencies are dropped:
#
# - **faiss.** ACE imports it and calls exactly one function, `normalize_L2`; the similarity search is
#   its own dense `numpy` dot product over all pairs, and no faiss index is ever built. So the wheel
#   bought one line of row normalization, which `_normalize` does here. Confirmed against the
#   pre-fork commit, so a run without faiss is faithful to ACE and not merely to the predecessor suite.
# - **The raw OpenAI SDK.** ACE calls `client.chat.completions.create` directly; merges here go
#   through the same JAZ `LiteLLM` backend as the reflector and curator, so one place decides how a
#   model string is routed and their cost lands in the same ledger as the rest of ACE's.
#
# What is kept bug-compatible with ACE: the greedy grouping order, the merge prompt's wording, the
# combined counts, taking the first bullet's id as the merged one's, and grouping across the whole
# document rather than within a section (so a `verification_checklist` bullet can absorb a
# `hard_rules_and_constraints` one, and the survivor keeps the first bullet's slug and position --
# upstream flattens the playbook the same way). Those decide which bullets survive a pass, so
# changing them would change the method rather than tidy it.
#
# NOT kept: upstream substitutes the group's first bullet when a merge reply does not parse while
# still dropping the rest, which deletes curator-written advice on a formatting slip. Here the group
# stays un-merged instead -- see `parse_merged`.

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from jaz_evals.harnesses.ace_playbook import Bullet, BulletBlock, format_bullet, split_bullet_blocks

# The sentence-transformer upstream uses. Named here rather than left to a default so a run records
# which embedding space its similarity threshold was calibrated against -- 0.90 means different things
# under different encoders.
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Upstream's default, and what the latest ACE configs set explicitly.
DEFAULT_THRESHOLD = 0.90

# `\s*` everywhere, matching `ace_playbook._BULLET_PARSE`: the stricter `\s+` this once used
# rejected replies the playbook itself would have parsed (`helpful=0 harmful=0::merged`), so a
# model that merged correctly but spaced it differently counted as a merge failure.
_MERGED_BULLET = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.+)", re.DOTALL)

_MERGE_PROMPT = """You are merging similar playbook bulletpoints into a single, comprehensive entry.

Given these similar bulletpoints:
{bullets}

Merge them into ONE bulletpoint that captures all important information while removing redundancy.

Requirements:
1. Keep the ID from the first entry: [{base_id}]
2. Use combined counts: helpful={helpful} harmful={harmful}
3. Combine the content to be comprehensive but concise
4. Output ONLY in this format: [{base_id}] helpful={helpful} harmful={harmful} :: [merged content]

Do NOT include any explanation, just output the merged bulletpoint."""


@dataclass(frozen=True)
class DedupResult:
    """What one pass did: the new playbook and how many bullets it removed."""

    playbook: str
    bullets_before: int
    bullets_after: int
    merges_failed: int = 0

    @property
    def removed(self) -> int:
        """How many bullets the pass eliminated."""
        return self.bullets_before - self.bullets_after


def similar_groups(embeddings: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    """Group bullet indices whose embeddings are at least `threshold` similar.

    Returns only groups of two or more, in the order they are found.
    """
    # Greedy and order-dependent, exactly as upstream: walk i in order, take every *later* j above the
    # threshold, and mark the whole group visited. This is not transitive closure -- two bullets that
    # are each similar to a third but not to each other land in one group anyway. Sections do not
    # bound it either: `deduplicate` flattens the playbook first, so a group can span headers.
    #
    # `visited` gates only `i`, NOT the `j` comprehension, so a bullet already absorbed into an
    # earlier group CAN be pulled into a later one. Witness, with a threshold t: four bullets where
    # sim(0,2) >= t, sim(1,2) >= t, sim(1,3) >= t and sim(0,1) < t give groups [0,2] then [1,2,3], and
    # bullet 2 is in both. The consequence is worse than the double billing to the merge model: its
    # content survives in TWO merged bullets, so a pass whose job is removing redundancy can create it.
    #
    # This is an upstream OVERSIGHT that this port reproduces knowingly -- not an upstream design it
    # adopts. The paper (arXiv 2510.04618, checked through v3) specifies no algorithm here at all: its
    # entire treatment is "a de-duplication step then prunes redundancy by comparing bullets via
    # semantic embeddings", with no pseudocode, no threshold, and no statement about grouping. So the
    # authors' only recorded intent is "prunes redundancy", which duplicating a bullet into two
    # survivors contradicts. Verbatim upstream all the same: `_find_similar_groups` in ACE's
    # `bulletpoint_analyzer.py` has the identical `visited.update(group)` with an unfiltered
    # `range(i + 1, ...)` scan, byte-identical between the pre-fork commit and the `jaz-lang/ace` fork
    # (the fork's patch touches only parsing and reconstruction).
    #
    # Why reproduce an oversight: the ordering of this arm's design criteria, which is
    # (1) same grading-report visibility as the JAZ meta, (2) no domain leakage into model-facing
    # text, (3) minimal diff subject to 1 and 2. Criterion 2 forced the prompt divergences recorded in
    # `ace.py`; this is not leakage, so criterion 3 governs and there is no cause to diverge.
    # NOT justified by "the published numbers were produced by it": those numbers came from few-shot
    # prompts carrying four AppWorld-specific worked
    # examples, all of which criterion 2 removed. This arm is zero-shot where upstream is few-shot, so
    # its numbers are not comparable with the paper's and paper fidelity is not a standard it holds.
    #
    # How much this actually costs, so a future reader can decide with a number rather than an
    # argument: re-running this function over the persisted playbooks of the three threshold-sweep
    # runs, at each run's own threshold and encoder, a bullet joined two groups in 4 of 300 rounds
    # (3 at 0.5, 1 at 0.6, 0 at 0.7) -- never at 0.7, which every shipped config sets. Approximate
    # upward-of-nothing: it re-embeds the post-dedup snapshots, the only ones persisted, while the
    # pass itself saw the playbook one curator delta earlier.
    # An absorbed bullet can still join a later group; that case is covered.
    #
    # What double membership does NOT break, checked because this is where a counting bug would hide:
    # a group's leader is never absorbed. `i` is skipped when visited, so a leader cannot be a member
    # of an earlier group; and a later group draws only from `range(i + 1, ...)`, which cannot reach an
    # earlier leader. So `replacement`'s keys and `absorbed` stay disjoint and
    # `bullets_after = len(blocks) - len(absorbed)` is exact even when a bullet is grouped twice.
    similarity = _similarity_matrix(embeddings)
    groups: list[list[int]] = []
    visited: set[int] = set()
    for i in range(len(embeddings)):
        if i in visited:
            continue
        similar = [j for j in range(i + 1, len(embeddings)) if similarity(i, j) >= threshold]
        if similar:
            group = [i, *similar]
            groups.append(group)
            visited.update(group)
    return groups


def _similarity_matrix(embeddings: Sequence[Sequence[float]]) -> Callable[[int, int], float]:
    """Return `sim(i, j)` over already-normalized `embeddings`.

    Uses numpy when it is importable and falls back to pure Python otherwise.
    """
    # The fallback is not just defensive: it keeps this module's algorithm testable with hand-written
    # vectors and no ML stack installed at all. numpy is always
    # present in a real run, since `sentence_transformers` depends on it -- and it matters there,
    # because the comparison is O(n^2 * d) with d=768: a few hundred bullets is tens of millions of
    # multiply-adds, seconds in Python and milliseconds as one matrix product.
    try:
        import numpy as np
    except ImportError:
        return lambda i, j: _cosine(embeddings[i], embeddings[j])

    matrix = np.asarray(embeddings, dtype=float)
    products = matrix @ matrix.T
    return lambda i, j: float(products[i, j])


def merge_prompt(bullets: Sequence[Bullet]) -> str:
    """The prompt asking a model to merge `bullets` into one."""
    listing = "\n".join(
        f"{i + 1}. [{b.id}] helpful={b.helpful} harmful={b.harmful} :: {b.content}"
        for i, b in enumerate(bullets)
    )
    return _MERGE_PROMPT.format(
        bullets=listing,
        base_id=bullets[0].id,
        helpful=sum(b.helpful for b in bullets),
        harmful=sum(b.harmful for b in bullets),
    )


def parse_merged(text: str) -> Bullet | None:
    """Parse a merge reply into a bullet, or None when the reply is not one."""
    # `re.DOTALL` because the model ignores the single-line instruction often enough to matter; without
    # it a multi-line merge would be truncated to its first line and silently lose content.
    #
    # Returns None rather than the group's first bullet so the caller can leave the group *un*-merged.
    # Substituting the first bullet here read as "these two stay separate" but was not: the caller
    # absorbed the rest of the group regardless, so one prose refusal from the merge model deleted
    # every other bullet in the group -- curator-written advice, gone, with `dedup.jsonl` recording it
    # as a successful merge. Keeping the group is the real pre-dedup state.
    match = _MERGED_BULLET.match(text.strip())
    if match is None:
        return None
    return Bullet(match.group(1), int(match.group(2)), int(match.group(3)), match.group(4).strip())


def deduplicate(
    playbook: str,
    *,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    merge: Callable[[str], str] | None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DedupResult:
    """Merge near-duplicate bullets in `playbook`.

    `embed` turns bullet contents into vectors; `merge` answers a merge prompt, or None to keep the
    first bullet of each group and drop the rest without a model call.
    """
    lines, blocks = split_bullet_blocks(playbook)
    if len(blocks) < 2:
        return DedupResult(playbook, len(blocks), len(blocks))

    vectors = _normalize(embed([block.bullet.content for block in blocks]))
    groups = similar_groups(vectors, threshold)
    if not groups:
        return DedupResult(playbook, len(blocks), len(blocks))

    replacement: dict[int, Bullet] = {}
    absorbed: set[int] = set()
    failed = 0
    for group in groups:
        members = [blocks[i].bullet for i in group]
        if merge is None:
            # `merge=None` is the paper's merge-free variant: dropping the rest of the group IS the
            # policy there, so absorbing without a model call is correct.
            replacement[group[0]] = members[0]
        else:
            merged = parse_merged(merge(merge_prompt(members)))
            if merged is None:
                # The group survives intact. A merge the model would not write is not evidence that
                # the bullets are redundant, and the dedup pass runs again after the next task.
                failed += 1
                continue
            replacement[group[0]] = merged
        absorbed.update(group[1:])

    return DedupResult(
        _rebuild(lines, blocks, replacement, absorbed), len(blocks), len(blocks) - len(absorbed), failed
    )


# Encoders by model name, shared process-wide. Keyed on the name because a config may change it, and
# two different models must not alias; unbounded only in the sense that a process running N distinct
# embedding models holds N encoders, which is what it asked for.
_ENCODERS: dict[str, Any] = {}
_ENCODER_LOCK = threading.Lock()


def build_deduplicator(
    *,
    merge: Callable[[str], str] | None,
    threshold: float = DEFAULT_THRESHOLD,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Callable[[str], DedupResult]:
    """Return a `playbook -> DedupResult` pass backed by a real sentence-transformer.

    Raises `RuntimeError` if the embedding backend is not installed.
    """
    # Why it raises rather than degrading to a no-op: a pass that silently does nothing leaves a run
    # looking like ACE-with-dedup while the playbook grows unbounded, with nothing in the artifacts to
    # reveal it -- an arm that is not the arm it is labelled as.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "ACE dedup needs an embedding backend that is not installed; run "
            "`uv sync --extra ace` (it pulls sentence-transformers), or set `dedup: false` to run "
            "the documented no-dedup variant"
        ) from exc

    # Shared across attempts, not merely across passes. `build_deduplicator` runs once per `run_task`,
    # and `eval_harness._run_attempts` runs attempts concurrently by default -- so keeping the encoder
    # per-call meant `--attempts 5` loaded five independent ~400MB encoders into one process, eagerly,
    # before task 0, and on a cold cache raced five threads on the same HuggingFace download. That is
    # the resident-memory multiplier the CPU pin below was added to avoid, just moved from VRAM to RAM.
    #
    # The raise-early property is kept and is why the import above stays eager: a missing backend must
    # fail at construction, not at task 1. Only the WEIGHTS are shared, and only they needed to be.
    # `SentenceTransformer` is read-only after load, so sharing one across attempt threads is safe;
    # `_ENCODER_LOCK` makes the load itself atomic so two threads cannot both pay for it.
    #
    # Pinned to CPU. `SentenceTransformer` defaults to CUDA whenever a card is visible, which cost a
    # run: three concurrent evals each took ~1GB of a GPU another tenant was already holding 44GB of,
    # and one died mid-dedup with `CUDA out of memory` after 5 tasks. The work does not want a GPU --
    # a pass embeds a few dozen short bullets, milliseconds either way -- so taking one is pure
    # downside: it competes with whatever else shares the box and makes the eval fail for reasons that
    # have nothing to do with the method under test.
    with _ENCODER_LOCK:
        encoder = _ENCODERS.get(embedding_model)
        if encoder is None:
            encoder = _ENCODERS[embedding_model] = SentenceTransformer(embedding_model, device="cpu")

    def embed(contents: Sequence[str]) -> Sequence[Sequence[float]]:
        return encoder.encode(list(contents), convert_to_numpy=True, show_progress_bar=False).tolist()

    def run(playbook: str) -> DedupResult:
        return deduplicate(playbook, embed=embed, merge=merge, threshold=threshold)

    return run


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two already-normalized vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """L2-normalize each row, so a dot product is a cosine similarity."""
    # This is the whole of what faiss was imported for upstream (`faiss.normalize_L2`). A zero vector
    # is left alone rather than divided by zero -- it can only come from an empty bullet, and mapping
    # it to zeros makes it similar to nothing, which is the right answer for content-free advice.
    out: list[list[float]] = []
    for vector in vectors:
        norm = sum(x * x for x in vector) ** 0.5
        out.append([x / norm for x in vector] if norm else list(vector))
    return out


def _rebuild(
    lines: list[str], blocks: list[BulletBlock], replacement: dict[int, Bullet], absorbed: set[int]
) -> str:
    """Rewrite the playbook, replacing merged bullets and dropping absorbed ones."""
    # Whole line ranges, never single lines: a merged bullet replaces its anchor *and* its
    # continuation lines, and an absorbed one takes its continuations with it. Rewriting by line would
    # leave orphaned continuation text under a bullet id that no longer exists.
    by_start = {block.line_start: index for index, block in enumerate(blocks)}
    out: list[str] = []
    line = 0
    while line < len(lines):
        index = by_start.get(line)
        if index is None:
            out.append(lines[line])
            line += 1
            continue
        block = blocks[index]
        if index in replacement:
            merged = replacement[index]
            out.append(format_bullet(merged.id, merged.helpful, merged.harmful, merged.content))
        elif index not in absorbed:
            out.extend(lines[block.line_start : block.line_end + 1])
        line = block.line_end + 1
    return "\n".join(out)
