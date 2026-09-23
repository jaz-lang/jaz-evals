"""Generate `tables/appworld_results.tex` from run artifacts, so no number is ever hand-typed.

Every figure in that file -- per-rep, mean, SEM, median, order-statistic interval -- is computed here
and written out in one pass. The file it produces is a build product: edit THIS, re-run, commit both.
Provenance (API-key labels, resolved-config hashes, the jaz commit under test, freeze verification) is
generated from the same run artifacts as the numbers, for the same reason -- see the comment below.

Usage:
    uv run python scripts/build_appworld_table.py            # writes tables/appworld_results.tex
    uv run python scripts/build_appworld_table.py --check    # exits 1 if the file is stale

Reproducing with your OWN runs: the arms' run directories live in `scripts/appworld_runs.json`, not in
this file. Copy it, replace each glob with yours, and pass `--runs-manifest yours.json`. Globs resolve
against `--runs-root` (default: the repo root), so `--runs-root /path/to/other/checkout` also works
unchanged. `--out` chooses where the table is written. `scripts/plot_appworld_curves.py` reads the same
manifest, so the table and the curves cannot be built from different runs.

`--check` is a gate you run by hand, and cannot be automated: it reads run directories,
which are gitignored, so a fresh clone has no inputs at all and the check would fail for the wrong
reason. Run it where the runs live. Nothing stops the committed `.tex` from drifting in the meantime;
that is a known gap, not an oversight.
"""

# Why a generator rather than a hand-maintained table: the first hand-written version had three wrong
# cells out of ~200, all of the same shape -- a median row transcribed from a script's stdout, where a
# value was truncated instead of rounded (31.6529 -> "31.6"), an adjacent column's figure was copied
# into the wrong cell (the solver MEAN 14.2825 landed in the solver MEDIAN cell), and a value was
# rounded the wrong way (42.4460 -> "42.5"). Every rep row and every mean row was correct; only the
# rows a human retyped were wrong. That is not a carelessness problem, it is a process problem, and
# the fix is to remove the retyping step.
#
# Provenance is derived, not asserted, for the same reason: the hand-written version claimed all
# three 6-rep arms split their API key between rep triples; in fact JAZ used one key for all six and
# the other two arms split in OPPOSITE directions. A claim about provenance is exactly as checkable
# as a number, and belongs in the same pipeline.

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, NamedTuple

sys.path.append(str(Path(__file__).resolve().parent))
from _artifacts import (
    REPO,
    aggregate_marks,
    d1,
    describe_text_difference,
    display,
    load_manifest,
    rerun_command,
    runs_root_line,
)

DEFAULT_MANIFEST = REPO / "scripts" / "appworld_runs.json"


OUT = REPO / "tables" / "appworld_results.tex"

# The official ReAct baseline's artifacts live OUTSIDE this repo, under AppWorld's *data* root -- not
# the code checkout. `run_appworld_official_react.sh` passes `--root "$APPWORLD_DATA_ROOT"`, and
# `path_store.experiment_outputs` resolves against that, so this is where `appworld run` wrote. The
# path is absolute and machine-specific; when it is missing the row falls back to `OFFICIAL_FALLBACK`
# below and the generated header says the row was not recomputed. See `_official_rows`.
OFFICIAL_ROOT = Path(
    "/scratch/zli11010/jaz/experiments/outputs/simplified_react_code_agent/openai/gpt-5.4-nano-high-reasoning"
)
OFFICIAL_LOGS = Path("/scratch/zli11010/appworld/logs_official_react_20260828T030427Z")

# Values recovered from the official harness's own evaluation JSON on 2026-09-03, kept so the table
# still builds on a machine without the sibling checkout. Regenerate by running with the artifacts
# present; `--check` compares against live artifacts when they exist.
OFFICIAL_FALLBACK = {
    "tgc": [46.762589928057555, 46.52278177458034, 51.31894484412471],
    "sgc": [15.827338129496404, 18.705035971223023, 26.618705035971225],
    "cost": [16.588626, 16.589859, 16.590883],
}

# WHAT TO PRINT FOR THAT ROW'S COST UNCERTAINTY. The SEM over its reps is either exactly $0.0 (from the
# run logs, which record two decimals, so all three reps read $16.59) or $0.0007 (from `OFFICIAL_FALLBACK`,
# which keeps six) -- and it prints as +/-0.0 either way, which draws the LEAST replicated arm as the most
# precisely measured one. Only one batch of that arm ever completed, so three near-identical totals are
# a cancellation whose replication is untestable. The honest figure is the paired per-task SEMs combined
# in quadrature (sqrt(sum SEM_i^2) = $0.197), derived from that arm's own per-task records.
#
# Every table and figure quoting this arm's cost uses it, with ONE exception: the per-rep table below
# keeps the literal SEM, because it prints the three totals the SEM was computed from and the paper's
# caption for that table explains the difference (executive call, user, 2026-09-22). An earlier pass
# made the opposite call -- the literal +/-0.0 everywhere, for internal consistency -- which left both
# tables and the cost figure stating a precision the arm does not have.
OFFICIAL_COST_SEM = 0.20


class Arm(NamedTuple):
    """One method's runs. `key` indexes the run manifest; `meta_source` names where Meta $ comes from."""

    label: str
    short: str  # plain-text name for the generated provenance block (LaTeX has no place there)
    key: str  # arm key in the run manifest; "" for arms whose artifacts live outside the repo
    meta_source: str  # "root_turns" | "ace_calls" | "none"
    prompt_only: str  # LaTeX for the aggregate table's prompt-only column
    self_improving: bool  # median row is emitted only for these (and only at n >= 6)


# THE LABELS ARE THE PAPER'S, VERBATIM (user, 2026-09-22) -- this generator writes the table the paper
# ships, so a label edited here without editing the paper is drift, not a fix.
#
# ONE SPELLING ACROSS BOTH RESULTS TABLES (executive call, user, 2026-09-22). The one arm that appears in
# both reads `CodeAct\textsubscript{\oursimpl{}} \citep{...} \scriptsize{(per task)}` here AND in
# `build_stulife_table.py`, which used to write the tag before the citation, brace it differently, and
# hyphenate it. The OTHER differences between the two tables are deliberate and stay: AppWorld subscripts
# `AppWorld \citep{}` where StuLife subscripts a bare `\citep{}`, and Letta takes no subscript at all.
#
# KEEP THE SIZE TAG LAST IN ITS CELL. `\scriptsize` is a declaration, not a command taking an argument, so
# `\scriptsize{(per task)}` shrinks everything after it to the end of the enclosing group rather than just
# the braced text. At the end of a cell the group ends immediately and it renders correctly; append
# anything to one of these labels and it silently comes out small too.
#
# Two conventions in them:
# the `\textsubscript` names the implementation an arm was run under (`AppWorld` for the published
# baseline, `\oursimpl{}` for everything this repo ran), and `\scriptsize{(per task)}` marks an arm
# that starts every task fresh. The three unmarked arms are the self_improving=True ones, which carry
# what they learn from one task to the next and are also the ones with median rows.
#
# An earlier revision tagged those three `\scriptsize{(CSI)}` (continual self-improvement) instead and
# left the implementation out of the name, so "CodeAct" meant two different systems one row apart.
ARMS = (
    Arm(
        r"CodeAct\textsubscript{AppWorld \citep{trivedi2024appworld}} \scriptsize{(per task)}",
        "official baseline",
        "",  # special-cased: artifacts are outside the repo
        "none",
        r"\xmark\,$^\dagger$",
        False,
    ),
    Arm(
        r"CodeAct\textsubscript{\oursimpl{}} \citep{wang2024codeact} \scriptsize{(per task)}",
        "CodeAct",
        "codeact",
        "none",
        r"\checkmark",
        False,
    ),
    Arm(
        r"CodeAct+subagents\textsubscript{\oursimpl{}}",
        "CodeAct + subagents",
        "codeact_subagents",
        "root_turns",
        r"\checkmark",
        True,
    ),
    Arm(
        r"ACE \citep{zhang2025ace} on CodeAct\textsubscript{\oursimpl{}}",
        "ACE on CodeAct",
        "ace_on_codeact",
        "ace_calls",
        r"\xmark",
        True,
    ),
    Arm(
        r"\oursimpl{} \lstinline|invoke|",
        "JAZ invoke",
        "jaz_invoke",
        "root_turns",
        r"\checkmark",
        True,
    ),
)

COLUMNS = ("tgc", "sgc", "cost", "meta", "solver")
# Lower is better for money; the aggregate table's bold/underline marks follow this.
LOWER_IS_BETTER = {"cost", "meta", "solver"}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def _tgc(results: list[dict[str, Any]]) -> float:
    return 100 * sum(1 for r in results if r["success"]) / len(results)


def _sgc(results: list[dict[str, Any]]) -> float:
    """Scenario goal completion: a scenario counts only if EVERY variant passed.

    The scenario id is the task id minus its trailing `_N`. AppWorld's test_challenge split is 139
    scenarios x 3 variants, so this quantizes to 100/139 = 0.72 points -- which is why exact ties
    between arms recur in the SGC column and are not a bug.
    """
    by_scenario: dict[str, list[bool]] = collections.defaultdict(list)
    for r in results:
        by_scenario[r["task_id"].rsplit("_", 1)[0]].append(bool(r["success"]))
    return 100 * sum(1 for v in by_scenario.values() if all(v)) / len(by_scenario)


# Attempts whose Meta $ came from a precomputed sidecar rather than from the trajectory. Module state
# because `_meta_cost` is called deep inside `collect` and the header reads it once at the end; it is
# CLEARED at the start of each `build` so a second build in one process reports its own runs rather
# than inheriting the first's. It stores `<run>/<attempt>`, not `attempt.name`: every attempt in this
# suite is called `attempt-0`, so keying on the name alone made a 21-run build record at most one
# entry -- a set that could only ever be empty or a single "attempt-0", i.e. a boolean wearing a
# set's clothes.
_PRECOMPUTED_META: set[str] = set()


def _meta_cost(attempt: Path, source: str) -> float | None:
    """The improver's own spend. Two machine-readable sources, chosen per arm -- see the docstring.

    `root_turns` reads `agent.atif.json`: the ROOT trajectory's `final_metrics.total_cost_usd` is the
    root agent's own spend, because a nested invoke's cost lives in its own entry under
    `subagent_trajectories` rather than in the parent's total. For a TTSI meta the root IS the
    improver and its subagents are the solver, so that field is exactly the split this column wants.
    `ace_calls` reads `ace/calls.jsonl` instead, because ACE has no root agent -- its improver is the
    reflector/curator/dedup calls the harness makes between tasks, and that file is their cost
    record. An arm with nothing carried between tasks has no improver at all and reports `---`.

    ATIF, never `agent.trace/overview.md`. The overview is a human-readable RENDER: reading cost by
    regexing `cost $X` out of it makes the metric a function of a display format nobody promises to
    keep stable -- measuring the file format rather than the run. ATIF is the canonical
    machine-readable record and carries real per-call `cost_usd`. The two agreed to under $0.0005 on
    all twelve runs across the two ATIF-sourced 6-rep arms (JAZ invoke, CodeAct+subagents) when this
    was switched, so the change moves no published digit -- it removes a dependency on prose, not an
    error.
    """
    if source == "none":
        return None
    # A redistributable bundle ships neither `agent.atif.json` nor `ace/calls.jsonl` -- both are the
    # agent's full text, which AppWorld's license does not let us republish in plaintext. It ships a
    # precomputed `meta_cost.json` instead. Preferring the real source keeps every normal build
    # derived; falling back keeps a bundle usable, and `build` records that it happened so the
    # generated header can say the column was READ rather than derived.
    sidecar = attempt / "meta_cost.json"
    if source == "ace_calls":
        calls = attempt / "ace" / "calls.jsonl"
        if not calls.is_file() and sidecar.is_file():
            _PRECOMPUTED_META.add(f"{attempt.parent.name}/{attempt.name}")
            return float(json.loads(sidecar.read_text())["meta_cost_usd"])
        return sum(json.loads(line).get("cost") or 0.0 for line in calls.open())
    atif = attempt / "agent.atif.json"
    if not atif.is_file() and sidecar.is_file():
        _PRECOMPUTED_META.add(f"{attempt.parent.name}/{attempt.name}")
        return float(json.loads(sidecar.read_text())["meta_cost_usd"])
    trace = json.loads(atif.read_text())
    return float(trace["final_metrics"]["total_cost_usd"])


def _provenance(run: Path) -> dict[str, str]:
    """Key fingerprint, resolved-config hash, and the jaz commit under test, per run.

    All three are recorded by `write_provenance` at the top of every run dir. Deriving them here is
    the point: the hand-written table asserted a key split that three of these six arms did not have.
    """
    prov = json.loads((run / "provenance.json").read_text())
    # `None`, not a sentinel STRING, for "this run has no recorded key". A sentinel had to be a value
    # `key` could also legitimately hold, so every reader downstream had to know its spelling to avoid
    # labelling it as a credential -- and one of them truncated `key` to 12 characters before
    # comparing, which any sentinel longer than that would have silently escaped. A `None` cannot be
    # truncated, misspelled, or renamed by a sibling branch.
    #
    # `launch` is absent from a run whose provenance was stripped for redistribution (it carries
    # local paths and key fingerprints), and `backend_keys` was added to `provenance.json` long after
    # the first runs in this tree; both used to raise `KeyError` here rather than reaching any
    # default.
    launch = prov.get("launch") or {}
    key = (launch.get("backend_keys") or {}).get("OPENAI_API_KEY")
    resolved = prov["config"]["resolved"]
    cfg_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:16]
    jaz = prov.get("repos", {}).get("jaz", {}).get("commit", "?")
    return {
        # Already a `sha256:` digest in the artifact -- this truncates it, never a key value.
        "key": key.replace("sha256:", "")[:12] if key else None,
        "config": cfg_hash,
        "jaz": jaz[:9],
    }


def collect(
    arm: Arm, manifest: dict[str, list[str]], runs_root: Path
) -> tuple[dict[str, list[float]], list[dict[str, str]]]:
    """Per-rep metrics and per-rep provenance for one arm, in rep order."""
    runs: list[Path] = []
    for pattern in manifest.get(arm.key, []):
        runs.extend(sorted(runs_root.glob(pattern)))
    data: dict[str, list[float]] = {c: [] for c in COLUMNS}
    prov: list[dict[str, str]] = []
    for run in runs:
        attempt = run / "attempt-0"
        results = _rows(attempt / "task_results.jsonl")
        total = json.loads((attempt / "results.json").read_text())["usage"]["cost_usd"]
        meta = _meta_cost(attempt, arm.meta_source)
        data["tgc"].append(_tgc(results))
        data["sgc"].append(_sgc(results))
        data["cost"].append(total)
        data["meta"].append(math.nan if meta is None else meta)
        data["solver"].append(total if meta is None else total - meta)
        prov.append(_provenance(run))
    return data, prov


def _official_evaluation(official_root: Path, rep: int) -> Path | None:
    """The evaluation JSON for one rep of the official baseline, whatever stamp it was run under.

    Returns `None` when the rep is absent, which is the signal to fall back to recorded constants.
    Ambiguity is refused rather than guessed: two stamped directories for one rep are two different
    runs, and picking either silently would put an unexplained number in a published table.
    """
    matches = sorted(official_root.glob(f"test_challenge_rep{rep}_*"))
    live = [m / "evaluations" / "test_challenge.json" for m in matches]
    live = [p for p in live if p.exists()]
    if not live:
        return None
    if len(live) > 1:
        raise SystemExit(
            f"--official-root holds {len(live)} runs for rep {rep}:\n"
            + "\n".join(f"  {p.parent.parent.name}" for p in live)
            + "\nkeep one, or point --official-root at a directory with a single run per rep"
        )
    return live[0]


def _official(
    official_root: Path = OFFICIAL_ROOT, official_logs: Path = OFFICIAL_LOGS
) -> tuple[dict[str, list[float]], bool, bool]:
    """The official ReAct baseline, from its own evaluation JSONs; falls back to recorded values.

    Returns `(data, scores_live, cost_live)`. The two liveness flags are separate because the scores
    and the cost come from different trees: the header states which of them was recomputed here.
    """
    # They were one flag, decided by the evaluation JSONs alone. A resolvable `--official-root` with
    # an absent `--official-logs` then shipped the recorded fallback COST while the header claimed the
    # whole row had been recomputed -- the one combination where a reader is told a number was
    # measured on this machine when it was not.
    # Parameters rather than module constants read directly, so `--official-root` can redirect them.
    # The defaults are one machine's absolute paths: without the flag, reproducing this one row meant
    # editing this file, which is not something a reader of a published artifact should have to do.
    #
    # The run-stamp is DISCOVERED, not pinned, for the same reason. `run_appworld_official_react.sh`
    # stamps each rep with `date -u`, so a reader's directories never carry ours -- pointing
    # `--official-root` at their own outputs would silently fall back to the recorded constants while
    # the header said the row was recomputed. Globbing means their stamp works without a second flag.
    data: dict[str, list[float]] = {c: [] for c in COLUMNS}
    live = True
    for rep in (1, 2, 3):
        evaluation = _official_evaluation(official_root, rep)
        if evaluation is None:
            live = False
            break
        individual = json.loads(evaluation.read_text())["individual"]
        rows = [{"task_id": k, "success": v["success"]} for k, v in individual.items()]
        data["tgc"].append(_tgc(rows))
        data["sgc"].append(_sgc(rows))
    cost_live = live
    if live:
        for rep in (1, 2, 3):
            log = official_logs / f"rep{rep}.log"
            costs = (
                re.findall(r"Overall  cost: \$([0-9.]+)", log.read_text(errors="ignore"))
                if log.exists()
                else []
            )
            data["cost"].append(float(costs[-1]) if costs else math.nan)
        if any(math.isnan(c) for c in data["cost"]):
            data["cost"] = list(OFFICIAL_FALLBACK["cost"])
            cost_live = False
    else:
        data["tgc"] = list(OFFICIAL_FALLBACK["tgc"])
        data["sgc"] = list(OFFICIAL_FALLBACK["sgc"])
        data["cost"] = list(OFFICIAL_FALLBACK["cost"])
    data["meta"] = [math.nan] * 3
    data["solver"] = list(data["cost"])
    return data, live, cost_live


def cell(values: list[float]) -> str:
    return "---" if math.isnan(values[0]) else ""


def mean_cell(values: list[float]) -> str:
    """`mean^{±SEM}`, or `---` for an arm with no improver."""
    if math.isnan(values[0]):
        return "---"
    sem = st.stdev(values) / math.sqrt(len(values))
    return f"{d1(st.mean(values))}$^{{\\pm{d1(sem)}}}$"


def median_cell(values: list[float]) -> str:
    """`median` with x_(2) and x_(5) as sub/superscript -- see `_ORDER_STAT_NOTE` for the coverage."""
    if math.isnan(values[0]):
        return "---"
    s = sorted(values)
    return f"{d1(st.median(values))}$^{{{d1(s[-2])}}}_{{{d1(s[1])}}}$"


def order_stat_coverage(n: int, k: int) -> float:
    """Exact coverage of [x_(k), x_(n+1-k)] as a CI for the population median.

    Inverts the sign test: the interval misses only when fewer than k or more than n-k observations
    fall below the true median, so coverage is 1 - 2*P(Bin(n, 1/2) < k). Distribution-free -- it
    assumes only that the distribution is continuous -- and exact, not asymptotic.
    """
    tail = sum(math.comb(n, i) for i in range(k)) / 2**n
    return 1 - 2 * tail


_ORDER_STAT_NOTE = """\
%   median -- sub/superscript are the 2nd and 5th order statistics, x_(2) and x_(5). For n=6 that
%             is a distribution-free CI for the POPULATION MEDIAN with EXACT coverage 78.13%
%             (it inverts the sign test: 1 - 2*P(Bin(6,1/2) <= 1) = 1 - 2*(7/64) = 50/64, depending
%             only on n, not on the distribution). Deliberately not 95%: at n=6 the only other
%             order-statistic interval is [x_(1), x_(6)] = the full range, at 96.88%. The first n
%             admitting a k=2 interval narrower than the range and still >=90% is n=8 (92.97%);
%             n=9 is the first at >=95% (96.09%).
%             Do NOT substitute a bootstrap here. At n=6 the percentile bootstrap on the mean covers
%             ~86% when it claims 95% (measured: 3000 simulated 6-rep experiments x 1000 resamples,
%             normal population; ~88% under a uniform one), because the empirical distribution is a
%             poor stand-in for the population at that n. The order-statistic coverage above is
%             exact by construction."""


def _letters(index: int) -> str:
    """Spreadsheet-style column letters: A..Z, then AA, AB, ... -- never a non-letter."""
    # `chr(ord("A") + index)` walks straight past Z into `[`, `\\`, `]`, `^` at 26 keys and silently
    # renders them into the shipped .tex. This suite has two keys, so it is a guard rather than a
    # feature -- but a wrong label here is indistinguishable from a right one, which is the kind of
    # error worth making impossible.
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _provenance_block(
    prov_by_arm: dict[str, list[dict[str, str]]], official_live: bool, official_cost_live: bool = True
) -> str:
    """The header's provenance section, DERIVED from each run's provenance.json.

    Written by the generator precisely because the hand-maintained version got this wrong: it claimed
    every 6-rep arm split its API key between rep triples. JAZ used one key for all six, and the two
    arms that did split, split in opposite directions.
    """
    # Opaque per-table labels, not the fingerprints `provenance.json` stores. A fingerprint is safe to
    # commit (see `_capture_backend_keys` in src/jaz_evals/provenance.py) but tells an outside reader
    # nothing they can act on; "key A"/"key B" answers the only question this table needs answered --
    # did two reps share a credential -- without publishing even a hash prefix. Sorted so the mapping
    # is a pure function of the fingerprints (stable across rebuilds), not of dict iteration order.
    # The sentinel means "this run recorded no credential", not "this key". Labelling it "key A" would
    # make the claim below false for exactly the runs whose key is unknown.
    all_keys = sorted({p["key"] for prov in prov_by_arm.values() for p in prov if p["key"]})
    key_label = {key: f"key {_letters(i)}" for i, key in enumerate(all_keys)}
    lines = [
        "% PROVENANCE (derived from each run's provenance.json by scripts/build_appworld_table.py --",
        "% do not edit these by hand; they are as machine-checked as the numbers).",
        "%",
        "%   'key' is an opaque per-table label (key A, key B, ...), not the credential or even its",
        "%   fingerprint. Two reps LABELLED share a label iff they billed the same OPENAI_API_KEY; a rep",
        "%   shown as '?' recorded no key at all, and two of those are not known to share anything.",
        "%   'jaz commit'",
        "%   is the commit of the jaz submodule (this project's agent framework dependency, installed",
        "%   editable) that rep actually ran; see src/jaz_evals/provenance.py for why the commit, not",
        "%   the version pin, is what is recorded.",
        "%",
        "%   arm                      per-rep API key label / resolved-config hash / jaz commit",
    ]
    for short, prov in prov_by_arm.items():
        # Comma-separated and one token per rep: the labels used to be space-joined, so a multi-word
        # one ("key not recorded") made `key A key not recorded key B` unparseable -- a reader could
        # recover neither the rep count nor the boundaries. Unlabelled reps are `?`.
        keys = ", ".join(key_label.get(p["key"]) or "?" for p in prov)
        cfgs = sorted({p["config"] for p in prov})
        jazs = sorted({p["jaz"] for p in prov})
        lines.append(f"%   {short:<24} keys: {keys}")
        lines.append(f"%   {'':<24} config: {'constant ' + cfgs[0] if len(cfgs) == 1 else ' '.join(cfgs)}")
        lines.append(f"%   {'':<24} jaz:    {' '.join(jazs)}")
    # Derived, not written out: `key_label` is assigned at build time from whichever fingerprints the
    # runs carry, so a hardcoded "1-3 on key A" sentence would relabel silently the moment the key set
    # changes -- describing the previous build while the rows above describe this one.
    patterns: list[str] = []
    for short, prov in prov_by_arm.items():
        seq = [key_label.get(p["key"]) for p in prov]
        if all(label is None for label in seq):
            patterns.append(f"{short} recorded no key for any of its {len(seq)} reps")
            continue
        seq = [label or "no recorded key" for label in seq]
        if len(set(seq)) == 1:
            patterns.append(f"{short} ran all {len(seq)} reps on {seq[0]}")
        else:
            # Run-length encoded, so a contiguous triple reads "reps 1-3" rather than three entries.
            spans: list[str] = []
            start = 0
            for i in range(1, len(seq) + 1):
                if i == len(seq) or seq[i] != seq[start]:
                    rng = f"rep {start + 1}" if i - start == 1 else f"reps {start + 1}-{i}"
                    spans.append(f"{rng} on {seq[start]}")
                    start = i
            patterns.append(f"{short} split them: {', '.join(spans)}")
    lines += [
        "%",
        "% What those rows mean, and the two confounders they expose:",
        "%   - API KEY is NOT split the same way across arms. Nothing about which key bills a request",
        "%     should affect a measurement, but the triples are not identically-provisioned draws and",
        "%     the split is not the tidy one it would be easy to assume:",
        *[f"%       {line}" for line in patterns],
        "%   - JAZ VERSION UNDER TEST differs between rep triples in every 6-rep arm: reps 1-3 ran at",
        "%     jaz 893715d00, reps 4-6 at cff8d3f62 (4 commits later). That diff touches",
        "%     src/jaz/template_loader.py and console.py (docstring/comment only) and adds an unused",
        "%     llm/copilot.py backend; the REPL, protocol, hook dispatcher and agent loop are untouched,",
        "%     so it should be inert for these arms. It is recorded because it is a difference in the",
        "%     system under test, which is a stronger caveat than the key split, and because 'should be",
        "%     inert' is a judgement a reader is entitled to re-check.",
        "%   - Within-arm resolved config is CONSTANT (hash above), so no arm changed its own settings",
        "%     mid-study. ACE additionally shares one hash across all six reps.",
        "%   - ACE freezing verified: freeze_after=42 in all six resolved configs, and ace/calls.jsonl",
        "%     records no adaptation at a task index above 41 in any of them.",
        "%   - The JAZ arms' delivered subagent prompt text is byte-identical across their six reps.",
    ]
    if _PRECOMPUTED_META:
        lines.append(
            "%   - Meta $ was READ from a precomputed `meta_cost.json`, not derived: this was built from"
        )
        lines.append("%     run directories whose trajectories were stripped for redistribution.")
    if not official_live:
        lines.append("%   - The official-baseline row was NOT recomputed on this machine (its artifacts live")
        lines.append(
            "%     outside the repo and were absent); recorded constants in the generator were used."
        )
    elif not official_cost_live:
        lines.append("%   - The official baseline's SCORES were recomputed here, but its cost was not: the")
        lines.append(
            "%     per-rep run logs (--official-logs) were absent, so the recorded cost constants were used."
        )
    return "\n".join(lines)


def _source_lines(manifest: dict[str, list[str]]) -> str:
    """The `%   <arm>  <glob>` provenance block, rendered from the manifest actually in use.

    Generated rather than written down: this block used to be a hand-maintained copy of the globs, so a
    run built from a different manifest would have carried a header naming runs it never read.
    """
    out: list[str] = []
    for arm in ARMS:
        if not arm.key:
            continue
        for i, pattern in enumerate(manifest.get(arm.key, [])):
            out.append(f"%   {arm.short if i == 0 else '':<20} {pattern}")
    return "\n".join(out)


def _header(
    prov_by_arm: dict[str, list[dict[str, str]]],
    official_live: bool,
    manifest: dict[str, list[str]],
    runs_root: Path,
    official_cost_live: bool = True,
) -> str:
    return f"""% AppWorld (test_challenge, 417 tasks, seed 42) results.
%
% GENERATED FILE -- do not hand-edit. Rebuild with:
%     uv run python scripts/build_appworld_table.py
% and verify it is current with `--check`. Every number below is computed from run artifacts by that
% script (see the comment below this file's module docstring for why a generator exists at all).
%
% Two tables: the aggregate (mean +/- SEM over reps) and the per-rep breakdown with mean and median
% summary rows. In the aggregate table, bold marks the best value in a column and underline the
% second-best, computed across all arms including baselines (per-rep rows carry neither -- see the
% comment above that table).
%
% An arm label's subscript is the implementation it was run under: AppWorld for the published
% baseline, \\oursimpl{{}} for every arm this repo ran. "(per task)" marks an arm that starts each task
% fresh; the three unmarked ones carry what they learn from one task to the next (CodeAct + subagents,
% ACE on CodeAct, JAZ invoke -- Arm.self_improving=True), and are the ones with median rows below.
%
% Sources, all globs relative to --runs-root, which for this build was:
%   {runs_root_line(runs_root)}
{_source_lines(manifest)}
%   official baseline    NOT in this repo -- AppWorld's own experiment outputs under its DATA root.
%                        Point --official-root at yours (and --official-logs at its per-rep run logs,
%                        which carry the cost); absent, that row falls back to recorded constants and
%                        the provenance block below says so. `runs/` is gitignored, so NONE of the
%                        paths above survive a fresh clone either; they name the machine this was
%                        built on, and the generator is what makes the table reproducible from them.
%
% How each column is computed:
%   TGC       task goal completion: 100 * successes / 417, from the `success` field of each row in
%             attempt-0/task_results.jsonl (AppWorld's own per-task pass/fail).
%   SGC       scenario goal completion: a scenario counts only if ALL its variants passed
%             (139 scenarios x 3 variants). Quantized to 100/139 = 0.72 points, so exact ties recur.
%   Cost      results.json usage.cost_usd -- the whole run.
%   Meta $    the improver's own spend. NOT the same source in every arm, deliberately:
%               JAZ / CodeAct+subagents -- attempt-0/agent.atif.json, the ROOT trajectory's
%                                          final_metrics.total_cost_usd. A nested invoke's cost lives
%                                          in its own subagent_trajectories entry, not the parent's
%                                          total, so that field is exactly root-vs-subagents. The meta
%                                          IS the root on these arms.
%               ACE                     -- summed `cost` in attempt-0/ace/calls.jsonl, i.e. the
%                                          reflector/curator/dedup calls. ACE has no root agent.
%               CodeAct / official      -- none; nothing carries between tasks.
%             Read from ATIF, never from agent.trace/overview.md. The overview is a human-readable
%             RENDER; regexing `cost $X` out of it would make this column a function of a display
%             format nobody promises to keep stable -- measuring the file format rather than the run.
%             The two agree to under $0.0005 on all twelve runs across the two ATIF-sourced 6-rep
%             arms (JAZ invoke, CodeAct+subagents), so this sources the same numbers from the
%             canonical record.
%   Solver $  Cost - Meta $.
%   mean      superscript is the SEM over that arm's reps (sd/sqrt(n)).
{_ORDER_STAT_NOTE}
%             Median rows are emitted only for the three self-improving arms, which have n=6; the two
%             n=3 baselines admit no informative order-statistic interval at all.
%
% Rounding: one decimal everywhere, round-half-up. The precision is set by the official row, which
% cannot go finer -- appworld's evaluator rounds AT COMPUTATION (`percentage_average(..., round_to=1)`,
% third_party/appworld/src/appworld/evaluator.py:349-358; upstream code, not a fork patch), so the
% value stored in evaluations/test_challenge.json is already 46.8. Its `individual` map does permit
% recomputing TGC/SGC at full precision, and this generator does exactly that when the artifacts are
% present -- but the published row is matched to the harness's own precision. Our arms are matched to
% it for consistency; at 417 tasks one task is 0.24 points, so nothing material is lost.
%
{_provenance_block(prov_by_arm, official_live, official_cost_live)}
%
% OTHER CAVEATS
%   - THE TWO TABLES REPORT DIFFERENT UNCERTAINTIES FOR THE OFFICIAL BASELINE'S COST, on purpose. Its
%     cost is $16.59 in all three reps (true totals $16.588626 / $16.589859 / $16.590883), so the SEM
%     over reps is $0.0 as computed here, from two-decimal log values ($0.0007 from the six-decimal
%     recorded fallback). The aggregate table prints +/-0.2 instead -- the paired per-task SEMs
%     combined in quadrature (sqrt(sum SEM_i^2) = $0.197) --
%     because three near-identical totals from the one batch of that arm that completed are a
%     cancellation whose replication is untestable, and +/-0.0 draws the least replicated arm as the
%     most precisely measured one. The per-rep table keeps +/-0.0: it prints the three totals that SEM
%     was computed from, and the paper's caption for that table explains the difference. Every other
%     table and figure quoting this arm's cost uses +/-0.20 (see `OFFICIAL_COST_SEM`).
%   - Not every run is error-free, though every one graded all 417 tasks: CodeAct reps 1 and 2 recorded
%     IterationLimitExhaustedError (x1 and x2), and CodeAct rep 2 and ACE rep 4 each recorded
%     submit_errors: 1. grader_errors is 0 in all 21 runs, so the measurements stand.
%   - THE UNIT OF ANALYSIS IS THE REP, NOT THE TASK, and the error bars here are SEM over reps.
%     It is tempting to treat the 417 tasks as independent Bernoulli trials and quote a binomial
%     standard error; that is wrong for every self-improving arm, because the improver's prompt and
%     tools at task k were shaped by tasks 1..k-1 -- the mechanism under test is what creates the
%     dependence. Measured on the 6 JAZ reps: sd of rep TGCs is 5.10 where 417 independent tasks
%     would give 2.14, a design effect of ~5.7x, i.e. an effective ~73 independent tasks per rep.
%     Standard errors computed the per-task way are understated by ~2.4x. Clustering by scenario does
%     not fix it: the run-level effect dominates the 3-variants-per-scenario structure.
%   - n=6 per arm is too few to separate the top arms: JAZ invoke vs CodeAct+subagents is t = 1.26
%     (df=10) on TGC, not significant.
%
% LaTeX requirements: booktabs (\\toprule/\\midrule/\\cmidrule/\\bottomrule), multirow, natbib or
% biblatex (\\citep), listings (\\lstinline), a \\checkmark and \\xmark source (amssymb + pifont, or
% bbding), and the paper's own \\oursimpl macro. The `*` and $^\\dagger$ markers in the aggregate
% table's header reference notes that live in the paper, not in this file.
"""


def _aggregate_table(
    labels: list[str], prompt_only: list[str], stats: dict[str, list[float]], sems: dict[str, list[float]]
) -> str:
    marks = aggregate_marks(stats, COLUMNS, LOWER_IS_BETTER)
    lines = [
        r"\begin{tabular}{rcccccc}",
        r"\toprule",
        r" & & \multicolumn{5}{c}{AppWorld (\texttt{test-challenge})} \\",
        r"\cmidrule(lr){3-7}",
        r" & \smash{\shortstack{prompt-\\only?\,*}} & TGC (\%) & SGC (\%)"
        r" & Cost (\$) & Meta \$ & Solver \$ \\",
        r"\midrule",
    ]
    for i, label in enumerate(labels):
        if i == len(labels) - 1:
            lines.append(r"\midrule")
        cells: list[str] = []
        for col in COLUMNS:
            v = stats[col][i]
            if math.isnan(v):
                cells.append("---")
                continue
            body = d1(v)
            if (mark := marks[col].get(i)) is not None:
                body = f"{mark}{{{body}}}"
            cells.append(f"{body} $\\pm$ {d1(sems[col][i])}")
        lines.append(f"{label} & {prompt_only[i]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _per_rep_table(labels: list[str], per_arm: list[dict[str, list[float]]], self_imp: list[bool]) -> str:
    lines = [
        r"\begin{tabular}{rlccccc}",
        r"\toprule",
        r" & & \multicolumn{5}{c}{AppWorld (\texttt{test-challenge})} \\",
        r"\cmidrule(lr){3-7}",
        r" & rep & TGC (\%) & SGC (\%) & Cost (\$) & Meta \$ & Solver \$ \\",
        r"\midrule",
    ]
    for i, label in enumerate(labels):
        if i:
            lines.append(r"\midrule")
        data = per_arm[i]
        n = len(data["tgc"])
        want_median = self_imp[i] and n >= 6
        span = n + 1 + (1 if want_median else 0)
        for rep in range(n):
            cells = ["---" if math.isnan(data[c][rep]) else d1(data[c][rep]) for c in COLUMNS]
            head = f"\\multirow{{{span}}}{{*}}{{{label}}} & {rep + 1}" if rep == 0 else f" & {rep + 1}"
            lines.append(f"{head} & " + " & ".join(cells) + r" \\")
        lines.append(r"\cmidrule(lr){2-7}")
        lines.append(" & mean & " + " & ".join(mean_cell(data[c]) for c in COLUMNS) + r" \\")
        if want_median:
            lines.append(" & median & " + " & ".join(median_cell(data[c]) for c in COLUMNS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build(
    manifest: dict[str, list[str]],
    runs_root: Path,
    manifest_path: Path,
    official_root: Path = OFFICIAL_ROOT,
    official_logs: Path = OFFICIAL_LOGS,
) -> str:
    _PRECOMPUTED_META.clear()
    labels: list[str] = []
    prompt_only: list[str] = []
    per_arm: list[dict[str, list[float]]] = []
    self_imp: list[bool] = []
    prov_by_arm: dict[str, list[dict[str, str]]] = {}
    official_live = True
    official_cost_live = True
    for arm in ARMS:
        if arm.key:
            data, prov = collect(arm, manifest, runs_root)
            prov_by_arm[arm.short] = prov
        else:
            data, official_live, official_cost_live = _official(official_root, official_logs)
        labels.append(arm.label)
        prompt_only.append(arm.prompt_only)
        per_arm.append(data)
        self_imp.append(arm.self_improving)
    # A checkout without `runs/` (any fresh clone -- it is gitignored) reaches here with empty column
    # lists and used to die in the comprehension below with `IndexError: list index out of range`,
    # which says nothing about the cause. Name it instead.
    # Only the glob-driven arms: the official baseline has no globs and tracks its own availability
    # through `official_live`, so an absent one is already handled and must not trip this.
    empty = [
        arm.label for arm, data in zip(ARMS, per_arm, strict=True) if arm.key and not data.get(COLUMNS[0])
    ]
    if empty:
        raise SystemExit(
            f"no run data found for: {', '.join(empty)}, under --runs-root {runs_root}.\n"
            "Either that tree has no `runs/` (it is gitignored, so a fresh clone has none -- point "
            "--runs-root at the checkout the runs were produced in), or an arm's globs in "
            f"{manifest_path} no longer match any directory because a run was renamed, moved or "
            "archived."
        )
    stats = {c: [st.mean(d[c]) if not math.isnan(d[c][0]) else math.nan for d in per_arm] for c in COLUMNS}
    sems = {
        c: [st.stdev(d[c]) / math.sqrt(len(d[c])) if not math.isnan(d[c][0]) else math.nan for d in per_arm]
        for c in COLUMNS
    }
    # See `OFFICIAL_COST_SEM`: the quadrature figure replaces the SEM over reps for the official
    # baseline's two money columns (`meta` is `---` for it, so `solver` carries the same total as
    # `cost`). The aggregate table only -- `_per_rep_table` computes its own SEMs from `per_arm`.
    #
    # ONLY for THESE runs. $0.197 was derived from this one batch's per-task pairing; it is not a
    # property of the arm, so stamping it onto a reproducer's own costs would print their mean beside
    # our uncertainty -- the "looks measured and is half borrowed" shape `appworld_points` warns about
    # on the other side of this same row. A redirected `--official-root`/`--official-logs` therefore
    # keeps the literal SEM over whatever reps it found.
    ours = official_root == OFFICIAL_ROOT and official_logs == OFFICIAL_LOGS
    if ours:
        for i, arm in enumerate(ARMS):
            if not arm.key:
                for c in ("cost", "solver"):
                    if not math.isnan(sems[c][i]):
                        sems[c][i] = OFFICIAL_COST_SEM
    return "\n".join(
        [
            _header(prov_by_arm, official_live, manifest, runs_root, official_cost_live),
            "",
            "% " + "-" * 73,
            "% Aggregate: mean +/- SEM over reps, except the official baseline's two money",
            "% columns -- see OTHER CAVEATS above for why that one row quotes +/-0.20 instead.",
            "% " + "-" * 73,
            _aggregate_table(labels, prompt_only, stats, sems),
            "",
            "% " + "-" * 73,
            "% Per-rep breakdown, with mean and (for the self-improving arms) median summary rows.",
            "% No bold/underline on rep rows: a single rep is not a 'best' result, and marking one",
            "% invites reading it as one.",
            "% " + "-" * 73,
            _per_rep_table(labels, per_arm, self_imp),
            "",
        ]
    )


def _rerun(args: argparse.Namespace) -> str:
    return rerun_command(
        "build_appworld_table.py",
        args,
        {
            "--runs-manifest": DEFAULT_MANIFEST,
            "--runs-root": REPO,
            "--out": OUT,
            "--official-root": OFFICIAL_ROOT,
            "--official-logs": OFFICIAL_LOGS,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed table is stale")
    parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON mapping arm key -> run-directory globs (default: scripts/appworld_runs.json)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO,
        help="directory the manifest's globs are resolved against (default: the repo root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help=f"where to write the table (default: {OUT.relative_to(REPO)})",
    )
    # Same flag name as `plot_appworld_curves.py`, which reads the same artifacts for its own
    # official-baseline line: one thing to redirect, spelled one way, for the table and the figures.
    parser.add_argument(
        "--official-root",
        type=Path,
        default=OFFICIAL_ROOT,
        help="AppWorld's experiment-output root for the official ReAct baseline "
        "(default: the path these runs were produced at; the row falls back to recorded "
        "values when it is absent)",
    )
    parser.add_argument(
        "--official-logs",
        type=Path,
        default=OFFICIAL_LOGS,
        help="directory of that baseline's per-rep run logs, which carry its cost",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.runs_manifest)
    out: Path = args.out
    rendered = build(
        manifest,
        args.runs_root.resolve(),
        args.runs_manifest,
        args.official_root,
        args.official_logs,
    )
    if args.check:
        if not out.exists():
            print(f"{display(out)} does not exist -- run `{_rerun(args)}`", file=sys.stderr)
            return 1
        current = out.read_text()
        if current == rendered:
            print(f"{display(out)} is up to date")
            return 0
        # Say WHAT differs, not just that something does. A gate that reports "STALE" and stops
        # sends the reader to diff two 200-line files by hand, and the most common difference by
        # far -- a build from a different --runs-root -- is not a data change at all.
        print(f"{display(out)} differs from a fresh build:", file=sys.stderr)
        for line in describe_text_difference(current, rendered):
            print(line, file=sys.stderr)
        # Echo the flags this invocation was given: a --check against someone else's runs is stale
        # with respect to THOSE inputs, and a fix-it line naming the bare script rebuilds from
        # different ones.
        print(f"\n  Rebuild with: {_rerun(args)}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(f"wrote {display(out)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
