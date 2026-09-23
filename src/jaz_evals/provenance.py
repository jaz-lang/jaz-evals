"""What a run records about itself: the config it ran, and the code that ran it.

Written to the top level of the run directory the moment a run starts -- before the first attempt, so a
run killed halfway still says what it was. Three things land there:

- `config.yaml`, the loaded eval config verbatim, comments intact;
- `provenance.json`, the record: run metadata (including the launch command), the resolved config, and
  each source tree's git state;
- `<tree>.diff`, one `git diff HEAD` patch per source tree that had uncommitted changes.

This repo is recorded as a source tree (commit, branch, dirty state, diff). JAZ is recorded by
**version only** -- it installs from PyPI at the version `pyproject.toml` pins, so there is no checkout
to describe.

Nothing here can fail a run. Every step records its own failure in place and the rest of the snapshot
is still written.

Limitations:

- Untracked files are listed by path and never copied, so a run whose behavior depends on an untracked
  source file is not reproducible from this snapshot -- the path list is there to make that visible.
- A superproject's `git diff HEAD` shows a submodule pointer move but none of the changes *inside* the
  submodule's working tree; the `+`/`-` markers in the recorded `git submodule status` are what surface
  those.
"""

# Naming: this is *run* provenance. It is unrelated to the per-message provenance JAZ stamps on trace
# steps (`extra.provenance` in an ATIF (Agent Trajectory Interchange Format) trace, read by
# `trace_to_directory`) and to `jaz.provenance`.
#
# Why JAZ's version and not its git state: JAZ installs from PyPI at the version `pyproject.toml`
# pins, so the pin fully determines the code that ran and the version string is the honest record.
# The internal repo consumes JAZ as an editable submodule checkout instead, where the pin documents
# intent only and the commit plus `git diff HEAD` is the one thing that says what actually ran -- that
# is why this module records a full git snapshot for `jaz_evals` and why it once did for `jaz` too.
#
# Executive calls made by the user when this was designed:
#   - record this repo only, not the ELL-StuLife or AppWorld submodules and not every editable
#     install (JAZ was in this set while it was an editable checkout; see above);
#   - record untracked files as paths, never contents (copying them would sweep in `runs/`, datasets,
#     caches, and `.env`-style secrets, at unbounded size);
#   - include a small run-metadata block, so the snapshot is self-describing without cross-referencing
#     `results.json`.
#
# Prior art, deliberately not reused: `jaz/evals/eval_harness.py:get_git_info` runs `git` with no
# working directory, so it snapshots whatever repo the *process* was launched from -- which for this
# suite driving a sibling JAZ is the wrong repo. Every command here is `git -C <tree>`.
#
# The field set is provisional, like `AttemptRecord`'s: exactly what a run must record to be
# reproducible is still unsettled, and this is one slice of it.

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from importlib import metadata, util
from pathlib import Path
from typing import Any, NamedTuple

from jaz_evals.config import EvalConfig

CONFIG_FILE = "config.yaml"
PROVENANCE_FILE = "provenance.json"

# A git call runs against the user's live checkout, which may be large or on a network filesystem; a
# slow `status` must not stall the start of a run, and a hung one must not hold it forever.
_GIT_TIMEOUT = 30.0

# A diff is diagnostics beside a run, not a backup. Patches are capped, and `--binary` is deliberately
# not passed: it base64-inlines blobs, so one stray image turns a 4 KB patch into tens of megabytes.
_MAX_DIFF_BYTES = 5 * 1024 * 1024

# The distribution shipping the `jaz` package has been spelled both ways (the checkout renamed it), and
# an editable install reports whichever its own pyproject declares. Try both: the version is a
# convenience, while the commit below is the actual answer to "what ran".
_JAZ_DISTRIBUTIONS = ("jaz", "jaz-lang")


def write_provenance(directory: Path, config: EvalConfig, *, run_id: str, attempts: int) -> None:
    """Record what this run is about to run, at the top level of `directory`.

    Writes `provenance.json`, the config verbatim as `config.yaml`, and a `<tree>.diff` for each source
    tree with uncommitted changes. Call once per run, before the first attempt.

    Never raises. A snapshot that cannot be taken is recorded as an error inside `provenance.json`, and
    a `provenance.json` that cannot be written is dropped.
    """
    # Mirrors `_write_analysis` in `eval_harness`: diagnostics must never turn a graded run into a
    # failed one, and the note lands where the record would have, so a break is visible rather than
    # silent. `Exception`, not `BaseException` -- a Ctrl-C during run start should still stop the run.
    try:
        payload = _snapshot(directory, config, run_id=run_id, attempts=attempts)
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    with suppress(OSError):  # a read-only or full disk must not kill the run either
        _write_json(directory / PROVENANCE_FILE, payload)


def _snapshot(directory: Path, config: EvalConfig, *, run_id: str, attempts: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "attempts": attempts,
        "env": config.env.name,
        "method": config.method.name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "launch": _capture_launch(),
        "config": _capture_config(directory, config),
        "repos": {
            "jaz_evals": _guarded(lambda: _git_snapshot(_this_source_dir(), "jaz_evals", directory)),
            "jaz": _guarded(_jaz_snapshot),
        },
    }


def _guarded(snapshot: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Take one tree's snapshot, recording a failure rather than losing the other tree's."""
    try:
        return snapshot()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _capture_launch() -> dict[str, Any]:
    """The process invocation that launched this run: raw argv, working directory, interpreter.

    For a `jaz-evals <config> [flags]` CLI run this is the reproduction recipe; for a programmatic run
    (a sweep driver, say) it is whatever launched that process. `cwd` is part of the recipe, not
    decoration: a relative config path in `argv` resolves against it, so the argv alone does not say
    what ran. `executable` names the interpreter -- which, for an editable install, is the venv whose
    `jaz` checkout the git snapshot describes.

    Also records `backend_keys`: for each backend credential in the environment, a short SHA-256 prefix
    of its VALUE -- never the value itself. See `_capture_backend_keys`.
    """
    # argv/executable cannot raise, but os.getcwd() does when the working directory was removed. Guard
    # the block and record a reason rather than let it escape: write_provenance's outer guard would
    # collapse the ENTIRE record -- git snapshots included -- to a single error, and this diagnostic
    # must not be able to cost the rest, like every other step here.
    record: dict[str, Any] = {}
    try:
        record["argv"] = list(sys.argv)
        record["executable"] = sys.executable
        record["cwd"] = os.getcwd()
    except OSError as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
    record["backend_keys"] = _capture_backend_keys()
    return record


# The backend credentials worth telling apart. Names only -- no value from this list is ever recorded.
_BACKEND_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "TURBOPUFFER_API_KEY",
    "LETTA_TPUF_API_KEY",
)
# How many hex characters of the digest to keep. 12 is ~48 bits: far beyond enough to tell a handful of
# accounts apart, and far too little to attack the key -- which is not the threat anyway, since a
# SHA-256 preimage of a high-entropy secret is not recoverable from any prefix length.
_KEY_FINGERPRINT_CHARS = 12


def _capture_backend_keys() -> dict[str, str]:
    """Which backend credential each run used, as `<var>: "sha256:<12 hex>"` -- never the key itself.

    Only a fingerprint of the value is stored, so `provenance.json` stays safe to commit while still
    answering "did these two runs use the same credential?" -- compare fingerprints. Note that is a
    question about the CREDENTIAL, not the account: a key rotated between two runs on one account
    fingerprints differently, so a match is conclusive and a mismatch is not. A variable that is unset or
    empty is recorded as `"unset"` rather than omitted, so silence means "not checked" and an explicit
    value means "checked, and there was none".
    """
    # Recorded because the alternative is unanswerable after the fact. A completed run's process is
    # gone, so `/proc/<pid>/environ` is unavailable, and nothing else in this file distinguishes two
    # accounts: costs cannot do it either (subset-sum over a run set is degenerate -- dozens of
    # combinations hit any given total, and spend exists that no run artifact accounts for). Two full
    # runs whose head-to-head numbers were being compared could not be shown to have used the same
    # account, which is a reproducibility gap this closes for a few lines.
    #
    # A fingerprint rather than the last four characters: a suffix is a substring of the real secret,
    # and this file is committed. A digest prefix leaks nothing and compares just as well.
    out: dict[str, str] = {}
    for name in _BACKEND_KEY_VARS:
        value = os.environ.get(name) or ""
        if not value:
            out[name] = "unset"
            continue
        # `surrogateescape`, not plain utf-8: `os.environ` decodes with it, so a credential holding
        # bytes that are not valid UTF-8 arrives as a lone surrogate and plain `.encode("utf-8")`
        # raises `UnicodeEncodeError`. That raise would escape `_capture_launch` -- the call there is
        # outside its try -- and `write_provenance`'s outer guard would then collapse the WHOLE record,
        # git snapshots and `config.yaml` included, which is the one thing this diagnostic must never
        # cost (see the comment in `_capture_launch`). Round-tripping the original bytes also keeps the
        # fingerprint a faithful function of the real value rather than of a lossy replacement.
        digest = hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()[:_KEY_FINGERPRINT_CHARS]
        out[name] = f"sha256:{digest}"
    return out


def _capture_config(directory: Path, config: EvalConfig) -> dict[str, Any]:
    """Write the config verbatim to `config.yaml` and describe what was captured."""
    # `resolved` is the post-load view (absolute `prompt_path`, opaque mappings). It is always present,
    # so a config built in code rather than loaded from a file still describes itself -- but it is not a
    # substitute for the file: re-serialising the dataclass would drop every comment and rewrite
    # `prompt_path`, producing a file that is *not* the config that ran. An honest absence beats that.
    record: dict[str, Any] = {
        "source": str(config.source) if config.source is not None else None,
        "resolved": {
            "method": {
                "name": config.method.name,
                # None -- not the string "None" -- when the pairing ships no domain-method prompt.
                "prompt_path": str(config.method.prompt_path) if config.method.prompt_path else None,
                "config": config.method.config,
            },
            "env": {"name": config.env.name, "config": config.env.config},
        },
    }
    try:
        text, note = _config_text(config)
        if text is not None:
            (directory / CONFIG_FILE).write_text(text, encoding="utf-8")
    except OSError as exc:
        text, note = None, f"cannot write {CONFIG_FILE}: {exc}"
    if text is None:
        return {**record, "captured": False, "file": None, "reason": note}
    return {**record, "captured": True, "file": CONFIG_FILE, **({"note": note} if note else {})}


def _config_text(config: EvalConfig) -> tuple[str | None, str | None]:
    """The config file's text, plus a note when it was not the text captured at load."""
    if config.source_text is not None:
        return config.source_text, None
    if config.source is None:
        return None, "config was constructed programmatically; it has no source file"
    # Re-reading answers "what does that file say now" rather than "what ran", so it is the fallback,
    # not the mechanism: a sweep driver that loads its configs up front can run for hours while
    # `configs/` is edited underneath it.
    try:
        return (
            config.source.read_text(encoding="utf-8"),
            "re-read from source at run start; the text was not captured at load",
        )
    except OSError as exc:
        return None, f"cannot read {config.source}: {exc}"


class _GitOutput(NamedTuple):
    """One git command's stdout, or the reason there is none."""

    text: str | None
    reason: str | None


def _git_snapshot(source: Path, name: str, out_dir: Path) -> dict[str, Any]:
    """Git state of the repository containing `source`, with its patch written to `<name>.diff`."""
    git = _git_executable()
    if git is None:
        return {"available": False, "path": str(source), "reason": "git was not found on PATH"}
    # Let git walk up from `source` rather than computing the repo root by path arithmetic: the answer
    # stays right if the package moves, and a real (non-editable) install into site-packages correctly
    # reports "not a repository" instead of naming some unrelated enclosing checkout.
    top = _run_git(git, source, "rev-parse", "--show-toplevel")
    if top.text is None:
        return {"available": False, "path": str(source), "reason": top.reason}

    # Everything after the probe runs from the repository root, so every path it prints is
    # repo-relative rather than relative to wherever the package happens to live.
    root = Path(top.text.strip())
    record: dict[str, Any] = {"available": True, "path": str(root)}
    record.update(_head_state(git, root))
    record.update(_worktree_state(git, root))
    record["diff"] = _write_diff(git, root, name, out_dir)
    submodules = _run_git(git, root, "submodule", "status", "--recursive")
    record["submodules"] = [line.rstrip() for line in submodules.text.splitlines()] if submodules.text else []
    return record


def _head_state(git: str, source: Path) -> dict[str, Any]:
    log = _run_git(git, source, "log", "-1", "--format=%H%n%cI%n%s")
    if log.text is None:
        # A repository with no commits has no HEAD. That is a state worth recording, not a failure:
        # everything else about the tree is still true.
        return {
            "commit": None,
            "commit_time": None,
            "subject": None,
            "branch": None,
            "head_reason": log.reason,
        }
    commit, _, rest = log.text.partition("\n")
    commit_time, _, subject = rest.partition("\n")
    branch = _run_git(git, source, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit.strip(),
        "commit_time": commit_time.strip(),
        "subject": subject.strip(),
        # `HEAD` when detached, which is the honest answer rather than an invented branch name.
        "branch": branch.text.strip() if branch.text is not None else None,
    }


def _worktree_state(git: str, source: Path) -> dict[str, Any]:
    # One `status` call answers both questions. `--untracked-files=all` lists individual files rather
    # than collapsing directories, and still excludes gitignored ones -- `--ignored` is deliberately not
    # passed, since `runs/` and every cache would otherwise flood the record.
    status = _run_git(git, source, "status", "--porcelain=v1", "--untracked-files=all")
    if status.text is None:
        return {"dirty": None, "untracked": None, "status_reason": status.reason}
    lines = status.text.splitlines()
    return {"dirty": bool(lines), "untracked": [line[3:] for line in lines if line.startswith("??")]}


def _write_diff(git: str, source: Path, name: str, out_dir: Path) -> dict[str, Any] | None:
    """Write this tree's `git diff HEAD` to `<name>.diff`; return where it went, or None when clean."""
    # A sidecar file rather than a string in the JSON: a patch embedded as an escaped one-line JSON
    # value makes `provenance.json` unreadable, which defeats the point of writing it.
    diff = _run_git(git, source, "diff", "HEAD")
    if diff.text is None:
        return {"file": None, "reason": diff.reason}
    if not diff.text:
        # No tracked changes. That is not the same as a clean tree -- untracked files and a dirty
        # submodule both leave this empty while `dirty` is true -- nor the same as a failed read, which
        # records a reason instead.
        return None
    data = diff.text.encode("utf-8")
    truncated = len(data) > _MAX_DIFF_BYTES
    payload = data[:_MAX_DIFF_BYTES] if not truncated else _truncate_utf8(data, _MAX_DIFF_BYTES)
    # Guard just this write. It is the only filesystem write inside the git snapshot, and it runs after
    # HEAD, worktree, and untracked state are already gathered -- letting an OSError escape would collapse
    # the whole tree's record to `available: false` up in `_guarded`, discarding all of that. Recording
    # only the diff's failure keeps the rest, as every other step here does.
    try:
        (out_dir / f"{name}.diff").write_bytes(payload)
    except OSError as exc:
        return {"file": None, "reason": f"cannot write {name}.diff: {exc}"}
    return {"file": f"{name}.diff", "bytes": len(data), "truncated": truncated}


def _truncate_utf8(data: bytes, limit: int) -> bytes:
    """`data` cut to at most `limit` bytes without leaving a split character at the end."""
    return data[:limit].decode("utf-8", errors="ignore").encode("utf-8")


def _jaz_snapshot() -> dict[str, Any]:
    # No git state and no diff for JAZ here: it installs from PyPI as the version pinned in
    # `pyproject.toml`, not as an editable checkout, so there is no repository to describe and no
    # working tree that could have diverged from the pin. The version IS the whole record, which is
    # what pinning it buys. Taking a `git diff HEAD` against site-packages would report a failure
    # reason on every run and tell a reader nothing.
    source = _jaz_source_dir()
    version = _jaz_version()
    if source is None:
        return {"available": False, "path": None, "reason": "jaz is not installed", "version": version}
    return {"available": True, "path": str(source), "version": version, "source": "installed distribution"}


def _jaz_version() -> str | None:
    for distribution in _JAZ_DISTRIBUTIONS:
        with suppress(metadata.PackageNotFoundError):
            return metadata.version(distribution)
    return None


def _jaz_source_dir() -> Path | None:
    """The directory the `jaz` package would be imported from, or None when it is not installed."""
    # Resolving the spec does not import jaz, which keeps this module free of the lazy-import rule the
    # JAZ harness lives under -- and keeps it working where jaz is absent entirely.
    #
    # `.resolve()` is load-bearing, not cosmetic: under setuptools' strict editable mode the origin
    # points into a symlink tree under `<checkout>/build/`, and only the resolved path lands in the
    # real checkout that git can describe.
    try:
        spec = util.find_spec("jaz")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin is not None and spec.origin != "namespace":
        return Path(spec.origin).resolve().parent
    locations = list(spec.submodule_search_locations or ())
    return Path(locations[0]).resolve() if locations else None


def _this_source_dir() -> Path:
    return Path(__file__).resolve().parent


def _git_executable() -> str | None:
    return shutil.which("git")


def _run_git(git: str, directory: Path, *args: str) -> _GitOutput:
    """Run one read-only git command in `directory`, returning its stdout or why there is none."""
    # `--no-optional-locks` because this runs inside the user's live checkout while they may be using
    # git themselves: without it, `status` refreshes and locks the index underneath them.
    command = [git, "--no-optional-locks", "--no-pager", "-C", str(directory), *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT,
            check=False,
            # A prompt would hang the run behind a terminal nobody is watching.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        return _GitOutput(None, f"`git {args[0]}` timed out after {_GIT_TIMEOUT:g}s")
    except OSError as exc:
        return _GitOutput(None, f"cannot run git: {exc}")
    if completed.returncode != 0:
        reason = _first_line(completed.stderr) or f"`git {args[0]}` exited {completed.returncode}"
        return _GitOutput(None, reason)
    return _GitOutput(completed.stdout, None)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # `default=str` because the opaque `env.config` / `method.config` mappings come straight from
    # `yaml.safe_load`, which turns an unquoted date into a `datetime.date` that plain JSON refuses.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
