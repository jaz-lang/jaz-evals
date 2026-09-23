"""Run ID generation.

A run ID names one logical run and is the directory that run writes to. It is deliberately *not* the key
used to isolate persistent state (see `jaz_evals.isolation`), which is fresh per attempt.

`new_run_id` stamps the current time, so it never reproduces an earlier ID: re-entering a run -- a
replay-resume finding the trace it replays -- means passing that run's existing ID to `run_attempt`
directly, not asking for a new one.

IDs are `<utc timestamp>-<name>`: the timestamp first so a listing of `runs/<env>/<method>/` comes out in
chronological order, the name second so a run is findable by what a person called it.
"""

# A run ID carries a NAME, not just entropy: `<utc timestamp>-<slug>`. Without one, the only way to
# tell two runs apart in a directory listing is to remember which timestamp was which, and the
# alternative operators reach for -- encoding meaning into a hex suffix -- is fragile, since the
# value has to stay valid hex.
#
# The random suffix survives only for an unnamed run, where nothing else separates two runs started in the
# same second. A named run has none, so two runs sharing a name and a second would share a directory --
# which `run_evaluation` refuses rather than interleaving their attempts.

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

_SUFFIX_BYTES = 6
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
# Long enough for a descriptive name, short enough that a run directory still reads at a glance in a
# terminal. Not a filesystem limit: with the 16-character timestamp a component tops out around 77
# bytes, far under the 255 every filesystem here allows, and `attempt-N` is a child directory rather
# than a suffix on this name.
_MAX_SLUG_LENGTH = 60

# Separators (both platforms') and control characters -- everything that stops a string being one
# directory name. Not a whitelist: an ID a caller supplies directly is theirs to choose, and spaces
# and punctuation are fine in a directory name.
_UNSAFE_IN_A_PATH = re.compile(r"[/\\\x00-\x1f]")
# Runs of anything that is not alphanumeric collapse to a single hyphen. Unicode-aware on purpose:
# an ASCII-only class silently *deleted* every other script's letters, so "Прогон 1" slugged to "1" --
# an unfindable, collision-prone name, which is the exact failure this module exists to remove -- and
# "实验一" slugged to nothing and was then rejected as having "no alphanumeric characters", which was
# false about the input. A directory name is UTF-8 on every platform this runs on, so the restriction
# bought nothing.
_NOT_SLUGGABLE = re.compile(r"[\W_]+", re.UNICODE)


def slugify(name: str) -> str:
    """Return `name` as a lowercase hyphen-separated fragment usable in a directory name.

    Collapses every run of non-alphanumeric characters to a single hyphen, trims hyphens from both
    ends, and truncates. Letters and digits of any script are kept. Returns `""` when `name` has
    none.
    """
    # Slugged rather than used verbatim, even though `check_run_id` would accept spaces and commas:
    # `runs/` is walked by this repo's own tooling and read in a terminal, and a directory whose name
    # needs quoting to `cd` into is a worse deal than a lossy one. The transform is deliberately
    # ASCII-only and lossy -- it produces a name to *find* a run by, not a record of what was typed;
    # `provenance.json` keeps the config and the run's own identity for anything that must be exact.
    slug = _NOT_SLUGGABLE.sub("-", name.lower()).strip("-")
    return slug[:_MAX_SLUG_LENGTH].rstrip("-")


def new_run_id(name: str | None = None, now: datetime | None = None) -> str:
    """Return a run ID for a run called `name`, of the form `<utc timestamp>-<name>`.

    `name` is free text -- "minimal meta v8, rep 1" becomes `20260822T051200Z-minimal-meta-v8-rep-1`.
    An unnamed run gets a random suffix instead, so two unnamed runs never collide. `now` is for
    tests; it defaults to the current UTC time.

    Raises `ValueError` if `now` is naive, or if `name` has no letters or digits to name a run with.
    """
    # A naive `now` is rejected rather than assumed to be UTC: `astimezone` would read it as system local
    # time, so a caller reaching for the legacy `datetime.utcnow()` would silently get an ID stamped with
    # the local offset.
    moment = datetime.now(UTC) if now is None else now
    if moment.utcoffset() is None:
        raise ValueError("`now` must be timezone-aware; a naive datetime would be read as local time")
    stamp = moment.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)
    if name is None:
        return f"{stamp}-{secrets.token_hex(_SUFFIX_BYTES)}"
    slug = slugify(name)
    if not slug:
        # Silently falling back to a random suffix would hand back an ID bearing no trace of the name
        # the caller asked for, and they would find out by going looking for the directory.
        raise ValueError(
            f"run name {name!r} has no letters or digits in any script; a run name has to survive as "
            "a directory name, so it needs at least one"
        )
    return f"{stamp}-{slug}"


def check_run_id(run_id: str) -> str:
    """Return `run_id` unchanged if it can name a directory, raising `ValueError` if it cannot.

    Rejects the empty string, `.` and `..`, path separators, and control characters.
    """
    # A run ID becomes one path component (`runs/<env>/<method>/<run_id>`), and this is the only thing
    # standing between a caller's string and `mkdir(parents=True)`. It must live here rather than
    # fall out of some other check: when the guard is a side effect of validating the ID's *format*,
    # removing that check for unrelated reasons takes the path guard with it, and `../../escaped`
    # then writes outside the pair directory while `.` collapses the run into it. Deliberately a
    # safety check and not a format one:
    # IDs from `new_run_id` are already safe, and what anyone else calls a run is their business.
    if not run_id or run_id in {".", ".."} or _UNSAFE_IN_A_PATH.search(run_id):
        raise ValueError(
            f"invalid run ID {run_id!r}: a run ID names one directory, so it cannot be empty, "
            "'.' or '..', or contain a path separator or control character"
        )
    return run_id
