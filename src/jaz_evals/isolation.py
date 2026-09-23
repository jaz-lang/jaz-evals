"""Per-run isolation of persistent state.

Every persistent store a method touches -- agent state, memory stores, learned artifacts, env state, on-disk
scratch -- is keyed on an `Isolation` so that state persists *within* an attempt and is never visible *across*
attempts.

The key is fresh random bytes generated per attempt, not the logical run ID. A replay-resume reuses its run ID
by construction (that is how it finds the trace it replays), so keying on the run ID would land a resume in a
non-empty namespace; keying on a fresh value means every attempt -- first run, resume, or re-run -- starts
empty with no reset step. Stale namespaces are garbage to collect later, not correctness to enforce during a
run: deleting shared state while other attempts are running is a race.

An `Isolation` knows nothing about the run it belongs to; callers needing both put them side by side, as
`AttemptRecord` does.
"""

# `Isolation` once carried the run ID and validated its format on the way in -- a value it never read,
# gatekept by the one module with no stake in it. That made this module the sole reason a run ID had to
# look machine-generated rather than be a name a human chose, so both the field and the check are gone.
# The format check was also, accidentally, the only guard on the ID as a path component; that job moved
# to `run_id.check_run_id`, called where the ID actually becomes a directory.
#
# It also once derived names for callers: `namespace(component)` -> `jazeval-<key>-<component>` for
# name-addressed stores, and `ensure_path(component)` -> `<root>/<key>/<component>` for file-backed
# ones. Both are gone because no store wanted that shape. Every consumer composes its own identifier
# from `key`, since each target imposes its own format: AppWorld needs `<experiment>/<key>` (a slash,
# which the component validator rejected), Letta needs a Docker-legal `jazevals-letta-<key[:12]>`, and
# `prompt_cache_key` needs `<run_id>::<key>` trimmed to OpenAI's 64-char cap. `namespace` ended with no
# caller anywhere; `ensure_path` had exactly one, which wanted the path and not the `mkdir` (it creates
# its workspace with `parents=True` regardless) and now builds the path directly.
# Exposing `key` and `root` and letting callers compose is what the callers were already doing.

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

_KEY_BYTES = 8


class IsolationError(ValueError):
    """Raised when an isolation namespace cannot be derived."""


@dataclass(frozen=True)
class Isolation:
    """The isolation scope for one attempt.

    Holds no reference to the run it scopes.
    """

    key: str
    root: Path

    @classmethod
    def new(cls, root: Path | str) -> Isolation:
        """Create an isolation scope with a fresh key. Call once per attempt.

        `root` is resolved eagerly: harnesses chdir into task workspaces, so a relative root would
        silently retarget mid-run.
        """
        return cls(key=secrets.token_hex(_KEY_BYTES), root=Path(root).resolve())
