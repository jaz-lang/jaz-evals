"""Pack the raw agent traces behind the paper into one encrypted, citable archive.

This is the whole published artifact: every trace file of every reported arm, unfiltered, so a reader
can both reproduce the paper's tables and figures and see what the agents actually did.

There was once a second, plaintext "scoreboard" bundle carrying only the fields the generators read.
It is gone. Once the traces themselves ship raw, it was a strict subset of this archive whose only
distinction was needing no passphrase -- and keeping a second copy of the same numbers, filtered by
different rules, is how two artifacts start disagreeing about what the run said.

    uv run python scripts/release_traces.py --out dist/traces.tar.zst.enc

WHY IT IS ENCRYPTED, AND WHY THAT COVERS BOTH BENCHMARKS. Both upstreams were written to by email
in September 2026 and both replied within the week, differently; encryption is what satisfies the two
answers at once. The replies are held by this repo's maintainers rather than reproduced here, since
they are private correspondence -- so what follows is a summary, and anyone relying on it for their
own release should ask the upstreams themselves rather than inherit a permission granted to us.

- **AppWorld** asked that AppWorld data or traces be released encrypted, any scheme. The reason is
  not ownership but TEST CONTAMINATION -- a model that has read the ground-truth tests off the open
  internet has an unearned advantage on the benchmark. Their license says the same for the protected
  portion and its derivatives, and a trace is a derivative: the meta-agent prints the grader's
  assertions into its own output.
- **ELL-StuLife** granted permission to release our traces publicly INCLUDING the task descriptions
  they contain, on two conditions: that the benchmark and its paper are cited, and that any
  third-party material inside a trace stays under its own licence.

So StuLife permits plaintext and AppWorld requires ciphertext. Encrypting everything is stricter than
StuLife asks and exactly what AppWorld asks, which is why this is one archive rather than two: a
reader gets one file, one command and one citation, and neither upstream's request is bent to fit the
other's.

NOTHING IS REWRITTEN. Every file is copied byte for byte, and the archive is bit-identical to what
the runs recorded. A trace's whole value is being an unaltered record, so an artifact that quietly
differs from the run it claims to document is worth less than no artifact at all.

Absolute paths are left exactly as recorded. AppWorld's API endpoints and its simulated filesystem
are themselves absolute paths, so a rewriter cannot separate them from the build machine's without
guessing, and guessing wrong in either direction is worse than not guessing: it either corrupts the
benchmark content or leaves the path it meant to remove.

WHAT THAT MEANS SHIPS, stated plainly because "raw" is a decision and not an absence of one. Every
`provenance.json` carries the build machine's `hostname`, and most carry `launch.cwd`/`launch.argv`
with the operator's username and a truncated SHA-256 fingerprint of each API key that was set. The
fingerprints are not reversible and are useless to a reader, but they are stable identifiers, and the
hostname and username identify the machine the runs were made on. None of that is a credential, and
the paper names its authors -- but it is published, and someone reusing this script elsewhere should
decide for themselves rather than inherit the decision.

WHAT LOOKS LIKE A LEAK AND IS NOT. AppWorld's simulated filesystem uses fictional personas --
`/home/cody`, `/home/cesar` and friends, tens of thousands of times over. Those are benchmark
content, not anyone's real home directory.
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from _artifacts import REPO, display, load_manifest
from build_appworld_table import DEFAULT_MANIFEST as APPWORLD_MANIFEST
from build_appworld_table import OFFICIAL_LOGS, OFFICIAL_RELPATH, OFFICIAL_ROOT, OFFICIAL_STAMP
from build_stulife_table import DEFAULT_MANIFEST as STULIFE_MANIFEST

#: The published passphrase. A secret one would defeat the purpose -- the archive is meant to be read
#: -- and AppWorld's own encrypted bundles carry their key in the source for the same reason. It
#: raises the cost of ingesting ground truth into a training corpus from "crawl it" to "know it is
#: there and decrypt it deliberately", which is the whole of what was asked for.
PASSPHRASE = "jaz-evals-traces"

#: AES-256-CBC with PBKDF2-HMAC-SHA256 at 100k iterations: the same construction AppWorld uses for
#: its own protected portion (`appworld/common/crypto.py`), so this is the scheme least surprising to
#: the community that asked for it -- and `openssl` is the only thing a reader needs to undo it.
_CIPHER = ("-aes-256-cbc", "-pbkdf2", "-iter", "100000", "-salt")

#: The only `--out` suffix the generated decrypt command can be written for: one strip gives the
#: compressed tar, two give the tar.
_SUFFIX = ".tar.zst.enc"


# Where the official ReAct baseline lands inside the archive: its own subtree in AppWorld's own
# layout, rather than beside the harness arms pretending to be a `runs/` directory. Imported rather
# than restated, because `build_appworld_table.resolve_official` looks for it at exactly this path --
# two spellings of one location is how the table would quietly stop finding the archived copy and
# fall back to recorded constants.
OFFICIAL_DEST = OFFICIAL_RELPATH

#: Omitted from that baseline, and the only place this archive is a filtered copy rather than a
#: complete one. `dbs/` is the AppWorld simulator's per-task SQLite state and `checkpoints/` its
#: restore points: together about 3.5 GB of the 3.8 GB, they are the environment's internals rather
#: than any record of what the agent did. Everything that evidences behaviour or produces a reported
#: number -- the per-task transcripts, the per-task and aggregate evaluations, the per-rep cost logs
#: -- is copied verbatim. The README says so in as many words, because "raw" is a claim this archive
#: makes everywhere else and a reader must not have to diff a tree to discover the exception.
OFFICIAL_EXCLUDE = frozenset({"dbs", "checkpoints"})


def collect_official(staging: Path, root: Path, logs: Path, stamp: str) -> int:
    """Copy the official ReAct baseline's three reps into `staging`. Returns the reps copied.

    Raises:
        SystemExit: a rep's directory or its cost log is missing, so the arm would ship incomplete.
    """
    # Checked per rep rather than once, and fatal rather than skipped. This arm was absent from the
    # first release precisely because nothing noticed it was absent: the manifests drive `collect`,
    # this tree is not in them, and a missing-but-silent arm is how a published artifact ends up
    # unable to support one of its own rows.
    reps = 0
    for rep in (1, 2, 3):
        src = root / f"test_challenge_rep{rep}_{stamp}"
        if not src.is_dir():
            raise SystemExit(
                f"official baseline rep{rep} not found at {display(src)}\n"
                "pass --official-root at AppWorld's experiment-output root for that batch, "
                "or --no-official to build an archive without this arm"
            )
        log = logs / f"rep{rep}.log"
        if not log.is_file():
            raise SystemExit(
                f"official baseline rep{rep} has no cost log at {display(log)}\n"
                "its cost column is read from these logs; pass --official-logs or --no-official"
            )
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            if OFFICIAL_EXCLUDE & set(rel.parts):
                continue
            _copy(path, staging / OFFICIAL_DEST / src.name / rel)
        _copy(log, staging / OFFICIAL_DEST / "logs" / log.name)
        reps += 1
    return reps


def _copy(src: Path, dest: Path) -> None:
    """Copy one artifact byte for byte."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def collect(staging: Path, runs_roots: Sequence[Path]) -> int:
    """Copy every trace of every run the two manifests name, verbatim. Returns the attempt count."""
    # Driven by the same manifests the tables read, so the release covers exactly the arms the paper
    # reports -- not whatever happens to be lying in `runs/`.
    #
    # Several roots merged into one staging tree, because the two domains' runs need not live in the
    # same checkout -- ours do not. One archive is the point of the design (one file, one command,
    # one citation), so the merge happens here rather than leaving a reader to stitch two downloads
    # together.
    attempts = 0
    # Keyed on the relative path because that IS the staging path: two runs sharing one cannot both
    # be in the archive whatever we do here. So the key deduplicates and the resolved source decides
    # which case it is -- the same run reached twice (skip) or two different runs colliding (stop).
    # A collision is an operator error in `--runs-root`, and the wrong response is to keep the first
    # and ship an archive quietly missing an arm, in whichever order the roots happened to be given.
    seen: dict[Path, Path] = {}
    for runs_root in runs_roots:
        for manifest_path in (APPWORLD_MANIFEST, STULIFE_MANIFEST):
            for globs in load_manifest(manifest_path).values():
                for pattern in globs:
                    for run in sorted(runs_root.glob(pattern)):
                        if not run.is_dir():
                            continue
                        rel, source = run.relative_to(runs_root), run.resolve()
                        if rel in seen:
                            if seen[rel] != source:
                                raise SystemExit(
                                    f"two different runs claim {rel}:\n"
                                    f"  {display(seen[rel])}\n  {display(source)}\n"
                                    "pass --runs-root roots that do not overlap"
                                )
                            continue
                        seen[rel] = source
                        for src in sorted(run.rglob("*")):
                            if src.is_file():
                                _copy(src, staging / src.relative_to(runs_root))
                        attempts += len(list(run.glob("attempt-*")))
    return attempts


_FIGURES_WITH_OFFICIAL = """\
The five figure generators additionally need matplotlib and seaborn, which the repository installs as
an extra -- so run these through `uv` rather than a bare interpreter, or they will not see it. The
three that draw AppWorld's official baseline are pointed at its subtree:

    OFFICIAL="$DIR/runs/appworld/official_react"
    uv run --extra plots python scripts/plot_stulife_curves.py --check --runs-root "$DIR"
    uv run --extra plots python scripts/plot_stulife_far_recall_curves.py --check --runs-root "$DIR"
    uv run --extra plots python scripts/plot_appworld_curves.py --check --runs-root "$DIR" \\
        --official-root "$OFFICIAL"
    uv run --extra plots python scripts/plot_combined_curves.py --check --runs-root "$DIR" \\
        --appworld-runs-root "$DIR" --official-root "$OFFICIAL"
    uv run --extra plots python scripts/plot_combined_cost.py --check --runs-root "$DIR" \\
        --appworld-runs-root "$DIR" --official-root "$OFFICIAL" --official-logs "$OFFICIAL/logs"

All five rebuild exactly, and report `is up to date` for every figure they write."""

_FIGURES_WITHOUT_OFFICIAL = """\
The five figure generators additionally need matplotlib and seaborn, which the repository installs as
an extra -- so run these through `uv` rather than a bare interpreter, or they will not see it:

    uv run --extra plots python scripts/plot_stulife_curves.py --check --runs-root "$DIR"
    uv run --extra plots python scripts/plot_stulife_far_recall_curves.py --check --runs-root "$DIR"
    uv run --extra plots python scripts/plot_appworld_curves.py --check --runs-root "$DIR"
    uv run --extra plots python scripts/plot_combined_curves.py --check --runs-root "$DIR" \\
        --appworld-runs-root "$DIR"
    uv run --extra plots python scripts/plot_combined_cost.py --check --runs-root "$DIR" \\
        --appworld-runs-root "$DIR"

The two StuLife generators rebuild exactly and report `is up to date` for all five of their PDFs. The
other three will report `Figure CONTENT differs`, and that is expected too, for the one reason this
archive cannot fix."""

_NO_OFFICIAL_SECTION = """\
**The one arm that is not in here.** AppWorld's own ReAct baseline was run through AppWorld's harness
rather than ours, and this archive was built without it. The AppWorld *table* reproduces anyway --
that row falls back to constants recorded from the original run, and the table's header says so. The
three *figures* that draw it (`appworld_curves_*`, `combined_curves`, `combined_cost`) instead omit
the line when its artifacts are absent, so what they rebuild really is a different picture, and
`--check` is right to say so. Pass `--official-root` at an AppWorld experiment output directory of
your own to draw that line."""


def _official_section(stamp: str) -> str:
    """The README's account of the official baseline's subtree, for a build that includes it."""
    # The layout is spelled out in full, down to the per-task `misc/` and `version/` directories,
    # because this subtree is the one place the archive is filtered: a reader deciding whether what
    # is missing matters needs to know exactly what is present, not a representative sample of it.
    return f"""\
**The official baseline, and the one place this archive is filtered.** AppWorld's own ReAct baseline
was run through AppWorld's harness rather than ours, so it is not a `runs/` directory and does not
look like the other arms. It is under `runs/appworld/official_react/`, in AppWorld's own layout:
three reps named `test_challenge_rep{{1,2,3}}_{stamp}`, each holding

- `configs/` -- the run's own configuration (`test_challenge.json`, `test_challenge.jsonnet`);
- `evaluations/` -- `test_challenge.json`, whose `individual` block is what that row's scores are
  computed from, and its text rendering `test_challenge.txt`;
- `tasks/<task_id>/`, one per task, holding `logs/` (the transcript: `lm_calls.jsonl`,
  `api_calls.jsonl`, `environment_io.md`, `logger.jsonl`, `logger.log`), `evaluation/` (that task's
  grading), `misc/` (`usage.json` and a `finished` marker) and `version/` (`code.txt`, `data.txt`).

The per-rep cost logs are beside the reps, in `runs/appworld/official_react/logs/rep{{1,2,3}}.log`.

Two directories AppWorld writes per task are **omitted**: `dbs/` (the simulator's SQLite state) and
`checkpoints/` (its restore points). They are about 3.5 GB of that arm's 3.8 GB and hold the
environment's internals, not any record of what the agent did -- no reported number reads them. Every
file that is here is byte-identical; this subtree is simply not the whole of what AppWorld wrote.

With this subtree the AppWorld table recomputes that row from these artifacts rather than from
recorded constants, and `build_appworld_table.py` finds it under `--runs-root` with no extra flag."""


def readme(attempts: int, arms: int, passphrase: str, name: str, official_reps: int) -> str:
    """The archive's own front page: how to open it, how to cite it, what is inside."""
    # Every substituted value is shell-quoted and every count is derived, because this text is the
    # only instruction a downloader gets. A hardcoded filename is wrong for any `--out` the operator
    # chooses; a hand-rolled quote breaks on a name or passphrase containing a space or an
    # apostrophe; and a hand-counted arm total drifts the first time the manifests change.
    #
    # `--out` is required to end `.tar.zst.enc` (`main` rejects anything else), so the two stems are
    # a suffix strip rather than a guess. Guessing was wrong: `--out x.enc` used to emit
    # `zstd -d x && tar xf x`, which zstd refuses outright -- unknown suffix -- and which would have
    # been reading the still-compressed file even if it had not.
    quoted = shlex.quote(passphrase)
    tar_zst = shlex.quote(name[: -len(".enc")])
    tar = shlex.quote(name[: -len(".zst.enc")])
    name = shlex.quote(name)
    # The official baseline changes three passages, and every one of them is wrong for the other kind
    # of build: a `--no-official` archive that described `runs/appworld/official_react/` would send a
    # reader looking for a directory it does not contain, and an archive WITH it that still said the
    # AppWorld figures cannot be rebuilt would be understating its own contents.
    included = official_reps > 0
    headline = (
        "One subtree is a filtered copy rather than a complete one, and it is the\n"
        "official baseline -- see the last section."
        if included
        else "(The paper reports a tenth arm, AppWorld's own ReAct baseline, which this\n"
        "build does not include -- see the last section.)"
    )
    figures = _FIGURES_WITH_OFFICIAL if included else _FIGURES_WITHOUT_OFFICIAL
    # `attempts` counts the manifest arms' run directories only; the official baseline is not one, so
    # its reps are named separately rather than silently missing from the total.
    reps_note = f" (plus the official baseline's {official_reps} reps)" if included else ""
    official = _official_section(OFFICIAL_STAMP) if included else _NO_OFFICIAL_SECTION
    return f"""# Raw agent traces

The full transcripts behind the paper's results: {attempts} attempts across {arms} arms on two
benchmarks{reps_note}, byte for byte as recorded. Every file here is identical to what the run wrote; nothing is
rewritten or redacted. {headline}

## Decrypting

    openssl enc -d {" ".join(_CIPHER)} \\
        -in {name} -out {tar_zst} -pass pass:{quoted}
    zstd -d {tar_zst} && tar xf {tar}

The passphrase is published on purpose: it is here so the archive can be read, and the archive is
encrypted so its contents are not swept into a training corpus by a crawler that never intended to.
**Please do not republish the decrypted contents in plaintext** -- that would undo the one thing the
AppWorld authors asked for, which is that their ground-truth tests stay out of pretraining data.

## Citing

Using these traces means using two benchmarks. Both asked to be cited, and StuLife's permission to
release its task descriptions is conditional on it:

- AppWorld -- Trivedi et al., *AppWorld: A Controllable World of Apps and People for Benchmarking
  Interactive Coding Agents* (ACL 2024).
- StuLife -- *Building a Self-Evolving Agent via Experience-Driven Lifelong Learning: A Framework and
  Benchmark*, arXiv:2508.19005.

Material inside a trace that originates with a third party remains under that party's own license.

## What is in here

Unpacks as `runs/`, alongside this file. One directory per run, one per attempt beneath it, holding
whatever that harness recorded: the ATIF (Agent Trajectory Interchange Format) trace, the flat agent
log, the per-task result rows, and each harness's own artifacts. `runs/appworld/...` and
`runs/stulife/...` are the two domains, merged here into one tree.

AppWorld's simulated filesystem uses fictional people (`/home/cody`, `/home/cesar`, ...). Those are
benchmark content, not real home directories.

**One field in the smolagents trace is a running total, not a per-step value.** In
`runs/stulife/smolagents/.../smolagents_trace.jsonl`, `cached_input_tokens` was snapshotted
cumulatively when these runs were made: the rows read 0, 6912, 14848, 22784, ..., so summing them
overstates the true figure by three orders of magnitude (296,991,124,992 against 144,481,792), and
feeding a row's value to a pricing function prices that step as if the whole run's cache had been
read. Take successive differences, or take the last row. Every other token field in that file, and
the attempt's own `results.json` totals, are correct as recorded -- `results.json` reports the true
144,481,792. The harness has since been changed to record this per-step like its neighbours, so a
trace produced by today's code does not need this treatment; these do.

## Rebuilding the paper's tables and figures from this tree

Everything in the code repository's `tables/` directory -- both `.tex` tables and every figure -- is
computed from these traces by the generators in its `scripts/`. Nothing is hand-typed, so the archive
and the repository together are the whole path from transcript to published number. (The repository
is linked from this archive's record, under *is supplemented by*.)

Unpack this archive anywhere and point a generator at it with `--runs-root`. The two tables need
nothing but Python 3.10 or newer, and run from any working directory:

    DIR=/path/to/unpacked
    python3 scripts/build_stulife_table.py  --check --runs-root "$DIR"
    python3 scripts/build_appworld_table.py --check --runs-root "$DIR"

`--check` rebuilds the artifact in memory, reports how it differs from the committed one, and writes
nothing; drop it to overwrite the file.

**Both tables will report a difference, exit 1, and end with a `Rebuild with:` line. That is the
expected outcome, not a failure.** The report goes to stderr under the heading `differs from a fresh
build`, and the line that matters is the one after it:

    Numbers are IDENTICAL. Only the provenance header differs, so the committed
    artifact describes the same results, built from a different tree or manifest

The header records the absolute `--runs-root` a build was given, which is your path and never ours;
the tables themselves match line for line. Nothing needs rebuilding, and the `Rebuild with:` line can
be ignored.

{figures}

{official}

`scripts/README.md` in the repository lists every generator and the artifact it writes.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip().split("\n\n")[0],
        epilog="See the module docstring in this file for why the archive is encrypted.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="encrypted archive to write; the name must end .tar.zst.enc",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        help="directory the manifests' globs resolve against; repeat it when the two domains' runs "
        "live in different checkouts (default: the repo root)",
    )
    parser.add_argument("--passphrase", default=PASSPHRASE, help="override the published passphrase")
    # Same flag names as the generators that read this arm, so there is one spelling across the
    # suite for "where that batch lives".
    parser.add_argument(
        "--official-root",
        type=Path,
        default=OFFICIAL_ROOT,
        help="AppWorld experiment-output root holding the official ReAct baseline's three reps",
    )
    parser.add_argument(
        "--official-logs",
        type=Path,
        default=OFFICIAL_LOGS,
        help="directory of that baseline's per-rep run logs, which carry its cost",
    )
    parser.add_argument(
        "--no-official",
        action="store_true",
        help="build without the official baseline (the AppWorld table then falls back to "
        "recorded constants for that row)",
    )
    parser.add_argument(
        "--level", type=int, default=19, help="zstd level (default 19; traces compress to ~5%%)"
    )
    args = parser.parse_args()
    # Checked before any work, because the archive's README derives its decrypt command by stripping
    # this suffix. A name it cannot strip would ship a command that does not run -- and the README is
    # the only instruction a downloader gets, so there is nowhere for them to find the right one.
    if not args.out.name.endswith(_SUFFIX):
        raise SystemExit(f"--out must name a file ending {_SUFFIX}, got {args.out.name!r}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "traces"
        staging.mkdir()
        roots = [root.resolve() for root in (args.runs_root or [REPO])]
        attempts = collect(staging, roots)
        if not attempts:
            raise SystemExit("no runs matched under: " + ", ".join(str(r) for r in roots))
        arms = len(load_manifest(APPWORLD_MANIFEST)) + len(load_manifest(STULIFE_MANIFEST))
        official_reps = (
            0
            if args.no_official
            else collect_official(staging, args.official_root, args.official_logs, OFFICIAL_STAMP)
        )
        # The official baseline is an arm of the paper but not of the manifests, so it is counted
        # here rather than derived -- the manifests drive which `runs/` directories ship and this
        # tree is not one of them.
        (staging / "README.md").write_text(
            readme(
                attempts, arms + (1 if official_reps else 0), args.passphrase, args.out.name, official_reps
            )
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Built beside the destination and moved into place only on success. Writing straight to
        # `--out` truncates it the moment the file is opened, so a missing `openssl` or a mid-stream
        # failure replaces a good archive with a 0-byte one -- while the previous run's `.sha256`
        # sits next to it still claiming the old digest, which is the shape of a corrupt release
        # that looks complete.
        partial = args.out.with_name(args.out.name + ".partial")
        checksums = args.out.with_name(args.out.name + ".sha256")
        # Checked before anything is written, so a missing tool is a message rather than a wrecked
        # destination.
        for tool in ("tar", "zstd", "openssl"):
            if shutil.which(tool) is None:
                raise SystemExit(f"{tool} is not on PATH; it is needed to build the archive")
        try:
            # Compress BEFORE encrypting. Ciphertext is incompressible, so the other order turns a
            # ~5% archive into a ~100% one -- hundreds of megabytes against gigabytes. Streamed
            # rather than staged through files: the plaintext tar is ~8 GB, and writing it to disk
            # to read it straight back costs that twice over.
            #
            # `-C staging` with `.` as the member, not the directory itself: the archive must unpack
            # AS `runs/`, which is where the generators look by default. Packing the wrapper gave
            # `traces/runs/...` and a reader had to `cd` before anything worked.
            tar = subprocess.Popen(["tar", "-C", str(staging), "-cf", "-", "."], stdout=subprocess.PIPE)
            assert tar.stdout is not None
            zstd = subprocess.Popen(
                ["zstd", "-q", f"-{args.level}", "-T0", "-c"], stdin=tar.stdout, stdout=subprocess.PIPE
            )
            tar.stdout.close()
            assert zstd.stdout is not None
            with partial.open("wb") as out:
                enc = subprocess.Popen(
                    ["openssl", "enc", *_CIPHER, "-pass", f"pass:{args.passphrase}"],
                    stdin=zstd.stdout,
                    stdout=out,
                )
                zstd.stdout.close()
                enc.wait()
            # Tail first. A failure downstream closes the pipe and kills its feeders with SIGPIPE, so
            # checking `tar` first reports the victim rather than the cause -- an openssl exit of 3
            # was being announced as "zstd failed with exit -13".
            for name, proc in (("openssl", enc), ("zstd", zstd), ("tar", tar)):
                code = proc.wait()
                if code != 0:
                    how = f"signal {-code}" if code < 0 else f"exit {code}"
                    raise SystemExit(f"{name} failed with {how}; {display(args.out)} is unchanged")
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        # Only now does the destination change, and the digest is rewritten in the same breath so the
        # two can never describe different bytes.
        partial.replace(args.out)

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    checksums.write_text(f"{digest}  {args.out.name}\n")
    size = args.out.stat().st_size / 1_048_576
    print(f"wrote {display(args.out)} ({size:.0f} MB, {attempts} attempts)")
    print(f"      official baseline reps: {official_reps}")
    print(f"      {display(checksums)}")
    print(f"      passphrase: {args.passphrase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
