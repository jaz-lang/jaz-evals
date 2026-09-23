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


def readme(attempts: int, arms: int, passphrase: str, name: str) -> str:
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
    return f"""# Raw agent traces

The full transcripts behind the paper's results: {attempts} attempts across the {arms} arms that are
computed from run artifacts, on two benchmarks, byte for byte as recorded: nothing is filtered,
rewritten or redacted. (The paper reports a tenth arm, AppWorld's own ReAct baseline, which was run
through AppWorld's harness rather than ours -- see the last section.)

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
nothing but a Python 3 interpreter, and run from any working directory:

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
archive cannot fix.

**The one arm that is not in here.** AppWorld's own ReAct baseline was run through AppWorld's harness
rather than ours, so no trace for it exists in this archive. The AppWorld *table* reproduces anyway --
that row falls back to constants recorded from the original run. The three *figures* that draw it
(`appworld_curves_*`, `combined_curves`, `combined_cost`) instead omit the line when its artifacts are
absent, so what they rebuild really is a different picture, and `--check` is right to say so. Pass
`--official-root` at an AppWorld experiment output directory of your own to draw that line.

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
        (staging / "README.md").write_text(readme(attempts, arms, args.passphrase, args.out.name))

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
    print(f"      {display(checksums)}")
    print(f"      passphrase: {args.passphrase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
