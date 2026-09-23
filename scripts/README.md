# `scripts/`

Everything needed to rebuild the paper's tables and figures from runs you produce yourself (the
paper is *Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity*), plus the
one script that runs AppWorld's own baseline agent.

These read `runs/`, which is **gitignored** and ships with nothing in it. You produce it by running
the configs in `configs/` (see the top-level `README.md`); then you point these scripts at it.

## Build the paper's artifacts

A table and a set of curves per domain, plus the far-recall and combined figures — seven scripts:

| script | writes | notes |
| --- | --- | --- |
| `build_appworld_table.py` | `tables/appworld_results.tex` | Every figure is computed from run artifacts so none is hand-typed. |
| `plot_appworld_curves.py` | `tables/appworld_curves_<stat>[_relative].pdf` | Per-piece curves over the AppWorld queue, one line per arm. Default stat is `median`. |
| `build_stulife_table.py` | `tables/stulife_results.tex` | The StuLife counterpart. Pass/Score over all scored tasks and over the far-recall subset, plus cost. |
| `plot_stulife_curves.py` | `tables/stulife_curves_<stat>[_relative].pdf` | Per-piece curves across the StuLife episode. Cut over the 939 **scored** tasks — the 345 trigger rows carry no score. |
| `plot_stulife_far_recall_curves.py` | `tables/stulife_far_recall_curves.pdf` | The same axis restricted to the **far-recall** tasks (gap > 50) — the subset the long-horizon claim rests on. All five arms, matching `stulife_results.tex` and both combined figures. |
| `plot_combined_curves.py` | `tables/combined_curves.pdf` | AppWorld and StuLife panels side by side under one shared legend. |
| `plot_combined_cost.py` | `tables/combined_cost.pdf` | Score against cost, both domains in one panel. |

`_curves.py` owns how a panel *looks* (`Series`, `draw_panel`, `legend_below`, colours, panel size) so
the figures cannot drift apart; `_artifacts.py` owns the data and verification half. The two combined
scripts take `--runs-root` **and** `--appworld-runs-root`, because the two domains' runs may live in
different checkouts.

The committed artifacts in `tables/` are the ones in the paper. Each domain has one manifest —
**`scripts/appworld_runs.json`** / **`scripts/stulife_runs.json`** — mapping each arm to its
run-directory globs, and every script for that domain reads it -- four readers for AppWorld (table,
curves, and the two combined figures) and five for StuLife (table, curves, far-recall, and the same
two combined figures, which load both manifests). So a domain's table and its figures cannot be built
from different runs.

**The shipped manifests name our run directories, which you will not have.** That is what
`--runs-manifest` is for: copy the manifest, replace each glob with your own run directories, and
pass it. Globs resolve against `--runs-root` (default: the repo root) and must be relative —
`Path.glob` refuses an absolute pattern, so put the prefix in `--runs-root`.

The two combined scripts read both domains, so they spell those four flags per domain instead:
`--stulife-manifest`/`--runs-root` and `--appworld-manifest`/`--appworld-runs-root`. Two roots
because the two domains' runs need not live in the same tree.

The arms in the two manifests are exactly the arms this repo ships configs for: four AppWorld
(`configs/appworld_*.yaml`) and five StuLife (`configs/stulife_*.yaml`). AppWorld's own baseline is
the tenth arm and is deliberately absent from the manifest — its artifacts are not under `runs/` at
all, so the generators reach it through `--official-root` instead. See *The one row you cannot
recompute* below.

A **rep** means different things in the two domains, which is why the manifests look different. The
AppWorld arms are several run directories of one attempt each (`…-rep-*`); the StuLife arms are one
run directory launched with `--attempts 3`. Both readers take every attempt of every matched run, so
either shape works.

The two tables need nothing but a Python interpreter. The five figure generators need matplotlib
and seaborn, which are the optional `plots` extra — `uv run --extra plots` is what puts them on the
path:

```bash
uv run python scripts/build_appworld_table.py --runs-manifest mine.json --out mytable.tex
uv run python scripts/build_stulife_table.py  --runs-manifest mine.json --out mystulife.tex
uv run --extra plots python scripts/plot_appworld_curves.py --runs-manifest mine.json --out-dir myfigs/
uv run --extra plots python scripts/plot_stulife_curves.py  --runs-manifest mine.json --out-dir myfigs/
uv run --extra plots python scripts/plot_stulife_far_recall_curves.py --runs-manifest mine.json --out myfar.pdf
uv run --extra plots python scripts/plot_combined_curves.py --stulife-manifest mine.json \
    --appworld-manifest mineaw.json --out mycombined.pdf
uv run --extra plots python scripts/plot_combined_cost.py --stulife-manifest mine.json \
    --appworld-manifest mineaw.json --out mycost.pdf
```

Pass `--out` / `--out-dir` when reproducing, or the artifacts built from your runs overwrite the
committed ones and you lose the comparison you were trying to make.

`scripts/_artifacts.py` holds what they share — the manifest loader, the `--check` reporting, the
best/second-best marking, and the half-up rounding both tables use.

### The one row you cannot recompute

The AppWorld table's `official baseline` row comes from AppWorld's **own** evaluation JSON, written
by `run_appworld_official_react.sh` under AppWorld's data root — an absolute, machine-specific path,
not `runs/`. When those artifacts are absent the generator falls back to constants recorded from our
run, and **says so in the generated header**:

```
%   - The official-baseline row was NOT recomputed on this machine (its artifacts live
%     outside the repo and were absent); recorded constants in the generator were used.
```

So on a fresh machine that row is ours, not yours, until you run that script and point the
generators at its output. The flags are `--official-root` (the experiment outputs, which carry the
per-task scores) and `--official-logs` (the per-rep run logs, which carry the cost) -- the table needs
both, since supplying only the first still falls back for cost. `plot_appworld_curves.py`,
`plot_combined_curves.py` and `plot_combined_cost.py` take `--official-root`; `plot_combined_cost.py`
takes `--official-logs` too, being the only figure that plots that arm's cost.

Those three figures need per-task outcomes, which the recorded constants do not carry, so without
`--official-root` they drop the official line entirely (with a note on stderr) rather than drawing a
line they cannot support -- which means `--check` reports `Figure CONTENT differs` for them, correctly.
The table is unaffected: it falls back to the constants and rebuilds identically. Every other row and
line comes from your runs either way.

### `--check`

All seven take `--check`: rebuild in memory (figures into a scratch directory), compare against the
committed artifact, write nothing, exit 1 if they differ. It reports **what** differs, separating the
two cases that mean opposite things:

- *numbers identical, provenance header differs* — same results, built from a different tree or
  manifest. This is what you get when you rebuild our **tables** from your own runs directory and the
  numbers reproduce. Nothing is wrong, and it still exits 1.
- *content differs* — the artifact no longer matches those runs, with a unified diff of the lines
  that moved (or, for a figure, a note that it changed and by how much).

Only the two `.tex` tables can report the first: it comes from the text-artifact comparison, and a
PDF has no provenance header to separate. A figure reports either *is up to date* or *Figure CONTENT
differs* — see the official-baseline note above for the one case where the latter is expected.

For the two `<stat>`-parameterised figures it builds **both** summary statistics, since all four of
each one's PDFs are committed and a gate that only looked at the selected `--stat` could pass while
the others were stale. `--print` still
prints under `--check`.

It reads run directories, so it only means something where the runs are. It is not a CI check.


## Run AppWorld's own baseline

| script | what |
| --- | --- |
| `run_appworld_official_react.sh` | Reps of AppWorld's **own** baseline agent on `test_challenge`. Deliberately not routed through `jaz-evals` — this arm exists to measure upstream, so running it through our harness would re-measure our REPL and prompt assembly instead. Its results feed the `official baseline` row of the AppWorld table, which is read from AppWorld's own evaluation JSON rather than from `runs/`. |

## Publish the artifacts

One artifact: **`release_traces.py`** packs every trace file of every reported arm into a single
encrypted archive. Nothing is filtered and nothing is rewritten — the files are bit-identical to what
the runs recorded, so the same download both reproduces the committed tables and figures and shows
what the agents actually did.

**Why encrypted, given that only one of the two benchmarks requires it** — both upstreams were asked
directly, and they answered differently:

- **StuLife** traces may be released **in plaintext**, task descriptions included, with citation.
- **AppWorld** traces may be released **encrypted**, any scheme. The authors' concern is test
  contamination rather than ownership: the meta *prints* the grader's assertions into its own output,
  and an LLM that reads those off the open internet has an unearned advantage on the benchmark.
  Measured on one attempt of the JAZ meta arm, two ways because the meta reformats what it prints:
  20 of that run's ground-truth assertion strings appear verbatim in its `agent.atif.json`, 44 times
  over; a looser scan for `assert ...` in the same file finds 300 distinct forms.

Encrypting everything is stricter than StuLife asks and exactly what AppWorld asks, so the two
permissions compose into one archive rather than two:

```bash
uv run python scripts/release_traces.py --out dist/traces.tar.zst.enc \
    --runs-root . --runs-root /path/to/another/checkout
```

`--runs-root` repeats, and may be given once, because the two domains' runs need not live in one
checkout: the published archive was built from two. The point of the design is that a reader gets one
file, one command and one citation rather than two downloads to stitch together.

The build compresses **before** encrypting (ciphertext does not
compress; the other order turns a ~5% archive into a ~100% one), writes a `.sha256` beside the
output, and gives the archive its own `README.md` carrying the decrypt command and both citations —
so a reader who unpacks it without ever seeing this page still has them.

Absolute paths are left as recorded, because AppWorld's API endpoints and its simulated filesystem
are themselves absolute paths and no rewriter can tell those from the build machine's. That is a
decision rather than an oversight, and it ships more than the traces: each run's `provenance.json`
carries the build machine's hostname, usually `launch.cwd`/`launch.argv` with the operator's
username, and a truncated fingerprint of each API key that was set. None of it is a credential and
none of it is reversible, but anyone reusing this script elsewhere should make the call themselves.

The passphrase is published, and that is the point: it exists so the archive can be read, and the
encryption exists so the contents are not swept into a training corpus by a crawler that never
intended to. AES-256-CBC with PBKDF2 at 100k iterations, matching AppWorld's own scheme, so `openssl`
is all a reader needs.

The archive's own `README.md` carries the round trip back: decrypt, unpack, and rebuild every table
and figure above with `--runs-root` pointed at the unpacked tree. Keep that text and this file in
step — it is the only reproduction instruction a downloader who never sees this repository gets, and
`readme()` in `release_traces.py` is where it lives.
