# Working in this repository

Repo: https://github.com/JacobFV/sc-wbd — site: https://sc-wbd.pages.dev

## Do not defer. Do not fall back. Fix it.

When you find a defect, fix it in the turn you find it. Filing it, deferring it
to the next run, or routing around it is not a neutral choice — in this repo it
is the single most expensive habit there is, and the evidence is in the file
below.

- **R12 has been "a design call for the next run" for three runs.** It now blocks
  the paper's headline claim, `scale_prolongations` is pinned empty by a test,
  and five tests are red. Deferring converted a two-hour decision into a
  structural blocker.
- **ISSUE-008 was visible from step 1** — `real_bold_nll` at 21.7 and climbing.
  It was not read until 46% of a 25-hour run. The whole fMRI likelihood is void.
- **ISSUE-010 was "fixed" four times by falling back** — `ckpt_every`, then
  `log_every`, then a logger redirect, then `out_dir` — and each partial fix left
  something still writing to a production path. It destroyed a checkpoint and a
  published report before the fifth attempt redirected the directories instead of
  enumerating the outputs.

The pattern in all three: a fallback that makes the symptom quieter leaves the
defect live and removes the pressure to fix it. A plausible number is worse than
an obviously broken one, because nobody looks again.

So: no `try/except` that swallows, no default that stands in for a measurement,
no "TODO(next run)", no disabling a check to make a run start. If something
genuinely cannot be fixed now, it gets a named ISSUE with what would discharge
it, a guard that fails loudly, and a line in the model card — and you say plainly
in the turn that you did not fix it and why. Silence is the failure mode.

## Mistakes made repeatedly in this repo

Each of these cost real time or real data. They are listed because they recur,
not because they are interesting.

### Shell

- **`pkill -f <pattern>` matches the shell running it.** Killed the turn five
  separate times (exit 144). Use explicit PIDs:
  `for p in $(ps -eo pid,args | grep "[p]attern" | awk '{print $1}'); do kill $p; done`
- **`$?` must be read on its own line, immediately.**
  `echo "$(basename $f) exit=$?"` records *basename's* status. A whole sweep
  reported `exit=0` for runs that `timeout` had killed. Guarded by
  `tests/release/test_shell_exit_capture.py`.
- **`git checkout -- <dir>` deletes everything uncommitted in it.** This
  destroyed the tail of run 2's training log — steps 4686→8700, unrecoverable,
  because a long job had been writing there and nothing had committed it. Use
  `git stash` or copy aside. Before restoring a path, ask what is writing to it.
- **`git clean -xff` would delete worktrees.** `-xfd` skips them; the double
  force does not.

### Measurement

- **Never conclude from a number measured while another job was running.**
  Three wrong conclusions in one session: a "300 s timeout" that passes in 65 s
  alone, "38 failing files" that was 10, and a cost-driver claim that was pure
  contention. Run `ps` first. Every time.
- **A test file that will not finish is not a test file that fails.** Say
  "unmeasured". `slow` is a registered marker, deselected by default; `make
  test-slow` runs the rest.
- **Read the whole error line, not the field you came for.** `exit=0` beside two
  `F` characters is a contradiction. `454` in a shape mismatch is not `62`.
- **A same-length mutation can leave stale bytecode behind.** Python validates
  a `.pyc` on `(size, mtime-seconds)`. `SIMULATOR_TIER = 4` → `= 1` → restore,
  all inside one second, reuses the *mutant's* bytecode: the guard then "failed"
  after the file was already restored, and the mutation result was not real.
  Run mutation sweeps under `PYTHONDONTWRITEBYTECODE=1`, and always re-run the
  restored file — a red on the restore means the sweep, not the guard.
- **The artifact and the code that generates it are two objects.** Green tests
  on the generator say nothing about bytes already published. Diff published
  output against freshly generated output before republishing — a routine
  republish nearly replaced the model card's headline figure with one describing
  a run that never happened.

### This codebase specifically

- **Several classes share a name across modules.** `derive_families`
  (`scwbd.anatomy.families` vs `scwbd.foundation.families`), `ClaimManifest`
  (`schema.claims` vs `foundation.manifest`), `RegionFamily` (`family_id` vs
  `name`), and R12 itself (`R12Violation` vs `CompilerRefusal`, unrelated
  hierarchies). **Read the import before concluding anything about a symbol.**
  See `ARCHITECTURE.md` O-3/O-7.
- **`--out` moves checkpoints, not logs.** Logs are keyed by `train.run_name`,
  so a scratch run appends to the production log. Set a distinct run name.
- **An unmatched glob is an empty permission set, not an error.** This is how
  88.8% of run 2's parameters went untrained while the loss fell. Guarded by
  `tests/foundation/test_card_patterns_reach_the_model.py`.
- **A guard is not accepted until it has been made to fail on purpose.**
  Mutation-test it, and say in the commit which mutation you ran.

### Writing

- Open with what the thing does, in the indicative. Never open a page, section,
  or paragraph with a negation of a claim nobody made.
- A caveat goes after what it qualifies, once.
- State measured numbers flat. "51.7%" is a measurement; "suggests that
  orientation may carry more information" throws away the result.
- The refusal machinery, claim manifests and integrity tiers are plumbing so the
  claims can be trusted. They are not the product and should be nearly invisible
  in prose.

## Layout

- `scwbd/` — the package. `foundation/` trains, `infer/` is the identifiability
  laboratory, `observe/` is forward models, `anatomy/` is the prior,
  `schema/`+`compiler/` are the type system, `release/` publishes.
- `configs/curriculum/source_cards/` is read by **training**;
  `configs/source_cards/` by **release**. They are not the same directory.
- `site/content/` → `site/build.py` → `site/_build/` → rsync to `docs/` →
  `npx wrangler pages deploy docs --project-name=sc-wbd`. `site/check.py`
  validates links first. Each arrow is a make target: `site`, `site-stage`,
  `site-deploy`.
- `paper/` builds with `make paper` (tectonic, at `~/.local/bin`).

## Commands

`make help` lists every target, grouped. Names are `<area>` or
`<area>-<action>` — `test-*`, `site-*`, `paper-*`, `video-*`, `release-*` — and
the bare area name is the default one. The ones used daily:

    make health          is training alive / complete
    make test            fast suite (slow deselected)
    make test-slow       the rest; hours
    make paper           rebuild the PDF
    make site            build + link-check
    make site-stage      copy the checked build into docs/, and diff it
    make site-deploy     publish docs/ to Cloudflare Pages

## Running several agents on this one worktree

There is ONE branch and ONE worktree. Agents do not get their own; they all
commit to master. That is deliberate — divergent branches were worse — but it
means two failure modes that have both already happened:

- **`git add -A` sweeps another job's in-flight files into your commit.** Three
  agents hit this in one session; one caught it before pushing and reset. Stage
  BY PATH, always. `git commit <paths>` or `git add <paths>`, never `-A`, never
  `.`, in a shared tree.
- **Mutation testing edits shared source.** A mutation is live on disk for the
  seconds it takes to run one file, and any other agent's test run in that
  window imports the mutant. Results from an overlapping window are not
  evidence. Say which window you ran in, or re-run after the other job lands.

And append-only edits to `reports/known_issues.md` have twice produced a stale
`Status:` line and once a duplicated heading. Update the entry AND the status
index at the top of that file, in the same commit.
