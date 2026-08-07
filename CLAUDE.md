# Working in this repository

Repo: https://github.com/JacobFV/sc-wbd — site: https://sc-wbd.pages.dev

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
  validates links first.
- `paper/` builds with `make paper` (tectonic, at `~/.local/bin`).

## Commands

    make health          is training alive / complete
    make test            fast suite (slow deselected)
    make test-slow       the rest; hours
    make paper           rebuild the PDF
    make site            build + link-check
