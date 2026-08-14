# findings/ — one thing that turned out to be true

`status: provisional | measured | refuted | superseded`

A finding is a claim with evidence attached. The bar is the repository's bar, because a note that
gets quoted into `reports/` inherits whatever rigour it was written with:

- **The number, flat.** `0.67%`, not "individualisation appears weak".
- **Where it came from** — a file and a command, so it can be re-run.
- **What was running at the time.** Never conclude from a number measured while another job was
  running (CLAUDE.md). If you did not check `ps`, write that you did not.
- **What would refute it.** No refutation condition means it is an opinion, not a finding.
- **The artifact and the code that generates it are two objects.** Say which one you measured.

`provisional` is for something you believe on one observation. It is a real status and it is better
than not writing the note; it is not a licence to quote the number elsewhere.

`unmeasured` is not a status here because it is not a finding. A test that will not finish, a gate
that could not run, a sweep that timed out — those are `questions/` with `status: open`, or a line
in `reports/known_issues.md`.
