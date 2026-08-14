# notes/ — the lab notebook

`reports/` is the record: curated, argued, and expected to stay true. `HANDOFF-<run>.md` is the
state of one run. This is the notebook underneath both — where work gets written down **as it
happens**, before anyone has decided what it means.

It exists because of one specific failure, and this repository has the receipts for it. An agent
learns something expensive — a mechanism, a dead end, a constraint that only appears at runtime —
and that knowledge lives in one context window and dies there. The next agent rediscovers it, or
does not, and repeats the dead end. CLAUDE.md's list of "mistakes made repeatedly" is exactly that
list: `pkill -f` matching its own shell, `$?` read after a command substitution, a same-length
mutation reusing stale bytecode. Every one of those was learned more than once.

`notes/` is where that knowledge survives a context reset.

## Why one file per note

Because agents work in **one shared tree**, and a single shared file is the thing that cannot
survive that. Two agents appending to a log conflict on every merge; two agents *creating*
`notes/findings/2026-08-14-the-gate-refuses-rather-than-guesses.md` and
`notes/findings/2026-08-14-fisher-needs-a-bound-map.md` merge silently and correctly.

So: **never append to a shared file from concurrent work. Create a new one.** The filename carries
the date and a slug, which makes collisions essentially impossible and the listing chronological.

This is the same reason CLAUDE.md forbids `git add -A`: the tree is shared, and the safe primitive
is the one that needs no coordination.

## What goes where

| directory | holds | written when |
|---|---|---|
| [`findings/`](./findings/) | one thing that turned out to be true, and what would refute it | after you measured it — not after you assumed it |
| [`questions/`](./questions/) | one idea, hunch or open question — **the parking lot** | the moment it is noticed, whether or not you will chase it |
| [`decisions/`](./decisions/) | one fork, the branch taken, and the branches rejected | when the fork is taken, *before* the work |

### `questions/` is a parking lot, and capture must stay cheap

The point is to **pile ideas up without opening a thread on each one**. Most of what gets noticed
mid-run is worth writing down and not worth doing now, and the alternative is that it lives in one
agent's context and dies there.

The rigour bar below applies to **findings, not to ideas**. A question note can legitimately be
three lines. Do not let "I have not thought it through" stop you writing it — unthought-through is
what the directory is for. Never delete an idea for being bad; mark it `status: rejected` with one
line saying why, because the next agent will otherwise have it again.

## Frontmatter

```yaml
---
date: 2026-08-14
author: <agent-slug | lead>
status: <see the vocabulary for your directory>
title: "The one-sentence claim"      # optional — overrides the `# heading`
task: CLAIM_GATES                    # optional — the scratch/<TASK>.md this came out of
related: [2026-08-14-other-note-slug]  # optional — slugs, without the .md
blocked_on: "no prospective perturbation dataset is held"   # optional
superseded_by: "run 5 measured it directly"                 # only when status is superseded
---
```

`status` is the load-bearing field, and **each directory has its own vocabulary**, because a
question and a measurement do not have the same life cycle:

| directory | statuses |
|---|---|
| `findings/` | `provisional` · `measured` · `refuted` · `superseded` |
| `questions/` | `idea` · `open` · `answered` · `rejected` |
| `decisions/` | `active` · `superseded` · `reversed` |

## The bar for a finding, in this repository specifically

This repo publishes negative results and refuses to publish unvalidated ones. A note that says
something is true has to carry the same evidence a `reports/` entry would:

- **The number, flat.** "51.7%", not "suggests that orientation may carry more information".
- **Where it came from.** A file and a command, so the next reader can re-run it.
- **What was running at the time.** CLAUDE.md: never conclude from a number measured while another
  job was running. If you did not check `ps`, say so.
- **What would refute it.** A finding with no refutation condition is an opinion.
- **`unmeasured` is a valid answer** and is not the same as a null result. A test file that will
  not finish is not a test file that fails.

## What does NOT go here

- Anything a `reports/` entry already carries — link to it instead.
- Run status. That is `HANDOFF-<run>.md`.
- In-flight task state. That is `scratch/<TASK>.md`; see below.

## `scratch/` is the other half

`notes/` is durable knowledge, written once and kept. `scratch/<TASK>.md` is the **live state of one
piece of work in flight** — objectives, what is done, what is next, what is blocking — and it is
edited continuously by whoever is working on that task.

The split matters on restart: a new agent reads `scratch/<TASK>.md` to learn *where the work is*,
and `notes/` to learn *what is already known*. Mixing them produces a status file nobody trusts and
a notebook nobody can search.
