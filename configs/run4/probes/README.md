# ISSUE-016 diagnostic arms — NOT run configs

These reproduce the four arms that identified why run 4's measured BOLD
likelihood degrades. They are kept because a number nobody can regenerate is a
number nobody can check, and moved out of `configs/run4/` because they sat beside
the run config where someone reaching for a config in a hurry could take one.

Arm A is not here: it is `configs/run4/scwbd-004.yaml` itself, the run as
launched. Its log is `reports/training/scwbd-004_train.jsonl` at the abort.

| file | intervention | `real_bold_nll` |
| --- | --- | --- |
| — (the run itself) | arm A, as launched | 3.21 @160, 12.96 @400 |
| `arm_b_balloon_frozen.yaml` | the five Balloon-Windkessel ODE constants frozen | 3.70 @160 — no better |
| `arm_c_trunk_frozen.yaml` | the shared trunk frozen, observation heads live | 1.92 @160, 1.86 @200, falling |
| `arm_d_bold_fast.yaml` | `bold.*` in its own group at 5× the stage LR | median 2.11, max 14.24 — oscillates |

Conclusion in `reports/known_issues.md` ISSUE-016 and `reports/RUN4.md`: the
shared trunk moves under the BOLD head, which is 4.13% of the mixture and
outvoted 23.2:1. Arm B rules out the ODE constants; arm C shows the head fits a
still trunk; arm D shows it can track a moving one but not stably.

**A first version of arm B was WRONG and is not preserved**, deliberately — it
dropped `bold.*` from the stage grants, which emptied `ds002336_real`'s
permission intersection and made the source be SKIPPED entirely. `real_bold_nll`
was then absent rather than flat, and a flat line would have "confirmed" the
hypothesis while measuring nothing. The version here grants the readout and noise
terms so the source stays admitted and only the ODE constants are frozen.
