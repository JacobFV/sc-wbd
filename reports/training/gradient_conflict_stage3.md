# Gradient conflict by module and source — Stage III

Deliverable requested by main. Source: `reports/training/mixture_III_sliced.json`,
written at the Stage III boundary (global step 4800). Conflict is measured every
10th step over a rolling window of 50 observations.

## Finding 1 — real EEG and the simulator disagree on the connectome operator

| pair | module | mean cos | min | frac neg | n |
|---|---|---|---|---|---|
| `eegmmidb_real` \| `sim_wholebrain` | **coupling** | **−0.259** | −0.999 | **0.64** | 22 |
| `eegmmidb_real` \| `sim_wholebrain` | assimilate | +0.001 | −0.446 | 0.44 | 50 |
| `eegmmidb_real` \| `sim_wholebrain` | local | +0.015 | −0.158 | 0.46 | 50 |
| `eegmmidb_real` \| `sim_wholebrain` | log_dt_scale | +0.016 | −0.144 | 0.38 | 50 |
| `eegmmidb_real` \| `sim_wholebrain` | context | +0.016 | −0.554 | 0.44 | 50 |
| `eegmmidb_real` \| `sim_wholebrain` | residual | +0.026 | −0.298 | 0.42 | 50 |
| `eegmmidb_real` \| `sim_wholebrain` | msg_readin | +0.032 | −0.432 | 0.42 | 50 |
| `eegmmidb_real` \| `sim_wholebrain` | msg_proj | +0.054 | −0.704 | 0.44 | 50 |

**Everywhere except `coupling`, the two sources are essentially orthogonal**
(|mean cos| ≤ 0.054, fraction-negative ≈ 0.42–0.46, i.e. indistinguishable from
chance). They are not helping each other and not fighting; they are working in
different subspaces.

**`coupling` is the exception, and it is the module that matters most** — 4,946,799
parameters, the largest in the model, and the one carrying the connectome. Real EEG
and the simulator pull it in opposing directions 64% of the time.

**The action was enforced, not just logged.** `ConflictPolicy` escalates
`adapter → partial_pool → freeze → reject`; the yielding source is the one with
lower mixture weight, which was `sim_wholebrain`. On reaching `freeze`/`reject` the
policy adds `coupling.*` to that source's frozen patterns and **rebuilds the
`GradientGate`** (`mixture.py:638-646`). I verified this is real enforcement and not
a decorative log. So for most of Stage III **the simulated corpus was frozen out of
the coupling operator**, which the real EEG then trained alone.

That is a substantive statement about the artifact and belongs in any description of
what trained what.

## Finding 2 — `anatomical_prior` and `sim_wholebrain` are NOT independent on `bold`

| pair | module | mean cos | min cos | n |
|---|---|---|---|---|
| `anatomical_prior` \| `sim_wholebrain` | bold | **0.99999998** | **0.99999988** | 50 |

`bold` holds **3,183 parameters across 8 tensors**, so this is not a
one-parameter degeneracy where cosine is trivially ±1.

Two gradient vectors that are parallel to float32 precision on **50 out of 50**
observations in a 3,183-dimensional space are not "agreeing" — **they are the same
vector up to scale.** Whatever loss reaches `bold` is shared between the two
sources.

**Consequence:** on this module the anatomical prior and the simulator are *one*
piece of evidence entering twice, not two that corroborate each other. Presenting
their agreement as mutual support would be double-counting — the Appendix D
"derived-data duplication" failure. Confirmation check for whoever picks this up:
dump both gradient vectors at a single step and compare directly.

## Two corrections to my own first reading

**(a) `sim_wholebrain`'s contribution of −0.185 is NOT negative transfer.** I first
read the negative sign that way. `per_source_contribution` is
`contribution[k] / Σcontribution` where `contribution[sid] += float(ln.detach()) * step_w[sid]`
(`mixture.py:630, 653`) — it is each source's **share of the normalised loss**, not a
measure of benefit. The simulated source's loss includes the NPE term, which is
**genuinely negative** (`npe_loss = −12.19` in the live log). A negative loss share
is arithmetic, not harm.

The real negative-transfer measurement is leave-one-family-out
(`evaluate.source_ablation`), **which has not been run.** Until it is, no claim about
whether a family earns its place is supported in either direction.

**(b) "241 conflict decisions" overstates by roughly 241×.** Every decision is
`module=coupling, yielding=sim_wholebrain` — it is **one sustained conflict re-logged
on every measurement step**, not 241 distinct events. `evaluate()` re-fires whenever
`n ≥ 20` and mean cosine < −0.15, and re-appends a decision even when the freeze it
prescribes is already in force. The count is an artefact of the logging loop and
should never be quoted as an event count.

## What is not measured here

Conflict is only observed between sources **co-active in the same step**, so absence
of a pair is not evidence of compatibility. `anatomical_prior` appears only against
`sim_wholebrain`, and only on `bold`. Stages I and II produced **zero** conflict
pairs — not because the sources agreed, but because fewer than two sources were
active on shared modules.
