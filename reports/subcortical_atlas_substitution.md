# Substituting the subcortical atlas: criterion, then measurement

🍃 Mendel, 2026-08-06. Sequel to `reports/ei_ordering_substitution.md` and
`reports/licence_audit.md`.

> **§1 is committed before any candidate is measured.** The commit that
> introduces this file contains §1 and nothing else. `git log --follow` on this
> path shows the two commits in that order.

## 1. Criterion, pre-committed

### 1.0 What is being replaced, and why the E/I criterion does not transfer

`harvardoxford` — `"FSL license (free for non-commercial research)"` — supplies
the **geometry** of `Aseg14`: the 14 subcortical parcels
`BrainPrior.load(include_subcortex=True)` loads by default. After the E/I
substitution it is the only established non-commercial term left on the default
path (`reports/licence_audit.md`, headline 1).

**The E/I criterion must not be reused unmodified, and the reason is
substantive.** The E/I prior consumed an *ordering*, so a differently-sourced
map that ranks parcels similarly is a legitimate replacement — the object being
supplied was a rank, and ranks are comparable. An atlas supplies **boundaries**:
where one structure ends and the next begins, and hence a centroid and a volume.
Two atlases can agree on gross anatomy and disagree on the thalamus/pallidum
border, and a correlation between their centroid coordinates would be ≈1.00 in
both cases — a statistic that cannot come out badly, which is the register's own
definition of a decorative instrument. **Correlation of centroids is therefore
excluded as a criterion here**, before I know what it would have said.

What `Aseg14` is actually used for, established by reading the call sites
(`priors.py:248`, `connectome.py:1020`) rather than assumed:

| consumer | uses | sensitivity |
|---|---|---|
| `BrainPrior.load` | `labels`, `hemi`, `structure`, `centroids_mni`, `volumes_mm3` | labels must match the connectome's row order exactly |
| `connectome.load_structural_prior` | `labels` (order-checked against `strucLabels`), `centroids_mni` → Euclidean distance → **conduction delay** for every subcortical edge | delays scale linearly with centroid separation |

So the substitution's real target is **14 centroids and 14 volumes**, under a
constraint that the 14 **names and their order** are fixed by the ENIGMA/HCP
connectome and cannot move. A candidate that supplies better anatomy but
different structures is not a candidate; it is a different model.

### 1.1 Gate 1 — licence admissibility (unchanged in spirit, strengthened in evidence)

Carried over from `ei_ordering_substitution.md` §1.2 and kept verbatim in
effect: **a candidate whose licence is a pointer rather than terms is
inadmissible; trading a known restriction for an unknown is laundering, not
substitution.**

One strengthening, stated before it is applied: the gate is evaluated against
**licence text available in this tree** — a vendored `LICENSE` file under
`assets/src/…` counts and is *better* evidence than the registry's one-line
`license` field, which is the thing the audit found unreliable. If a source's
registry field is a pointer *and* the pointed-to licence file is on disk, the
file decides, and the registry field gets corrected. If neither states terms,
the candidate is inadmissible.

Explicitly: **a candidate is not admissible merely because it is already a
dependency.** "We already accepted this licence elsewhere" is an argument about
precedent, not about terms, and the FreeSurfer and Tian entries both invite it.

### 1.2 Gate 2 — anatomical admissibility (pass/fail, before any score)

Applied second, and it can eliminate a licence-clean candidate. A candidate must:

1. **define all 14 structures** — accumbens, amygdala, caudate, hippocampus,
   pallidum, putamen, thalamus, bilaterally. Fewer is disqualifying, not a
   deduction: a missing structure orphans a connectome row.
2. **be resolvable to the connectome's label order** by structure identity, not
   by index. A join by position is how two atlases silently swap the pallidum
   and the putamen.
3. **be in, or convertible to, MNI152 coordinates**, since the centroids feed a
   Euclidean distance in millimetres that becomes a physical delay.

### 1.3 Scores, lexicographic, among candidates passing both gates

1. **Per-structure boundary/position agreement with the incumbent** — reported
   **per structure, never as a global mean.** A global mean is exactly the
   statistic that hides one structure being wrong. Where both candidates are
   volumetric, Dice overlap per structure; where one is a surface mesh, the
   comparable quantity is per-structure **centroid displacement in mm** and
   **volume ratio**, both reported for all 14 with the worst case named.
   *Lower displacement is better only in the sense of "less disruption to what
   is already validated"; see §1.5 — it is not evidence of correctness.*
2. **Delay consequence.** The centroids exist to produce conduction delays. The
   score is the change in the subcortical block of the median delay matrix, in
   milliseconds and as a fraction, with the largest single edge change named.
   A substitution that moves a delay by less than the width of the velocity
   prior it is drawn through has, for this model, made no difference — and that
   is the outcome to hope for.
3. **Whether anything is orphaned.** Does the connectome still load, does the
   label-order check still pass, do the subcortical priors still resolve.
4. **Provenance coherence** — does the geometry come from the same pipeline as
   the connectome weights it will be paired with? Currently it does not: ENIGMA
   supplies the weights and Harvard-Oxford supplies the geometry, and
   `atlases._build_aseg14` already records that "the two delineations differ" and
   that centroids are "approximate stand-ins". A candidate that closes that gap
   is preferred *on scientific grounds*, independently of its licence.

### 1.4 The declared falsifying reading (rec. 3)

State it now, so it cannot be reinterpreted later:

> **If no candidate passes both gates, the answer is that the subcortical
> geometry is only available under non-commercial terms, and the family stays
> NC.** That is a complete result and it will be reported as the headline, not
> buried. I will not relax the gate to manufacture a substitute.

And the second falsifying reading, which is the one I think is more likely to
bite:

> **If a candidate passes the gates but moves a structure's centroid by more
> than the incumbent's own delineation uncertainty, the substitution is a
> licence win bought with an anatomical change**, and it must be reported that
> way — with the changed delays — rather than as a free improvement.

### 1.5 Forward prediction, recorded before measuring (rec. 6)

1. **The ENIGMA toolbox's own subcortical surfaces** (`sctx_lh.gii`,
   `sctx_rh.gii` + `aparc_aseg_fsa5_with_sctx.csv`, under the repository's
   BSD-3-Clause `LICENSE`) will pass gate 1, and it is the candidate I expect to
   win — because it **removes** a source rather than substituting one, and
   because it is the delineation the connectome was actually built on.
2. **Tian S1–S4 and the FreeSurfer aseg will fail gate 1** on the evidence in
   the tree today. I expect Tian's vendored `LICENSE` to be readable and to be
   permissive, which would flip that — I am recording the expectation so that
   whichever way it goes is on the record before I look.
3. **Centroid displacements will be small** (order 1–3 mm) and **delay changes
   will be well inside the conduction-velocity prior's width**, i.e. the change
   will be anatomically real and dynamically negligible. If displacements come
   out large (>5 mm on any structure) the "approximate stand-in" note in
   `_build_aseg14` was understating the problem and that becomes the finding.

### 1.6 What this criterion deliberately does not do

- **It does not rank atlases by anatomical quality.** I am not competent to
  adjudicate whether Harvard-Oxford maxprob-thr25, FreeSurfer `aseg`, or ENIGMA's
  meshes segment the thalamus better, and no measurement available here decides
  it. Agreement with the incumbent is scored as *disruption*, not as *accuracy* —
  the incumbent is not ground truth, and a candidate that disagrees with it might
  be the better atlas.
- **It does not treat "commercially clean" as outranking the science.** Gate 2
  is applied before any score, and §1.4 commits in advance to reporting "no
  admissible substitute" as the result if that is what the gates say.
- **It does not resolve whether the ENIGMA/HCP *scans* impose terms of their
  own.** `enigma_hcp_sc` reads `"BSD-3-Clause code; HCP open-access data-use
  terms for the underlying scans"`, and the HCP half of that is recorded as
  **unknown** in `reports/licence_audit.md`. Any candidate drawn from the ENIGMA
  toolbox inherits that same unknown. It does not *add* one — the connectome is
  already on the default path — but it does not remove it either, and the
  honest claim available at the end of this exercise is at best *"no established
  non-commercial term remains"*, never *"commercially clear"*.
