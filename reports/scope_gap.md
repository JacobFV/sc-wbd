# The scope gap between `paper/body.tex` and SC-WBD-001-beta

Owner: architect. 2026-08-06. Written after re-reading `body.tex` in full.

Every claim below was regenerated from source in this checkout, not read off an
earlier report.

---

## 0. Headline

> **We built the control arm of §11.4's first required ablation and shipped it
> under the name of the treatment arm.**

§11.4 opens its list of mandatory comparisons with:

> *structured regional state versus one scalar or pooled vector per region*

SC-WBD-001-beta is the second of those two. Popper's ruling that it is beaten
by all five baselines (NLL 2.5552 vs. persistence 2.2787) is therefore not a
surprising failure of the thesis — it is the expected behaviour of the null arm,
measured correctly, with the treatment arm never built.

The process defect that produced this is mine and is stated plainly in §4.

---

## 1. What the paper specifies

`body.tex` §2.1, verbatim:

```
X_i(t) = (X_i^sheet, X_i^layer, X_i^population, X_i^frequency,
          X_i^memory, X_i^metabolic, X_i^uncertainty) ∈ X_i

"The components need not have equal shape or even be ordinary dense tensors.
They may be fields on a cortical mesh, arrays indexed by depth and cell class,
graphs of local populations, point processes of spike events, sparse
distributed codes, sets of particles..."
```

Note `∈ X_i` — the *space itself* is indexed by region. Equation (2) then
registers nine operator types (`flow ODE/PDE, field kernel, convolution,
delayed SSM, spectral transfer, attention, point process, surrogate,
composition`) to be assigned per region.

§0.2 names the two differentiators: **(i) heterogeneous operator-valued
regional state; (ii) non-nested, source-native spatial and temporal
resolutions.**

---

## 2. What the artifact contains

Four gaps, each verified against code in this checkout.

### G-1 — one operator for the whole brain

`scwbd/foundation/config.py:32`

```python
local_core: str = "learned"
```

A single string. `MechanisticCore.__init__` resolves it once
(`scwbd/foundation/model.py:275-283`) and applies that one backend to all
regions. Six backends exist and are genuinely interchangeable by config switch
— but the switch is global, not per region. Regional heterogeneity enters only
through the θ conditioning vector, i.e. as *parameters of one operator*, never
as *different operators*.

State is dense and uniform: `(B, T, N=454, D=28)`, the same 28 components for
every parcel. This is `X` with no `_i` on the space.

Differentiator (i): **absent.**

### G-2 — the resolution poset is declared trivial

`scwbd/foundation/compiler_bridge.py:597`

> `SC-WBD-001-beta declares no cross-scale prolongation, so R02 has nothing to`
> `check.`

`scwbd/schema/poset.py` and `scwbd/transforms/sheaf.py` implement restriction
and prolongation. The foundation model declares neither. §4.2's three authority
policies (fine-authoritative / consensus multilevel / coarse-authoritative with
sparse refinement) are not instantiated; the compatibility pseudo-likelihood
Ψ_ab is not formed; adaptive refinement does not exist.

What survives is the two-clock multirate split — fast 125 Hz, slow 5 Hz. That
is a real instance of temporal non-nesting and it works. It is also the entire
extent of it.

Differentiator (ii): **present in weak form** (temporal only, two levels, fixed).

### G-3 — the subsystems are built and unwired

`scwbd/dynamics/hippocampus.py` (H_t = {k,v,g,c,ρ}, §5.1),
`scwbd/dynamics/subcortical.py`, `scwbd/dynamics/plasticity.py` are
implemented and tested. The foundation model instantiates none of them. §5's
entire argument — that these systems *warrant more engineered backends than a
generic block* — has no expression in the trained artifact.

### G-4 — the curriculum is named after the paper, not built from it

§6.1 Stage I is **per-regional-family** phenotype pretraining: visual fields on
retinotopic dynamics, auditory on spectrotemporal, hippocampal on episodic and
replay, brainstem/hypothalamic on interoceptive series. Our Stage I trains one
uniform model on one corpus. §6.2 (interface and pathway calibration) and §6.4
(connectome assembly) then have no distinct motifs to calibrate or assemble —
they are stage labels over a single homogeneous optimisation.

`scwbd/foundation/train.py` docstrings cite §6.1/§6.2 correctly. The code below
them does something else.

---

## 3. What is *not* wrong

Stated so the gap is not overclaimed:

- The multirate co-simulation (§4.5) is real and the semigroup residual
  ε_sg is measured.
- Six mechanistic backends are implemented and interchangeable by config.
- The compiler, its eleven refusals, the schema kernel and the bias–variance
  ledger (§2.7) are built and fail closed.
- Restriction/prolongation machinery exists — it is *undeclared*, not missing.
- Stage V's hierarchical decomposition θ_{p,s} = μ + α + δ + ζ (§6.5) is
  implemented.
- The identifiability, SBC, and gate infrastructure is real, and it is what
  detected most of the defects in this list.

The gap is in **assembly and declaration**, not in the component library.

---

## 4. How this happened

`ARCHITECTURE.md` §5, which I wrote, specifies:

> per-parcel structured state over N_regions, each with E/I rates, adaptation,
> spectral modes, hemodynamic compartments, uncertainty channel

"each with" — every parcel gets the same list. That sentence is a narrowing of
§2.1 from operator-valued heterogeneous state to a uniform feature vector. It
was implemented faithfully. It was never flagged as a narrowing, so no agent
had cause to challenge it and no gate could fire on it.

**The controlling failure is not the narrowing. It is the undeclared
narrowing.** A stated one is a decision the fleet can attack; an unstated one
is invisible to a process built entirely out of attacking stated things.

Corrective: `ARCHITECTURE.md` gains a **Declared Narrowings** section. Every
divergence from `body.tex` is listed with the section it narrows, the reason,
and whether it is permanent or scheduled. Anything not listed there is a defect
by definition.

---

## 5. Consequence for the claim boundary

`reports/CLAIM_BOUNDARY.md` must record that the run-1 artifact is the
equal-capacity generic-operator **control** for §11.4's first ablation. Its
FAIL is a valid measurement of that control, and it may not be reported as a
test of the thesis. G1–G5 remain COULD_NOT_RUN; nothing here changes that.
