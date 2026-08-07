# ds002336 → parcel space: the registration, run

**Measured 2026-08-06.** `scwbd/anatomy/registration.py` had been on master with
a complete EPI ← T1w ← template chain, `labels_to_epi_grid` and
`ParcelCoverage`, imported by nothing but its own test file. Every source card
that needs it says the registration *"has not been run"* — true, and read by me
as "none is available".

It has now been run, on real data.

## Result, one subject, one run

```
subject xp102, task motorloc      sub-xp102_task-motorloc_bold.nii.gz
BOLD                              106 x 106 x 32 x 166, TR 2.0 s
EPI voxel                         15.70 mm3  (1.98 x 1.88 x 3.80 mm)
atlas                             Schaefer400x7, MNI152NLin6Asym, 1 mm
registration  EPI<-T1w            rigid 6-DOF, mutual information
              T1w<-template       affine, mutual information
wall clock                        160.8 s

parcels covered                   379 / 400
parcels unobserved                21   -> NaN, never 0.0
```

## What the numbers mean, and what they do not

**21 parcels have no observation.** The acquisition is 32 slices at 3.8 mm — a
121.6 mm slab, not a head. Those parcels are outside it in *every* frame of this
run, systematically, by acquisition. They are `NaN` in the timeseries and
`False` in the coverage mask, and `gaussian_nll` marginalises them out rather
than scoring them.

That distinction is the entire reason this was blocked. A BOLD value of `0.0`
for a parcel that was never in the field of view is a fabricated observation,
and once it is in an array it is indistinguishable from a real one.

**`coverage_basis` is `field_of_view`, not `brain_mask`** — and the registration
module says so itself, unprompted:

> 'covered' here means the parcel falls inside the acquisition's rectangular
> array, which is a far weaker statement than 'this parcel has signal'.

So 379 is an *upper bound* on usable parcels, not a count of good ones. A brain
mask would lower it. That is worth knowing before anyone quotes 379/400 as
coverage.

## What this does and does not unblock

| | |
|---|---|
| **done** | voxel BOLD → parcel-space timeseries with an honest coverage mask (`scwbd/sources/parcellate_bold.py`) |
| **done** | a parcel-space BOLD likelihood in the trainer (`FoundationTrainer.real_bold_losses`), which *requires* the mask and raises without it |
| **not done** | a dataset/loader yielding BOLD windows batched with their masks |
| **not done** | the card enabled — that needs the loader, and a card enabled without one trains on nothing |

The remaining piece is a loader, not a component. Every part it would compose —
registration, atlas, coverage, likelihood — now exists and has been run on real
bytes.

## Cost

161 s per run, dominated by the two affine registrations. The chain is per
*subject* and reusable across that subject's runs (`chain=` argument), so
ds002336's 55 runs cost roughly the number of subjects, not the number of runs.


---

## Full corpus, 2026-08-06

All 55 runs parcellated, all six tasks, all ten subjects.

```
runs discovered   55        runs cached           55
participants      10        dropped for coverage   0
windows           485       window frames         32   (64 s at TR 2 s)
wall clock      1837 s      invariant violations   0
```

Coverage across all 55 runs:

```
min 0.8925   median 1.0000   max 1.0000
below 0.95:  3 runs          below 0.90: 1 run
```

**The median run covers every parcel.** That is worth stating against the first
measurement, which happened to be `sub-xp102` at 379/400 and would have made
94.75% look typical. It is not — it is in the worst quartile. A single run is
not a coverage estimate.

The invariant — `NaN` rows exactly equal uncovered parcels, parcel by parcel —
holds on all 55 without exception. That is the property that keeps a fabricated
zero out of the likelihood, and it is now checked against 55 artifacts rather
than an argument about the function that writes them.

**What this does not establish.** Coverage is `field_of_view`, not `brain_mask`.
A parcel inside the acquisition's rectangular array is counted as covered
whether or not it contains brain, so these figures are upper bounds on usable
parcels. The registration module says so itself in every sidecar it writes.
