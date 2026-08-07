"""Regional **families**: the partition of parcels the anatomy prior can actually defend.

Why this module exists
----------------------
``body.tex`` §2.1 indexes the state *space* by region (``X_i ∈ 𝒳_i``) and §6.1
pretrains "each regional family" on the measurements informative about its
function.  Neither is expressible while the prior returns only per-parcel
scalars: a consumer that wants to give visual cortex a different operator than
hippocampus has nothing to ask.  ``ARCHITECTURE.md`` N-2 makes the dependency
explicit — operators are assigned at *family* granularity because "families are
the finest granularity the anatomy prior actually distinguishes."  This module
is the thing that decides what that sentence means, and it is built so that the
answer is measured rather than asserted.

The rule this module follows
----------------------------
**A family exists when evidence separates it, not when a name exists for it.**
Two tiers of evidence are recognised and they are never conflated:

``measured_separation``
    The families differ in a *measured regional profile* by more than a
    spatial-autocorrelation-preserving null allows.  The test is a Váša-style
    spin test (rotate the cortical sphere, re-match parcels by optimal
    assignment, permute the family labels), FDR-corrected across all family
    pairs.  A plain label shuffle is not admissible here: cortical maps are
    smooth, so *any* spatially contiguous partition separates under it, and the
    claim "families differ in receptor density" would be a restatement of
    smoothness rather than a finding.

``atlas_separation``
    The families are distinct segmented structures in the source atlas, but
    **no per-parcel measurement in this build distinguishes them.**  Their
    separation is an atlas fact.  It is not a measurement, and a family in this
    tier carries ``training_status="prior_only_untrained"`` per `stage1-data-limited`.

Nothing is placed in the first tier because it is anatomically obvious.  The
von Economo–Koskinas cytoarchitectonic classes are the worked example: they are
real, they crosswalk cleanly onto Schaefer-400, and they **fail** the spin test
on every measurement block we hold (see ``reports/anatomy_families.md`` §3).
They are therefore recorded as a per-family descriptive field and explicitly
*not* used to separate families.

What this module refuses to do
------------------------------
* It will not report a field value without a :class:`FieldProvenance` record
  saying where the value came from — :meth:`FamilyPartition.validate` raises.
* It will not let a synthetic partition claim ``measured_separation``.  The
  history here is concrete: ``_synthetic_prior`` returned
  ``provenance="synthetic_fallback"`` and production ran an entire training run
  on synthetic regions because the provenance field was correct and unread.  A
  fabricated taxonomy that is merely *labelled* fabricated repeats that.
* It will not invent a family for a system whose parcels are absent.  The
  cerebellum and the brainstem/hypothalamic systems named in §5 and §6.1 have
  **no parcels** in the 414-parcel prior; they are recorded in
  :attr:`FamilyPartition.declared_absent` rather than given empty membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from . import sources as S

__all__ = [
    "FieldProvenance",
    "RegionFamily",
    "FamilyPartition",
    "derive_families",
    "EVIDENCE_TIERS",
    "FIELD_STATUS",
    "TRAINING_STATUS",
    "FAMILY_FIELDS",
    "CORTICAL_FAMILY_DEFINITION",
    "SEPARATION_EVIDENCE",
]

#: How a family's boundary is justified.  See the module docstring.
EVIDENCE_TIERS = ("measured_separation", "atlas_separation", "synthetic")

#: What a per-family field value means.
#:
#: ``measured``
#:     Derived from a measured map covering the family's parcels.
#: ``measured_not_separating``
#:     Measured, and reported, but it does **not** separate this family from its
#:     neighbours; it may not be cited as the reason the family exists.
#: ``prior_only``
#:     A prior-initialised value with no measurement behind it for these parcels.
#: ``not_established``
#:     We hold nothing for this field.  The value is ``None``, never a fill-in.
FIELD_STATUS = ("measured", "measured_not_separating", "prior_only", "not_established")

#: Per `stage1-data-limited`: families we hold no pretraining data for are prior-initialised and
#: declared untrained rather than quietly trained on a proxy corpus.
TRAINING_STATUS = ("has_regional_data", "prior_only_untrained")

#: The per-family prior fields this declaration carries.  A consumer may rely on
#: these names; adding one is a contract change.
FAMILY_FIELDS = (
    "cytoarchitecture",
    "laminar_differentiation",
    "receptor_profile",
    "intrinsic_timescale_s",
    "ei_prior",
)

#: The cortical partition, as *declared constants*.
#:
#: The membership below is not a guess and it is not the paper's list.  It is
#: the finest grouping of the Schaefer-400 Yeo-7 networks in which **every pair
#: of families** separates under an FDR-corrected spin test on a measured
#: regional profile.  Coarser and finer candidates were tested on the same
#: ladder and are recorded in :data:`SEPARATION_EVIDENCE`; the 7-network,
#: 4-family and 3-family partitions each contain at least one pair that does not
#: separate, so they are not shipped.
#:
#: The ladder and its numbers are in ``reports/anatomy_families.md`` §3; the
#: separation is re-checked at 200 spins by
#: ``tests/anatomy/test_families.py::test_declared_partition_separates_but_a_matched_null_does_not``,
#: which also asserts that a smoothness-matched null partition separates *less*
#: -- without that clause the check would pass for any contiguous split.
CORTICAL_FAMILY_DEFINITION: dict[str, tuple[str, ...]] = {
    "cortex_unimodal": ("Vis", "SomMot"),
    "cortex_association": ("Default", "Cont", "DorsAttn", "SalVentAttn", "Limbic"),
}

#: Spin-test results behind :data:`CORTICAL_FAMILY_DEFINITION`, regenerated by
#: ``reports/anatomy_families.md`` §3.  ``q`` is Benjamini–Hochberg across the
#: pairs of the partition being tested.  Recorded here so the partition carries
#: its own falsification: if a rerun disagrees, the constant is wrong.
SEPARATION_EVIDENCE: dict[str, Any] = {
    "null": "vasa_spin_1000_fsLR32k_sphere_optimal_assignment",
    "seed": 20260806,
    "n_spin": 1000,
    "blocks": {
        "receptor_panel": "20 Hansen PET receptor/transporter maps, z-scored",
        "myelin_thickness": "HCP S1200 T1w/T2w myelin + cortical thickness, z-scored",
    },
    "shipped": {
        "pair": "cortex_unimodal vs cortex_association",
        "F_receptor": 46.17,
        "q_receptor": 0.0010,
        "F_myelin_thickness": 159.98,
        "q_myelin_thickness": 0.0010,
    },
    "rejected": {
        "C7_yeo7": "15 of 21 pairs do not separate (e.g. SomMot vs Vis q=0.49/0.78)",
        "C4_uni_dorsattn_salvent_assoc": "2 of 6 pairs do not separate "
        "(dorsal_attention vs association q=0.21/0.19; salience vs association q=0.078/0.57)",
        "C3_uni_attention_assoc": "attention_salience vs association q=0.75/0.38",
        "EconomoKoskinas5": "fails globally on every block "
        "(receptor p=0.19, timescale p=0.20, myelin+thickness p=0.34, metabolic p=0.79)",
    },
}

#: Systems named in ``body.tex`` §5 / §6.1 that have **no parcels** in this
#: prior.  Recorded so the absence is a declared fact rather than a silent gap.
_DECLARED_ABSENT: dict[str, str] = {
    "cerebellum": (
        "BrainPrior.load(include_cerebellum=False) is the shipped configuration and "
        "the 414-parcel prior contains zero cerebellar parcels. Even with "
        "include_cerebellum=True the cerebellar parcels arrive with no structural "
        "connectivity and no cortical surface maps (BrainPrior.unresolved records "
        "both), so a cerebellar family would carry membership and nothing else. "
        "§6.1 assigns cerebellum kinematic and error-correction data; we hold neither "
        "the parcels nor the corpus."
    ),
    "brainstem_hypothalamic_autonomic": (
        "The subcortical atlas in use (Aseg14) segments 7 bilateral structures and "
        "none of them is brainstem, hypothalamus, or an autonomic nucleus. §6.1's "
        "interoceptive family (cardiac, respiratory, endocrine, thermal, nociceptive "
        "series) has no anatomical support in this prior and no corpus behind it."
    ),
    "auditory": (
        "§6.1 names auditory fields as a family distinct from motor/somatosensory. "
        "The Yeo-7 partition carried by Schaefer-400 places auditory cortex inside "
        "SomMot and does not separate it. No parcel-level auditory delineation is "
        "held here, so an auditory family cannot be declared without inventing its "
        "boundary."
    ),
}


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FieldProvenance:
    """Where one per-family field value came from.

    ``licence_is_nc`` exists so the checkpoint policy can route: the Hansen
    receptor maps are CC-BY-NC-SA-4.0, which is accepted for the prior but must
    not reach a checkpoint emitted before the synthetic-data stage.  A consumer
    filters on this flag rather than re-deriving licence from the source key.
    """

    field: str
    source_key: str  # key into scwbd.anatomy.sources.SRC
    citation: str
    licence: str
    licence_is_nc: bool
    method: str  # exactly how the value was computed
    status: str  # one of FIELD_STATUS
    coverage: float  # fraction of the family's parcels with a measured value

    def __post_init__(self) -> None:
        if self.field not in FAMILY_FIELDS:
            raise ValueError(f"unknown family field {self.field!r}; have {FAMILY_FIELDS}")
        if self.status not in FIELD_STATUS:
            raise ValueError(f"unknown field status {self.status!r}; have {FIELD_STATUS}")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError(f"coverage must be in [0,1], got {self.coverage}")


@dataclass(frozen=True)
class RegionFamily:
    """One family: which parcels, what the prior knows, and on whose authority."""

    family_id: str
    label: str
    division: str  # "cortex" | "subcortex" | "cerebellum"
    parcels: tuple[int, ...]  # indices into the prior's parcel order
    evidence_tier: str  # one of EVIDENCE_TIERS
    training_status: str  # one of TRAINING_STATUS

    #: Where the family's *membership* comes from, as distinct from where its
    #: field values come from.  A family can have well-sourced boundaries and no
    #: measured content -- every subcortical family here is exactly that -- and
    #: collapsing the two would hide it.
    membership_source: str = ""
    membership_licence: str = ""

    # --- prior fields; None means "not established", never a filled-in value
    cytoarchitecture: str | None = None
    laminar_differentiation: str | None = None
    receptor_profile: tuple[float, ...] | None = None
    receptor_names: tuple[str, ...] = ()
    intrinsic_timescale_s: float | None = None
    ei_prior: float | None = None

    provenance: tuple[FieldProvenance, ...] = ()
    #: Which measured block separated this family from the others, if any.
    separating_evidence: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise ValueError(f"unknown evidence tier {self.evidence_tier!r}")
        if self.training_status not in TRAINING_STATUS:
            raise ValueError(f"unknown training status {self.training_status!r}")

    @property
    def n_parcels(self) -> int:
        return len(self.parcels)

    def field_value(self, name: str) -> Any:
        if name not in FAMILY_FIELDS:
            raise KeyError(f"unknown family field {name!r}; have {FAMILY_FIELDS}")
        return getattr(self, name)

    def provenance_for(self, name: str) -> FieldProvenance | None:
        for p in self.provenance:
            if p.field == name:
                return p
        return None

    def nc_fields(self) -> tuple[str, ...]:
        """Fields whose value derives from a non-commercial-licensed source."""
        return tuple(p.field for p in self.provenance if p.licence_is_nc)


@dataclass(frozen=True)
class FamilyPartition:
    """A partition of the prior's parcels into named families, with provenance."""

    atlas: str
    n_regions: int
    families: tuple[RegionFamily, ...]
    provenance: str
    declared_absent: dict[str, str] = field(default_factory=dict)
    separation_evidence: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # -- interface -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.families)

    def __iter__(self):
        return iter(self.families)

    def __getitem__(self, key: str | int) -> RegionFamily:
        if isinstance(key, int):
            return self.families[key]
        for f in self.families:
            if f.family_id == key:
                return f
        raise KeyError(f"unknown family {key!r}; have {[f.family_id for f in self.families]}")

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(f.family_id for f in self.families)

    def family_index(self) -> np.ndarray:
        """``(n_regions,)`` int: which family each parcel belongs to.

        This is the array a batched trainer wants.  It is only meaningful
        because :meth:`validate` has established the partition is exhaustive and
        disjoint; building it from an invalid partition would silently drop or
        double-count parcels, so it validates first.
        """
        self.validate()
        idx = np.full(self.n_regions, -1, dtype=np.int64)
        for i, f in enumerate(self.families):
            idx[list(f.parcels)] = i
        return idx

    def is_biological(self) -> bool:
        """False when any family's boundary is synthetic."""
        return all(f.evidence_tier != "synthetic" for f in self.families)

    def untrained(self) -> tuple[RegionFamily, ...]:
        """Families that must be marked untrained in the checkpoint manifest (`stage1-data-limited`)."""
        return tuple(f for f in self.families if f.training_status == "prior_only_untrained")

    def nc_licensed_fields(self) -> dict[str, tuple[str, ...]]:
        """``{family_id: (field, ...)}`` for every NC-SA-derived value.

        The checkpoint policy consumes this: a checkpoint emitted before the
        synthetic-data stage must not carry these fields.
        """
        return {f.family_id: f.nc_fields() for f in self.families if f.nc_fields()}

    # -- the guard -----------------------------------------------------
    def validate(self) -> None:
        """Raise unless the partition is a partition and every value is sourced.

        Each check below has a failing fixture in
        ``tests/anatomy/test_families.py``.  A guard with no test that makes it
        fire reads as coverage without being coverage
        (``reports/decorative_guards.md``), so every branch here is exercised.
        """
        if not self.families:
            raise ValueError("FamilyPartition has no families")

        ids = [f.family_id for f in self.families]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate family ids: {dup}")

        seen: dict[int, str] = {}
        for f in self.families:
            if not f.parcels:
                raise ValueError(
                    f"family {f.family_id!r} has no parcels. A system with no "
                    "anatomical support belongs in declared_absent, not in the "
                    "partition with empty membership."
                )
            for p in f.parcels:
                if not 0 <= p < self.n_regions:
                    raise ValueError(
                        f"family {f.family_id!r} references parcel {p} outside "
                        f"[0,{self.n_regions})"
                    )
                if p in seen:
                    raise ValueError(
                        f"parcel {p} is in both {seen[p]!r} and {f.family_id!r}; "
                        "families must be disjoint"
                    )
                seen[p] = f.family_id

        missing = sorted(set(range(self.n_regions)) - set(seen))
        if missing:
            raise ValueError(
                f"{len(missing)} of {self.n_regions} parcels belong to no family "
                f"(first few: {missing[:8]}). The partition must be exhaustive: an "
                "unassigned parcel has no state space and no operator."
            )

        # A value without provenance is exactly the defect this module exists to
        # prevent -- a number that looks measured because nothing says it isn't.
        for f in self.families:
            for name in FAMILY_FIELDS:
                val = f.field_value(name)
                prov = f.provenance_for(name)
                if val is not None and prov is None:
                    raise ValueError(
                        f"family {f.family_id!r} reports {name}={val!r} with no "
                        "FieldProvenance. Every field value must say where it came "
                        "from; an unsourced number is indistinguishable from a "
                        "fabricated one."
                    )
                if prov is not None and prov.status == "not_established" and val is not None:
                    raise ValueError(
                        f"family {f.family_id!r} marks {name} 'not_established' but "
                        f"still reports {val!r}"
                    )
                if prov is not None and prov.status == "measured" and prov.coverage <= 0.0:
                    raise ValueError(
                        f"family {f.family_id!r} claims {name} is 'measured' with "
                        "coverage 0.0 -- that is a prior, not a measurement"
                    )

            # The anti-fabrication rule that the synthetic-prior incident earned.
            if f.evidence_tier == "measured_separation" and not f.separating_evidence:
                raise ValueError(
                    f"family {f.family_id!r} claims tier 'measured_separation' but "
                    "names no separating evidence"
                )
            if f.evidence_tier == "atlas_separation" and f.separating_evidence:
                raise ValueError(
                    f"family {f.family_id!r} is tier 'atlas_separation' but names "
                    f"separating evidence {f.separating_evidence!r}. Atlas identity "
                    "is not a measured separation; conflating them is how a prior "
                    "gets credited with a finding it does not support."
                )
            if f.evidence_tier == "atlas_separation" and f.training_status != "prior_only_untrained":
                raise ValueError(
                    f"family {f.family_id!r} separates only by atlas identity but is "
                    "not marked prior_only_untrained (`stage1-data-limited`)"
                )

        if "synthetic" in self.provenance:
            bad = [f.family_id for f in self.families if f.evidence_tier != "synthetic"]
            if bad:
                raise ValueError(
                    f"partition provenance is {self.provenance!r} but families {bad} "
                    "claim a non-synthetic evidence tier. A fabricated taxonomy may "
                    "not present itself as a measured one."
                )

    def summary(self) -> dict[str, Any]:
        return {
            "atlas": self.atlas,
            "n_regions": self.n_regions,
            "n_families": len(self.families),
            "provenance": self.provenance,
            "is_biological": self.is_biological(),
            "families": [
                {
                    "family_id": f.family_id,
                    "division": f.division,
                    "n_parcels": f.n_parcels,
                    "evidence_tier": f.evidence_tier,
                    "training_status": f.training_status,
                    "membership_source": f.membership_source,
                    "separating_evidence": list(f.separating_evidence),
                    "fields_established": [
                        n for n in FAMILY_FIELDS if f.field_value(n) is not None
                    ],
                    "nc_fields": list(f.nc_fields()),
                }
                for f in self.families
            ],
            "declared_absent": sorted(self.declared_absent),
            "n_untrained": len(self.untrained()),
        }


# ----------------------------------------------------------------------
def _src(key: str) -> dict[str, Any]:
    try:
        return S.SRC[key]
    except KeyError as exc:  # pragma: no cover - registry is static
        raise KeyError(f"unknown source key {key!r} in scwbd.anatomy.sources.SRC") from exc


def _licence(key: str) -> tuple[str, bool, str]:
    """Licence text, non-commercial flag, citation for a source key.

    The NC test is delegated to :func:`scwbd.release.licence.is_noncommercial_text`
    rather than done here with a substring match.  That module is the contract
    the checkpoint policy already reads, and a bare ``"NC" in text`` matches
    "Encoding", "Inc." and "Franchise" -- a false NC routes as badly as a missed
    one.  Imported lazily so ``scwbd.anatomy`` keeps loading if the release
    package is not installed.
    """
    s = _src(key)
    lic = str(s.get("licence") or s.get("license") or "")
    try:
        from scwbd.release.licence import is_noncommercial_text

        nc = bool(is_noncommercial_text(lic))
    except ImportError:  # pragma: no cover - release package always present here
        raise ImportError(
            "scwbd.release.licence is unavailable, so the non-commercial status of "
            f"source {key!r} cannot be established. Refusing to fall back to a "
            "substring match: a licence flag that is wrong in the permissive "
            "direction is worse than no flag."
        ) from None
    return lic, nc, str(s.get("citation", ""))


def _prov(field_name: str, key: str, method: str, status: str, coverage: float) -> FieldProvenance:
    lic, nc, cite = _licence(key)
    return FieldProvenance(
        field=field_name,
        source_key=key,
        citation=cite,
        licence=lic,
        licence_is_nc=nc,
        method=method,
        status=status,
        coverage=coverage,
    )


def _mean_of_priors(seq: Sequence[Any], idx: Iterable[int]) -> float | None:
    vals: list[float] = []
    for i in idx:
        p = seq[i]
        m = getattr(p, "mean", None)
        m = m() if callable(m) else m
        if m is None:
            return None
        vals.append(float(m))
    return float(np.mean(vals)) if vals else None


def derive_families(prior: Any) -> FamilyPartition:
    """Build the family declaration from a :class:`~scwbd.anatomy.priors.BrainPrior`.

    Membership and every field value are recomputed from the atlas and the map
    set on each call.  The only declared constants are *how many families there
    are and which networks compose them* — that decision is the spin-test result
    recorded in :data:`SEPARATION_EVIDENCE`, not something re-derived at import
    time, because rerunning 1000 spins per pair on every load would be absurd.
    """
    labels = np.asarray(prior.labels)
    structure = np.asarray([str(x) for x in prior.structure])
    network = np.asarray([str(x) for x in prior.network])
    n = int(labels.shape[0])
    maps = prior.maps
    map_names = set(maps.names() if hasattr(maps, "names") else maps.maps)
    receptor_names = tuple(sorted(m for m in map_names if m.startswith("receptor_")))

    ei_seq = prior.ei_ratio_prior()
    ts_seq = prior.timescale_prior()

    families: list[RegionFamily] = []

    # -- cortex: the measured-separation tier --------------------------
    ctx = np.where(structure == "cortex")[0]
    if ctx.size:
        R = np.column_stack(
            [np.asarray(maps[k].values, dtype=float) for k in receptor_names]
        )
        R = (R - np.nanmean(R, axis=0)) / np.nanstd(R, axis=0).clip(1e-9)
        ek_class, ek_purity = _economo_classes(prior, ctx)

        for fid, nets in CORTICAL_FAMILY_DEFINITION.items():
            sel = np.array([i for i in ctx if network[i] in nets], dtype=int)
            if sel.size == 0:
                raise ValueError(
                    f"cortical family {fid!r} selected no parcels from networks "
                    f"{nets}; the atlas's network labels are "
                    f"{sorted(set(network[ctx]))}. The declared partition does not "
                    "match this parcellation."
                )
            prof = tuple(float(v) for v in np.nanmean(R[sel], axis=0))
            tsv = _mean_of_priors(ts_seq, sel)
            eiv = _mean_of_priors(ei_seq, sel)
            cyto, cyto_cov = _dominant_economo(ek_class, ek_purity, sel, ctx)

            families.append(
                RegionFamily(
                    family_id=fid,
                    label=fid.replace("_", " "),
                    division="cortex",
                    parcels=tuple(int(i) for i in sel),
                    evidence_tier="measured_separation",
                    training_status="has_regional_data",
                    membership_source="schaefer2018",
                    membership_licence=_licence("schaefer2018")[0],
                    cytoarchitecture=cyto,
                    laminar_differentiation=None,
                    receptor_profile=prof,
                    receptor_names=receptor_names,
                    intrinsic_timescale_s=tsv,
                    ei_prior=eiv,
                    separating_evidence=("receptor_panel", "myelin_thickness"),
                    provenance=(
                        _prov(
                            "receptor_profile",
                            "hansen_receptors",
                            f"family mean of {len(receptor_names)} z-scored PET "
                            "receptor/transporter maps over the family's parcels",
                            "measured",
                            1.0,
                        ),
                        _prov(
                            "intrinsic_timescale_s",
                            "hcps1200_maps",
                            "family mean of BrainPrior.timescale_prior() means; the "
                            "cortical prior is built from the HCP S1200 MEG "
                            "intrinsic-timescale map",
                            "measured",
                            1.0,
                        ),
                        _prov(
                            "ei_prior",
                            "hcps1200_maps",
                            "family mean of BrainPrior.ei_ratio_prior() means under "
                            "the declared ei_ordering",
                            "measured",
                            1.0,
                        ),
                        _prov(
                            "cytoarchitecture",
                            "voneconomo",
                            "modal von Economo-Koskinas class by area-weighted "
                            "crosswalk on fsLR-32k. Reported as description only: "
                            "the EK partition does NOT separate under the spin test "
                            "(see SEPARATION_EVIDENCE['rejected'])",
                            "measured_not_separating",
                            float(cyto_cov),
                        ),
                        _prov(
                            "laminar_differentiation",
                            "bigbrain_layers",
                            "NOT ESTABLISHED. The Mesulam laminar-differentiation "
                            "labels shipped with the Hansen repository are a bare "
                            "integer column with no region names; the positional "
                            "join onto Desikan-Killiany was tested against the "
                            "meaning of the classes (class 1 should be primary "
                            "sensorimotor, class 4 paralimbic) and FAILED (0.00 and "
                            "0.00 agreement). No laminar value is reported",
                            "not_established",
                            0.0,
                        ),
                    ),
                    notes=(
                        f"Yeo-7 networks {nets}. Separates from the other cortical "
                        "family on the receptor panel and on myelin+thickness under "
                        "a 1000-spin Váša null, FDR-corrected."
                    ),
                )
            )

    # -- subcortex: the atlas-separation tier --------------------------
    # Every map in this build is 400 long (cortex only), so no subcortical parcel
    # has a measured regional profile.  BrainPrior.ei_ratio_prior() and
    # timescale_prior() DO return a number for each of the 14 -- but it is one
    # identical value, the cortical mean, for all of them.  Reporting that as a
    # family's intrinsic timescale would be imputing missing data as an
    # average-brain label, which ARCHITECTURE.md §7 rule 1 forbids outright.  So
    # the fields are `not_established` and the reason travels with them.  The
    # per-parcel tensors are unchanged; a trainer may still initialise from them,
    # but it can no longer mistake the initialiser for regional knowledge.
    sub_atlas = _subcortical_atlas_provenance(prior)
    sub = np.where(structure == "subcortex")[0]
    for fid, sel in _subcortical_groups(labels, sub).items():
        families.append(
            RegionFamily(
                family_id=fid,
                label=fid.replace("_", " "),
                division="subcortex",
                parcels=tuple(int(i) for i in sel),
                evidence_tier="atlas_separation",
                training_status="prior_only_untrained",
                membership_source=sub_atlas["source_key"],
                membership_licence=sub_atlas["licence"],
                cytoarchitecture=None,
                laminar_differentiation=None,
                receptor_profile=None,
                receptor_names=(),
                intrinsic_timescale_s=None,
                ei_prior=None,
                separating_evidence=(),
                provenance=(
                    _prov(
                        "intrinsic_timescale_s",
                        sub_atlas["source_key"],
                        "NOT ESTABLISHED. No map in this build covers subcortex. "
                        "BrainPrior.timescale_prior() returns the same cortical-mean "
                        "value for all 14 subcortical parcels; reporting it as this "
                        "family's timescale would impute an average-brain label "
                        "(ARCHITECTURE.md §7 rule 1)",
                        "not_established",
                        0.0,
                    ),
                    _prov(
                        "ei_prior",
                        sub_atlas["source_key"],
                        "NOT ESTABLISHED. Identical across all 14 subcortical "
                        "parcels for the same reason",
                        "not_established",
                        0.0,
                    ),
                ),
                notes=(
                    f"Separated from the other subcortical families by the "
                    f"{sub_atlas['atlas']} segmentation only. No map in this build "
                    "covers subcortex, so no measured regional profile "
                    "distinguishes it. Prior-only and untrained per `stage1-data-limited`."
                ),
            )
        )

    part = FamilyPartition(
        atlas=str(getattr(prior, "atlas", "unknown")),
        n_regions=n,
        families=tuple(families),
        provenance="scwbd.anatomy.families.derive_families",
        declared_absent=dict(_DECLARED_ABSENT),
        separation_evidence=dict(SEPARATION_EVIDENCE),
        notes=(
            "Two cortical families separated by measurement; the remainder "
            "separated by atlas identity alone and declared untrained. The "
            "cerebellar, brainstem/hypothalamic and auditory families named in "
            "body.tex §5/§6.1 have no parcels here and are in declared_absent."
        ),
    )
    part.validate()
    return part


def _economo_classes(prior: Any, ctx: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Modal von Economo-Koskinas class per cortical parcel, by area crosswalk."""
    try:
        from .atlases import crosswalk, load_parcellation

        parc = prior.parcellation
        ek = load_parcellation("EconomoKoskinas", parc.space, parc.density)
        X = crosswalk(parc, ek)
    except Exception:  # noqa: BLE001 - descriptive field only; absence is reported
        return None, None
    tot = X.sum(axis=1)
    cls = X.argmax(axis=1)
    purity = X.max(axis=1) / np.maximum(tot, 1e-9)
    names = np.asarray([str(x) for x in ek.labels])
    return names[cls], purity


def _dominant_economo(
    ek_class: np.ndarray | None,
    ek_purity: np.ndarray | None,
    sel: np.ndarray,
    ctx: np.ndarray,
) -> tuple[str | None, float]:
    if ek_class is None or ek_purity is None:
        return None, 0.0
    pos = {int(p): k for k, p in enumerate(ctx)}
    rows = [pos[int(i)] for i in sel if int(i) in pos]
    if not rows:
        return None, 0.0
    vals, counts = np.unique(ek_class[rows], return_counts=True)
    top = str(vals[counts.argmax()])
    frac = float(counts.max() / counts.sum())
    return f"{top} ({frac:.0%} of parcels)", float(np.mean(ek_purity[rows]))


def _subcortical_atlas_provenance(prior: Any) -> dict[str, str]:
    """Which subcortical atlas the prior actually loaded, and under what licence.

    Read from the prior rather than hardcoded.  ``DEFAULT_SUBCORTICAL_ATLAS`` is
    ``Aseg14T`` (Melbourne/Tian delineation), **not** ``Aseg14``
    (Harvard-Oxford), and the substitution exists specifically to keep a
    non-commercial term off the default path
    (``reports/subcortical_atlas_substitution.md``).  Hardcoding the wrong one
    would flag these fields NC and mis-route the checkpoint policy in the
    direction that looks safe and is wrong.
    """
    rec = (prior.provenance or {}).get("subcortical_atlas") or {}
    if not isinstance(rec, dict):
        rec = {}
    key = str(rec.get("licence_key") or "")
    if not key:
        raise ValueError(
            "BrainPrior.provenance['subcortical_atlas'] carries no licence_key, so "
            "the licence of the subcortical family boundaries cannot be "
            "established. Refusing to guess: an unlabelled licence routes as "
            "permissive by default, which is the failure that matters."
        )
    lic, _, _ = _licence(key)
    return {"atlas": str(rec.get("name", "unknown")), "source_key": key, "licence": lic}


def _subcortical_groups(labels: np.ndarray, sub: np.ndarray) -> dict[str, np.ndarray]:
    """Group the subcortical parcels by the structure the atlas segments.

    The Aseg14 atlas labels are ``L``/``R`` + a structure stem (``thal``,
    ``hippo``, ``amyg``, ``caud``, ``put``, ``pal``, ``accumb``).  The stems are
    what the atlas actually distinguishes, so they are what the families are.
    Grouping caudate/putamen/pallidum/accumbens into a single "basal ganglia"
    family would be *my* neuroanatomy rather than the atlas's, and nothing in
    this build measures the difference either way — so the atlas's own
    granularity is shipped and the interpretation is left to the reader.
    """
    groups: dict[str, list[int]] = {}
    for i in sub:
        lab = str(labels[i])
        stem = lab[1:] if lab[:1] in ("L", "R") else lab
        groups.setdefault(f"subcortex_{stem.lower()}", []).append(int(i))
    return {k: np.asarray(v, dtype=int) for k, v in sorted(groups.items())}
