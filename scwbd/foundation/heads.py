"""Source-native observation heads: EEG lead field, Balloon-Windkessel BOLD, behaviour.

Agent F owns ``scwbd.observe``.  This module **prefers agent F's lead fields and
BOLD operators** and falls back to clearly-labelled analytic stand-ins so the
foundation model can train before they land.  A fallback lead field carries
``provenance="analytic_sphere_fallback"``; the claim manifest records that no
individual head model was used and therefore no source-localisation claim is
supported.

Every head returns a **distribution**, never a point: mean plus a log-variance,
so a likelihood can be evaluated and calibration can be measured (thesis §2.7 --
bias and variance never collapse into one score).

That log-variance is heteroscedastic **only when the model supplies an
observation interface** (``SCWBD.observation``, from ``uncertainty.py``).  With
``state_dependent_variance=False`` the EEG and BOLD heads fall back to a
broadcast per-channel constant, which is what run 1 shipped and what
``reports/training/p0_variance_channel.md`` measures: constant predictive
variance cost SC-WBD-001-beta +0.4467 nats of excess NLL, 1.62x its whole
deficit to persistence.  The module docstring previously asserted
heteroscedasticity unconditionally while two of the three heads could not
deliver it; the assertion is now conditional because the code is.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "LeadField",
    "build_lead_field",
    "build_bipolar_lead_field",
    "parse_bipolar_derivations",
    "EEGHead",
    "BOLDHead",
    "BehaviourHead",
    "gaussian_nll",
]

#: 64-channel montage of the PhysioNet EEG Motor Movement/Imagery database
#: (Sharbrough / extended 10-10).  Names as they appear in the EDF files.
EEGMMIDB_CHANNELS = (
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6", "Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz",
    "Af4", "Af8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "Ft7", "Ft8", "T7", "T8",
    "T9", "T10", "Tp7", "Tp8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "Po7", "Po3",
    "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz",
)


def gaussian_nll(y: Tensor, mean: Tensor, logvar: Tensor, *, mask: Tensor | None = None, reduce: str = "mean") -> Tensor:
    """Heteroscedastic Gaussian negative log-likelihood, in nats per observed element.

    ``mask`` marks *observed* elements.  Missing data are **marginalised out by
    masking**, never imputed as zero (ARCHITECTURE.md §7 rule 1).
    """
    logvar = logvar.clamp(-14.0, 14.0)
    nll = 0.5 * (math.log(2 * math.pi) + logvar + (y - mean) ** 2 * torch.exp(-logvar))
    if mask is not None:
        nll = nll * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(nll.numel()), device=nll.device)
    if reduce == "mean":
        return nll.sum() / denom
    if reduce == "sum":
        return nll.sum()
    return nll


# ======================================================================
# lead field
# ======================================================================
@dataclass
class LeadField:
    """``(n_channels, n_regions)`` gain matrix in V per unit source amplitude."""

    matrix: Tensor
    channel_names: tuple[str, ...]
    units: str = "V"
    frame: str = "synthetic_ellipsoid_RAS"
    provenance: str = "analytic_sphere_fallback"
    #: volts per unit of the normalised gain, so the normalisation is invertible
    physical_scale: float = 1.0
    note: str = ""
    #: Free-orientation gain, ``(n_channels, n_regions, 3)``, normalised by the
    #: same factor as :attr:`matrix`.  ``None`` when the forward solution only
    #: supplies a fixed-orientation field.
    #:
    #: This is what a 3-vector regional moment must be observed through.
    #: Contracting it against one mean normal per parcel -- which is what
    #: :attr:`matrix` is -- bakes the folding cancellation into the *operator*,
    #: and that is where the difference between eta = 0.321 and eta = 0.834
    #: lives, measured on the model's own 400 cortical parcels.  A vector state
    #: observed through a scalar lead field is still in the scalar regime.
    matrix_vec: Tensor | None = None

    def is_individual(self) -> bool:
        return self.provenance.startswith("mne_forward") or "subject" in self.provenance

    def to(self, device) -> "LeadField":
        return LeadField(
            self.matrix.to(device), self.channel_names, self.units, self.frame,
            self.provenance, self.physical_scale, self.note,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_channels": len(self.channel_names),
            "n_regions": int(self.matrix.shape[1]),
            "units": self.units,
            "frame": self.frame,
            "provenance": self.provenance,
            "physical_scale_volts_per_unit": self.physical_scale,
            "individual_head_model": self.is_individual(),
            "condition_number": float(torch.linalg.cond(self.matrix.float()).item())
            if min(self.matrix.shape) > 1
            else float("nan"),
            "note": self.note,
        }


def _montage_positions(names: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Electrode positions in mm from MNE's standard_1005 montage (real 10-10 geometry).

    The lookup is written as an explicit ``is None`` test rather than ``a or b``.
    ``lower.get(...)`` returns a length-3 ``ndarray``, and ``ndarray or ...``
    evaluates its truth value, which raises ``ValueError: The truth value of an
    array with more than one element is ambiguous`` -- **on the first electrode
    that is found**, not on a missing one.

    That is how this function raised for every montage this project ever built.
    :func:`build_lead_field` catches the exception and falls back to a Fibonacci
    spiral, so every lead field up to and including the one in the published
    run-2 checkpoint was built on synthetic geometry while the ``note`` on it
    said "electrodes at real 10-10 montage positions". All 64 eegmmidb
    electrodes are present in ``standard_1005``; none of them was ever used.
    Recorded in ``reports/known_issues.md`` because it changes a published
    artifact's description, not just this code.
    """
    import mne

    m = mne.channels.make_standard_montage("standard_1005")
    pos = m.get_positions()["ch_pos"]
    lower = {k.lower(): v for k, v in pos.items()}
    xyz, keep = [], []
    for n in names:
        v = lower.get(n.lower())
        if v is None:
            v = lower.get(n.lower().replace(".", ""))
        if v is None:
            continue
        xyz.append(np.asarray(v, dtype=np.float64) * 1000.0)  # m -> mm
        keep.append(n)
    if not xyz:
        raise ValueError(
            f"standard_1005 has no position for any of {list(names)[:8]}; "
            "refusing to return an empty montage"
        )
    return np.stack(xyz), tuple(keep)


def build_lead_field(
    anat,
    *,
    channel_names: Sequence[str] = EEGMMIDB_CHANNELS,
    device="cpu",
    conductivity: float = 0.33,
    allow_fallback: bool = True,
    positions: "np.ndarray | Sequence[Sequence[float]] | None" = None,
    positions_note: str = "",
) -> LeadField:
    """Prefer agent F's forward solution; else an analytic single-sphere lead field.

    The fallback assumes cortical dipoles oriented radially in a homogeneous
    conducting sphere.  It reproduces the *structure* that matters for training
    (spatial smoothing, depth bias, rank deficiency) and nothing else.  It is
    **not** a head model and supports no source-localisation claim.
    """
    dev = torch.device(device)
    try:
        obs = importlib.import_module("scwbd.observe")
        for name in ("build_lead_field", "lead_field", "EEGLeadField", "make_lead_field"):
            fn = getattr(obs, name, None)
            if fn is None:
                continue
            got = fn(anat, channel_names=list(channel_names)) if not isinstance(fn, type) else fn(anat)
            mat = getattr(got, "matrix", None)
            if mat is None and torch.is_tensor(got):
                mat = got
            if mat is not None:
                return LeadField(
                    torch.as_tensor(mat, dtype=torch.float32, device=dev),
                    tuple(getattr(got, "channel_names", channel_names)),
                    units=str(getattr(got, "units", "V")),
                    frame=str(getattr(got, "frame", "subject_MRI_RAS")),
                    provenance=str(getattr(got, "provenance", "scwbd.observe (agent F)")),
                    note="agent F forward solution",
                )
    except Exception:  # noqa: BLE001 - agent F not landed / different API
        if not allow_fallback:
            raise
    if not allow_fallback:
        raise RuntimeError("scwbd.observe lead field unavailable and allow_fallback=False")

    geometry = "real 10-10 montage positions (MNE standard_1005)"
    dropped: tuple[str, ...] = ()
    if positions is not None:
        # Measured digitisation, supplied by the caller. Preferred over the
        # standard_1005 lookup whenever a source ships one: `EEG001`..`EEG074`
        # are not 10-10 names and no table can resolve them, and a digitised
        # position is the electrode where it actually sat.
        elec = np.asarray(positions, dtype=np.float64)
        if elec.shape != (len(channel_names), 3):
            raise ValueError(
                f"positions must be ({len(channel_names)}, 3) in mm, got {elec.shape}"
            )
        if not np.isfinite(elec).all():
            raise ValueError("digitised positions contain non-finite values")
        kept = tuple(channel_names)
        geometry = positions_note or "caller-supplied digitised electrode positions"
    else:
        try:
            elec, kept = _montage_positions(channel_names)
            dropped = tuple(n for n in channel_names if n not in set(kept))
        except Exception:  # noqa: BLE001 - mne montage unavailable
            geometry = "a Fibonacci spiral on a sphere -- NOT electrode geometry"
            n_ch = len(channel_names)
            idx = np.arange(n_ch, dtype=np.float64)
            phi = math.pi * (3 - math.sqrt(5)) * idx
            z = 1 - 2 * (idx + 0.5) / n_ch
            r = np.sqrt(np.clip(1 - z * z, 0, None))
            elec = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1) * 95.0
            kept = tuple(channel_names)

    src = anat.positions.detach().float().cpu().numpy()
    # scale source positions into the montage's radius so the geometry is consistent
    r_src = np.linalg.norm(src, axis=1, keepdims=True)
    r_ele = np.linalg.norm(elec, axis=1).mean()
    src_s = src / max(r_src.max(), 1e-6) * (r_ele * 0.82)
    # radial dipole orientation for cortex, random-but-fixed for deep structures
    orient = src_s / np.maximum(np.linalg.norm(src_s, axis=1, keepdims=True), 1e-6)
    deep = np.asarray([d != "cortex" for d in anat.division])
    rng = np.random.default_rng(11)
    o2 = rng.normal(size=orient.shape)
    o2 /= np.linalg.norm(o2, axis=1, keepdims=True)
    orient[deep] = o2[deep]

    d = elec[:, None, :] - src_s[None, :, :]  # (C, N, 3)
    r = np.linalg.norm(d, axis=-1)
    r = np.maximum(r, 6.0)  # mm floor: no electrode sits on a dipole
    # Contract against the per-source orientation to get the fixed-orientation
    # gain, and keep the uncontracted (C, N, 3) tensor beside it.
    #
    # The contraction is where the 2.6x is lost.  A parcel's contribution to a
    # sensor is the *vector* sum of its dipoles; summing them against one mean
    # normal bakes in the folding cancellation irreversibly.  On the model's own
    # 400 cortical parcels a scalar-per-parcel support retains eta = 0.321 of
    # the whitened lead field against 0.834 for a 3-vector.  Keeping the three
    # components makes the cancellation a property of the *data* rather than of
    # the operator.
    kernel = 1.0 / (4 * math.pi * conductivity * r**3)  # (C, N)
    L_vec = d * kernel[..., None]  # (C, N, 3) -- free orientation
    L = (d * orient[None, :, :]).sum(-1) * kernel
    # Two scales, both recorded, neither hidden: `physical_scale` maps the
    # normalised gain back to volts per unit source current, and the matrix
    # itself is normalised so that L @ N(0,I) has unit variance. Without the
    # normalisation the head would start ~7 orders of magnitude below the data
    # and the reported likelihood would be dominated by a units mistake.
    physical_scale = float(np.abs(L).max() / 1e-5)  # V per unit source amplitude
    gain = float(np.sqrt((L**2).sum(axis=1).mean()))
    L = L / max(gain, 1e-30)
    # Normalise the free-orientation tensor by the SAME gain, so a 3-vector
    # moment and a scalar amplitude are in the same units and a run may switch
    # between them without a silent rescale.
    L_vec = L_vec / max(gain, 1e-30)
    return LeadField(
        torch.as_tensor(L, dtype=torch.float32, device=dev),
        tuple(kept),
        units="V",
        frame=str(getattr(anat, "frame", "unknown")),
        provenance="analytic_sphere_fallback",
        physical_scale=physical_scale,
        note=(
            "ANALYTIC SINGLE-SPHERE LEAD FIELD, NOT A HEAD MODEL. Radial dipoles in a "
            f"homogeneous conducting sphere with electrodes at {geometry}. "
            "Supports no source-localisation or individual-anatomy claim."
            + (
                f" {len(dropped)} requested channel(s) have no position in the "
                f"montage and are ABSENT from this operator: {list(dropped)[:8]}."
                if dropped
                else ""
            )
        ),
        matrix_vec=torch.as_tensor(L_vec, dtype=torch.float32, device=dev),
    )


def parse_bipolar_derivations(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """``("EEG Fpz-Cz", ...)`` -> ``(("Fpz", "Cz"), ...)``.

    Accepts the Sleep-EDF label form (an ``EEG `` prefix, an ASCII hyphen between
    the two electrodes).  A label that does not name exactly two electrodes is
    refused rather than passed through as a monopolar channel: a bipolar
    derivation observed through a monopolar row is a different operator, and the
    error would be invisible in the loss.
    """
    out: list[tuple[str, str]] = []
    for raw in names:
        label = str(raw).strip()
        for prefix in ("EEG ", "eeg "):
            if label.startswith(prefix):
                label = label[len(prefix) :]
                break
        parts = [p.strip() for p in label.split("-") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"channel {raw!r} is not a bipolar derivation: expected "
                "'<anode>-<cathode>' naming exactly two electrodes, got "
                f"{parts}. Refusing to treat it as monopolar -- the lead field "
                "row for a difference of two electrodes is not the row for "
                "either of them."
            )
        out.append((parts[0], parts[1]))
    return tuple(out)


def build_bipolar_lead_field(
    anat,
    *,
    derivations: Sequence[tuple[str, str]],
    device="cpu",
    conductivity: float = 0.33,
    allow_fallback: bool = True,
) -> LeadField:
    """Lead field for bipolar derivations: the difference of two monopolar rows.

    A bipolar channel measures ``V(anode) - V(cathode)``, and the forward
    operator is linear in the source amplitudes, so the correct gain row is
    exactly ``L[anode] - L[cathode]``.  Nothing is fitted, interpolated or
    padded here; the derivation is carried out on the same monopolar field every
    other montage is built from, so a bipolar source and a monopolar source
    observe *the same* physics through different operators.

    This is what ``sleepedf_real`` was blocked on.  The card recorded the block
    as "two bipolar derivations cannot constrain a 64-channel observation head",
    which was true of forcing Sleep-EDF through the eegmmidb head and is not a
    property of the data: a 2-channel montage constrains a 2-dimensional
    projection of the state, and the honest way to say so is a 2-row operator,
    not two padded rows in a 64-row one.  What it still cannot do is
    disambiguate within that projection -- the rank is 2, and the card says so.

    The alternative that was NOT taken: zero-padding Fpz-Cz and Pz-Oz into the
    64-channel montage. That asserts 62 measured-and-silent electrodes, which is
    fabricated data, and the model would learn that most of the scalp is quiet.
    """
    pairs = tuple((str(a), str(b)) for a, b in derivations)
    if not pairs:
        raise ValueError("no derivations given")
    # One monopolar solve over the union of electrodes, so both rows of a
    # derivation come from the same forward solution.
    electrodes: list[str] = []
    for a, b in pairs:
        for e in (a, b):
            if e not in electrodes:
                electrodes.append(e)
    mono = build_lead_field(
        anat,
        channel_names=electrodes,
        device=device,
        conductivity=conductivity,
        allow_fallback=allow_fallback,
    )
    idx = {n.lower(): i for i, n in enumerate(mono.channel_names)}
    missing = [e for e in electrodes if e.lower() not in idx]
    if missing:
        raise ValueError(
            f"the forward solution has no row for electrode(s) {missing}; a "
            "bipolar derivation cannot be formed from a missing electrode. "
            f"Solved rows: {list(mono.channel_names)}."
        )
    rows = [mono.matrix[idx[a.lower()]] - mono.matrix[idx[b.lower()]] for a, b in pairs]
    matrix = torch.stack(rows, dim=0)
    matrix_vec = None
    if mono.matrix_vec is not None:
        matrix_vec = torch.stack(
            [mono.matrix_vec[idx[a.lower()]] - mono.matrix_vec[idx[b.lower()]] for a, b in pairs],
            dim=0,
        )
    return LeadField(
        matrix,
        tuple(f"{a}-{b}" for a, b in pairs),
        units=mono.units,
        frame=mono.frame,
        provenance=f"bipolar_of({mono.provenance})",
        physical_scale=mono.physical_scale,
        note=(
            "Bipolar derivations, each row the difference of two monopolar rows of "
            f"the same {mono.provenance} forward solution. Rank is at most "
            f"{len(pairs)}: this operator constrains a {len(pairs)}-dimensional "
            "projection of the source space and supports no claim finer than that. "
            + mono.note
        ),
        matrix_vec=matrix_vec,
    )


# ======================================================================
# heads
# ======================================================================
class EEGHead(nn.Module):
    """State -> scalp potential through a (fixed) lead field plus learned nuisance.

    The lead field is a **buffer, not a parameter**: physics is compiled, not
    fitted.  What is learned is (i) how regional structured state maps onto a
    source-current amplitude, (ii) per-channel gain/offset nuisance (electrode
    impedance, amplifier calibration -- the ``calibration`` role of Appendix B),
    and (iii) the predictive log-variance.

    (iii) has two **separately parameterised** parts, and they are kept separate
    on purpose (RL-2).  ``log_noise`` is the per-channel *instrument floor*:
    electrode impedance and amplifier noise are genuinely not functions of neural
    state, so they stay a constant.  ``logvar_mix`` maps the model's own
    per-parcel ``X^uncertainty`` onto channels.  Sharing one parameterisation
    between them would let the floor silently absorb the state term and
    reintroduce the run-1 defect with more code to hide it in.

    There is no ``horizon=`` argument, deliberately (RL-1).  Horizon dependence
    arrives because ``X^uncertainty`` is integrated forward, so it grows
    differently per family and per parcel and is falsifiable.  A ``horizon=h``
    embedding would produce h-dependence whether or not the model knew anything,
    which is the decorative-guard failure rebuilt inside its own repair.
    """

    def __init__(self, layout, lead_field: LeadField, *, hidden: int = 64, n_nuisance_basis: int = 8) -> None:
        super().__init__()
        self.layout = layout
        self.register_buffer("L", lead_field.matrix.clone())
        # Free-orientation gain, when the forward solution supplies one.  A
        # 3-vector regional moment must be observed through this; contracting
        # it against one mean normal per parcel -- which is what ``L`` is --
        # bakes the folding cancellation into the operator.  Measured on this
        # forward model: eta 0.3795 for the scalar support against 1.0 for the
        # 3-vector, a ratio of 2.64x -- which is the BEM answer on the model's
        # own parcellation, 0.321 against 0.834, a ratio of 2.6x.  The two
        # forward models agreed only after the BEM pair was re-measured on
        # Schaefer400x7; on the 68-parcel Desikan-Killiany atlas the BEM ratio is
        # 9.2x, and that gap was read for two runs as a property of the sphere
        # approximation rather than of the parcellation.
        # persistent=False: the lead field is COMPILED PHYSICS, not fitted, so it
        # is rebuilt from the anatomy at construction and must never enter a
        # checkpoint.  Making it persistent broke every existing checkpoint --
        # `1 missing: eeg.L_vec` -- and the loader correctly refused rather than
        # scoring a model with an uninitialised buffer.  A derived quantity in a
        # state dict is a version-compatibility bomb with no upside.
        self.register_buffer(
            "L_vec",
            lead_field.matrix_vec.clone() if lead_field.matrix_vec is not None else None,
            persistent=False,
        )
        self.lead_field_meta = lead_field.summary()
        self.channel_names = lead_field.channel_names
        n_ch = self.L.shape[0]
        exported = list(layout.exported_names())
        in_dim = sum(layout.spec(n).dim for n in exported)
        self._exported = exported
        self.source_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # per-channel calibration nuisance
        self.log_gain = nn.Parameter(torch.zeros(n_ch))
        self.offset = nn.Parameter(torch.zeros(n_ch))
        # Per-channel INSTRUMENT FLOOR only -- see the class docstring. Unit-variance
        # init: a head that starts 6 orders below the data reports a units error as a
        # likelihood, which is the least honest kind of number.
        self.log_noise = nn.Parameter(torch.zeros(n_ch))
        # Parcel log-variance -> channel log-variance. Initialised from the lead
        # field's row-normalised magnitude, so at step 0 a channel's variance is the
        # |L|-weighted mean of the parcel variances actually feeding it: the physics,
        # not a guess, and NOT zero.
        #
        # Zero-init would be the obvious "start as a no-op" default and it would be a
        # trap: the state term would start dead, a firing test would pass while
        # measuring nothing, and heteroscedasticity would appear only if training
        # happened to find it. Hodgkin hit exactly this on the propagator's innovation
        # layer. `softplus` keeps the map non-negative so accumulating uncertainty can
        # only ever RAISE predicted variance, matching `_LogVarHead`'s guarantee.
        w0 = self.L.abs()
        w0 = w0 / w0.sum(dim=1, keepdim=True).clamp_min(1e-8)  # (C, N), rows sum to 1
        self.logvar_mix = nn.Parameter(torch.log(torch.expm1(w0.clamp_min(1e-6))))
        # low-rank spatial nuisance (reference/montage effects), not a source model
        self.nuisance = nn.Parameter(torch.zeros(n_ch, n_nuisance_basis))
        self.nuisance_drive = nn.Linear(in_dim, n_nuisance_basis)
        # Set by SCWBD via `set_observation`. Stored WITHOUT nn.Module registration:
        # the interface is already owned by SCWBD, and registering it here would
        # double-count its parameters in `parameter_report()` and double-apply weight
        # decay to them.
        object.__setattr__(self, "_observation", None)

    def set_observation(self, observation) -> None:
        """Wire the state-side boundary (``SCWBD.observation``), or ``None``.

        ``None`` restores run-1 behaviour -- a broadcast per-channel constant --
        and is kept so the repair itself can be ablated.
        """
        object.__setattr__(self, "_observation", observation)

    @property
    def state_dependent_variance(self) -> bool:
        """True iff this head can emit a variance that varies with the state."""
        return getattr(self, "_observation", None) is not None

    @torch.no_grad()
    def calibrate_noise_floor(self, residual: Tensor, *, state: Tensor | None = None) -> dict[str, Any]:
        """Set ``log_noise`` to its closed form given held-out residuals.

        The Gaussian NLL is stationary in ``log_noise`` at exactly
        ``log(mean residual variance)`` per channel.  Run 1 asked SGD to find
        that: stage V ran 900 steps at lr 5.77e-5 and left the parameter at
        0.273 when the optimum was ``log(3.97) = 1.379``.  The model therefore
        asserted a predictive variance of 1.31 against a realised 3.97 --
        uniformly overconfident by 3.0x -- which cost **+0.4467 nats** of excess
        NLL, 1.62x its entire deficit to persistence, and was the single largest
        term in the run-1 FAIL (`reports/training/p0_variance_channel.md`).

        It is a closed form.  It is now computed rather than searched for.

        ``residual`` is ``(B, T, C)`` observed-minus-predicted on windows the
        mean was **not** fitted on.  When a state term is active it is measured
        and subtracted, so the floor stays a floor and does not absorb the
        state-dependent part (RL-2).
        """
        r = residual.detach().float()
        target = r.pow(2).mean(dim=tuple(range(r.dim() - 1))).clamp_min(1e-8).log()  # (C,)
        obs = getattr(self, "_observation", None)
        state_term = torch.zeros_like(target)
        if obs is not None and state is not None:
            lv_n = obs.predictive_logvar(state)
            lv_n = lv_n.squeeze(-1) if lv_n.shape[-1] == 1 else lv_n.mean(-1)
            mix = torch.nn.functional.softplus(self.logvar_mix).to(lv_n.dtype)
            state_term = torch.einsum("cn,btn->btc", mix, lv_n).float().mean(dim=(0, 1))
        before = self.log_noise.detach().clone()
        self.log_noise.copy_((target - state_term).to(self.log_noise.dtype))
        return {
            "parameter": "eeg.log_noise",
            "closed_form": "log(mean residual variance) per channel, minus the mean state term",
            "state_term_subtracted": bool(obs is not None and state is not None),
            "log_noise_before_mean": float(before.mean()),
            "log_noise_after_mean": float(self.log_noise.mean()),
            "implied_variance_before": float(before.exp().mean()),
            "implied_variance_after": float(self.log_noise.detach().exp().mean()),
            "n_windows": int(r.shape[0]),
        }

    def noise_floor_report(self) -> dict[str, Any]:
        """Whether the floor was ever fitted -- detectable, not inferable.

        ``EEGHead.log_noise`` initialises to all-zeros and ``BOLDHead.log_noise``
        to all ``-4.0``.  Both are constant across channels, so a parameter that
        never received a gradient has **sd exactly 0.0** and sits exactly on its
        init value.  Run 1 shipped ``bold.log_noise`` in precisely that state and
        nothing in the artifact said so.  This makes it visible.
        """
        v = self.log_noise.detach()
        sd = float(v.std()) if v.numel() > 1 else 0.0
        at_init = bool(sd == 0.0 and torch.allclose(v, torch.zeros_like(v)))
        return {
            "parameter": "eeg.log_noise",
            "role": "per-channel instrument floor (electrode impedance, amplifier noise)",
            "mean": float(v.mean()),
            "sd_across_channels": sd,
            "at_initialisation": at_init,
            "warning": (
                "log_noise is exactly its initialisation value with zero spread: it "
                "never received a gradient and was never calibrated. Any likelihood "
                "reported from this head describes an unfitted noise model."
                if at_init
                else None
            ),
            "state_dependent_variance": self.state_dependent_variance,
        }

    def source_moment(self, x: Tensor) -> Tensor | None:
        """``(..., N, D) -> (..., N, 3)`` regional dipole moment, or ``None``.

        Returns ``None`` unless the state actually declares a ``dipole``
        component, so a scalar-state model keeps its old path exactly.  This is
        the only place a 3-vector reaches an observation; every other consumer
        sees the contracted scalar and cannot tell the difference, which is
        precisely why the contraction had to move out of ``build_lead_field``.
        """
        layout = self.layout
        has = getattr(layout, "__contains__", None)
        if has is None or "dipole" not in layout:
            return None
        m = layout.get(x, "dipole")
        return m if m.shape[-1] == 3 else None

    def source_amplitude(self, x: Tensor) -> Tensor:
        """``(..., N, D) -> (..., N)`` source-current amplitude (arbitrary units)."""
        feat = torch.cat([self.layout.get(x, n) for n in self._exported], dim=-1)
        return self.source_proj(feat).squeeze(-1)

    def forward(self, x: Tensor, *, gain: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``x (B,T,N,D) -> (mean (B,T,C), logvar (B,T,C))``."""
        m = self.source_moment(x)  # (B,T,N,3) or None
        if m is not None and self.L_vec is not None:
            # Observe the vector moment through the vector lead field.  No
            # projection onto a mean normal happens anywhere: the cancellation
            # becomes a property of the data rather than of the operator.
            s = m.norm(dim=-1)  # kept only for the nuisance/logvar path below
            y = torch.einsum("cnk,btnk->btc", self.L_vec.to(m.dtype), m)
        else:
            s = self.source_amplitude(x)  # (B,T,N)
            y = torch.einsum("cn,btn->btc", self.L.to(s.dtype), s)
        feat = torch.cat([self.layout.get(x, n) for n in self._exported], dim=-1).mean(-2)
        y = y + torch.einsum("ck,btk->btc", self.nuisance.to(s.dtype), self.nuisance_drive(feat))
        g = self.log_gain.exp()
        if gain is not None:
            g = g * gain.reshape(*gain.shape, *([1] * (y.dim() - gain.dim() - 1)))
        y = y * g + self.offset
        obs = getattr(self, "_observation", None)
        if obs is None:
            # Run-1 behaviour: constant per channel. Retained only as the ablation
            # arm; `p0_variance_channel.md` measures what it costs.
            lv = self.log_noise.expand_as(y)
        else:
            # (B,T,N,logvar_dim) -> (B,T,N); the interface emits logvar_dim=1.
            lv_n = obs.predictive_logvar(x)
            lv_n = lv_n.squeeze(-1) if lv_n.shape[-1] == 1 else lv_n.mean(-1)
            mix = torch.nn.functional.softplus(self.logvar_mix).to(lv_n.dtype)
            lv = self.log_noise.to(lv_n.dtype) + torch.einsum("cn,btn->btc", mix, lv_n)
        return y, lv.clamp(-14.0, 14.0)

    def extra_repr(self) -> str:
        return f"channels={len(self.channel_names)}, provenance={self.lead_field_meta['provenance']}"


class BOLDHead(nn.Module):
    """Balloon-Windkessel haemodynamics on the slow clock (Friston et al. 2000).

    The four compartments (s, f, v, q) live in the ``hemo`` component of the
    regional state, integrated in **fp32** at ``dt_slow``.  The BOLD signal is
    the standard Buxton output equation.  Parameters are per-region but shrunk to
    literature values, because regional haemodynamic variability is real and is
    also the classic confound for "neural" claims.
    """

    def __init__(self, layout, n_regions: int, *, dt_slow: float = 0.2, te: float = 0.04, v0: float = 0.04) -> None:
        super().__init__()
        self.layout = layout
        self.dt = float(dt_slow)
        self.te, self.v0 = float(te), float(v0)
        self.log_kappa = nn.Parameter(torch.full((n_regions,), math.log(0.65)))
        self.log_gamma = nn.Parameter(torch.full((n_regions,), math.log(0.41)))
        self.log_tau = nn.Parameter(torch.full((n_regions,), math.log(0.98)))
        self.alpha = nn.Parameter(torch.full((n_regions,), 0.32))
        self.rho = nn.Parameter(torch.full((n_regions,), 0.34))
        self.neural_gain = nn.Parameter(torch.ones(n_regions))
        # Per-region INSTRUMENT FLOOR only (scanner/physiological noise), kept
        # separately parameterised from the state term for the same reason as
        # `EEGHead.log_noise` -- see that class's docstring.
        self.log_noise = nn.Parameter(torch.full((n_regions,), -4.0))
        # Parcel log-variance -> BOLD log-variance. Diagonal by construction: the
        # Balloon output is already region-indexed, so no spatial mixing is needed
        # or justified. `softplus(0.5413) == 1.0`, i.e. identity at init -- non-zero,
        # for the reason given on `EEGHead.logvar_mix`.
        self.logvar_gain = nn.Parameter(torch.full((n_regions,), 0.5413248546))
        self.register_buffer("_priors", torch.tensor([math.log(0.65), math.log(0.41), math.log(0.98), 0.32, 0.34]))
        object.__setattr__(self, "_observation", None)

    def set_observation(self, observation) -> None:
        """Wire the state-side boundary (``SCWBD.observation``), or ``None``."""
        object.__setattr__(self, "_observation", observation)

    @property
    def state_dependent_variance(self) -> bool:
        return getattr(self, "_observation", None) is not None

    def noise_floor_report(self) -> dict[str, Any]:
        """Whether the BOLD noise floor was ever fitted.

        Run 1 shipped this parameter at exactly ``-4.0`` across all 454 regions,
        sd exactly ``0.0``: it never received a gradient, because no measured
        BOLD ever entered the corpus.  Nothing was scored on it, which is why it
        went unnoticed -- and which is exactly why it is dangerous.  The moment
        haemodynamic data lands, an unfitted noise model would be presented as a
        fitted one.  This check is the thing that fires instead.
        """
        v = self.log_noise.detach()
        sd = float(v.std()) if v.numel() > 1 else 0.0
        at_init = bool(sd == 0.0 and torch.allclose(v, torch.full_like(v, -4.0)))
        return {
            "parameter": "bold.log_noise",
            "role": "per-region instrument floor (scanner and physiological noise)",
            "mean": float(v.mean()),
            "sd_across_regions": sd,
            "at_initialisation": at_init,
            "warning": (
                "bold.log_noise is exactly its -4.0 initialisation across every region "
                "with zero spread: it never received a gradient. No BOLD likelihood "
                "from this head is fitted, and none may be reported as if it were."
                if at_init
                else None
            ),
            "state_dependent_variance": self.state_dependent_variance,
        }

    @torch.no_grad()
    def calibrate_noise_floor(self, residual: Tensor, *, state: Tensor | None = None) -> dict[str, Any]:
        """Closed-form BOLD noise floor -- see :meth:`EEGHead.calibrate_noise_floor`."""
        r = residual.detach().float()
        target = r.pow(2).mean(dim=tuple(range(r.dim() - 1))).clamp_min(1e-8).log()
        obs = getattr(self, "_observation", None)
        state_term = torch.zeros_like(target)
        if obs is not None and state is not None:
            lv_n = obs.predictive_logvar(state)
            lv_n = lv_n.squeeze(-1) if lv_n.shape[-1] == 1 else lv_n.mean(-1)
            gain = torch.nn.functional.softplus(self.logvar_gain).to(lv_n.dtype)
            state_term = (gain * lv_n).float().mean(dim=tuple(range(lv_n.dim() - 1)))
        before = self.log_noise.detach().clone()
        self.log_noise.copy_((target - state_term).to(self.log_noise.dtype))
        return {
            "parameter": "bold.log_noise",
            "log_noise_before_mean": float(before.mean()),
            "log_noise_after_mean": float(self.log_noise.mean()),
            "state_term_subtracted": bool(obs is not None and state is not None),
            "n_windows": int(r.shape[0]),
        }

    def initial(self, batch: int, n_regions: int, device, dtype=torch.float32) -> Tensor:
        z = torch.zeros(batch, n_regions, 4, device=device, dtype=dtype)
        z[..., 1:] = 1.0  # f = v = q = 1 at rest
        return z

    def step(self, hemo: Tensor, neural: Tensor) -> Tensor:
        """One slow-clock Euler step, fp32.  ``hemo (B,N,4)``, ``neural (B,N)``."""
        hemo = hemo.float()
        s, f, v, q = hemo[..., 0], hemo[..., 1], hemo[..., 2], hemo[..., 3]
        kappa, gamma = self.log_kappa.exp(), self.log_gamma.exp()
        tau, alpha = self.log_tau.exp(), self.alpha.clamp(0.1, 0.6)
        rho = self.rho.clamp(0.05, 0.9)
        f_ = f.clamp_min(1e-3)
        v_ = v.clamp_min(1e-3)
        ds = self.neural_gain * neural.float() - kappa * s - gamma * (f_ - 1.0)
        df = s
        fout = v_.pow(1.0 / alpha)
        dv = (f_ - fout) / tau
        dq = (f_ * (1 - (1 - rho).clamp_min(1e-3).pow(1.0 / f_)) / rho - fout * q / v_) / tau
        out = torch.stack(
            [s + self.dt * ds, (f + self.dt * df).clamp_min(1e-3), (v + self.dt * dv).clamp_min(1e-3), q + self.dt * dq],
            dim=-1,
        )
        return torch.nan_to_num(out, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-8.0, 8.0)

    def signal(self, hemo: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """BOLD percent-signal-change ``(B,...,N)`` and its log-variance.

        ``state`` is the regional structured state the log-variance is sourced
        from.  It is optional because callers outside this package
        (``scwbd.observe.bold``) drive the Balloon model without one; those
        callers get the instrument floor and no state term, which is honest
        rather than silent.
        """
        v = hemo[..., 2].clamp_min(1e-3)
        q = hemo[..., 3]
        rho = self.rho.clamp(0.05, 0.9)
        k1 = 7.0 * rho
        k2 = 2.0
        k3 = 2.0 * rho - 0.2
        y = self.v0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
        obs = getattr(self, "_observation", None)
        if obs is None or state is None:
            return y, self.log_noise.expand_as(y)
        lv_n = obs.predictive_logvar(state)
        lv_n = lv_n.squeeze(-1) if lv_n.shape[-1] == 1 else lv_n.mean(-1)
        gain = torch.nn.functional.softplus(self.logvar_gain).to(lv_n.dtype)
        lv = self.log_noise.to(lv_n.dtype) + gain * lv_n.reshape(y.shape)
        return y, lv.clamp(-14.0, 14.0)

    def prior_penalty(self) -> Tensor:
        """Shrinkage toward literature haemodynamics -- regional freedom is not free."""
        p = self._priors
        return (
            (self.log_kappa - p[0]).pow(2).mean()
            + (self.log_gamma - p[1]).pow(2).mean()
            + (self.log_tau - p[2]).pow(2).mean()
            + (self.alpha - p[3]).pow(2).mean()
            + (self.rho - p[4]).pow(2).mean()
        )


class BehaviourHead(nn.Module):
    """Pooled state -> behavioural readout (choice logits + log response time).

    Deliberately thin.  It exists so behavioural episodes can attach at a
    declared port (§6.3) rather than being smuggled in as a neural loss.
    """

    def __init__(self, layout, n_regions: int, *, n_out: int = 4, hidden: int = 128, n_pool: int = 32) -> None:
        super().__init__()
        self.layout = layout
        exported = list(layout.exported_names())
        self._exported = exported
        in_dim = sum(layout.spec(n).dim for n in exported)
        self.pool = nn.Parameter(torch.randn(n_pool, n_regions) / math.sqrt(n_regions))
        self.net = nn.Sequential(nn.Linear(in_dim * n_pool, hidden), nn.GELU(), nn.Linear(hidden, n_out + 2))
        self.n_out = n_out

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feat = torch.cat([self.layout.get(x, n) for n in self._exported], dim=-1)  # (B,T,N,F)
        pooled = torch.einsum("pn,btnf->btpf", self.pool.to(feat.dtype), feat).flatten(-2)
        out = self.net(pooled)
        return {
            "choice_logits": out[..., : self.n_out],
            "log_rt_mean": out[..., self.n_out],
            "log_rt_logvar": out[..., self.n_out + 1].clamp(-8, 4),
        }
