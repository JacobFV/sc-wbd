"""The three-region identifiability example (thesis sec. 0.3, equations T1-T3).

This is the fixture the linear identifiability laboratory (build-order item 2)
and the end-to-end synthetic slice (item 4) consume.  It declares:

* three latent regional states with directed coupling (T1), one of which
  carries the **unknown conduction delay** the benchmark must recover;
* a controlled input through a single impulse intervention port (``B u(t-tau_u)``);
* an EEG-like instrument observing instantaneous mixing ``L(l)`` at
  ``Delta_E = 1 ms`` (T2);
* an fMRI-like instrument observing a hemodynamic convolution ``h(s;rho)`` at
  ``Delta_B = 1 s`` (T3).

The lead field ``L(l)`` and hemodynamic kernel ``h(s;rho)`` are *declared* here
as observation models with their supports, clocks, units and nuisance parameter
groups.  Their physics belongs to agent F (``scwbd.observe``); this module owns
only the contract.

The schema compiles with **no overrides**: every refusal R01..R11 passes.  That
is the point of it - it is the reference for what a compliant declaration looks
like, and its ledgers, calibrations, lineage groups and safe set are all
populated rather than defaulted.
"""

from __future__ import annotations

from ..claims import ClaimManifest
from ..clocks import ClockSpec
from ..frames import CalibrationManifest, FrameEdge, FrameGraphSpec, FrameNode
from ..ids import ClockId, FrameId, ScaleId
from ..ledger import UncertaintyLedger
from ..lineage import Identity, LineageUnit
from ..operators import (
    Identification,
    OperatorSpec,
    ResidualPolicy,
    SemigroupTest,
)
from ..poset import (
    CocycleCheck,
    GluingPolicy,
    MapSpec,
    ResolutionPoset,
    ScaleMapPair,
    ScaleNode,
)
from ..priors import DiracPrior, LogNormalPrior, NormalPrior, UniformPrior
from ..regions import AtlasRef, Region
from ..schema import BrainSchema
from ..sources import (
    Governance,
    GradientPermission,
    HierarchicalEffect,
    InterventionModel,
    Missingness,
    ObservationModel,
    PopulationStructure,
    SafeSet,
    SignalSpec,
    SourceCard,
    SplitPolicy,
    TemporalHoldout,
)
from ..state import ComponentSpec, Port, StateSpec
from ..supports import PSF, Support, TemporalSupport
from ..units import Unit

__all__ = [
    "build_three_region_schema",
    "build_three_region_claim",
    "REGION_IDS",
    "DELTA_EEG",
    "DELTA_BOLD",
]

#: Region ids, in the order the state vector x in R^3 is packed.
REGION_IDS: tuple[str, str, str] = ("region_a", "region_b", "region_c")

#: T2: EEG sampling interval.
DELTA_EEG = 1e-3
#: T3: fMRI volume interval.
DELTA_BOLD = 1.0
#: Latent process step (the simulation clock of T1).
DELTA_SIM = 1e-3

_SESSION_VALIDITY = (0.0, 3600.0)


# ---------------------------------------------------------------------------
# small builders
# ---------------------------------------------------------------------------
def _estimable(
    units: str,
    *,
    estimator: str,
    measurement: float = 0.0,
    session: float = 0.0,
    parameter: float = 0.0,
    numerical: float = 0.0,
    bias: tuple[float, float] = (0.0, 0.0),
    discrepancy: float | None = None,
    domain: dict | None = None,
) -> UncertaintyLedger:
    """A design-estimable ledger: replication/randomization names its estimator."""
    variance = {
        "measurement": measurement,
        "between_session": session,
        "parameter": parameter,
        "numerical": numerical,
    }
    return UncertaintyLedger(
        variance={k: v for k, v in variance.items() if v > 0.0},
        bias_interval=bias,
        bias_status="design_estimable",
        bias_estimator=estimator,
        model_discrepancy=discrepancy,
        validity_domain=domain or {"regime": "linear_gaussian_synthetic"},
        units=units,
    )


def _bounded(
    units: str, *, source: str, bias: tuple[float, float], **variance: float
) -> UncertaintyLedger:
    """An externally-bounded ledger: a phantom/anchor supplies the interval."""
    return UncertaintyLedger(
        variance={k: v for k, v in variance.items() if v > 0.0},
        bias_interval=bias,
        bias_status="externally_bounded",
        external_bound_source=source,
        validity_domain={"regime": "linear_gaussian_synthetic"},
        units=units,
    )


def _sensitivity(units: str, *, bias: tuple[float, float], **variance: float) -> UncertaintyLedger:
    """A prior-specified-sensitivity ledger: swept over a declared range."""
    if bias[0] >= bias[1]:
        raise ValueError("a sensitivity bias must be a range, never a point")
    return UncertaintyLedger(
        variance={k: v for k, v in variance.items() if v > 0.0},
        bias_interval=bias,
        bias_status="prior_specified_sensitivity",
        validity_domain={"regime": "linear_gaussian_synthetic"},
        units=units,
    )


def _calibration(
    cid: str,
    layers: tuple[str, ...],
    *,
    residual: float,
    tolerance: float,
    residual_units: str = "m",
    inverse: bool = True,
) -> CalibrationManifest:
    return CalibrationManifest(
        id=cid,
        layers=layers,  # type: ignore[arg-type]
        n_observations=64,
        fitting_method="least_squares_with_bootstrap_residuals",
        residual=residual,
        residual_units=Unit(residual_units),
        residual_tolerance=tolerance,
        validity_interval=_SESSION_VALIDITY,
        validity_domain={"temperature_c": [19.0, 24.0], "session": "synthetic"},
        extrapolation_distance=0.0,
        recalibration_triggers=("device_change", "participant_regeometry", "session_end"),
        units_checked=True,
        handedness_checked=True,
        roundtrip_checked=True,
        inverse_available=inverse,
        ledger=_bounded(
            residual_units, source=f"{cid}_phantom", bias=(-residual, residual),
            measurement=residual**2,
        ),
    )


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def _frames() -> FrameGraphSpec:
    def node(fid: str, obj: str, origin: str) -> FrameNode:
        return FrameNode(
            id=FrameId(fid),
            object=obj,
            origin=origin,
            axes=("R", "A", "S"),
            handedness="right",
            units=Unit("m"),
            epoch="session_start",
            validity_interval=_SESSION_VALIDITY,
            coordinate_type="physical",
        )

    nodes = (
        node("scanner_RAS", "scanner bore", "isocentre"),
        node("subject_anat_RAS", "T1w anatomical volume", "anterior_commissure"),
        node("subject_surface_RAS", "cortical surface mesh", "anterior_commissure"),
        node("eeg_cap", "EEG electrode array", "nasion"),
        node("stim_device", "impulse delivery device", "device_tip"),
    )

    def edge(src: str, dst: str, lineage: str, cal: CalibrationManifest, resid: float) -> FrameEdge:
        return FrameEdge(
            src=FrameId(src),
            dst=FrameId(dst),
            transform="rigid",
            parameters={
                "tx": NormalPrior(loc=0.0, scale=1e-3, units=Unit("m")),
                "ty": NormalPrior(loc=0.0, scale=1e-3, units=Unit("m")),
                "tz": NormalPrior(loc=0.0, scale=1e-3, units=Unit("m")),
            },
            lineage=lineage,
            invertible=True,
            calibration=cal,
            ledger=_bounded("m", source=f"{src}_to_{dst}_landmarks", bias=(-resid, resid),
                            measurement=resid**2),
            units_in=Unit("m"),
            units_out=Unit("m"),
            roundtrip_residual=resid,
            roundtrip_tolerance=resid * 4.0,
        )

    edges = (
        edge(
            "scanner_RAS",
            "subject_anat_RAS",
            "rigid registration of T1w to scanner isocentre, ANTs v2.5 affine restricted to rigid",
            _calibration("cal_scanner_to_anat", ("physical_reference_frame",),
                         residual=5e-4, tolerance=2e-3),
            5e-4,
        ),
        edge(
            "subject_anat_RAS",
            "subject_surface_RAS",
            "FreeSurfer surface reconstruction, vertex correspondence to anatomical volume",
            _calibration("cal_anat_to_surface", ("deformation_projection",),
                         residual=3e-4, tolerance=1.5e-3),
            3e-4,
        ),
        edge(
            "subject_surface_RAS",
            "eeg_cap",
            "electrode digitization with a Polhemus stylus against three fiducials",
            _calibration("cal_surface_to_cap", ("sensor_device_geometry",),
                         residual=1.2e-3, tolerance=5e-3),
            1.2e-3,
        ),
        edge(
            "subject_surface_RAS",
            "stim_device",
            "device pose read from the neuronavigation tracker at impulse onset",
            _calibration("cal_surface_to_device", ("sensor_device_geometry", "intervention_dose"),
                         residual=1.5e-3, tolerance=6e-3),
            1.5e-3,
        ),
    )
    return FrameGraphSpec(root=FrameId("scanner_RAS"), nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------
def _clocks() -> list[ClockSpec]:
    return [
        ClockSpec(
            id=ClockId("sim"),
            label="latent process clock (T1)",
            dt=DELTA_SIM,
            epoch="session_start",
            reference=None,
            sync_evidence="shared_hardware",
            adaptive_stepping=False,
            interpolation="band_limited",
            ledger=_estimable("s", estimator="fixed_step_integrator", numerical=1e-18),
        ),
        ClockSpec(
            id=ClockId("eeg_amp"),
            label="EEG amplifier clock (T2, Delta_E = 1 ms)",
            dt=DELTA_EEG,
            epoch="session_start",
            reference=ClockId("sim"),
            offset=0.0,
            drift=0.0,
            jitter_sd=2e-5,
            group_delay=1.5e-3,
            sync_evidence="physical_trigger",
            interpolation="band_limited",
            dropped_sample_policy="marginalize",
            ledger=_estimable("s", estimator="trigger_crosscorrelation", measurement=4e-10),
        ),
        ClockSpec(
            id=ClockId("scanner_volume"),
            label="fMRI volume trigger clock (T3, Delta_B = 1 s)",
            dt=DELTA_BOLD,
            epoch="session_start",
            reference=ClockId("sim"),
            offset=0.0,
            drift=0.0,
            jitter_sd=5e-4,
            integration_window=0.06,
            sync_evidence="physical_trigger",
            interpolation="zero_order_hold",
            dropped_sample_policy="marginalize",
            ledger=_estimable("s", estimator="volume_trigger_log", measurement=2.5e-7),
        ),
        ClockSpec(
            id=ClockId("stim_computer"),
            label="impulse delivery clock",
            dt=DELTA_SIM,
            epoch="session_start",
            reference=ClockId("sim"),
            sync_evidence="shared_hardware",
            interpolation="event_exact",
            ledger=_estimable("s", estimator="shared_oscillator_audit", measurement=1e-12),
        ),
    ]


# ---------------------------------------------------------------------------
# resolution poset
# ---------------------------------------------------------------------------
def _poset() -> ResolutionPoset:
    nodes = (
        ScaleNode(
            id=ScaleId("surface_vertex"),
            label="cortical surface vertex",
            axis="spatial",
            characteristic_scale=1.5e-3,
            n_elements=10242,
        ),
        ScaleNode(
            id=ScaleId("parcel"),
            label="anatomical parcel",
            axis="spatial",
            characteristic_scale=2.0e-2,
            n_elements=3,
        ),
        ScaleNode(
            id=ScaleId("network"),
            label="distributed network summary",
            axis="spatial",
            characteristic_scale=8.0e-2,
            n_elements=1,
        ),
        # Deliberately incomparable with the spatial chain: a spectral band is
        # not coarser or finer than a parcel, and no defensible map exists.
        ScaleNode(
            id=ScaleId("spectral_band"),
            label="frequency band",
            axis="spectral",
            characteristic_scale=4.0,
            characteristic_units=Unit("Hz"),
            n_elements=5,
        ),
        ScaleNode(
            id=ScaleId("event_window"),
            label="task event window",
            axis="behavioral",
            characteristic_scale=2.0,
            characteristic_units=Unit("s"),
            n_elements=40,
        ),
    )
    relations = (
        (ScaleId("surface_vertex"), ScaleId("parcel")),
        (ScaleId("parcel"), ScaleId("network")),
    )
    maps = (
        ScaleMapPair(
            fine=ScaleId("surface_vertex"),
            coarse=ScaleId("parcel"),
            restriction=MapSpec(
                name="vertex_to_parcel_area_weighted_mean",
                kind="restriction",
                roundtrip_residual=0.04,
                roundtrip_tolerance=0.10,
                units=Unit("Hz"),
                ledger=_bounded("Hz", source="synthetic_ground_truth_fields",
                                bias=(-0.02, 0.02), measurement=1e-3),
            ),
            prolongation=MapSpec(
                name="parcel_to_vertex_conditional_distribution",
                kind="prolongation",
                returns_distribution=True,
                roundtrip_residual=0.04,
                roundtrip_tolerance=0.10,
                units=Unit("Hz"),
                ledger=_sensitivity("Hz", bias=(-0.08, 0.08), parameter=4e-3),
            ),
            landmark_coverage=0.92,
            required_coverage=0.80,
            roundtrip_tested=True,
            landmark_tested=True,
            out_of_support_policy="return_distribution",
        ),
        # parcel -> network has a restriction only. No prolongation is declared,
        # so no unsupported fine structure can be manufactured (R02 has nothing
        # to object to).
        ScaleMapPair(
            fine=ScaleId("parcel"),
            coarse=ScaleId("network"),
            restriction=MapSpec(
                name="parcel_to_network_mean",
                kind="restriction",
                roundtrip_residual=0.30,
                roundtrip_tolerance=0.50,
                units=Unit("Hz"),
                ledger=_bounded("Hz", source="synthetic_ground_truth_fields",
                                bias=(-0.15, 0.15), measurement=9e-3),
            ),
            roundtrip_tested=True,
            landmark_tested=True,
            landmark_coverage=0.85,
            out_of_support_policy="inflate_uncertainty",
        ),
    )
    gluing = GluingPolicy(
        materialize_global=False,
        cocycle_checks=(
            CocycleCheck(
                path=(ScaleId("surface_vertex"), ScaleId("parcel"), ScaleId("network")),
                residual=0.11,
                tolerance=0.40,
                units=Unit("Hz"),
                n_samples=512,
            ),
        ),
        on_failure="preserve_sections",
    )
    return ResolutionPoset(nodes=nodes, relations=relations, maps=maps, gluing=gluing)


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------
def _parcel_support(index: int) -> Support:
    return Support(
        kind="parcel",
        frame=FrameId("subject_surface_RAS"),
        units=Unit("m"),
        psf=PSF(
            kind="integration_kernel",
            fwhm=(0.02, 0.02, 0.006),
            units=Unit("dimensionless"),
            extent_units=Unit("m"),
            kernel_ref=f"parcel_{index}_membership",
        ),
        extent=(0.02, 0.02, 0.006),
        n_elements=1,
        resolution=ScaleId("parcel"),
        label=f"parcel {index}",
    )


def _sim_temporal() -> TemporalSupport:
    return TemporalSupport(clock=ClockId("sim"), dt=DELTA_SIM)


def _region(index: int, rid: str) -> Region:
    support = _parcel_support(index)
    latent = ComponentSpec(
        kind="population",
        shape=(1,),
        units=Unit("Hz"),
        support=support,
        temporal=_sim_temporal(),
        dtype="float32",
        boundary=True,
        resolution=ScaleId("parcel"),
        ledger=_estimable(
            "Hz",
            estimator="repeated_simulation_replicates",
            parameter=1e-2,
            numerical=1e-8,
            bias=(-0.05, 0.05),
        ),
        description="latent regional state x_i(t) of equation T1",
    )
    uncertainty = ComponentSpec(
        kind="uncertainty",
        shape=(1,),
        units=Unit("dimensionless"),
        support=support,
        temporal=_sim_temporal(),
        boundary=False,
        resolution=ScaleId("parcel"),
        ledger=_sensitivity("dimensionless", bias=(-0.10, 0.10), parameter=1e-3),
        description="per-region posterior variance channel",
    )
    hemo = ComponentSpec(
        kind="metabolic",
        shape=(4,),
        units=Unit("dimensionless"),
        support=support,
        temporal=TemporalSupport(
            clock=ClockId("scanner_volume"), dt=DELTA_BOLD, integration_window=0.06
        ),
        boundary=True,
        resolution=ScaleId("parcel"),
        ledger=_sensitivity("dimensionless", bias=(-0.20, 0.20), parameter=4e-2),
        description="Balloon-Windkessel compartments driven by h(s;rho) of T3",
    )

    eeg_port = Port(
        name="eeg_out",
        state_spec=StateSpec(
            components={
                "source_amplitude": ComponentSpec(
                    kind="channels",
                    shape=(1,),
                    units=Unit("A*m"),
                    support=Support(
                        kind="sensor",
                        frame=FrameId("eeg_cap"),
                        units=Unit("V"),
                        psf=PSF(
                            kind="lead_field",
                            units=Unit("V"),
                            extent_units=Unit("m"),
                            kernel_ref="leadfield_3shell_bem_19ch",
                            ledger=_sensitivity("V", bias=(-2e-6, 2e-6), model_class=1e-12),
                        ),
                        n_elements=19,
                        resolution=ScaleId("parcel"),
                        label="19-channel montage",
                    ),
                    temporal=TemporalSupport(
                        clock=ClockId("eeg_amp"), dt=DELTA_EEG, group_delay=1.5e-3,
                        jitter_sd=2e-5,
                    ),
                    boundary=True,
                    ledger=_bounded("A*m", source="dipole_phantom", bias=(-1e-10, 1e-10),
                                    measurement=1e-20),
                )
            }
        ),
        support=Support(
            kind="sensor",
            frame=FrameId("eeg_cap"),
            units=Unit("V"),
            psf=PSF(
                kind="lead_field",
                units=Unit("V"),
                extent_units=Unit("m"),
                kernel_ref="leadfield_3shell_bem_19ch",
            ),
            n_elements=19,
            resolution=ScaleId("parcel"),
        ),
        temporal=TemporalSupport(
            clock=ClockId("eeg_amp"), dt=DELTA_EEG, group_delay=1.5e-3, jitter_sd=2e-5
        ),
        direction="out",
        role="observation",
        modality="eeg",
        source_ids=("eeg_sim_v1",),
        ledger=_bounded("V", source="dipole_phantom", bias=(-2e-6, 2e-6), measurement=1e-12),
    )

    bold_port = Port(
        name="bold_out",
        state_spec=StateSpec(
            components={
                "bold_signal": ComponentSpec(
                    kind="channels",
                    shape=(1,),
                    units=Unit("dimensionless"),
                    support=support,
                    temporal=TemporalSupport(
                        clock=ClockId("scanner_volume"), dt=DELTA_BOLD,
                        integration_window=0.06,
                    ),
                    boundary=True,
                    ledger=_sensitivity("dimensionless", bias=(-0.05, 0.05), measurement=4e-4),
                )
            }
        ),
        support=Support(
            kind="parcel",
            frame=FrameId("subject_surface_RAS"),
            units=Unit("dimensionless"),
            psf=PSF(
                kind="hemodynamic",
                fwhm=(6.0,),
                units=Unit("dimensionless"),
                extent_units=Unit("s"),
                kernel_ref="balloon_windkessel_h_s_rho",
            ),
            extent=(0.02, 0.02, 0.006),
            n_elements=1,
            resolution=ScaleId("parcel"),
        ),
        temporal=TemporalSupport(
            clock=ClockId("scanner_volume"), dt=DELTA_BOLD, integration_window=0.06
        ),
        direction="out",
        role="observation",
        modality="fmri",
        source_ids=("fmri_sim_v1",),
        ledger=_sensitivity("dimensionless", bias=(-0.05, 0.05), measurement=4e-4),
    )

    ports = [eeg_port, bold_port]
    if rid == REGION_IDS[0]:
        ports.append(
            Port(
                name="impulse_in",
                state_spec=StateSpec(
                    components={
                        "drive": ComponentSpec(
                            kind="field",
                            shape=(1,),
                            units=Unit("A/m^2"),
                            support=Support(
                                kind="field",
                                frame=FrameId("stim_device"),
                                units=Unit("A/m^2"),
                                psf=PSF(
                                    kind="gaussian",
                                    fwhm=(0.012, 0.012, 0.010),
                                    units=Unit("A/m^2"),
                                    extent_units=Unit("m"),
                                ),
                                extent=(0.012, 0.012, 0.010),
                                n_elements=1,
                                resolution=ScaleId("parcel"),
                            ),
                            temporal=TemporalSupport(
                                clock=ClockId("stim_computer"), dt=DELTA_SIM
                            ),
                            boundary=True,
                            ledger=_bounded(
                                "A/m^2", source="field_solver_phantom",
                                bias=(-0.5, 0.5), measurement=0.04,
                            ),
                        )
                    }
                ),
                support=Support(
                    kind="field",
                    frame=FrameId("stim_device"),
                    units=Unit("A/m^2"),
                    psf=PSF(
                        kind="gaussian",
                        fwhm=(0.012, 0.012, 0.010),
                        units=Unit("A/m^2"),
                        extent_units=Unit("m"),
                    ),
                    extent=(0.012, 0.012, 0.010),
                    n_elements=1,
                    resolution=ScaleId("parcel"),
                ),
                temporal=TemporalSupport(clock=ClockId("stim_computer"), dt=DELTA_SIM),
                direction="in",
                role="intervention",
                modality="simulated_impulse",
                source_ids=("impulse_sim_v1",),
                ledger=_bounded("A/m^2", source="field_solver_phantom", bias=(-0.5, 0.5),
                                measurement=0.04),
            )
        )

    return Region(
        id=rid,
        label=f"synthetic parcel {index}",
        state=StateSpec(
            components={
                "latent": latent,
                "uncertainty": uncertainty,
                "hemodynamic": hemo,
            }
        ),
        ports=ports,
        atlas_refs=[
            AtlasRef(
                atlas="synthetic_three_parcel",
                version="1.0.0",
                index=index,
                label=rid,
                frame=FrameId("subject_surface_RAS"),
                asset_hash="0" * 64,
                coverage=1.0,
            )
        ],
        authority="fine",
        resolution=ScaleId("parcel"),
        system="cortex",
    )


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def _operators() -> list[OperatorSpec]:
    ops: list[OperatorSpec] = []

    # Self-dynamics: the diagonal of A(vartheta).  Mechanistic, identified by a
    # stated leak-plus-recurrence assumption set rather than by correlation.
    for rid in REGION_IDS:
        ops.append(
            OperatorSpec(
                id=f"self_{rid}",
                src=rid,
                dst=rid,
                family="flow_ode",
                evidence_class="hard",
                mechanistic_status="mechanistic",
                delay_prior=DiracPrior(value=0.0, units=Unit("s")),
                params={
                    "leak_rate": LogNormalPrior(mu=-2.3, sigma=0.4, units=Unit("Hz")),
                    "self_gain": NormalPrior(loc=0.0, scale=0.2, units=Unit("dimensionless")),
                },
                ledger=_estimable(
                    "Hz",
                    estimator="profile_likelihood_over_replicates",
                    parameter=4e-3,
                    numerical=1e-9,
                    bias=(-0.01, 0.01),
                ),
                clock=ClockId("sim"),
                identification=Identification(
                    basis=("assumption_set", "anatomical_prior"),
                    assumption_set="linear_leak_plus_recurrence_with_known_noise_covariance_Q",
                    evidence_source_ids=("eeg_sim_v1", "fmri_sim_v1"),
                ),
                differentiable=True,
                solver="exponential_euler",
                units_in=Unit("Hz"),
                units_out=Unit("Hz"),
            )
        )

    # a -> b carries the unknown conduction delay tau of T1.
    ops.append(
        OperatorSpec(
            id="couple_a_b",
            src="region_a",
            dst="region_b",
            family="delayed_ssm",
            evidence_class="hard",
            mechanistic_status="effective",
            delay_prior=LogNormalPrior(
                mu=-4.6, sigma=0.5, units=Unit("s"),
                provenance="tract length / conduction velocity prior; the benchmark target",
            ),
            params={"gain": NormalPrior(loc=0.4, scale=0.2, units=Unit("dimensionless"))},
            ledger=_estimable(
                "Hz",
                estimator="impulse_response_recovery_under_randomized_input",
                parameter=9e-3,
                numerical=1e-9,
                bias=(-0.02, 0.02),
            ),
            clock=ClockId("sim"),
            identification=Identification(
                basis=("intervention", "randomization"),
                intervention_source_ids=("impulse_sim_v1",),
                evidence_source_ids=("eeg_sim_v1", "impulse_sim_v1"),
                notes="delay and gain are identified by the randomized impulse, not by correlation",
            ),
            distance_m=0.045,
            differentiable=True,
            solver="delayed_exponential_euler",
            units_in=Unit("Hz"),
            units_out=Unit("Hz"),
        )
    )

    ops.append(
        OperatorSpec(
            id="couple_b_c",
            src="region_b",
            dst="region_c",
            family="delayed_ssm",
            evidence_class="hard",
            mechanistic_status="effective",
            delay_prior=DiracPrior(value=5e-3, units=Unit("s")),
            params={"gain": NormalPrior(loc=0.3, scale=0.15, units=Unit("dimensionless"))},
            ledger=_estimable(
                "Hz",
                estimator="impulse_response_recovery_under_randomized_input",
                parameter=7e-3,
                numerical=1e-9,
                bias=(-0.02, 0.02),
            ),
            clock=ClockId("sim"),
            identification=Identification(
                basis=("intervention",),
                intervention_source_ids=("impulse_sim_v1",),
                evidence_source_ids=("eeg_sim_v1", "impulse_sim_v1"),
            ),
            distance_m=0.038,
            units_in=Unit("Hz"),
            units_out=Unit("Hz"),
        )
    )

    # c -> a is supported only by passive covariance, so it is labelled
    # functional. Calling it effective here is exactly what R04 refuses.
    ops.append(
        OperatorSpec(
            id="couple_c_a",
            src="region_c",
            dst="region_a",
            family="spectral_transfer",
            evidence_class="soft",
            mechanistic_status="functional",
            delay_prior=UniformPrior(low=0.0, high=0.02, units=Unit("s")),
            params={"gain": NormalPrior(loc=0.0, scale=0.1, units=Unit("dimensionless"))},
            ledger=_sensitivity("Hz", bias=(-0.08, 0.08), parameter=2e-2, model_class=1e-2),
            clock=ClockId("sim"),
            identification=Identification(
                basis=("passive_correlation",),
                evidence_source_ids=("eeg_sim_v1",),
                notes="statistical association only; common input is not excluded",
            ),
            distance_m=0.061,
            units_in=Unit("Hz"),
            units_out=Unit("Hz"),
        )
    )

    # A deliberately misspecified learned residual, kept inside a preregistered
    # gain ratio and semigroup-tested (thesis sec. 0.3 asks for a misspecified
    # module in the end-to-end slice).
    ops.append(
        OperatorSpec(
            id="residual_b_a",
            src="region_b",
            dst="region_a",
            family="surrogate",
            evidence_class="proposed",
            mechanistic_status="surrogate",
            delay_prior=DiracPrior(value=0.0, units=Unit("s")),
            params={"kernel_scale": LogNormalPrior(mu=-3.0, sigma=0.5, units=Unit("dimensionless"))},
            ledger=_sensitivity("Hz", bias=(-0.05, 0.05), model_class=3e-2, numerical=1e-8),
            clock=ClockId("sim"),
            is_learned=True,
            identification=Identification(
                basis=("simulation_ground_truth",),
                evidence_source_ids=("eeg_sim_v1",),
            ),
            residual=ResidualPolicy(
                rho_max=0.25,
                measured_ratio=0.11,
                validity_set="linear_gaussian_synthetic_regime_sweep",
                report_violations=True,
                norm="l2_energy",
            ),
            semigroup=SemigroupTest(
                tested_step_pairs=((1e-3, 1e-3), (1e-3, 2e-3), (2e-3, 2e-3)),
                permitted_step_pairs=((1e-3, 1e-3), (1e-3, 2e-3), (2e-3, 2e-3)),
                max_residual=3.2e-4,
                tolerance=1e-3,
                units=Unit("Hz"),
                folded_into_intervals=True,
            ),
            distance_m=0.045,
            units_in=Unit("Hz"),
            units_out=Unit("Hz"),
        )
    )
    return ops


# ---------------------------------------------------------------------------
# lineage and splits
# ---------------------------------------------------------------------------
_PARTICIPANTS = ("sub-01", "sub-02", "sub-03")
_SESSIONS = {
    "sub-01": ("sub-01_ses-01", "sub-01_ses-02"),
    "sub-02": ("sub-02_ses-01",),
    "sub-03": ("sub-03_ses-01",),
}
_FOLDS = {"sub-01": "train", "sub-02": "val", "sub-03": "test"}


def _lineage_units() -> tuple[LineageUnit, ...]:
    units: list[LineageUnit] = []
    for p in _PARTICIPANTS:
        units.append(LineageUnit(id=p, kind="participant", n_observations=0))
        for s in _SESSIONS[p]:
            units.append(
                LineageUnit(id=s, kind="session", parent_ids=(p,), n_observations=600)
            )
    return tuple(units)


def _fold_assignments() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in _PARTICIPANTS:
        out[p] = _FOLDS[p]
        for s in _SESSIONS[p]:
            out[s] = _FOLDS[p]
    return out


def _split_policy() -> SplitPolicy:
    return SplitPolicy(
        grouping_keys=("participant", "session"),
        temporal_holdout=TemporalHoldout(key="session_start", boundary=1800.0),
        fold_assignments=_fold_assignments(),  # type: ignore[arg-type]
        leakage_barrier="parent_lineage",
        role_locked=True,
        stimuli_shared_across_folds=False,
    )


def _population(effect_name: str, level: str) -> PopulationStructure:
    return PopulationStructure(
        levels=("participant", "session"),
        units=_lineage_units(),
        n_participants=len(_PARTICIPANTS),
        demographics={"synthetic": True, "age_range": [20, 40]},
        selection_mechanism="complete enumeration of the simulated cohort",
        effects=(
            HierarchicalEffect(
                name=effect_name,
                level=level,  # type: ignore[arg-type]
                parameterization="sum_to_zero",
                shrinkage_prior=NormalPrior(loc=0.0, scale=0.1, units=Unit("dimensionless")),
                recovery_tested=True,
                recovery_report="reports/identifiability/effect_recovery.json",
            ),
        ),
        min_subgroup_n=1,
    )


def _identity(pid: str, label: str) -> Identity:
    return Identity(
        persistent_id=pid,
        version="1.0.0",
        release="synthetic-2026-01",
        file_hashes={f"{pid}.h5": "a" * 64},
        parent_ids=(),
        software_hash="b" * 40,
        container_hash="c" * 40,
        preprocessing_hash="d" * 40,
        mutable_download=False,
        label=label,
    )


def _governance() -> Governance:
    return Governance(
        license="CC0-1.0",
        dua=None,
        purpose_limits=("methods_development",),
        consent_scope="not applicable: fully simulated data, no human participants",
        redistribution="full",
        withdrawal_status="none_pending",
        privacy_class="public",
        retention="indefinite",
        may_release_weights=True,
        may_release_examples=True,
        contains_biometric=False,
    )


# ---------------------------------------------------------------------------
# source cards
# ---------------------------------------------------------------------------
def _eeg_source() -> SourceCard:
    return SourceCard(
        identity=_identity("eeg_sim_v1", "simulated EEG at 1 kHz (T2)"),
        governance=_governance(),
        population=_population("session_gain_offset", "session"),
        spatial=Support(
            kind="sensor",
            frame=FrameId("eeg_cap"),
            units=Unit("V"),
            psf=PSF(
                kind="lead_field",
                units=Unit("V"),
                extent_units=Unit("m"),
                kernel_ref="leadfield_3shell_bem_19ch",
            ),
            n_elements=19,
            resolution=ScaleId("parcel"),
        ),
        temporal=TemporalSupport(
            clock=ClockId("eeg_amp"), dt=DELTA_EEG, group_delay=1.5e-3, jitter_sd=2e-5
        ),
        calibration=_calibration(
            "cal_eeg_chain",
            ("amplitude_unit", "clock_graph", "sensor_device_geometry"),
            residual=1.0e-6,
            tolerance=5.0e-6,
            residual_units="V",
        ),
        observation=ObservationModel(
            name="eeg_instantaneous_mixing",
            forward_physics="y_E[k] = L(l) x(k*Delta_E) + eps_E[k]  (T2)",
            observed_variables=("scalp_potential",),
            preprocessing=("highpass_0p5Hz_zero_phase", "notch_50Hz"),
            likelihood_kind="generative",
            noise_model=NormalPrior(loc=0.0, scale=2e-6, units=Unit("V")),
            calibration_status="calibrated_empirically",
            leakage_checked=True,
            target_ports=tuple(f"{r}.eeg_out" for r in REGION_IDS),
            ledger=_bounded("V", source="dipole_phantom", bias=(-2e-6, 2e-6), measurement=4e-12),
        ),
        intervention=None,
        missingness=Missingness(
            mechanism="mar",
            handling="marginalize",
            planned_missing_channels=(),
            unplanned_missing_fraction=0.02,
            artifact_rejection="amplitude threshold at 150 uV, windows marked not zeroed",
            dropout=0.0,
            attrition=0.0,
        ),
        ledger=_estimable(
            "V",
            estimator="within/between-session variance components over repeated runs",
            measurement=4e-12,
            session=1e-12,
            bias=(-2e-6, 2e-6),
            discrepancy=1e-13,
        ),
        gradient_permission=GradientPermission(
            modules=("operator:*:params", "operator:*:delay", "region:*:state:latent"),
            ports=("*.eeg_out",),
            scales=("parcel",),
            parameter_groups=("observation:eeg_sim_v1:nuisance",),
            frozen=("frame_edge:*", "scale_map:*"),
            evaluation_only=False,
            adapters=(),
            max_weight=1.0,
            notes="EEG may update coupling, delay and its own nuisance terms",
        ),
        role="likelihood",
        split_policy=_split_policy(),
        signal=SignalSpec(
            name="scalp_potential",
            units=Unit("uV"),
            reference="average",
            dynamic_range=(-2.0e2, 2.0e2),
            saturation=2.0e2,
            quantization=1e-2,
            n_channels=19,
        ),
        label="simulated EEG",
        effective_n=180.0,
    )


def _fmri_source() -> SourceCard:
    return SourceCard(
        identity=_identity("fmri_sim_v1", "simulated BOLD at TR = 1 s (T3)"),
        governance=_governance(),
        population=_population("site_intensity_offset", "site"),
        spatial=Support(
            kind="parcel",
            frame=FrameId("subject_surface_RAS"),
            units=Unit("dimensionless"),
            psf=PSF(
                kind="hemodynamic",
                fwhm=(6.0,),
                units=Unit("dimensionless"),
                extent_units=Unit("s"),
                kernel_ref="balloon_windkessel_h_s_rho",
            ),
            extent=(0.02, 0.02, 0.006),
            n_elements=3,
            resolution=ScaleId("parcel"),
        ),
        temporal=TemporalSupport(
            clock=ClockId("scanner_volume"), dt=DELTA_BOLD, integration_window=0.06
        ),
        calibration=_calibration(
            "cal_fmri_chain",
            ("physical_reference_frame", "clock_graph", "deformation_projection"),
            residual=6.0e-4,
            tolerance=2.0e-3,
        ),
        observation=ObservationModel(
            name="hemodynamic_convolution",
            forward_physics=(
                "y_B[n] = M * int_0^{T_h} h(s;rho) x(n*Delta_B - s) ds + eps_B[n]  (T3)"
            ),
            observed_variables=("bold_percent_signal_change",),
            preprocessing=("motion_regression", "parcel_averaging"),
            likelihood_kind="generative",
            noise_model=NormalPrior(loc=0.0, scale=0.5, units=Unit("dimensionless")),
            calibration_status="calibrated_empirically",
            leakage_checked=True,
            target_ports=tuple(f"{r}.bold_out" for r in REGION_IDS),
            ledger=_sensitivity("dimensionless", bias=(-0.10, 0.10), measurement=0.25),
        ),
        intervention=None,
        missingness=Missingness(
            mechanism="mar",
            handling="marginalize",
            unplanned_missing_fraction=0.05,
            artifact_rejection="framewise displacement > 0.5 mm, volumes masked not zeroed",
            dropout=0.0,
            attrition=0.0,
        ),
        ledger=_estimable(
            "dimensionless",
            estimator="phantom-anchored intensity replication across sessions",
            measurement=0.25,
            session=0.04,
            bias=(-0.10, 0.10),
            discrepancy=0.02,
        ),
        gradient_permission=GradientPermission(
            modules=("region:*:state:hemodynamic",),
            ports=("*.bold_out",),
            scales=("parcel",),
            parameter_groups=("observation:fmri_sim_v1:nuisance",),
            frozen=("frame_edge:*",),
            max_weight=1.0,
            notes="BOLD constrains hemodynamics and its own nuisance terms only",
        ),
        role="likelihood",
        split_policy=_split_policy(),
        signal=SignalSpec(
            name="bold_percent_signal_change",
            units=Unit("percent"),
            reference="parcel mean over run",
            dynamic_range=(-5.0, 5.0),
            n_channels=3,
        ),
        label="simulated fMRI",
        effective_n=30.0,
    )


def _impulse_source() -> SourceCard:
    return SourceCard(
        identity=_identity("impulse_sim_v1", "calibrated simulated impulse (B u(t - tau_u))"),
        governance=_governance(),
        population=_population("session_dose_offset", "session"),
        spatial=Support(
            kind="field",
            frame=FrameId("stim_device"),
            units=Unit("A/m^2"),
            psf=PSF(
                kind="gaussian",
                fwhm=(0.012, 0.012, 0.010),
                units=Unit("A/m^2"),
                extent_units=Unit("m"),
            ),
            extent=(0.012, 0.012, 0.010),
            n_elements=1,
            resolution=ScaleId("parcel"),
        ),
        temporal=TemporalSupport(clock=ClockId("stim_computer"), dt=DELTA_SIM),
        calibration=_calibration(
            "cal_impulse_dose",
            ("intervention_dose", "material_field", "calibration_validity"),
            residual=0.02,
            tolerance=0.10,
            residual_units="A/m^2",
        ),
        observation=None,
        intervention=InterventionModel(
            name="single_calibrated_impulse",
            modality="simulated_impulse",
            waveform="monophasic rectangular, 1 ms",
            dose={"peak_current_density": 1.0, "pulse_width": 1e-3},
            dose_units={
                "peak_current_density": Unit("A/m^2"),
                "pulse_width": Unit("s"),
            },
            pose_frame="stim_device",
            field_solver="quasi_static_fem_v1",
            target_ports=("region_a.impulse_in",),
            sham="amplitude-zero pulse with identical timing and audible profile",
            randomized=True,
            a_safe=SafeSet(
                id="A_safe_simulated_impulse",
                constraints={
                    "peak_current_density": (0.0, 2.0),
                    "pulse_width": (1e-4, 2e-3),
                    "duty_cycle": (0.0, 0.05),
                },
                constraint_units={
                    "peak_current_density": Unit("A/m^2"),
                    "pulse_width": Unit("s"),
                    "duty_cycle": Unit("dimensionless"),
                },
                independently_validated=True,
                validator="simulation_safety_envelope_v1 (independent of the optimizer)",
                review_reference="not applicable: simulated cohort, no human participants",
                deferral_policy="refuse",
            ),
            is_prospective_human=False,
            ethics_review=None,
            dose_independently_calibrated=True,
            ledger=_bounded(
                "A/m^2", source="field_solver_phantom", bias=(-0.05, 0.05), measurement=0.01
            ),
        ),
        missingness=Missingness(mechanism="planned", handling="design_factor"),
        ledger=_estimable(
            "A/m^2",
            estimator="randomized dose ladder with repeated delivery",
            measurement=0.01,
            parameter=0.004,
            bias=(-0.05, 0.05),
        ),
        gradient_permission=GradientPermission(
            modules=("operator:couple_a_b:*", "operator:couple_b_c:*"),
            parameter_groups=("intervention:impulse_sim_v1:nuisance",),
            frozen=("frame_edge:*", "scale_map:*"),
            max_weight=1.0,
            notes="the impulse identifies coupling direction, gain and delay",
        ),
        role="likelihood",
        split_policy=_split_policy(),
        signal=SignalSpec(
            name="delivered_current_density",
            units=Unit("A/m^2"),
            reference="device tip",
            dynamic_range=(0.0, 2.0),
            n_channels=1,
        ),
        label="simulated impulse intervention",
        effective_n=120.0,
    )


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------
def build_three_region_schema() -> BrainSchema:
    """Three regions, EEG + fMRI observation ports, one impulse intervention port."""
    return BrainSchema(
        id="three_region_identifiability",
        label="T1-T3 identifiability example (thesis sec. 0.3)",
        version="scwbd-schema/1.0.0",
        identity=Identity(
            persistent_id="schema:three_region_identifiability",
            version="1.0.0",
            release="beta",
            file_hashes={"three_region.py": "e" * 64},
            software_hash="b" * 40,
            container_hash="c" * 40,
        ),
        regions=[_region(i, rid) for i, rid in enumerate(REGION_IDS)],
        operators=_operators(),
        resolution_poset=_poset(),
        clocks=_clocks(),
        frames=_frames(),
        sources=[_eeg_source(), _fmri_source(), _impulse_source()],
        metadata={
            "equations": ["T1", "T2", "T3"],
            "unknown_parameters": ["couple_a_b.delay", "couple_a_b.gain", "leak_rate", "l", "rho"],
            "designs": [
                "eeg_only",
                "fmri_only",
                "joint_native_clock",
                "joint_naive_resampling",
                "joint_plus_impulse",
            ],
        },
    )


def build_three_region_claim() -> ClaimManifest:
    """The claim this example is allowed to make: effective, no overrides."""
    return ClaimManifest(
        id="three_region_identifiability_claim",
        claim_class="effective",
        posterior_class="calibrated_bayesian",
        requires_global_section=False,
        optimizes_intervention=False,
        prospective_human=False,
        target_gates=("G1", "G4"),
        overrides=(),
        disabling_evidence=(
            "If joint native-clock inference and the calibrated impulse fail to raise "
            "likelihood Fisher information for the preregistered subset "
            "(couple_a_b delay and gain) above the EEG-only and fMRI-only designs, or "
            "if interval coverage is below nominal across held-out simulation regimes, "
            "the cross-method integration claim is narrowed to a provenance claim."
        ),
        notes={
            "reference": "paper/thesis_contract.tex sec. 0.3, equations T1-T4",
            "consumer": "scwbd.bench identifiability laboratory",
        },
    )
