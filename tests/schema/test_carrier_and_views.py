"""O-1/O-2 guards, each broken on purpose to check that it can fire.

``reports/decorative_guards.md`` catalogues guards that could not fail. Every
test here has a partner that *breaks the thing being guarded* and asserts the
refusal, so a guard that stopped working would take a test with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.schema.annotations import (
    Annotation,
    AnnotationProvenance,
    AnnotationSet,
    Coverage,
    InadmissibleEvidence,
    Licence,
    SupportMismatch,
    derived_evidence_tier,
    derived_training_status,
)
from scwbd.schema.carrier import Carrier, CarrierElement, View, ViewMismatch, ViewOperator
from scwbd.schema.supports import (
    ArityError,
    ElementSpec,
    OntologyError,
    Support,
    TemporalSupport,
)
from scwbd.schema.support_algebra import (
    common_refinement,
    element_join,
    embed_along,
    project_along,
    restriction_between,
    temporal_common_refinement,
)

F = "subject_surface_RAS"
VEC = ElementSpec.vector3("A*m", F)
SCA = ElementSpec.scalar("A*m")
SCA_N = ElementSpec.scalar("A*m", projected_along="surface_normal")


def parcels(n: int, element: ElementSpec, label: str = "parcel") -> Support:
    return Support(
        kind="parcel", frame=F, units=element.units, n_elements=n,
        element=element, label=label,
    )


# ======================================================================
# ElementSpec -- the arity that Support was missing
# ======================================================================
class TestElementSpec:
    def test_a_vector_of_arity_one_is_not_representable(self):
        with pytest.raises(ValueError, match="arity 3, not 1"):
            ElementSpec(kind="vector", arity=1, units="A*m")

    def test_vector_components_need_a_frame(self):
        with pytest.raises(ValueError, match="component_frame"):
            ElementSpec(kind="vector", arity=3, units="A*m")

    def test_a_vector_is_not_a_projection_of_anything(self):
        with pytest.raises(ValueError, match="only a scalar is a projection"):
            ElementSpec(
                kind="vector", arity=3, units="A*m",
                component_frame=F, projected_along="surface_normal",
            )

    def test_n_elements_is_not_the_dimension(self):
        s = parcels(68, VEC)
        assert s.n_elements == 68
        assert s.n_dof == 204

    def test_a_support_with_no_element_has_no_dimension(self):
        # the pre-2b state: refuses rather than assuming arity 1
        s = Support(kind="parcel", frame=F, units="A*m", n_elements=68)
        with pytest.raises(ValueError, match="no ElementSpec"):
            _ = s.n_dof


# ======================================================================
# element_join -- where orientation is enforced
# ======================================================================
class TestElementJoin:
    def test_bare_scalar_with_vector_refuses(self):
        with pytest.raises(ArityError, match="declares no direction"):
            element_join(SCA, VEC)

    def test_declared_projection_with_vector_joins_to_the_vector(self):
        assert element_join(SCA_N, VEC).arity == 3

    def test_different_dimensions_refuse(self):
        with pytest.raises(OntologyError, match="different dimensions"):
            element_join(ElementSpec.scalar("V"), VEC)

    def test_same_arity_different_frames_refuse(self):
        other = ElementSpec.vector3("A*m", "mni152_RAS")
        with pytest.raises(OntologyError, match="different frames|frames"):
            element_join(VEC, other)

    # -- mutation: if the guard is removed the wrong thing becomes buildable
    def test_the_scalar_vector_join_is_what_blocks_a_silent_orientation_invention(self):
        a = parcels(4, SCA, "scalar_parcels")
        b = parcels(4, VEC, "vector_parcels")
        ia = np.repeat(np.arange(4), 3)
        with pytest.raises(ArityError):
            common_refinement(a, b, ia, ia)
        # and it *succeeds* the moment the direction is declared, so the guard
        # is discriminating rather than blanket
        a2 = parcels(4, SCA_N, "scalar_parcels")
        r = common_refinement(a2, b, ia, ia)
        assert r.support.element.arity == 3
        assert r.provenance["orientation_manufactured_from_a"] is True


# ======================================================================
# the algebra
# ======================================================================
class TestCommonRefinement:
    def setup_method(self):
        self.a = parcels(4, VEC, "A")
        self.b = parcels(3, VEC, "B")
        self.ia = np.repeat(np.arange(4), 3)  # 12 atoms
        self.ib = np.repeat(np.arange(3), 4)
        self.w = np.ones(12)

    def test_cells_are_the_nonempty_intersections(self):
        r = common_refinement(self.a, self.b, self.ia, self.ib, atom_weights=self.w)
        # 4x3 grid of 12 atoms in this arrangement gives 6 non-empty cells
        assert r.n_cells == 6
        assert r.support.n_dof == 18

    def test_restrictions_out_are_exact_and_prolongations_in_are_not(self):
        r = common_refinement(self.a, self.b, self.ia, self.ib, atom_weights=self.w)
        assert r.to_a.direction == "restriction"
        assert r.to_a.manufactures_dof is False
        assert r.from_a.direction == "prolongation"
        assert r.from_a.manufactures_dof is True
        assert r.from_a.unresolved_rank > 0

    def test_a_prolongation_refuses_to_be_used_before_it_is_measured(self):
        r = common_refinement(self.a, self.b, self.ia, self.ib, atom_weights=self.w)
        with pytest.raises(OntologyError, match="declares no prior sd"):
            r.from_a.as_prolongation_inputs()
        # and stops refusing once a *measured* sd is attached
        ok = r.from_a.with_uncertainty(1e-9)
        assert ok.as_prolongation_inputs()["prior_sd_unresolved"] == 1e-9

    def test_zero_is_not_an_admissible_uncertainty(self):
        r = common_refinement(self.a, self.b, self.ia, self.ib, atom_weights=self.w)
        with pytest.raises(OntologyError, match="positive and finite"):
            r.from_a.with_uncertainty(0.0)

    def test_restriction_then_prolongation_is_the_identity_on_the_coarse_side(self):
        r = common_refinement(self.a, self.b, self.ia, self.ib, atom_weights=self.w)
        x = np.arange(self.a.n_dof, dtype=float)
        assert np.allclose(r.to_a.apply(r.from_a.apply(x)), x)

    def test_unassigned_atoms_are_reported_not_averaged_away(self):
        ia = self.ia.copy()
        ia[:3] = -1
        r = common_refinement(self.a, self.b, ia, self.ib, atom_weights=self.w)
        assert r.n_unassigned_atoms == 3

    def test_supports_in_different_frames_refuse(self):
        b = Support(
            kind="parcel", frame="mni152_RAS", units="A*m", n_elements=3,
            element=ElementSpec.vector3("A*m", "mni152_RAS"), label="B",
        )
        with pytest.raises(OntologyError, match="different frames"):
            common_refinement(self.a, b, self.ia, self.ib)

    def test_disjoint_supports_refuse_rather_than_returning_an_empty_map(self):
        ia = np.full(12, -1)
        with pytest.raises(OntologyError, match="share no atom"):
            common_refinement(self.a, self.b, ia, self.ib)

    def test_composed_psf_is_at_least_as_blurred_as_either_parent(self):
        from scwbd.schema.supports import PSF

        a = self.a.model_copy(
            update={"psf": PSF(kind="gaussian", fwhm=(0.02,), units="A*m")}
        )
        b = self.b.model_copy(
            update={"psf": PSF(kind="gaussian", fwhm=(0.03,), units="A*m")}
        )
        r = common_refinement(a, b, self.ia, self.ib, atom_weights=self.w)
        assert r.composed_psf is not None
        assert max(r.composed_psf.fwhm) >= 0.03


class TestProjectionAndEmbedding:
    def test_projection_lowers_arity_and_must_name_its_direction(self):
        v = parcels(5, VEC, "v")
        d = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
        dst, m = project_along(v, d, name="surface_normal")
        assert m.lowers_arity and m.projected_along == "surface_normal"
        assert dst.element.projected_along == "surface_normal"
        assert m.matrix.shape == (5, 15)

    def test_an_unnamed_projection_cannot_be_constructed(self):
        from scwbd.schema.support_algebra import SupportMap

        v = parcels(5, VEC, "v")
        s = parcels(5, SCA, "s")
        with pytest.raises(ArityError, match="without naming the direction"):
            SupportMap(
                src=v, dst=s, matrix=np.zeros((5, 15)), direction="projection",
                method="hand-written", manufactures_dof=False, lowers_arity=True,
            )

    def test_embedding_requires_the_scalar_to_have_declared_its_direction(self):
        s = parcels(5, SCA, "s")
        d = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
        with pytest.raises(ArityError, match="no projected_along"):
            embed_along(s, d, name="surface_normal")

    def test_embedding_under_a_different_name_refuses(self):
        s = parcels(5, SCA_N, "s")
        d = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
        with pytest.raises(ArityError, match="silently change what the numbers mean"):
            embed_along(s, d, name="radial")

    def test_project_then_embed_recovers_the_normal_component_only(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=(5, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        v = parcels(5, VEC, "v")
        sca, proj = project_along(v, d, name="surface_normal")
        _, emb = embed_along(sca, d, name="surface_normal")
        x = rng.normal(size=15)
        back = emb.apply(proj.apply(x))
        # the tangential part is gone and the normal part survives exactly
        assert np.allclose(proj.apply(back), proj.apply(x))
        assert not np.allclose(back, x)

    def test_retained_energy_separates_a_good_map_from_a_bad_one(self):
        # a forward operator that lives entirely in one element's subspace
        v = parcels(4, VEC, "v")
        F_ = np.zeros((2, 12))
        F_[0, 0] = 1.0
        F_[1, 1] = 1.0
        keeps = restriction_between(v, parcels(4, VEC, "v2"), np.arange(4))
        assert keeps.retained_energy(F_) == pytest.approx(1.0)
        misses = restriction_between(v, parcels(1, VEC, "one"), np.zeros(4, int))
        assert misses.retained_energy(F_) < 0.3


class TestTemporalRefinement:
    def test_eeg_against_bold_is_a_computation_over_declared_supports(self):
        eeg = TemporalSupport(clock="eeg_amp", dt=1.0 / 5000.0)
        bold = TemporalSupport(
            clock="scanner_volume", dt=2.0, integration_window=2.0, group_delay=0.0
        )
        r = temporal_common_refinement(eeg, bold)
        assert r.support.dt == pytest.approx(1.0 / 5000.0)
        assert r.to_b_stride == 10000
        assert r.to_b_kernel.shape[0] == 10000  # a boxcar, not a pick
        assert r.delay_b == pytest.approx(1.0)

    def test_incommensurate_clocks_report_their_residual(self):
        a = TemporalSupport(clock="eeg_amp", dt=1.0 / 512.0)
        b = TemporalSupport(clock="scanner_volume", dt=1.0 / 60.1)
        r = temporal_common_refinement(a, b)
        assert r.incommensurability_b > 0.0


# ======================================================================
# Carrier and View
# ======================================================================
class TestCarrierAndView:
    def setup_method(self):
        self.carrier = Carrier(
            id="cortical_moment_field",
            support=parcels(3, VEC),
            frame=F,
            element_ids=("a", "b", "c"),
        )

    def test_n_dof_is_the_sum_of_arities(self):
        assert self.carrier.n_dof == 9
        assert self.carrier.offsets() == (0, 3, 6, 9)

    def test_heterogeneous_elements_give_a_ragged_layout(self):
        het = self.carrier.model_copy(
            update={
                "overrides": (
                    CarrierElement(id="b", spec=ElementSpec.scalar("A*m")),
                )
            }
        )
        assert het.arities() == (3, 1, 3)
        assert het.n_dof == 7
        assert het.offsets() == (0, 3, 4, 7)
        assert het.span("c") == (4, 7)
        assert het.is_homogeneous is False

    def test_an_override_keyed_by_an_unknown_element_refuses(self):
        with pytest.raises(ValueError, match="overrides unknown element"):
            Carrier(
                id="c",
                support=parcels(3, VEC),
                frame=F,
                element_ids=("a", "b", "c"),
                overrides=(CarrierElement(id="zzz", spec=ElementSpec.scalar("A*m")),),
            )

    def test_a_carrier_whose_support_has_no_arity_refuses(self):
        with pytest.raises(ValueError, match="no ElementSpec"):
            Carrier(
                id="c",
                support=Support(kind="parcel", frame=F, units="A*m", n_elements=3),
                frame=F,
            )

    def _view(self, n_ch: int, n_carrier_dof: int, carrier_id: str) -> View:
        return View(
            id="montage",
            carrier=carrier_id,
            operator=ViewOperator(
                kind="lead_field", shape=(n_ch, n_carrier_dof), units="V/(A*m)",
                operator_ref="G",
            ),
            support=Support(
                kind="sensor", frame="subject_head_RAS", units="V",
                n_elements=n_ch, element=ElementSpec.scalar("V"), label="montage",
            ),
            temporal=TemporalSupport(clock="eeg_amp", dt=1e-3),
            ledger=_dummy_ledger(),
        )

    def test_two_montages_with_different_channel_counts_share_one_carrier(self):
        a = self._view(32, 9, self.carrier.id)
        b = self._view(27, 9, self.carrier.id)
        a.check_against(self.carrier)
        b.check_against(self.carrier)
        assert a.operator.shape[0] != b.operator.shape[0]
        assert a.operator.shape[1] == b.operator.shape[1] == self.carrier.n_dof

    def test_a_free_orientation_operator_on_a_scalar_carrier_refuses(self):
        scalar = Carrier(
            id="cortical_moment_field", support=parcels(3, SCA_N), frame=F,
            element_ids=("a", "b", "c"),
        )
        v = self._view(32, 9, scalar.id)  # 9 dof operator, 3 dof carrier
        with pytest.raises(ArityError, match="free-orientation"):
            v.check_against(scalar)

    def test_the_arity_guard_is_what_fires_not_a_generic_shape_check(self):
        # same carrier, an operator that is simply the wrong width for no
        # orientation-related reason -> still refuses, but without the hint
        v = self._view(32, 7, self.carrier.id)
        with pytest.raises(ArityError) as e:
            v.check_against(self.carrier)
        assert "free-orientation" not in str(e.value)

    def test_a_view_naming_another_carrier_refuses(self):
        v = self._view(32, 9, "some_other_field")
        with pytest.raises(ViewMismatch, match="names carrier"):
            v.check_against(self.carrier)

    def test_units_must_compose_not_merely_shapes(self):
        v = View(
            id="montage",
            carrier=self.carrier.id,
            operator=ViewOperator(
                kind="lead_field", shape=(32, 9), units="T/(A*m)", operator_ref="G"
            ),
            support=Support(
                kind="sensor", frame="subject_head_RAS", units="V",
                n_elements=32, element=ElementSpec.scalar("V"), label="m",
            ),
            temporal=TemporalSupport(clock="eeg_amp", dt=1e-3),
            ledger=_dummy_ledger(),
        )
        with pytest.raises(ViewMismatch, match="not dimensionally"):
            v.check_against(self.carrier)

    def test_a_view_support_whose_element_count_contradicts_its_operator_refuses(self):
        with pytest.raises(ValueError, match="dof but its support has"):
            View(
                id="montage",
                carrier=self.carrier.id,
                operator=ViewOperator(
                    kind="lead_field", shape=(31, 9), units="V/(A*m)", operator_ref="G"
                ),
                support=Support(
                    kind="sensor", frame="subject_head_RAS", units="V",
                    n_elements=32, element=ElementSpec.scalar("V"), label="m",
                ),
                temporal=TemporalSupport(clock="eeg_amp", dt=1e-3),
                ledger=_dummy_ledger(),
            )


def _dummy_ledger():
    from scwbd.schema.ledger import UncertaintyLedger

    return UncertaintyLedger(
        variance={"measurement": 0.0},
        bias_interval=(0.0, 0.0),
        bias_status="design_estimable",
    )


# ======================================================================
# O-3 / O-4 / O-7 -- annotations
# ======================================================================
def _prov(source: str = "hansen2022", nc: bool = False) -> AnnotationProvenance:
    return AnnotationProvenance(
        source_key=source,
        citation="Hansen et al. 2022",
        method="parcel mean of the PET map",
        licence=Licence(
            spdx="CC-BY-NC-SA-4.0" if nc else "CC-BY-4.0",
            non_commercial=nc,
            may_reach_checkpoint=not nc,
        ),
    )


def _ann(
    name: str,
    n: int,
    *,
    label: str,
    status: str = "measured",
    admissible: tuple[str, ...] = ("family_separation",),
    bar_reason: str = "",
    n_with_value: int | None = None,
    values_ref: str | None = "asset:x",
) -> Annotation:
    n_val = n if n_with_value is None else n_with_value
    return Annotation(
        id=f"{name}@{label}",
        name=name,
        support=Support(
            kind="parcel", frame=F, units="dimensionless", n_elements=n,
            element=ElementSpec.scalar("dimensionless"), label=label,
        ),
        units="dimensionless",
        status=status,
        provenance=_prov(),
        coverage=Coverage(n_elements=n, n_with_value=n_val),
        admissible_for=admissible,
        bar_reason=bar_reason,
        values_ref=values_ref if n_val else None,
    )


class TestAnnotations:
    def test_a_family_record_cannot_be_read_as_a_parcel_vector(self):
        """The three-times-in-one-day bug, as a type error."""
        fam = _ann("family_id", 9, label="family")
        parcel_support = Support(
            kind="parcel", frame=F, units="dimensionless", n_elements=414,
            element=ElementSpec.scalar("dimensionless"), label="parcel",
        )
        with pytest.raises(SupportMismatch, match="9 elements"):
            fam.assert_on(parcel_support)

    def test_same_size_different_support_still_refuses(self):
        """The dangerous case: two supports that happen to have equal length."""
        a = _ann("x", 68, label="parcel")
        b = Support(
            kind="parcel", frame=F, units="dimensionless", n_elements=68,
            element=ElementSpec.scalar("dimensionless"), label="parcel_scalar_v2",
        )
        with pytest.raises(SupportMismatch, match="same size"):
            a.assert_on(b)

    def test_arity_mismatch_refuses(self):
        a = _ann("normals", 68, label="parcel")
        b = Support(
            kind="parcel", frame=F, units="dimensionless", n_elements=68,
            element=ElementSpec.vector3("dimensionless", F), label="parcel",
        )
        with pytest.raises(SupportMismatch, match="arity"):
            a.assert_on(b)

    def test_the_matching_support_is_accepted(self):
        a = _ann("x", 68, label="parcel")
        assert a.assert_on(a.support) is a

    def test_a_barred_annotation_raises_when_cited(self):
        cyto = _ann(
            "cytoarchitecture", 414, label="parcel",
            status="measured_not_separating", admissible=(),
            bar_reason="von Economo classes fail the spin null globally on every "
            "measured block (receptor p=0.19, myelin+thickness p=0.34)",
        )
        assert cyto.is_barred
        with pytest.raises(InadmissibleEvidence, match="not admissible for"):
            cyto.assert_admissible("family_separation")
        # it is still carried and describable
        assert cyto.status == "measured_not_separating"

    def test_an_empty_admissibility_must_say_why(self):
        with pytest.raises(ValueError, match="admissible for nothing and gives no"):
            _ann("cytoarchitecture", 414, label="parcel", admissible=())

    def test_the_licence_surface_is_one(self):
        with pytest.raises(ValueError, match="non-commercial but declares"):
            Licence(spdx="CC-BY-NC-SA-4.0", non_commercial=True,
                    may_reach_checkpoint=True)

    def test_coverage_must_be_over_this_support(self):
        with pytest.raises(ValueError, match="coverage is over"):
            Annotation(
                id="x", name="x",
                support=Support(
                    kind="parcel", frame=F, units="dimensionless", n_elements=414,
                    element=ElementSpec.scalar("dimensionless"), label="parcel",
                ),
                units="dimensionless", status="measured", provenance=_prov(),
                coverage=Coverage(n_elements=9, n_with_value=9),
                admissible_for=("description",), values_ref="asset:x",
            )

    def test_a_weighted_coverage_must_name_its_weight(self):
        with pytest.raises(ValueError, match="must name the weight"):
            Coverage(n_elements=10, n_with_value=5, weighted_fraction=0.4)

    def test_ambiguous_names_across_supports_refuse_rather_than_taking_the_first(self):
        s = AnnotationSet(
            regions=Support(
                kind="parcel", frame=F, units="dimensionless", n_elements=414,
                element=ElementSpec.scalar("dimensionless"), label="parcel",
            ),
            region_ids=tuple(f"r{i}" for i in range(414)),
            annotations=(
                _ann("timescale", 414, label="parcel"),
                _ann("timescale", 9, label="family"),
            ),
        )
        with pytest.raises(SupportMismatch, match="name the support you want"):
            s.get("timescale")
        assert s.get("timescale", on="family").support.n_elements == 9


class TestDerivedStatus:
    def test_status_follows_the_annotations_and_cannot_be_declared(self):
        with_data = (_ann("receptor", 414, label="parcel",
                          admissible=("state_prior",)),)
        assert derived_training_status(with_data) == "has_regional_data"

    def test_inadmissible_evidence_does_not_make_a_family_trained(self):
        barred = (
            _ann("cytoarchitecture", 414, label="parcel", admissible=(),
                 bar_reason="fails the spin null globally"),
        )
        assert derived_training_status(barred) == "prior_only_untrained"

    def test_an_admissible_annotation_with_no_data_does_not_either(self):
        empty = (
            _ann("receptor", 414, label="parcel", admissible=("state_prior",),
                 status="not_established", n_with_value=0, values_ref=None),
        )
        assert derived_training_status(empty) == "prior_only_untrained"

    def test_evidence_tier_is_derived_the_same_way(self):
        sep = (_ann("receptor", 414, label="parcel",
                    admissible=("family_separation",)),)
        assert derived_evidence_tier(sep) == "measured_separation"
        desc = (_ann("cyto", 414, label="parcel", admissible=("description",)),)
        assert derived_evidence_tier(desc) == "atlas_separation"
        assert derived_evidence_tier(()) == "synthetic"
