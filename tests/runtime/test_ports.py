"""The port contract: declared names in, refusals out, no raw slices anywhere.

The failure this guards against is specific and quiet.  Run 1's state is a
uniform ``(B, T, 454, 28)`` tensor; run 2's is per-family with heterogeneous
widths.  A consumer that learned ``state[..., 0]`` keeps working across that
change and keeps returning a real tensor full of plausible numbers -- of a
different quantity.  Nothing raises, nothing looks wrong, and the number is
wrong.

So each test below either fires a refusal or proves a read carries its own
declaration.  As in ``test_admission.py``, every refusal is paired with the
neighbouring case that must *not* refuse.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.runtime.ports import (
    UNIFORM_FAMILY,
    DeclaredPort,
    LayoutNotDeclared,
    PortContract,
    PortedState,
    RawStateAccessRefused,
    SpanViolation,
    UndeclaredPort,
    UnexportedPort,
)

#: Verbatim from the run-1 checkpoint's own ``state_layout`` key
#: (``checkpoints/scwbd-001-beta/last.pt``), read on 2026-08-06.
RUN1_LAYOUT = {
    "dim": 28,
    "components": [
        {"name": "rate_e", "dim": 1, "offset": 0, "units": "Hz", "clock": "fast",
         "exported": True, "stochastic": True, "description": "excitatory mean-field rate"},
        {"name": "rate_i", "dim": 1, "offset": 1, "units": "Hz", "clock": "fast",
         "exported": True, "stochastic": True, "description": "inhibitory mean-field rate"},
        {"name": "adaptation", "dim": 2, "offset": 2, "units": "dimensionless",
         "clock": "fast", "exported": False, "stochastic": False,
         "description": "adaptation currents"},
        {"name": "spectral", "dim": 16, "offset": 4, "units": "dimensionless",
         "clock": "fast", "exported": True, "stochastic": True,
         "description": "quadrature spectral modes"},
        {"name": "hemo", "dim": 4, "offset": 20, "units": "dimensionless",
         "clock": "slow", "exported": False, "stochastic": False,
         "description": "Balloon-Windkessel s,f,v,q"},
        {"name": "uncertainty", "dim": 4, "offset": 24, "units": "log_var",
         "clock": "meta", "exported": False, "stochastic": False,
         "description": "per-region predictive log-variance"},
    ],
}

#: What run 2 must emit: region/family-indexed, heterogeneous widths.
RUN2_LAYOUT = {
    "entries": [
        {"region": "hippocampal", "component": "k", "numel": 8, "units": "dimensionless",
         "clock": "fast", "boundary": True, "elem_offset": 0},
        {"region": "hippocampal", "component": "rho", "numel": 3, "units": "dimensionless",
         "clock": "slow", "boundary": False, "elem_offset": 8},
        {"region": "cortical_visual", "component": "rate_e", "numel": 1, "units": "Hz",
         "clock": "fast", "boundary": True, "elem_offset": 0},
        {"region": "cortical_visual", "component": "retinotopic", "numel": 32,
         "units": "dimensionless", "clock": "fast", "boundary": True, "elem_offset": 1},
        {"region": "brainstem", "component": "rate_e", "numel": 1, "units": "Hz",
         "clock": "fast", "boundary": True, "elem_offset": 0},
    ]
}


@pytest.fixture
def run1() -> PortContract:
    return PortContract.from_state_layout(RUN1_LAYOUT)


@pytest.fixture
def run2() -> PortContract:
    return PortContract.from_state_layout(RUN2_LAYOUT)


# ---------------------------------------------------------------------------
# reading a layout
# ---------------------------------------------------------------------------

class TestReadingADeclaredLayout:
    def test_the_run_one_layout_reads_as_one_family(self, run1):
        assert run1.families == (UNIFORM_FAMILY,)
        assert run1.is_uniform
        assert run1.width_of(UNIFORM_FAMILY) == RUN1_LAYOUT["dim"] == 28

    def test_the_run_two_layout_reads_as_several_families(self, run2):
        assert set(run2.families) == {"hippocampal", "cortical_visual", "brainstem"}
        assert not run2.is_uniform
        # heterogeneous widths -- the thing a fixed D cannot express
        assert run2.width_of("hippocampal") == 11
        assert run2.width_of("cortical_visual") == 33
        assert run2.width_of("brainstem") == 1

    def test_only_declared_exports_are_consumable(self, run1):
        assert {p.name for p in run1.exported_ports()} == {"rate_e", "rate_i", "spectral"}

    def test_a_missing_layout_refuses_rather_than_returning_an_empty_contract(self):
        for layout in (None, {}):
            with pytest.raises(LayoutNotDeclared) as exc:
                PortContract.from_state_layout(layout)
            assert "did not declare" in str(exc.value)

    def test_an_unrecognised_layout_shape_is_refused_not_guessed(self):
        with pytest.raises(LayoutNotDeclared) as exc:
            PortContract.from_state_layout({"tensor_shape": [454, 28]})
        assert "refuses to guess" in str(exc.value)

    def test_a_port_without_units_cannot_be_declared(self):
        with pytest.raises(ValueError) as exc:
            DeclaredPort(family="f", name="p", dim=1, units="", clock="fast")
        assert "no units" in str(exc.value)

    def test_duplicate_port_declarations_are_refused(self):
        p = DeclaredPort(family="f", name="p", dim=1, units="Hz", clock="fast")
        with pytest.raises(ValueError) as exc:
            PortContract(ports=(p, p))
        assert "duplicate" in str(exc.value)


# ---------------------------------------------------------------------------
# the digest: what a consumer pins
# ---------------------------------------------------------------------------

class TestTheDigestPinsTheConsumableContract:
    def test_repacking_the_same_ports_does_not_change_the_digest(self, run1):
        """Offsets are storage detail; a consumer must not be broken by them."""
        moved = {
            "dim": 28,
            "components": [
                {**c, "offset": 27 - c["offset"]} for c in RUN1_LAYOUT["components"]
            ],
        }
        assert PortContract.from_state_layout(moved).digest() == run1.digest()

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda c: {**c, "name": "rate_excitatory"}, id="rename"),
            pytest.param(lambda c: {**c, "dim": 2}, id="width"),
            pytest.param(lambda c: {**c, "units": "spikes/s"}, id="units"),
            pytest.param(lambda c: {**c, "clock": "slow"}, id="clock"),
            pytest.param(lambda c: {**c, "exported": False}, id="export"),
        ],
    )
    def test_changing_what_a_consumer_depends_on_changes_the_digest(self, run1, mutate):
        comps = [mutate(c) if c["name"] == "rate_e" else c
                 for c in RUN1_LAYOUT["components"]]
        assert PortContract.from_state_layout({"dim": 28, "components": comps}).digest() \
            != run1.digest()

    def test_run_one_and_run_two_have_different_digests(self, run1, run2):
        """The pin fails at load when the model changes underneath the consumer."""
        assert run1.digest() != run2.digest()

    def test_the_exported_digest_ignores_internal_ports(self, run1):
        comps = [
            {**c, "dim": 99} if c["name"] == "hemo" else c
            for c in RUN1_LAYOUT["components"]
        ]
        other = PortContract.from_state_layout({"dim": 123, "components": comps})
        assert other.exported_digest() == run1.exported_digest()
        assert other.digest() != run1.digest()


# ---------------------------------------------------------------------------
# reading values
# ---------------------------------------------------------------------------

class TestReadingThroughPorts:
    def test_a_read_returns_the_declared_span_with_its_units(self, run1):
        state = torch.arange(2 * 454 * 28, dtype=torch.float64).reshape(2, 454, 28)
        ported = PortedState(run1, state)
        v = ported.read(UNIFORM_FAMILY, "spectral")
        assert v.values.shape == (2, 454, 16)
        assert v.units == "dimensionless"
        assert v.clock == "fast"
        assert torch.equal(v.values, state[..., 4:20])

    def test_reading_an_unexported_port_is_refused_by_name(self, run1):
        ported = PortedState(run1, torch.zeros(454, 28, dtype=torch.float64))
        for internal in ("adaptation", "hemo", "uncertainty"):
            with pytest.raises(UnexportedPort) as exc:
                ported.read(UNIFORM_FAMILY, internal)
            assert internal in str(exc.value)
            assert "model-internal state" in str(exc.value)

    def test_reading_an_undeclared_port_is_an_error_not_a_zero(self, run1):
        ported = PortedState(run1, torch.zeros(454, 28, dtype=torch.float64))
        with pytest.raises(UndeclaredPort) as exc:
            ported.read(UNIFORM_FAMILY, "dopamine")
        assert "dopamine" in str(exc.value)

    def test_reading_an_undeclared_family_is_an_error(self, run2):
        ported = PortedState(
            run2,
            {
                "hippocampal": torch.zeros(4, 11, dtype=torch.float64),
                "cortical_visual": torch.zeros(4, 33, dtype=torch.float64),
                "brainstem": torch.zeros(4, 1, dtype=torch.float64),
            },
        )
        with pytest.raises(UndeclaredPort) as exc:
            ported.read("cerebellar", "rate_e")
        assert "cerebellar" in str(exc.value)

    def test_read_many_returns_every_exported_port_of_a_family(self, run2):
        ported = PortedState(
            run2,
            {
                "hippocampal": torch.zeros(4, 11, dtype=torch.float64),
                "cortical_visual": torch.zeros(4, 33, dtype=torch.float64),
                "brainstem": torch.zeros(4, 1, dtype=torch.float64),
            },
        )
        got = ported.read_many("cortical_visual")
        assert {v.port.name for v in got} == {"rate_e", "retinotopic"}
        # the non-exported hippocampal port is not in read_many's output
        assert {v.port.name for v in ported.read_many("hippocampal")} == {"k"}

    def test_the_same_port_name_in_two_families_reads_two_different_spans(self, run2):
        """The exact confusion a flat index cannot express."""
        storage = {
            "hippocampal": torch.zeros(11, dtype=torch.float64),
            "cortical_visual": torch.full((33,), 2.0, dtype=torch.float64),
            "brainstem": torch.full((1,), 7.0, dtype=torch.float64),
        }
        ported = PortedState(run2, storage)
        assert float(ported.read("cortical_visual", "rate_e").values[0]) == 2.0
        assert float(ported.read("brainstem", "rate_e").values[0]) == 7.0


# ---------------------------------------------------------------------------
# the refusals that make it structural
# ---------------------------------------------------------------------------

class TestRawAccessIsRefused:
    @pytest.fixture
    def ported(self, run1):
        return PortedState(run1, torch.zeros(454, 28, dtype=torch.float64))

    def test_indexing_is_refused(self, ported):
        with pytest.raises(RawStateAccessRefused) as exc:
            ported[..., 0]
        assert "read(family, name)" in str(exc.value)

    def test_iteration_is_refused(self, ported):
        with pytest.raises(RawStateAccessRefused):
            list(ported)

    def test_len_is_refused(self, ported):
        with pytest.raises(RawStateAccessRefused) as exc:
            len(ported)
        assert "storage detail" in str(exc.value)

    @pytest.mark.parametrize("attr", ["tensor", "state", "raw", "values", "data"])
    def test_no_attribute_hands_out_the_underlying_tensor(self, ported, attr):
        with pytest.raises(RawStateAccessRefused) as exc:
            getattr(ported, attr)
        assert "no raw tensor" in str(exc.value)

    def test_the_supported_reads_still_work(self, ported):
        """Paired with the refusals above so this is not a wall around nothing."""
        assert ported.contract.is_uniform
        assert ported.families() == (UNIFORM_FAMILY,)
        assert ported.read(UNIFORM_FAMILY, "rate_e").values.shape == (454, 1)


class TestSpanEnforcementUnderNarrowingN1:
    """`padded-family-state` permits padded family state *only* if out-of-span reads are impossible."""

    def test_a_read_past_the_family_span_raises_instead_of_returning_padding(self, run2):
        # cortical_visual declares 33 elements; hand it a 33-wide buffer padded
        # into a 64-wide one by giving it only what a *smaller* family needs.
        ported = PortedState(
            run2,
            {
                "hippocampal": torch.zeros(11, dtype=torch.float64),
                "cortical_visual": torch.zeros(8, dtype=torch.float64),  # truncated
                "brainstem": torch.zeros(1, dtype=torch.float64),
            },
        )
        with pytest.raises(SpanViolation) as exc:
            ported.read("cortical_visual", "retinotopic")
        assert "narrowing `padded-family-state`" in str(exc.value)
        assert "instead of returning padding" in str(exc.value)
        # the port that *does* fit still reads
        assert ported.read("cortical_visual", "rate_e").values.shape == (1,)

    def test_heterogeneous_state_cannot_be_passed_as_one_tensor(self, run2):
        with pytest.raises(SpanViolation) as exc:
            PortedState(run2, torch.zeros(454, 45, dtype=torch.float64))
        assert "per-family mapping" in str(exc.value)
        assert "read into another's span" in str(exc.value)

    def test_a_uniform_contract_may_be_passed_as_one_tensor(self, run1):
        assert PortedState(run1, torch.zeros(454, 28, dtype=torch.float64)) is not None

    def test_missing_family_storage_is_refused(self, run2):
        ported = PortedState(run2, {"hippocampal": torch.zeros(11, dtype=torch.float64)})
        with pytest.raises(SpanViolation) as exc:
            ported.read("brainstem", "rate_e")
        assert "no storage was supplied" in str(exc.value)


class TestTheRemainingRefusalsAlsoFire:
    """Guards I added must not join the class this work exists to close."""

    @pytest.mark.parametrize(
        "kwargs, fragment",
        [
            ({"dim": 0}, "declares dim=0"),
            ({"dim": -1}, "declares dim=-1"),
            ({"clock": ""}, "declares no clock"),
        ],
    )
    def test_a_malformed_port_declaration_is_refused(self, kwargs, fragment):
        base = dict(family="f", name="p", dim=1, units="Hz", clock="fast")
        base.update(kwargs)
        with pytest.raises(ValueError) as exc:
            DeclaredPort(**base)
        assert fragment in str(exc.value)

    def test_ports_of_an_undeclared_family_is_refused(self, run2):
        with pytest.raises(UndeclaredPort) as exc:
            run2.ports_of("cerebellar")
        assert "cerebellar" in str(exc.value)
        assert "declared families" in str(exc.value)
        # the declared ones still answer
        assert len(run2.ports_of("hippocampal")) == 2

    def test_width_of_an_undeclared_family_is_refused(self, run2):
        with pytest.raises(UndeclaredPort):
            run2.width_of("cerebellar")

    def test_ports_can_be_selected_by_clock(self, run1, run2):
        assert {p.name for p in run1.ports_on_clock("slow")} == {"hemo"}
        assert {p.name for p in run1.ports_on_clock("meta")} == {"uncertainty"}
        assert run1.ports_on_clock("nonexistent") == ()
        assert {(p.family, p.name) for p in run2.ports_on_clock("slow")} == {
            ("hippocampal", "rho")
        }

    def test_a_port_with_no_offset_cannot_be_located(self):
        """A layout that names ports but does not place them is refused."""
        contract = PortContract(
            ports=(
                DeclaredPort(family="f", name="p", dim=1, units="Hz",
                             clock="fast", exported=True, offset=None),
            )
        )
        ported = PortedState(contract, torch.zeros(4, dtype=torch.float64))
        with pytest.raises(SpanViolation) as exc:
            ported.read("f", "p")
        assert "declares no offset" in str(exc.value)

    def test_a_region_indexed_entry_may_declare_its_width_as_a_shape(self):
        """`scwbd.compiler.layout` emits `shape`; `numel` is not always present."""
        contract = PortContract.from_state_layout(
            {"entries": [
                {"region": "r", "component": "c", "shape": [4, 3], "units": "Hz",
                 "clock": "fast", "boundary": True, "elem_offset": 0},
            ]}
        )
        assert contract.port("r", "c").dim == 12

    def test_the_summary_names_the_shape_of_the_contract(self, run1, run2):
        assert "6 ports, 3 exported, 1 family" in run1.summary()
        assert "3 families" in run2.summary()
        assert run1.digest()[:12] in run1.summary()

    def test_a_port_value_reports_its_own_identity(self, run1):
        v = PortedState(run1, torch.zeros(454, 28, dtype=torch.float64)).read(
            UNIFORM_FAMILY, "rate_e"
        )
        assert v.qualified == "all_regions.rate_e"
        assert "units='Hz'" in repr(v)
