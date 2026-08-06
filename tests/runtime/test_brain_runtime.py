"""The generic research runtime of ``ARCHITECTURE.md`` Sec. 6b.

Defects foreclosed:

* a read that cannot be supported returning a number.  Every unsupported read
  returns ``Unresolved(reason=...)``, and ``Unresolved`` is falsey so that
  ``if runtime.read(port):`` fails closed rather than open.
* a port whose clock has not ticked being interpolated and presented as an
  observation.
* a readout without a ledger, or with a partial one.
* a unit or frame mismatch being silently coerced into a scale factor.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.runtime import BrainRuntime, PortSpec, Readout, Unresolved
from scwbd.schema.ledger import VARIANCE_COMPONENTS

_DT = torch.float32


def _runtime(**kwargs) -> BrainRuntime:
    ports = [
        PortSpec(
            name="tms_impulse",
            direction="in",
            units="V/m",
            frame="cortex_surface",
            clock="device",
            dt=1e-3,
            shape=(4,),
            drive=lambda v: v * 10.0,
            description="simulated field impulse into the latent state",
        ),
        PortSpec(
            name="eeg",
            direction="out",
            units="V",
            frame="sensor_cap",
            clock="eeg_amp",
            dt=4e-3,
            shape=(2,),
            readout=lambda x: x[:2] * 1e-6,
            group_delay=2e-3,
            integration_window=1e-3,
        ),
        PortSpec(
            name="bold",
            direction="out",
            units="dimensionless",
            frame="scanner_RAS",
            clock="scanner_volume",
            dt=0.72,
            shape=(1,),
            readout=lambda x: x[:1] * 0.01,
        ),
    ]
    A = -torch.eye(4, dtype=_DT) * 5.0
    return BrainRuntime(
        n_state=4,
        drift=lambda x, t: A @ x,
        ports=ports,
        dt=1e-3,
        seed=7,
        validity_domain={"scope": "simulation_only", "model": "toy_linear"},
        **kwargs,
    )


class TestUnsupportedReadsAreUnresolved:
    def test_an_unknown_port_is_unresolved(self):
        result = _runtime().read("motor_cortex_spike_rate")
        assert isinstance(result, Unresolved)
        assert "no port named" in result.reason
        assert result.missing == ("motor_cortex_spike_rate",)

    def test_reading_an_input_port_is_unresolved(self):
        result = _runtime().read("tms_impulse")
        assert isinstance(result, Unresolved)
        assert "will not be faked" in result.reason

    def test_a_port_that_has_not_fired_is_unresolved_not_zero(self):
        result = _runtime().read("eeg")
        assert isinstance(result, Unresolved)
        assert "has not fired yet" in result.reason
        assert "does not interpolate" in result.reason

    def test_a_slow_port_stays_unresolved_while_a_fast_one_fires(self):
        runtime = _runtime()
        runtime.advance(0.02)
        assert isinstance(runtime.read("eeg"), Readout)
        assert isinstance(runtime.read("bold"), Unresolved)

    def test_unresolved_is_falsey_so_a_careless_branch_fails_closed(self):
        result = _runtime().read("eeg")
        assert not result


class TestMultirateAdvance:
    def test_each_port_fires_on_its_own_clock(self):
        runtime = _runtime()
        step = runtime.advance(0.05)
        assert "eeg" in step.ports_fired
        assert "bold" not in step.ports_fired
        assert step.n_substeps == 50

    def test_a_long_advance_fires_the_slow_port_too(self):
        runtime = _runtime()
        step = runtime.advance(1.5)
        assert {"eeg", "bold"} <= set(step.ports_fired)

    def test_a_readout_reports_its_own_age(self):
        runtime = _runtime()
        runtime.advance(0.02)
        readout = runtime.read("eeg")
        assert isinstance(readout, Readout)
        assert 0.0 <= readout.age_s < 4e-3 + 1e-9
        assert readout.clock == "eeg_amp"
        assert readout.frame == "sensor_cap"

    def test_the_runtime_is_deterministic_under_a_seed(self):
        a, b = _runtime(), _runtime()
        for runtime in (a, b):
            runtime.write("tms_impulse", [1.0, 0.0, 0.0, 0.0], units="V/m", frame="cortex_surface")
            runtime.advance(0.02)
        assert torch.equal(a.read("eeg").value, b.read("eeg").value)

    def test_reset_clears_the_fired_history(self):
        runtime = _runtime()
        runtime.advance(0.02)
        runtime.reset()
        assert isinstance(runtime.read("eeg"), Unresolved)


class TestTypedPorts:
    def test_a_unit_mismatch_is_an_error_not_a_scale_factor(self):
        runtime = _runtime()
        with pytest.raises(ValueError) as exc:
            runtime.write("tms_impulse", [1.0] * 4, units="mV/m", frame="cortex_surface")
        assert "not a scale factor" in str(exc.value)

    def test_a_frame_mismatch_is_an_error(self):
        runtime = _runtime()
        with pytest.raises(ValueError):
            runtime.write("tms_impulse", [1.0] * 4, units="V/m", frame="scanner_RAS")

    def test_writing_to_an_output_port_is_refused(self):
        runtime = _runtime()
        with pytest.raises(ValueError):
            runtime.write("eeg", [0.0, 0.0], units="V", frame="sensor_cap")

    def test_an_out_port_without_a_readout_map_cannot_be_declared(self):
        with pytest.raises(ValueError):
            PortSpec(
                name="x",
                direction="out",
                units="V",
                frame="f",
                clock="c",
                dt=1e-3,
                shape=(1,),
            )

    def test_a_port_without_units_or_a_clock_cannot_be_declared(self):
        with pytest.raises(ValueError):
            PortSpec(
                name="x",
                direction="out",
                units="",
                frame="f",
                clock="c",
                dt=1e-3,
                shape=(1,),
                readout=lambda x: x[:1],
            )
        with pytest.raises(ValueError):
            PortSpec(
                name="x",
                direction="out",
                units="V",
                frame="f",
                clock="",
                dt=1e-3,
                shape=(1,),
                readout=lambda x: x[:1],
            )


class TestEveryReadoutCarriesALedger:
    def test_the_ledger_is_complete(self):
        runtime = _runtime()
        runtime.advance(0.02)
        readout = runtime.read("eeg")
        assert set(readout.ledger.variance) == set(VARIANCE_COMPONENTS)
        assert readout.ledger.has_estimator()

    def test_the_ledger_records_the_group_delay_as_a_bias(self):
        runtime = _runtime()
        runtime.advance(0.02)
        readout = runtime.read("eeg")
        lo, hi = readout.ledger.bias_interval
        assert (hi - lo) == pytest.approx(2 * 2e-3)

    def test_the_ledger_records_staleness_and_the_native_period(self):
        runtime = _runtime()
        runtime.advance(0.02)
        domain = runtime.read("eeg").ledger.validity_domain
        assert domain["native_dt_s"] == 4e-3
        assert domain["scope"] == "simulation_only"
        assert "staleness_periods" in domain

    def test_the_numerical_term_is_measured_not_asserted(self):
        runtime = _runtime()
        runtime.write(
            "tms_impulse", [1.0, 0.5, 0.0, 0.0], units="V/m", frame="cortex_surface"
        )
        runtime.advance(0.05)
        readout = runtime.read("eeg")
        # a real coarse-vs-fine comparison, not an asserted zero
        assert readout.ledger.variance["numerical"] > 0.0

    def test_a_state_that_never_leaves_zero_has_zero_numerical_error(self):
        runtime = _runtime()
        runtime.advance(0.05)
        assert runtime.read("eeg").ledger.variance["numerical"] == 0.0

    def test_the_ledger_says_it_has_no_observation_model_attached(self):
        runtime = _runtime()
        runtime.advance(0.02)
        assert "attach one (scwbd.observe)" in runtime.read("eeg").ledger.notes


class TestTheRuntimeIsNotTheTargetingPath:
    def test_it_imports_nothing_from_the_targeting_path(self):
        """Sec. 6b: the generic runtime is *not* what tms-robotics consumes."""
        import ast
        from pathlib import Path

        import scwbd.runtime.brain_runtime as module

        tree = ast.parse(Path(module.__file__).read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
        forbidden = {"targeting", "head", "backends", "compare", "serving", "frames"}
        assert forbidden.isdisjoint(
            {name.lstrip(".").split(".")[0] for name in imported}
        ), f"brain_runtime imports from the targeting path: {imported}"

    def test_it_has_no_notion_of_a_coil_a_scalp_or_a_feasible_set(self):
        public = {n for n in dir(BrainRuntime) if not n.startswith("_")}
        for token in ("coil", "scalp", "safe", "pose", "efield", "dose"):
            assert not any(token in n.lower() for n in public), f"{token!r} in {public}"

    def test_describe_states_the_scope(self):
        assert _runtime().describe()["scope"] == "simulation_only"
