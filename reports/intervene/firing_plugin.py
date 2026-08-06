"""pytest plugin: record which declared A_safe bounds actually FIRE during a run.

A bound that never appears in the output was never made to refuse by any test
in the selected set -- it is green because nothing exercised it, which is the
`reports/decorative_guards.md` failure mode.
"""
import json
import os

FIRED = set()
CHECKED = set()


def pytest_configure(config):
    from scwbd.intervene.safety import LimitSpec

    original = LimitSpec.check

    def instrumented(self, value):
        CHECKED.add(self.key)
        v = original(self, value)
        if v is not None:
            FIRED.add(f"{self.key}:{v.kind}")
        return v

    LimitSpec.check = instrumented


def pytest_sessionfinish(session, exitstatus):
    from scwbd.intervene.safety import SafetyLimits

    limits = SafetyLimits.load()
    sides = []
    for s in limits.all_specs():
        if s.minimum is not None:
            sides.append(f"{s.key}:below_minimum")
        if s.maximum is not None:
            sides.append(f"{s.key}:above_maximum")
    out = {
        "declared_bound_sides": sorted(sides),
        "fired": sorted(FIRED),
        "never_fired": sorted(set(sides) - FIRED),
        "axes_never_even_checked": sorted(
            {s.key for s in limits.all_specs()} - CHECKED
        ),
        "exitstatus": exitstatus,
    }
    path = os.environ.get("FIRING_OUT", "/tmp/firing.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[firing] wrote {path}")
    print(f"[firing] {len(FIRED)}/{len(sides)} declared bound sides fired")
    print(f"[firing] never fired: {out['never_fired']}")
