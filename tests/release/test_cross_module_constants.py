"""Release identity constants are defined in several modules; bind them.

Found by running the whole tree at once (`reports/integration.md`).  Three
modules each define ``SCHEMA_VERSION`` and ``MODEL_DESIGNATION`` as their own
module-level literal, and two define ``THESIS_VERSION``:

===================== ================================================
constant              defined independently in
===================== ================================================
``SCHEMA_VERSION``    ``scwbd.schema.schema``, ``scwbd.runtime.provenance``,
                      ``scwbd.bench.report``
``MODEL_DESIGNATION`` ``scwbd.schema.designation``, ``scwbd.runtime.provenance``,
                      ``scwbd.bench.report``
``THESIS_VERSION``    ``scwbd``, ``scwbd.bench.report``
===================== ================================================

They agree today.  **Nothing made them agree** — no import, no test, no
assertion — so they agree by coincidence of three people typing the same
string, and the coincidence is not load-bearing on anything that would notice
if it broke.  Each is stamped into artifacts by a different writer: the
benchmark reports from ``bench.report``, the runtime provenance block from
``runtime.provenance``, the schema envelope from ``schema.schema``.  A release
that bumped one would emit artifacts disagreeing about which schema they are,
and every individual module's tests would still pass, because no module's
tests can see another module's copy.

This is the cheapest possible repair: not a refactor to a single source (that
touches three owners' surfaces), just an executable statement that the copies
must not drift.  When it fires, the fix is to make them equal *or* to record
deliberately that a module has been versioned separately.

The test is verified by mutation in `reports/integration.md` §4: changing any
single copy makes it fail.
"""

from __future__ import annotations

import scwbd
import scwbd.bench.report as bench_report
import scwbd.runtime.provenance as runtime_provenance
import scwbd.schema.designation as schema_designation
import scwbd.schema.schema as schema_schema


def test_schema_version_agrees_across_every_module_that_stamps_it():
    """Three writers, three literals, one artifact format."""
    copies = {
        "scwbd.schema.schema": schema_schema.SCHEMA_VERSION,
        "scwbd.runtime.provenance": runtime_provenance.SCHEMA_VERSION,
        "scwbd.bench.report": bench_report.SCHEMA_VERSION,
    }
    assert len(set(copies.values())) == 1, (
        "SCHEMA_VERSION has drifted between modules that each stamp it into "
        f"released artifacts: {copies}"
    )


def test_model_designation_agrees_across_every_module_that_stamps_it():
    copies = {
        "scwbd.schema.designation": schema_designation.MODEL_DESIGNATION,
        "scwbd.runtime.provenance": runtime_provenance.MODEL_DESIGNATION,
        "scwbd.bench.report": bench_report.MODEL_DESIGNATION,
    }
    assert len(set(copies.values())) == 1, (
        "MODEL_DESIGNATION has drifted; artifacts would disagree about which "
        f"model they describe: {copies}"
    )


def test_thesis_version_agrees_between_the_package_and_the_bench_report():
    copies = {
        "scwbd": scwbd.THESIS_VERSION,
        "scwbd.bench.report": bench_report.THESIS_VERSION,
    }
    assert len(set(copies.values())) == 1, (
        f"THESIS_VERSION has drifted between the package and its reports: {copies}"
    )
