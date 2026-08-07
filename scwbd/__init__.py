"""SC-WBD-001-beta - compiled, multiresolution, multirate whole-brain modeling.

This package is *not* a validated digital twin of any person, a clinical device,
or evidence that any admitted operator is neurally realized.  Per
``paper/thesis_contract.tex`` sec. 0.6 the build order stops at item 5; build
order item 6 (prospective human TMS/tFUS) is out of scope.

The two subpackages published here are the contract layer every other module
codes against:

* :mod:`scwbd.schema` - typed regions, ports, operators, clocks, supports,
  frames, authority policies, uncertainty status, source roles, immutable
  lineage.
* :mod:`scwbd.compiler` - ``compile(schema, *, claim) -> CompiledModel``, which
  fails closed on all eleven refusals of Table ``tab:compiler-refusals``.
"""

from __future__ import annotations

__version__ = "0.1.0"
#: Thesis version this implementation is bound to.
THESIS_VERSION = "V6"

from . import compiler, schema  # noqa: E402

#: Model designation from ARCHITECTURE.md.  Re-exported from the one definition
#: in ``scwbd.schema.designation`` -- a package constant that repeats the literal
#: is a fifth place for the name to be wrong.
from .schema.designation import MODEL_DESIGNATION as DESIGNATION  # noqa: E402
from .compiler import CompiledModel, compile  # noqa: E402
from .schema import (  # noqa: E402
    SCHEMA_VERSION,
    BrainSchema,
    ClaimManifest,
    CompilerRefusal,
)

__all__ = [
    "__version__",
    "DESIGNATION",
    "THESIS_VERSION",
    "SCHEMA_VERSION",
    "schema",
    "compiler",
    "BrainSchema",
    "ClaimManifest",
    "CompiledModel",
    "CompilerRefusal",
    "compile",
]
