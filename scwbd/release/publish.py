"""Build, gate and (only when told to) push SC-WBD artifacts to the Hub.

Posture
-------
``ARCHITECTURE.md`` §7a: *ship the artifact and label it, never refuse to
produce it*.  This module extends that outward to distribution.  It is built so
that publishing is one command, and so that the honest label is **generated**
rather than written by whoever happens to run it.

Three properties this module is designed around, in order of how badly the
opposite would hurt:

**1. Dry run is the default, and it reaches no network.**
:func:`publish` takes ``dry_run=True`` and the CLI defaults to it.  Nothing
imports ``huggingface_hub`` until the push branch is actually entered, so a dry
run cannot create a repo, cannot upload, and cannot mint remote state through a
library side effect.  ``--push`` is the only way past it and it is a visible
act at the call site, in the same spirit as
``ProvenanceBlock.save(require_attribution=False)``.

**2. The namespace is never guessed.**
``HfApi().whoami()`` resolving to some account is not consent to publish under
it.  :func:`resolve_namespace` reads ``--namespace`` or ``$SCWBD_HF_NAMESPACE``
and raises if neither is set.  There is deliberately no fallback to the logged-in
user: the owner may be mid-switch or publishing under an org, and a default that
silently picks the wrong namespace publishes to the wrong place *successfully*,
which is the failure mode with no error message.

**3. Cards are derived, never restated.**
Every number on a card is read at build time from a file in this repository —
``reports/training/evaluation.json`` for scores, ``assets/MANIFEST.json`` for
asset provenance, the corpus index for corpus size, the registries for
citations.  Nothing here contains a metric of its own.  A figure typed into
this module would drift from its source the moment the source changed, and the
drift would be invisible; this project has already withdrawn several relayed
figures for exactly that reason.

The gate
--------
Attribution is computed by :mod:`scwbd.sources.attribution` from the artifact's
own provenance and ``require_complete()`` is called before any bytes move.  For
several inputs — ``scwbd.anatomy.sources.SRC['tian2020']`` most sharply —
citation *is* the licence condition, so an artifact that cannot state what it
was built from may not be released.  This module does not decide licences; it
reports :func:`scwbd.release.licence.union_of`'s answer so a reader can audit
it, and that union remains the authority.
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "NAMESPACE_ENV",
    "TOKEN_ENV_OVERRIDES",
    "NamespaceError",
    "IdentityMismatch",
    "PublishBlocked",
    "observed_identity",
    "verify_identity",
    "FileSpec",
    "ArtifactPlan",
    "resolve_namespace",
    "plan_anatomy_prior",
    "plan_run1_checkpoint",
    "plan_sim_corpus",
    "plan_run2_pilot",
    "PLANNERS",
    "publish",
]

SCHEMA = "scwbd-publish-plan/1.0.0"

#: Read instead of ``whoami()``. No default: see the module docstring.
NAMESPACE_ENV = "SCWBD_HF_NAMESPACE"

REPO_ROOT = Path(__file__).resolve().parents[2]


class NamespaceError(RuntimeError):
    """No publishing namespace was supplied, and none may be inferred."""


class PublishBlocked(RuntimeError):
    """The artifact cannot be published as specified."""


# ---------------------------------------------------------------------------
# namespace
# ---------------------------------------------------------------------------
def resolve_namespace(
    explicit: str | None = None, *, env: Mapping[str, str] | None = None
) -> str:
    """Return the Hub namespace to publish under, or raise.

    Order: ``explicit`` (the ``--namespace`` flag), then ``$SCWBD_HF_NAMESPACE``.
    There is **no** third step.  In particular this never calls ``whoami()``:
    the account a token happens to resolve to is not a statement about where
    the owner wants these artifacts to live, and an org publish looks exactly
    like a personal publish until it lands in the wrong place.
    """
    environ = os.environ if env is None else env
    value = (explicit if explicit is not None else environ.get(NAMESPACE_ENV, "")) or ""
    value = value.strip().strip("/")
    if not value:
        raise NamespaceError(
            "No publishing namespace. Pass --namespace <user-or-org> or set "
            f"${NAMESPACE_ENV}. This is intentionally not defaulted to the "
            "logged-in account: publishing to the wrong namespace succeeds "
            "silently, so the namespace must be stated."
        )
    if "/" in value or value.startswith("."):
        raise NamespaceError(
            f"{value!r} is not a bare namespace. Pass the user or org only "
            "(the artifact name is supplied by the planner)."
        )
    return value


# ---------------------------------------------------------------------------
# file specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FileSpec:
    """One file that would be uploaded, with the provenance key it resolves to."""

    local: Path
    repo_path: str
    n_bytes: int
    #: Key into ``assets/MANIFEST.json``, when the file is a built asset.
    manifest_key: str | None = None
    #: Anatomy source keys (``scwbd.anatomy.sources.SRC``) this file derives from.
    inputs: tuple[str, ...] = ()
    #: Why this file could not be attributed, if it could not be.
    unattributable: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "local": str(self.local),
            "repo_path": self.repo_path,
            "n_bytes": self.n_bytes,
            "manifest_key": self.manifest_key,
            "inputs": list(self.inputs),
            "unattributable": self.unattributable,
        }


@dataclass
class ArtifactPlan:
    """Everything needed to publish one artifact, and nothing that pushes it."""

    name: str
    repo_type: str  # "model" | "dataset"
    files: tuple[FileSpec, ...] = ()
    card: str = ""
    #: ``AttributionBlock``; typed loosely to keep this module import-light.
    attribution: Any = None
    licence: Any = None
    #: Hard stops. A non-empty list means :func:`publish` refuses.
    blockers: tuple[str, ...] = ()
    #: Things a reader must know that do not stop the publish.
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def repo_id(self, namespace: str) -> str:
        return f"{namespace}/{self.name}"

    @property
    def n_bytes(self) -> int:
        return sum(f.n_bytes for f in self.files)

    def as_dict(self) -> dict[str, Any]:
        att = self.attribution
        return {
            "schema": SCHEMA,
            "name": self.name,
            "repo_type": self.repo_type,
            "n_files": len(self.files),
            "n_bytes": self.n_bytes,
            "files": [f.as_dict() for f in self.files],
            "attribution": att.as_dict() if att is not None else None,
            "licence": self.licence.as_dict() if self.licence is not None else None,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# asset-manifest resolution
# ---------------------------------------------------------------------------
def _load_asset_manifest(assets_manifest: str | Path) -> tuple[Path, dict[str, Any]]:
    from .licence import _resolve_assets_manifest

    mp = _resolve_assets_manifest(assets_manifest)
    if mp is None:
        raise PublishBlocked(
            f"{assets_manifest} could not be resolved. The asset manifest is "
            "what supplies each file's inputs; without it nothing can be "
            "attributed and no anatomy artifact may be published."
        )
    payload = json.loads(mp.read_text())
    return mp, dict(payload.get("assets") or {})


def _inputs_from_artifact(path: Path) -> tuple[str, ...]:
    """Read an asset's contributing source keys out of the asset itself.

    Built ``.npz`` assets carry a ``_meta`` block holding the prior's **own**
    provenance.  Two things in it resolve to ``scwbd.anatomy.sources.SRC`` keys:

    * ``provenance.streams[*].source_key`` — already a registry key;
    * ``provenance.source.url`` / ``provenance.source_url`` — matched against
      the registry's own ``url`` field.

    This is the route ``attribution_for_anatomy`` asks for ("keys must come from
    the prior's own provenance"), and it is strictly more trustworthy than the
    asset manifest because it travels inside the file it describes.  It is used
    as the fallback when ``assets/MANIFEST.json`` has no entry, which is
    currently the case for the two files that make the parcellation 414.
    """
    try:
        import numpy as np

        from ..anatomy.sources import SRC
    except Exception:
        return ()
    try:
        with np.load(path, allow_pickle=True) as z:
            if "_meta" not in z:
                return ()
            meta = z["_meta"].item()
    except Exception:
        return ()
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return ()
    if not isinstance(meta, dict):
        return ()
    prov = meta.get("provenance") or {}
    if not isinstance(prov, dict):
        return ()

    keys: set[str] = set()
    for s in prov.get("streams") or ():
        if isinstance(s, dict) and s.get("source_key"):
            keys.add(str(s["source_key"]))

    by_url: dict[str, str] = {}
    for k, v in SRC.items():
        u = str(v.get("url") or "").rstrip("/")
        if u:
            by_url.setdefault(u, k)
    urls: list[str] = []
    src = prov.get("source")
    if isinstance(src, dict) and src.get("url"):
        urls.append(str(src["url"]))
    if prov.get("source_url"):
        urls.append(str(prov["source_url"]))
    for u in urls:
        hit = by_url.get(u.rstrip("/"))
        if hit:
            keys.add(hit)
    return tuple(sorted(keys))


def _spec_for_asset(
    key: str, assets: Mapping[str, Any], root: Path, *, repo_prefix: str = ""
) -> FileSpec:
    """Build a :class:`FileSpec` for a built asset, resolving its inputs.

    Order: the asset manifest, then the asset's own embedded provenance.  A file
    that resolves to **no** inputs by either route is *not* silently given an
    empty input set.  Empty inputs would read as "derives from nothing", which
    for a licence computation is indistinguishable from "unrestricted" — the
    precise failure ``reports/subcortical_atlas_substitution.md`` exists to
    prevent.  It is recorded as unattributable instead, and the gate refuses it.
    """
    local = root / key
    repo_path = f"{repo_prefix}{key}" if repo_prefix else key
    exists = local.exists()
    n_bytes = local.stat().st_size if exists else 0
    if not exists:
        return FileSpec(
            local=local,
            repo_path=repo_path,
            n_bytes=0,
            unattributable=f"{local} does not exist on disk",
        )
    meta = assets.get(key)
    inputs = tuple((meta or {}).get("inputs") or ())
    if inputs:
        return FileSpec(
            local=local,
            repo_path=repo_path,
            n_bytes=n_bytes,
            manifest_key=key,
            inputs=inputs,
        )
    derived = _inputs_from_artifact(local)
    if derived:
        return FileSpec(
            local=local, repo_path=repo_path, n_bytes=n_bytes, inputs=derived
        )
    return FileSpec(
        local=local,
        repo_path=repo_path,
        n_bytes=n_bytes,
        manifest_key=key if meta is not None else None,
        unattributable=(
            f"{key!r} has no entry in assets/MANIFEST.json and its own _meta "
            "block yields no resolvable source key, so nothing states which "
            "sources it derives from. Its licence and its citation set are "
            "both underivable."
        ),
    )


def _anatomy_licence(keys: Iterable[str], *, policy: Mapping[str, str] | None = None):
    """Union the anatomy source terms for ``keys``, read from ``SRC``."""
    from ..anatomy.sources import SRC
    from .licence import term_from_licence_text, union_of

    terms = []
    for k in sorted(set(keys)):
        rec = SRC.get(k)
        if rec is None:
            continue
        terms.append(
            term_from_licence_text(
                k,
                rec.get("license"),
                provenance="scwbd/anatomy/sources.py::SRC",
                verified=bool(rec.get("license")),
                url=rec.get("url"),
            )
        )
    return union_of(terms, policy=policy)


# ---------------------------------------------------------------------------
# card rendering helpers
# ---------------------------------------------------------------------------
def _yaml_front_matter(fields: Mapping[str, Any]) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            lines.append(f"{k}:")
            lines += [f"- {x}" for x in v]
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _licence_section(union: Any, attribution: Any) -> str:
    """Render the licence + citation section entirely from computed objects."""
    out = ["## Licence and attribution", ""]
    if union is not None:
        out.append(f"Computed union: `{union.summary()}`")
        out.append("")
        nc = union.noncommercial_effective
        sa = union.share_alike_effective
        out.append(f"- non-commercial: **{nc}**")
        if nc is True:
            out.append(f"  - forced by: {', '.join(union.sources_forcing('noncommercial'))}")
        out.append(f"- share-alike: **{sa}**")
        if sa is True:
            out.append(f"  - forced by: {', '.join(union.share_alike_sources)}")
        if union.unknown_sources:
            out.append(
                f"- sources stating no terms: {', '.join(union.unknown_sources)}"
            )
        out.append("")
        out.append(
            "These are derived from each source's own licence text by "
            "`scwbd.release.licence.union_of`, not asserted here."
        )
        out.append("")
    if attribution is not None:
        out.append("### Citations (a licence condition, not a courtesy)")
        out.append("")
        out.append(
            "The Melbourne Subcortex Atlas grants unrestricted use *subject to "
            "citation*; several other inputs carry attribution as their only "
            "obligation. Using this artifact requires reproducing these:"
        )
        out.append("")
        for c in attribution.citations():
            out.append(f"- {c}")
        out.append("")
        out.append("<details><summary>Full attribution block (generated)</summary>")
        out.append("")
        out.append("```")
        out.append(attribution.render().rstrip())
        out.append("```")
        out.append("")
        out.append("</details>")
        out.append("")
    return "\n".join(out)


#: The public source repository. Every card links it so a reader can check any
#: figure against the code that generated it.
PROJECT_URL = "https://github.com/JacobFV/sc-wbd"

#: The project's own licence, set by the owner to the most restrictive term any
#: input imposes. It is a **floor**, not a substitute for the per-artifact
#: computation: an artifact may inherit more than this, never less.
PROJECT_LICENCE = "CC-BY-NC-SA-4.0"


def _project_section() -> str:
    return "\n".join(
        [
            "## The project",
            "",
            f"Source code: <{PROJECT_URL}>",
            "",
            f"The SC-WBD repository itself is licensed **{PROJECT_LICENCE}**, "
            "the most restrictive term any of its inputs imposes. Treat that as "
            "a floor: the licence section above is computed from *this "
            "artifact's* own inputs and an artifact can inherit more than the "
            "floor, never less.",
            "",
            "This is research code. It is not a medical device, not a clinical "
            "tool, and nothing here should be used to make a decision about a "
            "person.",
            "",
        ]
    )


def _provenance_footer(sources: Sequence[tuple[str, str]]) -> str:
    out = [
        "## How this card was produced",
        "",
        "Every figure above was read at build time from a file in the SC-WBD "
        "repository by `scwbd/release/publish.py`. None of them is typed into "
        "the card generator. The sources:",
        "",
    ]
    out += [f"- `{path}` — {what}" for path, what in sources]
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# planner: the anatomy prior
# ---------------------------------------------------------------------------
#: The production 414-parcel path. Kept as data, not scattered through code, so
#: that "what we publish" is one reviewable list.
ANATOMY_CORE_ASSETS: tuple[str, ...] = (
    "derived/parcellations/Schaefer400x7__fsLR-32k.npz",
    "derived/parcellations/Aseg14T__MNI152-1mm.npz",
    "derived/connectome/Schaefer400x7__enigma_hcp__with-Aseg14Tsctx__euclidean.npz",
    "derived/geometry/Schaefer400x7__fsLR-32k__geom.npz",
)

#: The 33 regional maps. Held separate because 20 of them are Hansen-derived and
#: including them flips the whole artifact to CC-BY-NC-SA-4.0. This is a third
#: Hansen route, distinct from the theta route and the connectome route, and it
#: is the one that actually bites on Schaefer400x7.
ANATOMY_MAP_ASSETS: tuple[str, ...] = (
    "derived/maps/Schaefer400x7__fsLR-32k__maps.npz",
)


def plan_anatomy_prior(
    *,
    assets_manifest: str | Path = "assets/MANIFEST.json",
    include_maps: bool = False,
    name: str | None = None,
    repo_root: Path | None = None,
) -> ArtifactPlan:
    """Plan the 414-parcel anatomical prior as a dataset repo."""
    from ..sources.attribution import attribution_for_anatomy

    root = repo_root or REPO_ROOT
    mp, assets = _load_asset_manifest(root / assets_manifest)
    asset_root = mp.parent

    keys = list(ANATOMY_CORE_ASSETS) + (list(ANATOMY_MAP_ASSETS) if include_maps else [])
    specs = tuple(_spec_for_asset(k, assets, asset_root) for k in keys)

    inputs = sorted({i for s in specs for i in s.inputs})
    att = attribution_for_anatomy(inputs)
    # A file with no derivable provenance is an attribution hole, and the block
    # is where holes are recorded so require_complete() can refuse on them.
    att.unattributable = att.unattributable + tuple(
        (s.repo_path, s.unattributable) for s in specs if s.unattributable
    )
    union = _anatomy_licence(inputs)

    blockers = tuple(f"{k}: {why}" for k, why in att.unattributable)
    warnings: list[str] = []
    stale = [s.repo_path for s in specs if s.manifest_key is None and s.inputs]
    if stale:
        warnings.append(
            "assets/MANIFEST.json has no entry for "
            + ", ".join(stale)
            + "; their provenance was read from the assets' own `_meta` blocks "
            "instead. The manifest is stale and should be regenerated by "
            "scwbd.anatomy.build."
        )
    if include_maps:
        warnings.append(
            "Includes the 33 regional maps, 20 of which are Hansen-derived: "
            "this artifact is CC-BY-NC-SA-4.0 (non-commercial AND share-alike)."
        )
    else:
        warnings.append(
            "Excludes the regional maps. The maps file is the one production "
            "Schaefer400x7 asset that carries Hansen (CC-BY-NC-SA-4.0); "
            "omitting it is what keeps this artifact free of the NC-SA term."
        )

    plan = ArtifactPlan(
        name=name or ("scwbd-anatomy-prior-414" + ("-maps" if include_maps else "")),
        repo_type="dataset",
        files=specs,
        attribution=att,
        licence=union,
        blockers=blockers,
        warnings=tuple(warnings),
    )
    plan.card = _anatomy_card(plan, assets=assets, asset_root=asset_root, mp=mp)
    return plan


def _npz_fact(path: Path, key: str) -> Any:
    """Read one array's shape from an artifact. Numbers come from artifacts."""
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as z:
            if key not in z:
                return None
            return z[key]
    except Exception:
        return None


def _anatomy_card(
    plan: ArtifactPlan, *, assets: Mapping[str, Any], asset_root: Path, mp: Path
) -> str:
    conn_key = "derived/connectome/Schaefer400x7__enigma_hcp__with-Aseg14Tsctx__euclidean.npz"
    conn = asset_root / conn_key
    labels = _npz_fact(conn, "labels")
    n_parcels = int(labels.shape[0]) if labels is not None else None
    subcortex = [str(x) for x in labels[-14:]] if labels is not None else []

    nc = plan.licence.noncommercial_effective if plan.licence else None
    fm = _yaml_front_matter(
        {
            "license": "other",
            "license_name": "see-licence-section",
            "task_categories": ["graph-ml"],
            "tags": ["neuroscience", "connectome", "brain", "parcellation", "anatomy"],
            "pretty_name": "SC-WBD anatomical prior (Schaefer-400 + 14 subcortical)",
        }
    )

    n = n_parcels if n_parcels is not None else "unread"
    body = [
        fm,
        "",
        "# SC-WBD anatomical prior",
        "",
        f"A group-average anatomical prior over **{n} parcels** — 400 Schaefer-2018 "
        "cortical parcels (Yeo-7 networks, fsLR-32k) plus 14 subcortical "
        "structures — packaged for whole-brain dynamics modelling.",
        "",
        "## What this is",
        "",
        "- **Parcellation**: Schaefer-400/Yeo-7 on fsLR-32k, with centroids, areas and vertex labels.",
        "- **Connectome**: group-average structural connectivity with an explicit "
        "*evidence grammar* — each edge is classed `hard`, `soft` or `proposed` "
        "rather than presented as a single weight you must trust.",
        "- **Geometry**: Euclidean and geodesic inter-parcel distances.",
        "- **Delays**: not stored as a matrix. The artifact ships *priors* "
        "(log-normal conduction velocity and tortuosity) so a consumer derives "
        "delays with uncertainty rather than inheriting a point estimate.",
        "",
    ]
    if subcortex:
        body += [
            f"The {len(subcortex)} subcortical labels: `" + "`, `".join(subcortex) + "`.",
            "",
        ]
    body += [
        "## What this is not",
        "",
        "- **Not a subject's brain.** Every object is a group average: the "
        "connectome from unrelated healthy adults aged 22–37, the surfaces from "
        "a template. It cannot support an inference about an individual.",
        "- **Not directed.** Tractography does not resolve direction; no edge "
        "here is afferent or efferent.",
        "- **Not synaptic strength.** Weights are streamline counts after "
        "consistency thresholding, not physiology.",
        "- **A zero is not independence**, and a permitted edge is not an active edge.",
        "- **The subcortex is coarse** — the thalamus is a single node.",
        "- **No cerebellum.** Cerebellar parcels are declared absent rather than "
        "included with zero edges.",
        "",
        "The 14 subcortical structures are the aseg structures the HCP connectome "
        "resolves, delineated using Melbourne Subcortex Atlas (Tian 2020) "
        "boundaries. That is **not** the same object as Tian S1, which has 16 "
        "parcels — do not cite it as such.",
        "",
        "## Contents",
        "",
        "| file | bytes | derives from |",
        "|---|---:|---|",
    ]
    for f in plan.files:
        src = ", ".join(f.inputs) if f.inputs else "**unresolved**"
        body.append(f"| `{f.repo_path}` | {f.n_bytes:,} | {src} |")
    body += [
        "",
        _licence_section(plan.licence, plan.attribution),
    ]
    if nc is not True:
        body += [
            "### The Hansen question, stated precisely",
            "",
            "The Hansen receptor atlas is CC-BY-NC-SA-4.0 and it propagates. On "
            "the Schaefer-400 production path there are three routes it could "
            "enter by, and they do not have the same answer:",
            "",
            "| route | carries Hansen? |",
            "|---|---|",
            "| E/I ordering (`theta`) | no — default ordering is `hcp_hierarchy` |",
            "| connectome | no — ENIGMA/HCP streams only |",
            "| regional maps | **yes** — 20 of 33 maps are receptor maps |",
            "",
            "This artifact ships the first two and **omits the maps**, which is "
            "why it is not NC-SA. Note that `Schaefer100x7` and "
            "`DesikanKilliany` *do* carry Hansen through the connectome route — "
            "so a 'no Hansen' check run against the test atlas would be checking "
            "the wrong configuration.",
            "",
        ]
    body += [
        _project_section(),
        _provenance_footer(
            [
                (str(mp), "asset inputs, sizes and per-source licences"),
                ("scwbd/anatomy/sources.py::SRC", "citations and licence text"),
                (conn_key, "parcel count and subcortical labels, read from the array"),
            ]
        ),
    ]
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# planner: the run-1 checkpoint
# ---------------------------------------------------------------------------
def plan_run1_checkpoint(
    *,
    checkpoint_dir: str | Path,
    evaluation: str | Path = "reports/training/evaluation.json",
    config: str | Path = "configs/scwbd_001_beta.yaml",
    name: str = "scwbd-001-beta",
    repo_root: Path | None = None,
) -> ArtifactPlan:
    """Plan the run-1 checkpoint as a model repo, card-first.

    ``checkpoint_dir`` has no default. The weights are not in this worktree and
    ``checkpoints/`` is git-ignored, so any default here would be a guess about
    another agent's filesystem.
    """
    from ..sources.attribution import attribution_from_manifest
    from .manifest import build_manifest

    root = repo_root or REPO_ROOT
    ckpt = Path(checkpoint_dir)
    eval_path = root / evaluation

    blockers: list[str] = []
    warnings: list[str] = []

    if not eval_path.is_file():
        # Report everything visible WITHOUT the evaluation, not just the
        # evaluation.  This used to raise immediately, so the plan reported a
        # single blocker -- which reads as "one thing left" and actually meant
        # "one thing visible".  Two run-1 filenames hid behind this raise for
        # the whole of run 2 and would have refused a correct artifact at the
        # end of a nine-hour job.
        also: list[str] = []
        if not ckpt.is_dir():
            also.append(f"checkpoint directory {ckpt} does not exist or is not a directory")
        else:
            for w in (_final_stage_file(ckpt, root / config), "config.yaml", "provenance.json"):
                if not (ckpt / w).is_file():
                    also.append(f"{ckpt / w} is missing")
        msg = (
            f"{eval_path} not found. The card's every score is read from it; "
            "without it there is no honest card to publish."
        )
        if also:
            msg += "\n    also blocking, independent of the evaluation: " + "; ".join(also)
        raise PublishBlocked(msg)
    ev = json.loads(eval_path.read_text())

    # A stale evaluation is a wrong name on a card, and the card still renders
    # perfectly -- about something else. The verdict string opens with the
    # designation; if it names a different one than the evaluation's own
    # model_id, that JSON came from an older build.
    _mid = str(ev.get("model_id") or "")
    _verdict = str(((ev.get("real_eeg_holdout") or {}).get("verdict")) or "")
    _wrong = {
        n for n in re.findall(r"(?:SC-WBD|scwbd)[-_]\d{3}[-\w]*", _verdict) if n != _mid
    }
    if _mid and _wrong:
        blockers.append(
            f"the evaluation's verdict names {sorted(_wrong)} but its model_id is "
            f"{_mid!r}; that JSON was written by an older build and the card would "
            "quote the wrong model in its most prominent line. Re-run the evaluation."
        )

    if not ckpt.is_dir():
        blockers.append(
            f"checkpoint directory {ckpt} does not exist or is not a directory"
        )
        files: tuple[FileSpec, ...] = ()
    else:
        # The final-stage checkpoint, DERIVED from the config's stage list
        # rather than hardcoded.  This read ["stage_V_individual.pt", ...] --
        # run 1's stage name -- so run 2 would have been blocked at the finish
        # line by a filename belonging to the previous run.  Same class as the
        # designation literals: a run-1 name reached for by a run-2 path.
        weights = _final_stage_file(ckpt, root / config)
        wanted = [weights, "config.yaml", "provenance.json"]
        files = tuple(
            FileSpec(local=ckpt / w, repo_path=w, n_bytes=(ckpt / w).stat().st_size)
            for w in wanted
            if (ckpt / w).is_file()
        )
        for w in wanted:
            if not (ckpt / w).is_file():
                blockers.append(f"{ckpt / w} is missing")

    anat = (ev.get("anatomy") or {})
    is_bio = anat.get("is_biological")
    card_dir, card_notes = _card_dir(root, ckpt, root / config)
    blockers += card_notes
    warnings += _enabled_but_unconsumed(ckpt, card_dir)
    warnings += _unreachable_parameters(ckpt, card_dir)
    manifest = build_manifest(
        card_dir=card_dir,
        config=str(root / config),
        anatomy_is_biological=is_bio,
        assets_manifest=str(root / "assets/MANIFEST.json"),
    )
    att = attribution_from_manifest(manifest, tag=name)
    union = manifest.licence()
    blockers += [f"{k}: {why}" for k, why in att.unattributable]

    if is_bio is False:
        warnings.append(
            "Trained on a SYNTHETIC anatomical prior (provenance "
            "'synthetic_fallback'), not the 414-parcel real prior. This is the "
            "reason it inherits no Hansen term — by accident, not by design."
        )
    split = ((ev.get("real_eeg_holdout") or {}).get("real_split") or {})
    if split.get("verified") is False:
        warnings.append(
            "The evaluation split is NOT verified identical to the training "
            "split; every score on this card rests on that unproven assumption."
        )

    plan = ArtifactPlan(
        name=name,
        repo_type="model",
        files=files,
        attribution=att,
        licence=union,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
    plan.card = _run1_card(plan, ev=ev, eval_rel=str(evaluation))
    return plan


def _run1_card(plan: ArtifactPlan, *, ev: Mapping[str, Any], eval_rel: str) -> str:
    ho = ev.get("real_eeg_holdout") or {}
    arms: Mapping[str, Any] = {
        k: v
        for k, v in (ho.get("results") or {}).items()
        if isinstance(v, dict) and "nll_per_sample" in v
    }
    metric = ho.get("metric", "(metric string absent from evaluation.json)")
    beaten = list(ho.get("scwbd_beaten_by") or [])
    verdict = ho.get("verdict", "")
    paired = ho.get("paired_vs_scwbd") or {}
    split = ho.get("real_split") or {}
    indiv = ho.get("individualization") or {}
    anat = ev.get("anatomy") or {}

    # Identity and framing are DERIVED, because this same function builds the
    # card for run 2.  It hardcoded "SC-WBD-001-beta" as the title, tagged every
    # artifact "control-arm", and opened by calling it a negative result -- so
    # publishing the run-2 treatment arm through it would have shipped the model
    # under the previous model's name, described as its own control. That is the
    # exact state R12 refuses, reached through the card instead of the weights.
    _cfg = ev.get("config") or {}
    _model_cfg = (_cfg.get("model") or {}) if isinstance(_cfg, dict) else {}
    name = str(ev.get("model_id") or plan.name)
    is_treatment = bool(_model_cfg.get("family_state"))

    # Whether ANY stage actually took a gradient on measured data.  `train.py`
    # computes a real-data loss only for these three stage names; run 2 renamed
    # every stage, so none matched and the model trained on simulation alone
    # while its stages were called things like "T1_measured_founding".  Derived
    # from the evaluation's own recorded config rather than assumed, because the
    # honest description of the artifact depends on it.
    _REAL_STAGES = {"III_sliced", "IV_assembly", "V_individual"}
    _stages = ((_cfg.get("train") or {}).get("stages") or []) if isinstance(_cfg, dict) else []
    _stage_names = {
        str(s.get("name")) for s in _stages if isinstance(s, dict) and s.get("name")
    }
    trained_on_measured = bool(_stage_names & _REAL_STAGES)
    # E1: SC-WBD is scored on target/s and the baselines on the raw target, so
    # the two sides are different random variables. True for every artifact this
    # evaluate.py has produced; kept as a flag rather than hardcoded so that
    # fixing evaluate.py removes the disclosure instead of leaving it stale.
    _units_defect = True
    sim_only = bool(_stage_names) and not trained_on_measured
    arm_word = "treatment" if is_treatment else "control"
    lost = bool(beaten)

    fm = _yaml_front_matter(
        {
            "license": "other",
            "license_name": "see-licence-section",
            "tags": (
                ["neuroscience", "eeg", "brain-dynamics"]
                + (["negative-result"] if lost else [])
                + ([] if is_treatment else ["control-arm"])
            ),
            "pretty_name": f"{name} ({arm_word} arm"
            + (", negative result)" if lost else ")"),
        }
    )

    body = [
        fm,
        "",
        f"# {name}",
        "",
        "> **Read this first: this checkpoint loses to copying the last observed "
        "sample forward.** It is published as a negative result and as a "
        + ("reference" if is_treatment else "control")
        + " artifact for others, not as a working model. If you are looking for a "
        "brain-dynamics model that works, this is not it.",
        "",
        (
            "> **This model never saw measured data during training, and four "
            "other training mechanisms were silently off.** This run renamed its "
            "training stages; six gates in the trainer match on the *previous* "
            "run's stage names, and five of the six therefore gave the wrong "
            "answer:\n"
            ">\n"
            "> | gate decides | result for this run |\n"
            "> |---|---|\n"
            "> | admit measured sources | **refused** -- no gradient was ever taken on real EEG |\n"
            "> | per-stage gradient allowlist | **wildcard** -- no restriction applied |\n"
            "> | boundary randomisation of sim inputs | **off** |\n"
            "> | haemodynamic state in the rollout | **off** |\n"
            "> | build the individualizer | **off** -- the stage named for it ran ordinary training |\n"
            "> | admit simulated sources | admitted, and correct only by accident |\n"
            ">\n"
            "> Nothing raised, because every gate fails toward *permissive*: an "
            "unmatched stage name means \"no restriction\" rather than \"unknown "
            "stage\". The scores below are therefore a "
            "**simulation-to-measurement transfer** result from a partially "
            "configured trainer -- not a held-out-performance result -- and a "
            "stage named for measurement is not evidence that measurement "
            "occurred. See `reports/RUN2.md` section 2b."
            if sim_only
            else ""
        ),
        "",
        (
            "> **The comparison below flatters this model, and the correction is "
            "not applied to the numbers.** `evaluate.py` scores SC-WBD on "
            "`y = target / s`, where `s` is each window's own standard deviation, "
            "with the Jacobian folded into the log-variance. Every baseline is "
            "scored on the **raw** target. The algebra is exact and "
            "model-independent: `NLL_scaled = NLL_raw - log s`, so the two sides "
            "are different random variables and SC-WBD's figure is the smaller "
            "one.\n"
            ">\n"
            "> Measured on this test fold, `mean(log s) = 0.5694` nats -- against "
            "a spread of 0.035 nats across the three non-trivial baselines, so the "
            "offset is roughly **17x the entire spread it is being compared "
            "against**. In the baselines' units SC-WBD's NLL is approximately "
            "**3.75** rather than the 3.179 tabulated, and the gap to the best "
            "baseline is about **1.70 nats** rather than 1.13. MSE is off by "
            "`1/s^2` and does not cancel.\n"
            ">\n"
            "> The rescale is harmless during training -- `s` does not depend on "
            "the parameters -- and is a pure unearned advantage at evaluation. No "
            "verdict changes: every paired interval already excluded zero and the "
            "correction moves all of them further from SC-WBD. The numbers are "
            "left as measured rather than silently adjusted, because re-scoring "
            "both sides on the raw target is the fix and arithmetic on published "
            "figures is not."
            if _units_defect
            else ""
        ),
        "",
        "## The headline",
        "",
    ]
    if verdict:
        body += [f"> {verdict}", ""]
    if arms:
        body += [
            f"Metric: {metric}",
            "",
            "| arm | NLL | 95% CI | MSE | params |",
            "|---|---:|---|---:|---:|",
        ]
        for key, a in sorted(arms.items(), key=lambda kv: kv[1]["nll_per_sample"]):
            ci = a.get("nll_ci95") or []
            ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if len(ci) == 2 else ""
            star = " **←**" if key.startswith("scwbd") else ""
            body.append(
                f"| `{key}`{star} | {a['nll_per_sample']:.4f} | {ci_s} | "
                f"{a.get('mse', float('nan')):.4f} | {a.get('n_parameters', 0):,} |"
            )
        body.append("")
    if beaten:
        body += [
            f"Beaten by **{len(beaten)}** baselines: "
            + ", ".join(f"`{b}`" for b in beaten)
            + ".",
            "",
        ]
    if paired:
        body += [
            "Paired participant-clustered differences (positive = SC-WBD worse):",
            "",
            "| vs | Δ NLL | 95% CI | excludes zero |",
            "|---|---:|---|---|",
        ]
        for k, d in sorted(paired.items(), key=lambda kv: -kv[1].get("delta", 0)):
            lo, hi = d.get("ci_lo"), d.get("ci_hi")
            ci_s = f"[{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else ""
            body.append(
                f"| `{k}` | {d.get('delta', float('nan')):+.4f} | {ci_s} | "
                f"{d.get('excludes_zero')} |"
            )
        body.append("")

    body += [
        "## Why it lost — the diagnosis",
        "",
        "Two things, and neither is 'the architecture does not work'. Both are "
        "documented in the repository, not inferred here.",
        "",
        (
            "**1. It is the control arm of our own ablation, shipped under the "
            "treatment arm's name.** The project's thesis requires comparing a "
            "*structured regional state* against *one scalar or pooled vector per "
            "region*. This checkpoint is the second of those. The treatment arm was "
            "never built, so this result is not a test of the thesis and must not be "
            "reported as one. It is still an unexplained defect: a 1.76M-parameter "
            "model losing to persistence is not what the control arm was predicted "
            "to do either."
            if not is_treatment
            else "**1. It is the treatment arm, and that is still not the "
            "ablation.** Regional state here is family-indexed and heterogeneous, "
            "so unlike run 1 these numbers describe the architecture the thesis "
            "argues for rather than its control. But the thesis claim is "
            "*comparative* -- structured regional state against pooled state at "
            "matched capacity -- and the pre-registered ablation needs five "
            "further arms that do not exist: two capacity-matched pooled "
            "controls, a scalar floor, a theta-conditioned control, and a "
            "permuted-family attribution control. Against generic forecasting "
            "baselines this artifact can lose or win without either outcome "
            "attributing anything to the structure. Read it as the candidate arm "
            "measured, not as the hypothesis tested."
        ),
        "",
        "**2. The whole loss is in the variance channel.** On the conditional "
        "*mean* this model beats every baseline including persistence — its MSE "
        "is the best in the table above. It loses on NLL because a single "
        "per-channel scalar (`eeg.log_noise`) sets the predictive variance, was "
        "left to SGD instead of its closed-form optimum, and ended up uniformly "
        "overconfident. The scalar cannot represent horizon-dependence at all; "
        "the baselines' variance can.",
        "",
        "This is a useful shape to know about: **a model can win on point "
        "prediction and still lose decisively on likelihood** because one "
        "uncalibrated scalar dominates the score.",
        "",
        "## What it was trained on",
        "",
    ]
    if anat:
        # Summarise the provenance rather than dumping it.  This printed the
        # entire assembled dict -- roughly four thousand characters of nested
        # JSON on one line -- which is not disclosure, it is the appearance of
        # disclosure. The full object ships in the artifact itself; the card
        # names the atlas, the space, and the sources, and points at the file.
        _prov = anat.get("provenance") or {}
        if isinstance(_prov, str):
            # It arrives as a Python repr string, not a mapping. The first
            # version of this summary checked `isinstance(_prov, dict)`, fell
            # through, and printed the 4 kB dump it was written to prevent --
            # a guard that silently declines to act looks exactly like no guard.
            import ast as _ast

            try:
                _prov = _ast.literal_eval(_prov)
            except (ValueError, SyntaxError):
                pass
        if isinstance(_prov, dict):
            _atlas = _prov.get("atlas", "?")
            _space = f"{_prov.get('space', '?')}/{_prov.get('density', '?')}"
            _srcs = sorted((_prov.get("sources") or {}).keys())
            _sub = (_prov.get("subcortical_atlas") or {}).get("name", "?")
            _prov_line = (
                f"atlas `{_atlas}` in `{_space}`, subcortex `{_sub}`, "
                f"{len(_srcs)} declared sources ({', '.join(_srcs)})"
            )
        else:
            _prov_line = f"`{_prov}`"
        body += [
            f"- Anatomy: {anat.get('n_regions')} regions — {_prov_line}; "
            f"`is_biological = {anat.get('is_biological')}`. Full provenance, "
            "including every licence and citation, is carried inside the "
            "checkpoint under `extra.anatomy` and in `reports/anatomy_prior.md`.",
        ]
        if anat.get("is_biological") is False:
            body.append(
                "  - **This is a synthetic ellipsoid stand-in, not real anatomy.** "
                "The real 414-parcel prior never reached the model; an attribute "
                "lookup silently fell back. Anything on this card about anatomy "
                "describes the stand-in."
            )
    lf = ev.get("lead_field") or {}
    if lf:
        body.append(
            f"- Lead field: {lf.get('n_channels')} channels, provenance "
            f"`{lf.get('provenance')}`, individual head model: "
            f"`{lf.get('individual_head_model')}`."
        )
    if ho.get("n_train_participants") is not None:
        # "train" here names the EVALUATION's fitting split for the baselines,
        # not anything this model was fit to. Under a heading called "what it was
        # trained on", the unqualified word was the most misleading line on the
        # card for a sim-only run.
        _split_what = (
            "Evaluation split (this model was **not** fit to any of it — see the "
            "disclosure above; the baselines are fit to the first half)"
            if sim_only
            else "Split"
        )
        body.append(
            f"- {_split_what}: {ho.get('n_train_windows')} fitting windows / "
            f"{ho.get('n_train_participants')} participants; "
            f"{ho.get('n_test_windows')} test windows / "
            f"{ho.get('n_test_participants')} participants, participant-disjoint."
        )
    body.append("")

    body += ["## Known defects", ""]

    # Derived from the artifact, not asserted: two baselines identical means the
    # subject-specific arm never ran, and that is a fact about THIS evaluation
    # rather than a general caveat.
    _res = ho.get("results") or {}
    _ss, _ar = _res.get("subject_specific_ar"), _res.get("ar16")
    if _ss and _ar and _ss.get("nll_per_sample") == _ar.get("nll_per_sample"):
        body.append(
            "- **The hardest baseline was not actually run, and reports itself "
            "healthy.** `subject_specific_ar` is bit-for-bit identical to `ar16`: "
            "the participant-disjoint split leaves no test participant with a "
            "fitted model, so every scored window routes to the pooled fallback. "
            "Its own `describe()` reports `n_subject_models=8, fallback_subjects=0` "
            "— which reads as healthy. A field only ever written on success is not "
            "a record. **Read the table as five distinct comparators, not six**, "
            "and note that the strongest one the thesis names is absent."
        )

    if (ho.get("individualization") or {}).get("applied") is False:
        body.append(
            "- **Individualisation cannot be measured on this holdout at all — "
            "not by retraining, not by patching the evaluation.** Refusal R10 "
            "makes the folds participant-disjoint, so no held-out person has a "
            "fitted person effect: the between-participant spread of the applied "
            "theta shift on the test fold is exactly `0.000e+00`. Every held-out "
            "person receives the identical population term. Measuring "
            "individualisation needs a *within-participant temporal* split, "
            "reported as a different claim. This is a property of the design, not "
            "a defect of this run, and it is why the individualisation figures "
            "here are absent rather than poor."
        )

    if sim_only:
        body.append(
            "- **Five of six training-stage gates gave the wrong answer** (see the "
            "disclosure at the top). No measured-data gradient, no per-stage "
            "gradient restriction, no boundary randomisation, no haemodynamic "
            "state in the rollout, and no individualizer. A complete fix existed "
            "in the repository before this run started and was not applied; six "
            "tests naming the defect were failing on the main branch throughout. "
            "This is the defect that most changes how the scores should be read."
        )
    for w in plan.warnings:
        body.append(f"- {w}")
    if split.get("verified") is False and split.get("note"):
        body.append(f"- Verbatim from `{eval_rel}`: *{split['note']}*")
    if indiv.get("n_individualised_participants") == 0:
        body.append(
            f"- Individualisation did not happen: "
            f"{indiv.get('n_participants_scored')} participants scored, "
            f"{indiv.get('n_individualised_participants')} individualised, "
            f"{indiv.get('n_at_initialisation')} still at initialisation."
        )
    # Was UNCONDITIONAL, and said "four distinct baselines" beside a block two
    # dozen lines above that says five. Both were literals about one particular
    # evaluation; the count and the condition are now read off the artifact, so a
    # protocol that no longer emits the duplicate row does not leave the card
    # asserting a duplication that is not there.
    _dropped = (ho.get("dropped_baselines") or {}).get("subject_specific_ar")
    _n_baselines = len([k for k in arms if k != name])
    if _ss and _ar and _ss.get("nll_per_sample") == _ar.get("nll_per_sample"):
        body.append(
            f"- Two baselines in the table, `ar16` and `subject_specific_ar`, are "
            f"bit-identical: the participant-disjoint split routes every test "
            f"window to the `ar16` fallback. Read the table as "
            f"{max(_n_baselines - 1, 0)} distinct baselines."
        )
    elif _dropped:
        body.append(f"- `subject_specific_ar` is not in the table. {_dropped}")
    body.append("")

    body += [
        "## What it is legitimately good for",
        "",
        (
            "- A **control artifact**: an equal-capacity pooled-state model with "
            "published weights and a published loss, for anyone running the same "
            "ablation."
            if not is_treatment
            else "- A **treatment-arm artifact**: family-indexed heterogeneous "
            "regional state with published weights and a published loss, for "
            "anyone running the same ablation against their own control"
            + (
                " — provided that control is trained the same way. These weights "
                "saw simulation only (see above), so a control fitted to "
                "recordings is not a comparison of *state structure*; it is a "
                "comparison of what each model was shown."
                if sim_only
                else "."
            )
        ),
        "- A **worked example of a variance-channel failure**, with the mean/"
        "variance decomposition available in the repository.",
        "- It is **not** evidence for or against the SC-WBD thesis.",
        "",
        _licence_section(plan.licence, plan.attribution),
        _project_section(),
        _provenance_footer(
            [
                (eval_rel, "every score, CI, parameter count and split size"),
                ("configs/scwbd_001_beta.yaml", "the training mixture"),
                ("scwbd/sources/cards/*.yaml", "dataset citations and licences"),
                ("reports/scope_gap.md, reports/training/p0_variance_channel.md",
                 "the two diagnoses, stated in prose above"),
            ]
        ),
    ]
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# planner: the simulated corpus (documented subset)
# ---------------------------------------------------------------------------
def plan_sim_corpus(
    *,
    corpus_dir: str | Path = "/data/scwbd/sim_corpus_414",
    n_shards: int = 1,
    name: str = "scwbd-sim-corpus-414-subset",
    repo_root: Path | None = None,
) -> ArtifactPlan:
    """Plan a *documented subset* of the simulated corpus.

    The full corpus is tens of gigabytes and grows while training runs. A subset
    that exists and is described beats a full mirror that never gets uploaded,
    and the index is published in full so a reader can see exactly what was left
    out.
    """
    from ..sources.attribution import attribution_for_anatomy

    root = repo_root or REPO_ROOT
    cdir = Path(corpus_dir)
    blockers: list[str] = []
    index_path = cdir / "index_fast.json"
    if not index_path.is_file():
        raise PublishBlocked(
            f"{index_path} not found; the corpus index is the only thing that "
            "states what the corpus contains, so no honest subset can be cut."
        )
    index = json.loads(index_path.read_text())

    shards = list(index.get("shards") or [])
    chosen = shards[:n_shards]
    files = [
        FileSpec(
            local=index_path, repo_path="index_fast.json", n_bytes=index_path.stat().st_size
        )
    ]
    for sh in chosen:
        rel = sh.get("path") or sh.get("file") or ""
        p = (cdir / rel) if rel else None
        if p is not None and p.is_file():
            files.append(
                FileSpec(local=p, repo_path=f"shards/{p.name}", n_bytes=p.stat().st_size)
            )
        else:
            blockers.append(f"shard {rel!r} listed in the index is not on disk at {p}")

    # The corpus was simulated on the real 414 prior; its anatomy attribution is
    # the prior's. Derived from the index's own anatomy block, not assumed.
    mp, assets = _load_asset_manifest(root / "assets/MANIFEST.json")
    specs = [_spec_for_asset(k, assets, mp.parent) for k in ANATOMY_CORE_ASSETS]
    inputs = sorted({i for s in specs for i in s.inputs})
    att = attribution_for_anatomy(inputs)
    att.unattributable = att.unattributable + tuple(
        (s.repo_path, s.unattributable) for s in specs if s.unattributable
    )
    blockers += [f"{k}: {why}" for k, why in att.unattributable]
    union = _anatomy_licence(inputs)

    plan = ArtifactPlan(
        name=name,
        repo_type="dataset",
        files=tuple(files),
        attribution=att,
        licence=union,
        blockers=tuple(blockers),
        warnings=(
            f"This is a subset: {len(chosen)} of {len(shards)} shards. The full "
            "index is published so the omission is visible.",
        ),
    )
    plan.card = _corpus_card(plan, index=index, n_total_shards=len(shards),
                             n_published=len(chosen), index_rel=str(index_path))
    return plan


def _corpus_card(
    plan: ArtifactPlan, *, index: Mapping[str, Any], n_total_shards: int,
    n_published: int, index_rel: str,
) -> str:
    anat = index.get("anatomy") or {}
    fm = _yaml_front_matter(
        {
            "license": "other",
            "license_name": "see-licence-section",
            "tags": ["neuroscience", "simulation", "brain-dynamics", "time-series"],
            "pretty_name": "SC-WBD simulated whole-brain corpus (414-parcel, subset)",
        }
    )
    backends: dict[str, int] = {}
    for sh in index.get("shards") or []:
        b = str(sh.get("backend") or sh.get("spec", {}).get("backend") or "unknown")
        backends[b] = backends.get(b, 0) + 1

    body = [
        fm,
        "",
        "# SC-WBD simulated whole-brain corpus — documented subset",
        "",
        f"Whole-brain simulated trajectories on the {anat.get('n_regions')}-parcel "
        "anatomical prior, across multiple neural-mass backends.",
        "",
        f"**This repository holds {n_published} of {n_total_shards} shards.** The "
        "complete index is published alongside, so you can see every shard that "
        "exists and exactly which ones are here.",
        "",
        "## Corpus totals (the full corpus, not this subset)",
        "",
        f"- Trajectories: {index.get('total_trajectories'):,}",
        f"- Trajectory-seconds: {index.get('total_trajectory_seconds'):,}",
        f"- Shards: {n_total_shards}",
        f"- Generated at git `{index.get('git_sha')}`, {index.get('created_utc')}",
        "",
    ]
    if backends:
        body += ["Shards per backend:", ""]
        body += [f"- `{b}`: {n}" for b, n in sorted(backends.items())]
        body.append("")
    if anat:
        body += [
            "## Anatomy it was simulated on",
            "",
            f"- {anat.get('n_regions')} regions "
            f"({anat.get('n_cortex')} cortex, {anat.get('n_subcortex')} subcortex, "
            f"{anat.get('n_cerebellum')} cerebellum)",
            f"- density {anat.get('density')}, mean tract length "
            f"{anat.get('mean_tract_length_mm')} mm, frame `{anat.get('frame')}`",
            "",
        ]
    body += [
        "## What this is not",
        "",
        "- **Not measured data.** Every trajectory is simulated. It carries the "
        "biases of its generative model, not of a brain.",
        "- **Not a benchmark with a ground-truth answer key** for anything about "
        "real neural dynamics.",
        "- The per-shard `edges_by_class` counts recorded in the index come from "
        "a fallback heuristic, **not** from the anatomical prior's tractography "
        "evidence grammar. Use the prior's own `evidence` array if you need the "
        "real edge classes.",
        "",
        _licence_section(plan.licence, plan.attribution),
        _project_section(),
        _provenance_footer(
            [
                (index_rel, "all corpus totals, shard list and anatomy block"),
                ("assets/MANIFEST.json", "the anatomical prior's inputs"),
            ]
        ),
    ]
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# planner: the run-2 pilot (path prepared; artifact not yet on disk)
# ---------------------------------------------------------------------------
def _unreachable_parameters(ckpt: Path, card_dir: str) -> list[str]:
    """Parameters no enabled card's grant pattern can name, measured on the weights.

    The largest thing wrong with this artifact, and it is invisible in every
    number on the card. The regional modules were renamed ``local`` ->
    ``family_local``, ``residual`` -> ``family_residual``, ``readout`` ->
    ``family_readout`` when the family-padded architecture landed; the source
    cards still grant ``local.*``, ``residual.*``, ``readout.*``. An unmatched
    glob is not an error -- it is an empty permission set -- so the run trained
    whatever *was* matched, converged, and shipped.

    Derived here rather than asserted, from the checkpoint's own parameter names
    and the cards the run recorded, so it cannot drift from either.
    """
    import fnmatch

    try:
        import torch
        import yaml
    except Exception:
        return []

    stages = sorted(Path(ckpt).glob("stage_*.pt"))
    if not stages:
        return []

    names: set[str] = set()
    sizes: dict[str, int] = {}
    for f in stages:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        for container, sub in (("", ck.get("model")), ("posterior", ck.get("posterior"))):
            if not isinstance(sub, dict):
                continue
            for k, v in sub.items():
                full = f"{container}.{k}" if container else k
                names.add(full)
                try:
                    sizes[full] = int(v.numel())
                except Exception:
                    sizes.setdefault(full, 0)
    if not names:
        return []

    # The cards AS OF THE RUN, not as of today.
    #
    # This read the live card directory, so it characterised a finished run using
    # current configuration. Caught by comparing the published card against a
    # freshly generated one: the live card says 88.7% and regenerating now says
    # 0.9%, because the card patterns were repaired after the run. Republishing
    # would have silently erased the artifact's headline finding and replaced it
    # with a number describing a run that never happened.
    #
    # Same class as `_card_dir` reading the LEGACY directory and as the test that
    # pins run 2's defect: fixing the configuration does not retrain the weights,
    # so anything said about the weights must be computed from the configuration
    # they were trained under. The checkpoint records `git_sha`; the cards at that
    # commit are the ones that governed it.
    sha = ""
    for f in stages:
        try:
            sha = str(torch.load(f, map_location="cpu", weights_only=False).get("git_sha") or "")
        except Exception:
            continue
        if sha:
            break

    def _cards_at(rev: str) -> list[tuple[str, str]]:
        import subprocess

        rel = Path(card_dir)
        try:
            rel = rel.relative_to(Path.cwd())
        except ValueError:
            pass
        try:
            names = subprocess.run(
                ["git", "ls-tree", "--name-only", f"{rev}:{rel}"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.split()
        except Exception:
            return []
        out = []
        for n in names:
            if not n.endswith(".yaml"):
                continue
            try:
                out.append((n, subprocess.run(
                    ["git", "show", f"{rev}:{rel}/{n}"],
                    capture_output=True, text=True, timeout=30, check=True,
                ).stdout))
            except Exception:
                return []
        return out

    # Run 2's checkpoint records `af568cf...-dirty`: the working tree carried
    # uncommitted changes when it trained. `-dirty` is not a rev, so it is
    # stripped and the base commit used -- and the fact is disclosed, because the
    # cards that governed the run may have differed from the ones at that commit
    # in exactly the way the suffix warns about.
    dirty = sha.endswith("-dirty")
    base = sha[: -len("-dirty")] if dirty else sha
    historical = _cards_at(base) if base else []
    if not historical:
        return [
            "The share of this model that could not receive a gradient is NOT "
            f"stated here. It must be computed from the source cards as of the run "
            f"(git_sha {sha or 'unrecorded'}), and those could not be read. "
            "Computing it from the cards currently on disk would describe a run "
            "that never happened -- the cards have been edited since, and editing "
            "a card does not retrain a checkpoint."
        ]

    pats: list[str] = []
    for name, text in historical:
        try:
            card = yaml.safe_load(text) or {}
        except Exception:
            continue
        # A card that grants nothing cannot make anything reachable, and one
        # carrying a bare "*" would make everything reachable -- neither belongs
        # in a question about which patterns actually name a parameter.
        if not card.get("enabled", True):
            continue
        for pat in card.get("gradient_permission") or []:
            pat = str(pat).split("#")[0].strip()
            if pat and pat != "*":
                pats.append(pat)
    if not pats:
        return []

    dead = [n for n in sorted(names) if not any(fnmatch.fnmatch(n, p) for p in pats)]
    if not dead:
        return []
    mods = sorted({n.split(".")[0] for n in dead})

    # Counted from the checkpoint's own parameter report, not from state-dict
    # tensor sizes. A state dict carries buffers as well as parameters -- here
    # the 4.1M-element connectome among them -- and dividing by that total gives
    # 26.8% where the honest figure is 88.7%. A buffer is not a parameter that
    # failed to train; it is not a parameter.
    report = {}
    for f in stages:
        try:
            rep = (torch.load(f, map_location="cpu", weights_only=False).get("extra") or {}).get(
                "parameter_report"
            )
        except Exception:
            continue
        if isinstance(rep, dict) and rep:
            report = rep
            break
    if not report:
        return [
            f"The modules {mods} are named by no enabled card's "
            "`gradient_permission`, so they received no gradient during this run. "
            "The checkpoint carries no parameter report, so the share of the "
            "model that represents is not stated here rather than being "
            "estimated from tensor sizes, which count buffers as parameters."
        ]
    n_all = int(report.get("TOTAL") or 0) or sum(
        int(v) for k, v in report.items() if k != "TOTAL"
    )
    n_dead = sum(int(v) for k, v in report.items() if k != "TOTAL" and k in set(mods))
    pct = (100.0 * n_dead / n_all) if n_all else 0.0
    provenance = (
        f" Computed from the source cards at `{base[:7]}`, the commit this "
        "checkpoint records."
        + (
            " That commit is recorded with a `-dirty` suffix, so the tree that "
            "trained carried uncommitted changes and the cards it used may differ "
            "from the cards at the commit."
            if dirty
            else ""
        )
    )
    # The renaming story is RUN 2's and must not be told about a run that fixed
    # it. `local` -> `family_local` was run 2's defect; run 3 trained 99.98% of
    # its parameters and the cards grant the current names.
    renamed = {"local", "residual", "readout", "family_local", "family_residual", "family_readout"}
    tells_the_rename_story = bool(set(mods) & renamed)

    if n_dead == 0:
        # Modules named by no card that carry no COUNTED parameters. Emitting the
        # "N of M parameters (0.0%)" headline here is self-contradictory and reads
        # as an alarm; it did, on run 3's card, beside a module with no entry in
        # the parameter report at all.
        return [
            f"The modules {mods} are named by no card's `gradient_permission`. "
            "They carry no parameters in this checkpoint's parameter report, so "
            "the share of the model that could not receive a gradient is "
            "**0.0%** -- this is a completeness note about the cards, not a "
            "finding about the weights." + provenance
        ]

    head = (
        f"**{n_dead:,} of {n_all:,} parameters ({pct:.1f}%) could not receive a "
        f"gradient from any enabled source card during this run.** The modules "
        f"{mods} are named by no card's `gradient_permission`, so they sat at "
        "their initialisation for every step while still taking part in the "
        "forward pass."
    )
    if tells_the_rename_story:
        head += (
            " This is a string mismatch, not a curriculum decision: the "
            "regional modules were renamed `local` -> `family_local`, `residual` "
            "-> `family_residual`, `readout` -> `family_readout`, and the cards "
            "still grant the old names. An unmatched glob is an empty permission "
            "set, not an error, so the loss fell and the run finished. Read the "
            "result accordingly -- it does not show that heterogeneous regional "
            "state fails, because the heterogeneous regional state never trained."
        )
    return [head + provenance]


def _enabled_but_unconsumed(ckpt: Path, card_dir: str) -> list[str]:
    """Sources the mixture enables that these particular weights never saw.

    Deriving the card directory from the run (see :func:`_card_dir`) fixed the
    artifact reading the wrong card set, and immediately produced the mirror
    error.  The corrected set has ``ds002336_real`` **enabled**, so its licence
    and citation now appear under "DATASET INPUTS" -- for a checkpoint whose
    recorded split contains no participant of that dataset.  A licence claim
    ahead of its evidence is the same defect as a link ahead of its scores, and
    an over-declared input is not the harmless direction: it credits a corpus
    for weights it did not shape.

    ``enabled`` is a statement about the *mixture*, not about a checkpoint.  A
    card switched on after a run finished is enabled and unconsumed at once, and
    nothing in the card can distinguish those.  The checkpoint can: it carries
    the participant split it actually trained on.

    The test is participant counts rather than name matching, because the two
    corpora label participants differently (``S001`` against ``sub-xp101``) and
    a string comparison across those returns zero overlap for *both* -- a
    confident answer about the wrong thing.  So: if the recorded split's size
    equals exactly one enabled likelihood card's declared ``n_participants``,
    every other enabled likelihood source contributed no participant to it.
    Where that is not decidable, this returns nothing rather than a guess.
    """
    last = ckpt / "last.pt"
    if not last.is_file():
        return []
    try:
        import torch
        import yaml

        rec = torch.load(last, map_location="cpu", weights_only=False)
        extra = rec.get("extra") or {}

        # DIRECT EVIDENCE BEATS THE INFERENCE BELOW.
        #
        # The heuristic that follows reads the recorded participant split and,
        # if its count matches exactly one card's declared `n_participants`,
        # concludes every OTHER source contributed nothing. That was sound for
        # run 2, which had one real source and whose `real_split` was the whole
        # story. It is FALSE for a multi-source run: run 3's `real_split` is
        # eegmmidb's split specifically, the other six sources carry their own
        # loaders and their own splits, and the heuristic would have put "`X`
        # contributed nothing to these weights" on a public model card for six
        # sources that each contributed a gradient every step.
        #
        # Run 3's checkpoint records `contributed_sources`, derived in
        # `MixtureTrainer` from the losses that actually ran. Where that exists
        # it is evidence, not inference, and the inference must not override it.
        # UNION with the training log. `extra.contributed_sources` under-reports
        # on run 3's checkpoints -- see `contributed_sources_union` -- and taking
        # it at face value would have put "`eegmmidb_real` produced no loss term"
        # on a public card for the largest source in the run.
        from ..foundation.release import contributed_sources_union

        run_name = ((rec.get("config") or {}).get("train") or {}).get("run_name") or ""
        contributed, _only_log = contributed_sources_union(
            extra, Path("reports/training") / f"{run_name}_train.jsonl"
        )
        if contributed:
            known = set(contributed)
            silent = []
            for f in sorted(Path(card_dir).glob("*.yaml")):
                c = yaml.safe_load(f.read_text()) or {}
                sid = str(c.get("id") or f.stem)
                if c.get("role") == "likelihood" and c.get("enabled", True) and sid not in known:
                    silent.append(
                        f"`{sid}` is enabled in the training mixture and produced no "
                        "loss term in this run: it is absent from the checkpoint's "
                        "own `contributed_sources`, which is recorded from the losses "
                        "that ran rather than inferred from a participant count. Its "
                        "licence and citation are listed below as terms this artifact "
                        "inherits, not as a corpus that shaped it."
                    )
            return silent

        folds = (extra.get("real_split") or {}).get(
            "participants_per_fold"
        ) or {}
        split = {p for v in folds.values() for p in v}
        if not split:
            return []
        cards = {}
        for f in sorted(Path(card_dir).glob("*.yaml")):
            c = yaml.safe_load(f.read_text()) or {}
            if c.get("role") == "likelihood" and c.get("enabled", True):
                n = c.get("n_participants")
                if isinstance(n, int) and n > 0:
                    cards[str(c.get("id") or f.stem)] = n
    except Exception:
        return []

    matches = [sid for sid, n in cards.items() if n == len(split)]
    if len(matches) != 1 or len(cards) < 2:
        return []
    consumed = matches[0]
    return [
        f"`{sid}` is enabled in the training mixture but contributed nothing to "
        f"these weights: the checkpoint's recorded split holds {len(split)} "
        f"participants, exactly `{consumed}`'s declared count, and `{sid}` "
        f"declares {cards[sid]} more that appear in no fold. Its licence and "
        "citation are listed below because the mixture enables it for the next "
        "run -- read them as terms this artifact will inherit, not as a corpus "
        "that shaped it."
        for sid in sorted(cards)
        if sid != consumed
    ]


def _card_dir(root: Path, ckpt: Path, config_path: Path) -> tuple[str, list[str]]:
    """The source-card directory that ACTUALLY governed this checkpoint.

    This was ``configs/source_cards`` as a literal, and that is the directory
    ``tests/curriculum/test_tiers.py`` names ``LEGACY`` -- the corrected set
    lives at ``configs/curriculum/source_cards`` and is what every run-2 config
    selects through its ``base:``.  So the artifact's licence and attribution
    manifest was built from a different card set than the one that trained.

    Today the two agree on every licence-bearing field, which is why nothing was
    visibly wrong.  They do not agree on ``enabled``: ``ds002336_real`` is on in
    the corrected set and off in the legacy one.  The moment a run takes a
    gradient on that BOLD, a manifest built from the legacy directory publishes
    a model whose card omits a dataset that contributed to it -- an attribution
    failure produced by reading the wrong directory, not by any mistake in the
    cards themselves.

    So the directory is *derived*, preferring the checkpoint's own recorded
    config over the config file, because the checkpoint is evidence of what ran
    and the file is only evidence of what was intended.  A disagreement between
    them is a blocker rather than a precedence rule: it means the config on disk
    is not the config that trained, and every provenance claim on the card
    inherits that gap.
    """
    notes: list[str] = []
    from_ckpt = from_file = None

    last = ckpt / "last.pt"
    if last.is_file():
        try:
            import torch

            rec = torch.load(last, map_location="cpu", weights_only=False).get("config")
            if isinstance(rec, dict):
                from_ckpt = rec.get("mixture_cards")
        except Exception as exc:  # unreadable checkpoint -> say so, do not assume
            notes.append(f"could not read mixture_cards from {last}: {exc}")

    try:
        from ..foundation.config import _resolve_base

        from_file = (_resolve_base(Path(config_path)) or {}).get("mixture_cards")
    except Exception as exc:
        notes.append(f"could not read mixture_cards from {config_path}: {exc}")

    if from_ckpt and from_file and from_ckpt != from_file:
        notes.append(
            f"the checkpoint trained with mixture_cards={from_ckpt!r} but "
            f"{config_path} declares {from_file!r}; the published provenance would "
            "describe a mixture the weights never saw"
        )
    chosen = from_ckpt or from_file
    if not chosen:
        notes.append(
            "neither the checkpoint nor the config names a mixture_cards directory, "
            "so the card set that trained this model cannot be identified. Falling "
            "back to a default here would build the licence manifest from a "
            "directory chosen by this function rather than by the run."
        )
        return str(root / "configs/source_cards"), notes
    return str(root / chosen), notes


def _final_stage_file(ckpt: Path, config_path: Path) -> str:
    """Name of the checkpoint holding the final stage's weights.

    Prefers the named stage file, because it is unambiguous about *which* stage
    produced the weights.  Falls back to ``last.pt``, which is the same tensors
    under a name that does not say so.  Returns the run-1 name only if that is
    what is actually on disk, so a run-1 directory still plans correctly.
    """
    try:
        from ..foundation.config import load_config

        stages = [s for s in load_config(str(config_path)).train.stages if s.steps > 0]
        if stages:
            named = f"stage_{stages[-1].name}.pt"
            if (ckpt / named).is_file():
                return named
    except Exception:
        pass  # a config we cannot read is not a reason to publish nothing
    for fallback in ("stage_V_individual.pt", "last.pt"):
        if (ckpt / fallback).is_file():
            return fallback
    return "last.pt"


def plan_run2_pilot(
    *,
    checkpoint_dir: str | Path,
    evaluation: str | Path = "reports/training/evaluation_run2.json",
    # The run-2 config actually on disk.  This defaulted to
    # "configs/scwbd_002_pilot.yaml", which has never existed -- a second blocker
    # sitting behind the missing-evaluation one, invisible until the first was
    # cleared. A blocker list that stops at the first item hides the rest.
    config: str | Path = "configs/run2/pilot-families.yaml",
    name: str = "scwbd-002-pilot",
    repo_root: Path | None = None,
) -> ArtifactPlan:
    """Plan the run-2 pilot checkpoint.

    Deliberately the same code path as run 1 rather than a new one: when the
    pilot lands, the only thing that should need to change is the paths. If the
    files are not there yet this returns a plan whose blockers say so, which is
    the honest state rather than an error.
    """
    try:
        return plan_run1_checkpoint(
            checkpoint_dir=checkpoint_dir,
            evaluation=evaluation,
            config=config,
            name=name,
            repo_root=repo_root,
        )
    except PublishBlocked as exc:
        return ArtifactPlan(
            name=name,
            repo_type="model",
            blockers=(str(exc),),
            notes=("run-2 pilot is still training; this plan is prepared, not ready",),
        )


def plan_run3(
    *,
    checkpoint_dir: str | Path,
    evaluation: str | Path = "reports/training/evaluation_run3.json",
    config: str | Path = "configs/run3/scwbd-003.yaml",
    name: str = "scwbd-003",
    repo_root: Path | None = None,
) -> ArtifactPlan:
    """Plan the SC-WBD-003 checkpoint.

    Same code path as runs 1 and 2 -- only the paths change, which is the point:
    a publisher with a per-run branch is a publisher whose gates differ per run.

    THE CARD MUST CARRY ISSUE-008. Run 3's weights were trained through a BOLD
    path that never integrated the Balloon ODE: the five haemodynamic parameters
    are bit-identical to their initialisation in all five stages and
    `real_bold_nll` diverged to 4.4e6. The code is fixed; these weights are not
    and cannot be. No fMRI or haemodynamic claim may be read off this artifact,
    and the card says so rather than leaving a reader to infer it from a
    `contributed_sources` list that truthfully includes `ds002336_real`.
    """
    plan = plan_run1_checkpoint(
        checkpoint_dir=checkpoint_dir,
        evaluation=evaluation,
        config=config,
        name=name,
        repo_root=repo_root,
    )

    # ISSUE-008 goes at the TOP of the card, not in a footnote. The card is
    # generated from the checkpoint, and the checkpoint truthfully lists
    # `ds002336_real` among its contributing sources -- so a reader who is not
    # told otherwise will reasonably conclude this model has an fMRI likelihood.
    # It does not. Leaving them to infer it from a `contributed_sources` list is
    # exactly the shape of defect this project exists to avoid.
    limitation = (
        "> ## No fMRI or haemodynamic claim may be read off this model\n>\n"
        "> These weights were trained through a BOLD path that **never integrated "
        "the Balloon-Windkessel ODE**. `BOLDHead.step` was not called in any of "
        "the five stages, the five haemodynamic parameters (`log_kappa`, "
        "`log_gamma`, `log_tau`, `alpha`, `neural_gain`) are **bit-identical to "
        "their initialisation**, and `real_bold_nll` diverged monotonically from "
        "21.7 to 4.4e6 over the run.\n>\n"
        "> `ds002336_real` appears in `contributed_sources` and that is accurate: "
        "its BOLD channel contributed a **gradient** and no **information**. The "
        "two are different statements and only the first was ever checked.\n>\n"
        "> The defect is repaired in the code (ISSUE-008, closed 2026-08-09) and "
        "cannot be repaired in these weights, which cannot be un-trained. Any "
        "fMRI, haemodynamic or neurovascular claim about this artifact is "
        "unsupported.\n>\n"
        "> Two further limits from the same run: the amortised posterior is "
        "well-calibrated and **uninformative** (R^2 ~ 0 on all six parameters), "
        "and the EEG lead field is an **analytic sphere**, not a head model, so "
        "no source-localisation claim is available either.\n\n"
    )
    return replace(plan, card=limitation + (plan.card or ""))


def plan_run4(
    *,
    checkpoint_dir: str | Path,
    evaluation: str | Path = "reports/training/evaluation_run4.json",
    config: str | Path = "configs/run4/scwbd-004.yaml",
    name: str = "scwbd-004",
    repo_root: Path | None = None,
) -> ArtifactPlan:
    """Plan the SC-WBD-004 checkpoint.

    Same code path as runs 1-3; only the paths change.

    THE CARD MUST CARRY ISSUE-016 AND ISSUE-012, and for the same reason run 3's
    had to carry ISSUE-008: the card is generated from the checkpoint, the
    checkpoint truthfully lists `ds002336_real` among its contributing sources,
    and a reader who is not told otherwise will reasonably conclude this model
    has a working fMRI likelihood.

    Run 4's is different from run 3's, and the difference is the finding. Run 3's
    BOLD path never integrated the ODE, so the term was inert. Run 4's DOES
    integrate it -- and the likelihood then degrades during training, because
    `ds002336_real` is 5.39% of the mixture and is outvoted 17.6:1 by the
    EEG-like sources. The fMRI claim is withdrawn either way; what changed is
    that we now know why, and can say it.
    """
    plan = plan_run1_checkpoint(
        checkpoint_dir=checkpoint_dir,
        evaluation=evaluation,
        config=config,
        name=name,
        repo_root=repo_root,
    )

    # The posterior paragraph is DERIVED, not written. Its first version was
    # composed before the run and said calibration "is UNKNOWN" and that the
    # posterior "should be informative (log_G R^2 0.674-0.766 across four
    # seeds)". Both were true when written and false by the time the card would
    # have been generated: production measured 0.284 and a KS p of 1.0e-147. A
    # card that quotes a pre-run sweep as though it were the run's own result is
    # exactly what `reports/publishing.md` records as the near-miss -- so the
    # numbers now come off the artifact and cannot go stale again.
    _root = repo_root or REPO_ROOT
    posterior_note = _run4_posterior_note(_root / evaluation)
    individualisation_note = _run4_individualisation_note(_root / evaluation)
    # The ablation lives in its OWN artifact -- `make release-004-ablate` writes
    # evaluation_run4_ablation.json, not the file above -- so it is looked up
    # beside the evaluation rather than inside it.
    ablation_note = _run4_ablation_note(
        (_root / evaluation).with_name("evaluation_run4_ablation.json")
    )

    limitation = (
        "> ## No fMRI claim may be read off this model, and this time we know why\n>\n"
        "> The measured BOLD path in this run **does** integrate the "
        "Balloon-Windkessel ODE -- that is what run 4 fixed (ISSUE-008). The "
        "haemodynamic likelihood is real here in a way it was not in run 3.\n>\n"
        "> **It diverges during training, and the full run settled how far.** "
        "`real_bold_nll` ran **1.99 to 36,472** over 14,600 steps -- a factor of "
        "about 18,000 -- while `eeg_nll` IMPROVED (1.74 to 1.50), the total loss "
        "stayed flat near 1.0, and `bold_log_scale` held at 5.3-5.9. So this is "
        "not a variance explosion and the fMRI term is not dominating the "
        "mixture: it is the one term getting worse while everything around it "
        "gets better. It never plateaus: T4 alone spans 1,530 to 650,815, four "
        "orders of magnitude inside one stage. And **T5's measured return does "
        "not repair it** -- that stage grants `bold.*` again and ends at 36,472. "
        "`bold_parcels_covered` held at full value throughout, so the ODE ran on "
        "every parcel for 46 hours and the likelihood it produced is "
        "worthless.\n>\n"
        "> Four diagnostic arms located the cause: the **shared trunk moves out "
        "from under the BOLD head**. `ds002336_real` is **5.39%** of the source "
        "mixture and is outvoted **17.6 : 1** by the EEG-like sources, so the "
        "latent state converges on what they want. Freezing the five Balloon "
        "parameters changes nothing; freezing the trunk makes the BOLD "
        "likelihood *improve*.\n>\n"
        "> `ds002336_real` appears in `contributed_sources` and that is accurate: "
        "its BOLD channel contributed a **gradient**. It did not thereby acquire "
        "**information**, and in this run the two move in opposite directions.\n>\n"
        "> No fMRI, haemodynamic or neurovascular claim about this artifact is "
        "supported. ISSUE-016 is open; the remedy is unshared capacity for the "
        "slow modality, which is a later run's design, not a caveat on this "
        "one.\n>\n"
        f"{posterior_note}"
        f"{individualisation_note}"
        f"{ablation_note}"
        "> The EEG lead field remains an **analytic sphere**, not a head model, "
        "so no source-localisation claim is available.\n\n"
    )
    return replace(plan, card=limitation + (plan.card or ""))


def _run4_posterior_note(eval_path: Path) -> str:
    """ISSUE-012's paragraph, read off the evaluation rather than remembered.

    Returns a `> `-quoted block ending in a blank quoted line, or a refusal if
    the evaluation is absent or carries no calibration -- never a silent empty
    string, because an omitted caveat reads as an absent problem.
    """
    import json as _json

    if not Path(eval_path).is_file():
        return (
            "> **The amortised posterior's calibration is UNREAD.** No evaluation "
            f"artifact at `{eval_path}`, so ISSUE-012's status cannot be stated. "
            "No inference claim may be read off this model.\n>\n"
        )
    cal = (_json.loads(Path(eval_path).read_text()).get("posterior_calibration") or {})
    if not cal.get("available"):
        return (
            "> **The amortised posterior's calibration is UNREAD.** The evaluation "
            "carries no `posterior_calibration` block. No inference claim may be "
            "read off this model.\n>\n"
        )

    names = list(cal.get("param_names") or [])
    r2 = [float(v) for v in (cal.get("posterior_r2") or [])]
    zsd = [float(v) for v in (cal.get("posterior_z_sd") or [])]
    worst = min(cal.get("sbc_ks_pvalue") or [1.0])
    mae = float(cal.get("coverage_mae", 1.0))
    dead = list(cal.get("uninformative_parameters") or [])
    calibrated = worst > 0.01 and mae < 0.12

    best_i = max(range(len(r2)), key=lambda i: r2[i]) if r2 else None
    best = f"`{names[best_i]}` R^2 {r2[best_i]:.3f}" if best_i is not None else "unavailable"
    worst_z = f"{max(zsd):.1f}" if zsd else "unavailable"

    # THREE states, not two. Collapsing the middle one is how the first draft of
    # this function described run 3's posterior -- calibrated, z-sd 1.0, and
    # explaining no variance in anything -- as "confidently wrong", which is the
    # opposite failure. Exercising it against run 3's artifact is what caught it.
    if calibrated and not dead:
        return (
            f"> **The amortised posterior is calibrated and informative** on this "
            f"run's own `posterior_calibration`: SBC KS p_min {worst:.3g}, "
            f"coverage MAE {mae:.3f}, best {best}. ISSUE-012's discharge condition "
            "is met; read the block before relying on the interval widths.\n>\n"
        )
    if calibrated:
        # Calibrated AND uninformative: the posterior returns the prior, which
        # satisfies SBC and coverage by construction. This is run 3's state and
        # the reason the calibration block alone cannot support a claim.
        return (
            "> **No inference or parameter-recovery claim may be read off this "
            "model.** ISSUE-012 is open. The posterior is well calibrated -- SBC "
            f"KS p_min {worst:.3g}, coverage MAE {mae:.3f}, worst "
            f"`posterior_z_sd` {worst_z} -- and that certifies nothing, because "
            f"{len(dead)} of {len(names)} parameters explain no variance "
            f"({', '.join(f'`{d}`' for d in dead)}). **A posterior that ignores "
            "its conditioning and returns the prior is calibrated by "
            "construction**: its ranks are uniform because the truth is a draw "
            "from the distribution it reported. Calibration does not qualify the "
            f"R^2. Best {best}.\n>\n"
        )
    return (
        "> **No inference or parameter-recovery claim may be read off this "
        "model.** ISSUE-012 is open and this run MEASURED it rather than "
        f"inheriting it. The learning-rate repair worked -- best {best}, against "
        "~0 on every parameter in run 3, so the flow now reads its conditioning "
        "-- and it overshot: worst `posterior_z_sd` "
        f"**{worst_z}** where a calibrated posterior sits near 1.0, SBC KS p_min "
        f"**{worst:.3g}**, coverage MAE **{mae:.3f}**. The posterior narrowed far "
        "more than its accuracy earned, so it is confidently wrong rather than "
        "uninformative. "
        + (
            f"{len(dead)} of {len(names)} parameters still explain no variance "
            f"({', '.join(f'`{d}`' for d in dead)}). "
            if dead
            else ""
        )
        + "Run 3's posterior was uninformative and honest; this one is partly "
        "informative and overconfident. Neither supports inference.\n>\n"
    )


PLANNERS: Mapping[str, Any] = {
    "anatomy-prior": plan_anatomy_prior,
    "run1-checkpoint": plan_run1_checkpoint,
    "sim-corpus": plan_sim_corpus,
    "run2-pilot": plan_run2_pilot,
    "run3": plan_run3,
    "run4": plan_run4,
}


# ---------------------------------------------------------------------------
# identity: an RL-11 guard, not an assumption
# ---------------------------------------------------------------------------
#: Set on this box, and it **silently overrides the stored CLI login**:
#: ``hf auth whoami`` reports ``brandonin`` while ``env -u HF_TOKEN hf auth
#: whoami`` reports ``jacob-valdez``. Authenticating as the wrong user is
#: invisible on its own — it only shows up as an artifact in someone else's
#: namespace — so this is checked and reported rather than assumed.
TOKEN_ENV_OVERRIDES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class IdentityMismatch(RuntimeError):
    """The authenticated account cannot write to the requested namespace."""


def observed_identity(token: str | None = None) -> dict[str, Any]:
    """Ask the Hub who we actually are. Returns the observation, not a verdict.

    Reported rather than reduced to a boolean on purpose: "identity ok" tells
    you nothing when it is wrong, and the interesting case here is precisely
    the one where a caller believes one thing and the network believes another.
    """
    from huggingface_hub import HfApi

    who = HfApi(token=token).whoami()
    return {
        "user": who.get("name"),
        "orgs": [o.get("name") for o in (who.get("orgs") or [])],
        "auth_type": who.get("auth", {}).get("type"),
        "token_env_set": [v for v in TOKEN_ENV_OVERRIDES if os.environ.get(v)],
    }


def verify_identity(namespace: str, *, token: str | None = None) -> dict[str, Any]:
    """Refuse unless the authenticated account may write to ``namespace``.

    This is the guard that makes the push falsifiable. Two failure modes it
    closes, which compose badly with each other:

    * a token environment variable overriding the stored login, so the caller
      authenticates as someone else while believing otherwise; and
    * ``create_repo(exist_ok=True)``, which on a namespace collision uploads
      *into an existing repo* instead of failing.

    Wrong identity plus silent-merge-on-collision puts an artifact somewhere
    nobody intended with nothing reporting it. So identity is asserted against
    the network before any bytes move, and the observation travels in the
    result either way.
    """
    ident = observed_identity(token=token)
    user, orgs = ident["user"], ident["orgs"]
    if namespace != user and namespace not in orgs:
        raise IdentityMismatch(
            f"Authenticated as {user!r} (orgs: {orgs or 'none'}), which cannot "
            f"write to namespace {namespace!r}.\n"
            + (
                f"  {'/'.join(ident['token_env_set'])} is set in the "
                "environment and overrides the stored CLI login. Re-run with "
                f"`env -u {' -u '.join(ident['token_env_set'])} ...`.\n"
                if ident["token_env_set"]
                else ""
            )
            + "  Refusing rather than publishing to the wrong account."
        )
    return ident


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------
def publish(
    plan: ArtifactPlan,
    *,
    namespace: str,
    dry_run: bool = True,
    private: bool = True,
    stage_dir: str | Path | None = None,
    token: str | None = None,
    require_attribution: bool = True,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Gate the plan and, only when ``dry_run`` is False, push it.

    The gate runs **before** the dry-run/push branch, so a dry run exercises
    exactly the check a real publish would hit. A gate that only fires on the
    push path is a gate nobody has ever seen fire.

    On the push path two further preconditions are asserted against the
    network, in this order and before any write: :func:`verify_identity`, then
    repository non-existence. Both refuse rather than proceed.
    """
    repo_id = plan.repo_id(namespace)
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": plan.repo_type,
        "dry_run": dry_run,
        "n_files": len(plan.files),
        "n_bytes": plan.n_bytes,
        "blockers": list(plan.blockers),
        "warnings": list(plan.warnings),
    }

    if require_attribution and plan.attribution is not None:
        plan.attribution.require_complete()  # raises AttributionError
    if plan.blockers:
        raise PublishBlocked(
            f"{repo_id} has {len(plan.blockers)} blocker(s) and will not be "
            "published:\n  - " + "\n  - ".join(plan.blockers)
        )

    stage = Path(stage_dir) if stage_dir else None
    if stage is not None:
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "README.md").write_text(plan.card)
        (stage / "publish_plan.json").write_text(
            json.dumps(plan.as_dict(), indent=2) + "\n"
        )
        result["staged_to"] = str(stage)

    if dry_run:
        result["action"] = "none — dry run, no remote state created"
        return result

    # Everything below this line touches the network. Imported here, not at
    # module scope, so that a dry run cannot reach the Hub even by accident.
    from huggingface_hub import HfApi

    # 1. Who are we, really? Refuses before any write.
    ident = verify_identity(namespace, token=token)
    result["identity"] = ident

    api = HfApi(token=token)

    # 2. Is the name free? A pre-existing repo we did not create is refused
    #    rather than uploaded into: exist_ok=True would silently merge, and on
    #    a first publish that is indistinguishable from success.
    if api.repo_exists(repo_id=repo_id, repo_type=plan.repo_type):
        if not allow_existing:
            raise PublishBlocked(
                f"{repo_id} ({plan.repo_type}) already exists. Refusing to "
                "upload into a repository this run did not create — that would "
                "merge into whatever is already there and report success. Pass "
                "allow_existing=True / --allow-existing if merging is intended."
            )
        result["preexisting"] = True

    api.create_repo(
        repo_id=repo_id,
        repo_type=plan.repo_type,
        private=private,
        exist_ok=allow_existing,
    )
    uploaded: list[str] = []
    # The card is NOT conditional on --stage-dir.  It was: the upload sat inside
    # `if stage is not None`, and --stage-dir is a debugging flag for inspecting
    # the rendered card locally.  So `make publish-002`, which does not pass it,
    # shipped scwbd-002-pilot public with weights and no model card at all --
    # no scores, no licence section, and none of the disclosure about the
    # curriculum defect.
    #
    # It hid because every dry run passed --stage-dir in order to READ the card,
    # so the verification path and the shipping path differed in exactly the
    # place the defect lived. An artifact's card is not an optional attachment;
    # publishing weights without one is publishing a claim with its caveats
    # removed.
    import tempfile as _tempfile

    _card_dir = stage if stage is not None else Path(_tempfile.mkdtemp(prefix="scwbd-card-"))
    _card = _card_dir / "README.md"
    if not _card.exists():
        _card.write_text(plan.card)
    api.upload_file(
        path_or_fileobj=str(_card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=plan.repo_type,
    )
    uploaded.append("README.md")
    for f in plan.files:
        api.upload_file(
            path_or_fileobj=str(f.local),
            path_in_repo=f.repo_path,
            repo_id=repo_id,
            repo_type=plan.repo_type,
        )
        uploaded.append(f.repo_path)
    result["uploaded"] = uploaded
    result["url"] = (
        f"https://huggingface.co/{'datasets/' if plan.repo_type == 'dataset' else ''}"
        f"{repo_id}"
    )
    result["action"] = (
        f"created {repo_id} as {ident['user']} and uploaded "
        f"{len(uploaded)} file(s)"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    from ..sources.attribution import AttributionError

    ap = argparse.ArgumentParser(
        prog="python -m scwbd.release.publish",
        description="Build, gate and publish SC-WBD artifacts. Dry run by default.",
    )
    ap.add_argument("artifact", choices=sorted(PLANNERS))
    ap.add_argument(
        "--namespace",
        default=None,
        help=f"Hub user or org. Required; or set ${NAMESPACE_ENV}. Never inferred.",
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="Actually create the repo and upload. Without this, nothing leaves the box.",
    )
    ap.add_argument("--public", action="store_true", help="Publish public (default private).")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--corpus-dir", default="/data/scwbd/sim_corpus_414")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--include-maps", action="store_true",
                    help="anatomy-prior: include Hansen-derived maps (makes it NC-SA).")
    ap.add_argument("--stage-dir", default=None,
                    help="Write the generated card and plan here for review.")
    ap.add_argument("--allow-existing", action="store_true",
                    help="Upload into a repo that already exists. Off by "
                         "default: silent merge on collision is a footgun.")
    ap.add_argument("--whoami", action="store_true",
                    help="Report the authenticated identity and exit. Creates "
                         "no state.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    namespace = resolve_namespace(args.namespace)

    for var in TOKEN_ENV_OVERRIDES:
        if os.environ.get(var):
            print(
                f"!! {var} is set and OVERRIDES the stored CLI login. If this "
                f"is not deliberate, re-run with `env -u {var} ...`."
            )

    if args.whoami:
        try:
            ident = verify_identity(namespace)
        except IdentityMismatch as exc:
            print(f"IDENTITY MISMATCH for namespace {namespace!r}")
            for line in str(exc).splitlines():
                print(f"  {line}")
            return 3
        print(f"OK  authenticated as {ident['user']!r} "
              f"(orgs: {ident['orgs'] or 'none'}) -> may write to {namespace!r}")
        return 0

    kwargs: dict[str, Any] = {}
    if args.artifact == "anatomy-prior":
        kwargs["include_maps"] = args.include_maps
    elif args.artifact in ("run1-checkpoint", "run2-pilot", "run3", "run4"):
        if not args.checkpoint_dir:
            raise SystemExit(
                "--checkpoint-dir is required: the weights are not in this "
                "worktree and checkpoints/ is git-ignored, so there is no "
                "default that would not be a guess."
            )
        kwargs["checkpoint_dir"] = args.checkpoint_dir
    elif args.artifact == "sim-corpus":
        kwargs["corpus_dir"] = args.corpus_dir
        kwargs["n_shards"] = args.n_shards

    plan = PLANNERS[args.artifact](**kwargs)
    try:
        res = publish(
            plan,
            namespace=namespace,
            dry_run=not args.push,
            private=not args.public,
            stage_dir=args.stage_dir,
            allow_existing=args.allow_existing,
        )
    except IdentityMismatch as exc:
        print(f"IDENTITY MISMATCH  refusing to publish {plan.repo_id(namespace)}")
        for line in str(exc).splitlines():
            print(f"  {line}")
        return 3
    except (PublishBlocked, AttributionError) as exc:
        # A blocked artifact is a normal outcome to report, not a crash. The
        # traceback would bury the one thing the operator needs to read.
        print(f"NOT PUBLISHABLE  {plan.repo_id(namespace)} ({plan.repo_type})")
        for line in str(exc).splitlines():
            print(f"  {line}")
        return 2
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"{'PUSH' if args.push else 'DRY RUN'}  {res['repo_id']} "
              f"({res['repo_type']})")
        print(f"  files: {res['n_files']}  bytes: {res['n_bytes']:,}")
        ident = res.get("identity")
        if ident:
            # The observation, not a boolean: "identity ok" tells you nothing
            # when it is wrong.
            print(f"  identity: user={ident['user']!r} orgs={ident['orgs'] or 'none'}"
                  f" auth={ident['auth_type']}")
            if ident["token_env_set"]:
                print(f"  identity: {'/'.join(ident['token_env_set'])} was set "
                      "in the environment")
        for w in res["warnings"]:
            print(f"  warning: {w}")
        print(f"  {res['action']}")
        if res.get("url"):
            print(f"  url: {res['url']}")
        if res.get("staged_to"):
            print(f"  card staged: {res['staged_to']}/README.md")
    return 0




def _run4_individualisation_note(eval_path: Path) -> str:
    """The individualisation caveat, read off the evaluation.

    Run 4 is the first artifact in this project with a MEASURED individualisation
    result rather than an unmeasurable one, and the measurement is negative. The
    numbers come off `session_individualisation`; a missing block is an explicit
    refusal, never silence.
    """
    import json as _json

    unread = (
        "> **No individualisation or personalisation claim may be read off this "
        "model.** The evaluation carries no `session_individualisation` block, so "
        "the person effect was not measured. It is NOT thereby supported.\n>\n"
    )
    if not Path(eval_path).is_file():
        return unread
    rep = _json.loads(Path(eval_path).read_text())
    si = rep.get("session_individualisation") or {}
    if not si.get("ok"):
        reason = str(si.get("reason") or "the block is absent or refused").strip()
        return (
            "> **No individualisation or personalisation claim may be read off "
            f"this model.** The person effect was not measured: {reason} It is "
            "NOT thereby supported.\n>\n"
        )

    shift = si.get("theta_shift") or {}
    spread = shift.get("spread_pooled")
    ratio = shift.get("spread_over_prior_sd")
    nll = si.get("held_out_session_nll")
    ci = si.get("held_out_session_nll_ci95") or []
    n_p = si.get("n_participants_individualisable")
    n_w = si.get("n_test_windows")
    zero_rows = shift.get("n_rows_exactly_zero")

    # The ratio is what makes `spread_pooled` interpretable. Artifacts written
    # before `spread_over_prior_sd` was added carry `prior_sd_person` or neither,
    # so derive it where possible rather than falling through.
    if not isinstance(ratio, (int, float)):
        prior = shift.get("prior_sd_person")
        if isinstance(spread, (int, float)) and isinstance(prior, list) and prior:
            ratio = float(spread) / (sum(float(v) for v in prior) / len(prior))

    ratio_txt = ""
    if isinstance(ratio, (int, float)):
        ratio_txt = (
            f" -- **{100 * float(ratio):.2f}%** of the scale the model allocated "
            "for that effect"
        )

    # `at or near zero` is the falsifier the evaluation itself declares. 1% of
    # the effect's own prior sd is the threshold for "near": below it the
    # individualiser has moved theta by less than a hundredth of what it was
    # built to move it by.
    #
    # A ratio that cannot be established takes the STRICT branch. An unmeasurable
    # shift is not a small one, and defaulting the other way would publish the
    # permissive caveat for a capability nobody has checked -- which is what this
    # function did on its first run, because `spread_over_prior_sd` postdates the
    # artifact it was first pointed at.
    applied_nothing = not isinstance(ratio, (int, float)) or float(ratio) < 0.01

    head = (
        "> **No individualisation or personalisation claim may be read off this "
        "model, and this run MEASURED that rather than inheriting it.**\n>\n"
        if applied_nothing
        else "> **Individualisation: read the numbers before claiming it.**\n>\n"
    )

    body = (
        f"> `session_individualisation` scored **{n_p} participants** over "
        f"**{n_w} held-out second-night windows** -- the same people on both "
        "sides of a SESSION split, which is the only arrangement on which a "
        "person effect is measurable at all. Held-out session NLL "
        f"**{float(nll):.4f}**"
        + (f" [{float(ci[0]):.4f}, {float(ci[1]):.4f}]" if len(ci) == 2 else "")
        + ", bootstrapped over participants rather than windows.\n>\n"
    )

    if applied_nothing:
        body += (
            "> **That score is not the finding.** The between-participant spread "
            f"of the applied theta shift is **{float(spread):.3g}**{ratio_txt}. "
            + (
                f"{zero_rows} of the scored person-effect rows are exactly zero. "
                if isinstance(zero_rows, int) and zero_rows
                else ""
            )
            + "The individualiser applied essentially nothing on a split built "
            "specifically to let it apply something, which is the falsifier this "
            "evaluation declares for the capability. Earlier runs reported "
            "individualisation as *unmeasurable* on a participant-disjoint "
            "split; this one built the split, trained the effect and measured "
            "it, and the effect is a fraction of a percent of its own scale.\n>\n"
            "> The held-out NLL is reported because withholding a measured number "
            "is its own distortion. It answers a different question than it "
            "appears to: what separates an individualised model from the "
            "population model here is the shift, not the score.\n>\n"
        )
    else:
        body += (
            f"> Applied theta shift: spread **{float(spread):.3g}**{ratio_txt}. "
            "Read `session_individualisation.theta_shift` before relying on any "
            "per-person behaviour.\n>\n"
        )
    return head + body


def _run4_ablation_note(eval_path: Path) -> str:
    """Leave-one-source-out, read off the artifact.

    Reports the MEASURED-holdout arm as the result and the simulated arm as a
    comparability figure, in that order, because they answer different questions
    and run 3 published only the second. Scored on `_sim_val_nll` alone the
    ablation asks "does dropping this measured source help the model fit the
    simulator?", and 200 steps of retraining answer yes by construction: run 3
    returned negative transfer on nine of ten families and the direction was
    predictable before it ran.

    A missing block is an explicit refusal. An ablation that was not run is not
    an ablation that found nothing.
    """
    import json as _json

    if not Path(eval_path).is_file():
        return (
            "> **No leave-one-source-out result is available.** The ablation "
            f"artifact `{Path(eval_path).name}` is absent, so no claim about which "
            "sources carry this model's performance may be read off it.\n>\n"
        )
    sa = (_json.loads(Path(eval_path).read_text()).get("source_ablation") or {})
    if not sa:
        return (
            "> **No leave-one-source-out result is available.** The evaluation "
            "carries no `source_ablation` block. Which sources earn their place "
            "in this model is UNMEASURED, not measured-and-neutral.\n>\n"
        )

    measured = sa.get("measured") or {}
    steps = sa.get("steps_per_arm")
    sim_neg = list(sa.get("negative_transfer") or [])
    n_fam = len(list(sa.get("families") or []))

    if not measured:
        return (
            "> **The leave-one-source-out result here is not evidence about the "
            "sources.** Every arm was scored on the SIMULATED validation set, so "
            "the question asked was whether dropping a measured source helps the "
            "model fit the simulator -- and during the "
            f"{steps or 'retraining'} steps each arm retrains, every measured "
            "gradient pulls parameters away from exactly that. "
            f"{len(sim_neg)} of {n_fam} families duly came back as negative "
            "transfer. That is the design, not a finding. No attribution claim "
            "may be read off it.\n>\n"
        )

    contributed = list(measured.get("contributed") or [])
    neg = list(measured.get("negative_transfer") or [])
    deltas = {
        k[len("delta_") :]: v for k, v in measured.items() if k.startswith("delta_")
    }
    # POSITIVE deltas only. Taking the top 3 unconditionally pads the list with
    # negatives when fewer than three families contributed, and then labels them
    # "largest positive deltas" -- run 4 has two contributors and the third slot
    # was filled with -0.0031.
    ranked = sorted(
        ((k, float(v)) for k, v in deltas.items() if float(v) > 0), key=lambda kv: -kv[1]
    )
    top = ", ".join(f"`{k}` {v:+.4f}" for k, v in ranked[:3]) or "none"

    head = (
        "> ## Which sources earn their place: leave-one-source-out on the "
        "MEASURED holdout\n>\n"
        "> Each arm drops one source family, retrains "
        f"{steps or 'briefly'} steps, and is scored on the same held-out "
        "participants the headline rests on. Positive delta means removing the "
        "family made measured prediction WORSE, i.e. the family contributed.\n>\n"
    )
    body = (
        f"> **{len(contributed)} of {len(deltas)} families contributed** on the "
        f"measured holdout; {len(neg)} showed negative transfer"
        + (f" ({', '.join('`' + n + '`' for n in neg)})" if neg else "")
        + f". Largest positive deltas: {top}.\n>\n"
    )
    # State the simulated count; do NOT assert what it "must" be. Run 3 returned
    # 9 of 10 negative there and the direction was structural. Run 4 returns 5 of
    # 10, so a blanket "this is what the simulated metric produces by
    # construction" would be rhetoric the number does not support.
    caveat = (
        "> The simulated-holdout arm of the same run is retained for "
        f"comparability with earlier runs and is NOT the result: {len(sim_neg)} of "
        f"{n_fam} families come back as negative transfer there. Scoring a "
        "measured source against the simulator asks whether dropping it helps "
        "the model fit the simulator, which is a different question from the one "
        "above and is not evidence about the source.\n>\n"
    )
    return head + body + caveat


# The entry point is LAST on purpose: anything defined below it is dead when
# the module runs as __main__. Appending helpers after this block is how
# `_run4_individualisation_note` raised NameError from `plan_run4` -- the same
# defect `evaluate.py` had. Guarded by
# tests/release/test_entry_points_are_last.py.
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
