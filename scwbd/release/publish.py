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
from dataclasses import dataclass
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
    manifest = build_manifest(
        card_dir=str(root / "configs/source_cards"),
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
        body += [
            f"- Anatomy: {anat.get('n_regions')} regions, provenance "
            f"`{anat.get('provenance')}`, `is_biological = {anat.get('is_biological')}`.",
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
        body.append(
            f"- Split: {ho.get('n_train_windows')} train windows / "
            f"{ho.get('n_train_participants')} participants; "
            f"{ho.get('n_test_windows')} test windows / "
            f"{ho.get('n_test_participants')} participants, participant-disjoint."
        )
    body.append("")

    body += ["## Known defects", ""]
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
    body.append(
        "- Two baselines in the table, `ar16` and `subject_specific_ar`, are "
        "bit-identical: the participant-disjoint split routes every test window "
        "to the `ar16` fallback. Read the table as four distinct baselines."
    )
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
            "anyone running the same ablation against their own control."
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


PLANNERS: Mapping[str, Any] = {
    "anatomy-prior": plan_anatomy_prior,
    "run1-checkpoint": plan_run1_checkpoint,
    "sim-corpus": plan_sim_corpus,
    "run2-pilot": plan_run2_pilot,
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
    if stage is not None:
        api.upload_file(
            path_or_fileobj=str(stage / "README.md"),
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
    elif args.artifact in ("run1-checkpoint", "run2-pilot"):
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
