"""Build every cached anatomical artifact and write ``assets/MANIFEST.json``.

    python -m scwbd.anatomy.build            # build what is missing
    python -m scwbd.anatomy.build --rebuild  # rebuild everything
    python -m scwbd.anatomy.build --verify   # re-hash what the manifest claims

The manifest records, for every upstream directory and every derived array, a
sha256, the source URL, the license and the version.  A derived artifact also
records what produced it and which inputs it consumed, so a stale cache is
detectable rather than silently trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from . import sources as S
from .atlases import ATLAS_SPECS, load_parcellation
from .connectome import (
    _ENIGMA_KEYS,
    DEFAULT_SUBCORTICAL_ATLAS,
    load_structural_prior,
    structural_cache_tag,
)
from .geometry import parcel_geometry
from .manifest import Manifest, git_commit
from .maps import load_maps
from .paths import assets_root, derived_dir
from .priors import BrainPrior

#: Upstream clones and caches, with the source-registry key that describes them.
UPSTREAM: list[tuple[str, str, str]] = [
    ("src/hansen_receptors", "hansen_receptors", "git"),
    ("src/tian_subcortex", "tian2020", "git"),
    ("src/cerebellar_atlases", "buckner2011", "git"),
    ("src/enigma", "enigmatoolbox", "git"),
    ("cache/nilearn", "schaefer2018", "downloader"),
    ("cache/neuromaps", "neuromaps", "downloader"),
    ("cache/netneurotools", "netneuro_lausanne_sc", "downloader"),
]

#: Parcellations to materialise.  The surface ones are what the connectome and
#: the maps attach to; the volumetric ones supply subcortex and cerebellum.
BUILD_SURFACE = [
    ("Schaefer100x7", "fsLR", "32k"),
    ("Schaefer200x7", "fsLR", "32k"),
    ("Schaefer300x7", "fsLR", "32k"),
    ("Schaefer400x7", "fsLR", "32k"),
    ("DesikanKilliany", "fsLR", "32k"),
    ("Glasser360", "fsLR", "32k"),
    ("EconomoKoskinas", "fsLR", "32k"),
    ("Destrieux", "fsaverage5", "10k"),
]
BUILD_VOLUME = [
    ("Schaefer400x7", "MNI152", "1mm"),
    ("Schaefer100x7", "MNI152", "1mm"),
    ("Schaefer200x7", "MNI152", "1mm"),
    ("Schaefer300x7", "MNI152", "1mm"),
    ("Schaefer500x7", "MNI152", "1mm"),
    ("Schaefer600x7", "MNI152", "1mm"),
    ("Schaefer800x7", "MNI152", "1mm"),
    ("Schaefer1000x7", "MNI152", "1mm"),
    ("Schaefer100x17", "MNI152", "1mm"),
    ("Schaefer200x17", "MNI152", "1mm"),
    ("Schaefer300x17", "MNI152", "1mm"),
    ("Schaefer400x17", "MNI152", "1mm"),
    ("Schaefer500x17", "MNI152", "1mm"),
    ("Schaefer600x17", "MNI152", "1mm"),
    ("Schaefer800x17", "MNI152", "1mm"),
    ("Schaefer1000x17", "MNI152", "1mm"),
    ("DesikanKilliany", None, None),  # surface-only; skipped by the guard below
    ("TianS1", "MNI152", "1mm"),
    ("TianS2", "MNI152", "1mm"),
    ("TianS3", "MNI152", "1mm"),
    ("TianS4", "MNI152", "1mm"),
    ("Aseg14", "MNI152", "1mm"),
    # The DEFAULT subcortical atlas (DEFAULT_SUBCORTICAL_ATLAS). Its absence
    # here is why the 414-parcel prior's subcortex was unattributed and the
    # attribution gate refused the whole anatomy prior.
    ("Aseg14T", "MNI152", "1mm"),
    ("Buckner7", "MNI152", "1mm"),
    ("Buckner17", "MNI152", "1mm"),
    ("SUITAnatom", "MNI152", "1mm"),
]


def _register_upstream(man: Manifest) -> list[str]:
    problems = []
    for rel, key, kind in UPSTREAM:
        p = assets_root() / rel
        if not p.exists():
            problems.append(f"missing upstream {rel}")
            continue
        src = S.SRC[key]
        version = git_commit(p) if kind == "git" else src["version"]
        man.register(
            p,
            kind="upstream",
            source_url=src["url"],
            license=src["license"],
            version=version,
            citation=src["citation"],
            notes=src.get("bias", ""),
        )
    return problems


def _register_derived(man: Manifest, path: Path, produced_by: str, inputs: list[str]) -> None:
    man.register(
        path,
        kind="derived",
        source_url="built by scwbd.anatomy",
        license="derived work; inherits the most restrictive upstream license "
        "of its inputs (see the `inputs` field)",
        version=f"scwbd.anatomy/{time.strftime('%Y%m%d')}",
        citation="; ".join(S.SRC[k]["citation"] for k in inputs if k in S.SRC),
        produced_by=produced_by,
        inputs=inputs,
        notes="Cached build product. Delete and re-run scwbd.anatomy.build to regenerate.",
    )


def build(*, rebuild: bool = False, verbose: bool = True) -> dict[str, Any]:
    man = Manifest()
    report: dict[str, Any] = {"built": [], "failed": [], "skipped": []}

    def log(*a: object) -> None:
        if verbose:
            print(*a, flush=True)

    log("== upstream ==")
    for msg in _register_upstream(man):
        report["failed"].append(msg)
        log("  !", msg)
    for rel, _k, _t in UPSTREAM:
        if (assets_root() / rel).exists():
            log(f"  ok {rel}")

    log("== parcellations ==")
    for name, space, density in BUILD_SURFACE + BUILD_VOLUME:
        if space is None or space not in ATLAS_SPECS[name]["spaces"]:
            report["skipped"].append(f"{name}/{space}")
            continue
        try:
            p = load_parcellation(name, space, density, rebuild=rebuild)
            f = derived_dir("parcellations") / f"{name}__{space}-{density}.npz"
            _register_derived(man, f, "scwbd.anatomy.atlases.load_parcellation",
                              _atlas_inputs(name, p))
            report["built"].append(str(f.name))
            log(f"  ok {name} {space}/{density} n={p.n_parcels}")
        except Exception as exc:  # noqa: BLE001
            report["failed"].append(f"parcellation {name}/{space}: {exc!r}")
            log(f"  ! {name} {space}: {exc!r}")
            if verbose:
                traceback.print_exc()

    log("== geometry ==")
    for name, space, density in BUILD_SURFACE:
        try:
            p = load_parcellation(name, space, density)
            parcel_geometry(p, rebuild=rebuild)
            f = derived_dir("geometry") / f"{name}__{space}-{density}__geom.npz"
            _register_derived(man, f, "scwbd.anatomy.geometry.parcel_geometry",
                              _atlas_inputs(name, p) + ["conte69"])
            report["built"].append(str(f.name))
            log(f"  ok {name}")
        except Exception as exc:  # noqa: BLE001
            report["failed"].append(f"geometry {name}: {exc!r}")
            log(f"  ! {name}: {exc!r}")

    log("== maps ==")
    for name, space, density in BUILD_SURFACE:
        if space != "fsLR":
            continue
        try:
            p = load_parcellation(name, space, density)
            ms = load_maps(p, rebuild=rebuild)
            f = derived_dir("maps") / f"{name}__{space}-{density}__maps.npz"
            _register_derived(man, f, "scwbd.anatomy.maps.load_maps",
                              _maps_inputs(ms))
            report["built"].append(str(f.name))
            log(f"  ok {name}: {len(ms.maps)} maps, {len(ms.receptor_names)} receptors")
            # A map that was expected and could not be built is a result, not a
            # non-event: surface it here rather than letting a short panel ship
            # silently the way the Schaefer300/400 5HT4 gap did.
            for k, why in sorted(ms.unavailable.items()):
                report.setdefault("unavailable", []).append(f"{name}: {k}: {why}")
                log(f"    unavailable {k}: {why}")
        except Exception as exc:  # noqa: BLE001
            report["failed"].append(f"maps {name}: {exc!r}")
            log(f"  ! {name}: {exc!r}")

    log("== connectomes ==")
    for name in _ENIGMA_KEYS:
        for include_sub in (True, False):
            try:
                sp = load_structural_prior(name, include_subcortex=include_sub,
                                           rebuild=rebuild)
                # Derived, never reconstructed: see structural_cache_tag.__doc__.
                tag = structural_cache_tag(name, include_sub, None, "euclidean")
                f = derived_dir("connectome") / f"{tag}.npz"
                # The subcortical parcels come from the subcortical atlas, and
                # the Melbourne licence's one condition is that work using it
                # cites Tian 2020. Attributing the 414-parcel connectome to
                # ENIGMA alone omits a required citation.
                ci = _connectome_inputs(sp)
                if include_sub:
                    ci = sorted(set(ci) | set(_atlas_inputs(DEFAULT_SUBCORTICAL_ATLAS)))
                _register_derived(man, f, "scwbd.anatomy.connectome.load_structural_prior", ci)
                report["built"].append(str(f.name))
                c = sp.class_counts()
                log(f"  ok {name} sctx={include_sub}: hard={c['hard']} soft={c['soft']} "
                    f"proposed={c['proposed']} absent={c['absent']}")
            except Exception as exc:  # noqa: BLE001
                report["failed"].append(f"connectome {name}: {exc!r}")
                log(f"  ! {name}: {exc!r}")

    log("== brain priors ==")
    for name in ("Schaefer100x7", "Schaefer400x7", "DesikanKilliany"):
        try:
            bp = BrainPrior.load(name, include_subcortex=True)
            log(f"  ok {name}: {json.dumps(bp.summary())[:160]}")
        except Exception as exc:  # noqa: BLE001
            report["failed"].append(f"brainprior {name}: {exc!r}")
            log(f"  ! {name}: {exc!r}")

    man.meta.update(
        {
            "produced_by": "scwbd.anatomy.build",
            "agent": "C (adult human anatomical priors)",
            "note": (
                "Binaries live on /data/scwbd/assets and are not tracked. This "
                "manifest is. Licenses are per-asset and some are non-commercial "
                "(the Hansen receptor atlas is CC-BY-NC-SA-4.0)."
            ),
        }
    )
    p = man.save()
    log(f"\nmanifest: {p} ({len(man.entries)} assets, "
        f"{sum(e.n_bytes for e in man.entries.values()) / 1e9:.2f} GB)")
    report["manifest"] = str(p)
    return report


def _maps_inputs(ms: Any) -> list[str]:
    """Source keys this ``MapSet`` actually loaded.

    Read off the built object rather than declared beside the call, because a
    literal cannot be wrong in a way anybody notices.  The hardcoded list this
    replaced was wrong in both directions on every maps asset: it omitted
    ``hill2010`` and ``raichle_metabolism`` (two unknown-licence sources that
    are genuinely read) and listed ``neuromaps`` (which is not a source key on
    any loaded map).  An audit driven off ``inputs`` could not see either.
    """
    return sorted({m.source_key for m in ms.maps.values()})


def _connectome_inputs(sp: Any) -> list[str]:
    """Source keys this ``StructuralPrior`` actually loaded.

    Two parts, and the second is easy to miss.  ``provenance["streams"]`` names
    the independent streams layered on; ``provenance["source"]`` is the *base*
    weight matrix and does not appear among them -- on ``DesikanKilliany`` the
    stream list is ``["hansen_lausanne_sc"]`` alone, yet the weights are
    ENIGMA/HCP.  Recovering the base key by identity against ``S.SRC`` keeps it
    derived instead of restating a literal that could drift from the loader.
    """
    keys = {st["source_key"] for st in sp.provenance.get("streams", [])}
    base = sp.provenance.get("source") or {}
    # Match on ``name``, not identity. The provenance dict is JSON round-tripped
    # through the .npz cache, so ``is`` against ``S.SRC`` fails for any prior
    # loaded from disk -- which is every prior in a normal build. That bug was
    # in the first version of this function and it dropped ``enigma_hcp_sc``
    # from DesikanKilliany, i.e. it under-reported a real dependency while
    # looking like it worked.
    name = base.get("name")
    if name is not None:
        hit = next((k for k, v in S.SRC.items() if v.get("name") == name), None)
        # An unregistered base source is recorded, never dropped: the point of
        # this field is that it can say what loaded, so a miss has to be
        # visible in the manifest rather than absent from it.
        keys.add(hit if hit is not None else f"unregistered:{name}")
    return sorted(keys)


#: Extra source keys an atlas reads beyond the one its own provenance names --
#: chiefly the ENIGMA toolbox, which supplies the label files for the atlases it
#: redistributes. Kept as a literal because it records a *packaging* fact that
#: no built object carries; the primary source is always derived, never listed.
_ATLAS_EXTRA_INPUTS: dict[str, list[str]] = {
    "DesikanKilliany": ["enigmatoolbox"],
    "Glasser360": ["enigmatoolbox"],
    "EconomoKoskinas": ["enigmatoolbox"],
}


def _atlas_inputs(name: str, parc: Any | None = None) -> list[str]:
    """Source keys an atlas actually reads, derived from its own provenance.

    This used to be a hand-maintained ``{name: [keys]}`` literal, and it failed
    exactly the way literals fail: ``Aseg14T`` -- the **default** subcortical
    atlas, and half of what makes the prior 414 parcels -- was absent, so it
    fell through to ``[]``. An asset with no declared inputs has no attribution,
    which is what made the attribution gate refuse the whole anatomy prior.

    The primary key is now recovered by matching the parcellation's own
    ``provenance.source_url`` against :data:`scwbd.anatomy.sources.SRC`, the
    same identity trick :func:`_connectome_inputs` uses for its base matrix. A
    literal cannot be wrong in a way anybody notices; a derived value fails
    loudly when the thing it derives from moves.
    """
    # The atlas that DEFINES the parcellation. Its citation is required by
    # several of these licences and is not recoverable from the redistributor's
    # url, so it is named here and always included.
    if name.startswith("Schaefer"):
        primary = ["schaefer2018"]
    elif name.startswith("Tian") or name.startswith("Aseg14T"):
        primary = ["tian2020"]
    else:
        primary = {
            "DesikanKilliany": ["desikan2006"],
            "Glasser360": ["glasser2016"],
            "EconomoKoskinas": ["voneconomo"],
            "Destrieux": ["destrieux2010"],
            "Aseg14": ["harvardoxford"],
            "Buckner7": ["buckner2011"],
            "Buckner17": ["buckner2011"],
            "SUITAnatom": ["diedrichsen2009"],
        }.get(name, [])

    # Whoever we actually read the bytes from, derived from the built object.
    # This is UNIONED with the primary, never substituted for it: the Schaefer
    # labels arrive via the ENIGMA toolbox, so a url match alone silently
    # replaced "schaefer2018" with "enigmatoolbox" and dropped the citation the
    # parcellation's own licence asks for.
    derived: list[str] = []
    if parc is not None:
        url = str(getattr(getattr(parc, "provenance", None), "source_url", "") or "")
        if url:
            derived = [k for k, v in S.SRC.items() if str(v.get("url", "")) == url]

    out = sorted(set(primary) | set(derived) | set(_ATLAS_EXTRA_INPUTS.get(name, [])))
    if not out:
        raise ValueError(
            f"atlas {name!r} resolves to NO source keys, so anything derived from "
            "it would ship unattributed. Refusing to register an asset with an "
            "empty inputs list -- that is the state the attribution gate exists "
            "to catch, and it is how Aseg14T reached the manifest with no licence."
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="ignore cached artifacts")
    ap.add_argument("--verify", action="store_true", help="re-hash manifest entries and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        man = Manifest()
        status = man.verify()
        bad = {k: v for k, v in status.items() if v != "ok"}
        for k, v in sorted(bad.items()):
            print(f"{v:16s} {k}")
        print(f"{len(status) - len(bad)}/{len(status)} assets verified")
        return 1 if bad else 0

    rep = build(rebuild=args.rebuild, verbose=not args.quiet)
    if rep["failed"]:
        print("\nFAILURES:")
        for f in rep["failed"]:
            print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
