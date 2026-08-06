"""Machine-readable source cards (Appendix B, ``tab:source-card``).

A named dataset is not a training instruction.  Every dataset enters SC-WBD
through a versioned card

.. math:: D_k = (I_k, G_k, P_k, S_k, T_k, C_k, O_k, U_k, M_k, B_k, V_k,
                 \\Delta_k, A_k, R_k)

serialised as YAML in ``scwbd/sources/cards/*.yaml``.  This module loads,
structurally validates and *interprets* those cards.  The interpretation that
matters is :meth:`SourceCardDoc.effective_gradient_permission`: an ``allow``
entry whose declared prerequisites are ``unknown`` is **disabled**, not
guessed.  Per Appendix B that is the contract, not a failure.

When ``scwbd.schema.SourceCard`` (agent A) becomes importable, :func:`load_card`
additionally validates the YAML against it; until then the structural checks in
this module are the whole of validation and :attr:`SourceCardDoc.typed` is
``None``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

CARD_DIR = Path(__file__).parent / "cards"
CARD_SCHEMA_VERSION = "scwbd-source-card/1.0.0"

UNKNOWN = "unknown"

#: Top-level sections, in Appendix B order.  ``observation`` and
#: ``intervention`` may be ``null`` (a source that manipulates nothing has no
#: intervention operator) but the key must be present and explicit.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "identity",
    "governance",
    "population",
    "signal",
    "spatial",
    "temporal",
    "calibration",
    "observation",
    "intervention",
    "missingness",
    "ledger",
    "gradient_permission",
    "role",
    "split_policy",
)

NULLABLE_SECTIONS = frozenset({"observation", "intervention"})

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": (
        "id",
        "name",
        "version",
        "release_date",
        "doi",
        "source_url",
        "retrieved_utc",
        "parent_dataset",
        "derived_from",
        "software",
        "container_hash",
        "local_path",
        "file_manifest",
    ),
    "governance": (
        "status",
        "license",
        "license_url",
        "license_text_excerpt",
        "dua_required",
        "credentials_required",
        "purpose_limits",
        "consent_scope",
        "redistribution",
        "withdrawal",
        "privacy",
        "retention",
        # machine-readable siblings consumed by the typed projection
        "redistribution_class",
        "withdrawal_status",
        "privacy_class",
        "contains_biometric",
        "may_release_weights",
        "may_release_examples",
    ),
    "population": (
        "n_participants",
        "participant_ids",
        "families",
        "sites",
        "devices",
        "sessions_per_participant",
        "runs_per_session",
        "trials_per_run",
        "demographics",
        "selection_mechanism",
        "grouping_keys",
    ),
    "signal": (
        "modalities",
        "n_channels",
        "channel_types",
        "units",
        "reference",
        "dynamic_range",
        "saturation",
        "quantization",
        # machine-readable SI siblings consumed by the typed projection
        "si_units",
        "si_dynamic_range",
        "si_saturation",
        "si_quantization",
    ),
    "spatial": ("kind", "frame", "units", "psf", "extent", "n_elements"),
    "temporal": (
        "clock",
        "dt",
        "sfreq_hz",
        "integration_window",
        "group_delay",
        "jitter_sd",
        "event_clock",
        "offset_drift",
    ),
    "calibration": (
        "frames",
        "fiducials",
        "transforms",
        "calibration_points",
        "residuals",
        "device_pose",
        "raw_metadata",
    ),
    "missingness": (
        "planned",
        "unplanned",
        "artifact_rejection",
        "dropout",
        "attrition",
        "policy",
        "mechanism",
        "handling",
    ),
    "ledger": (
        "variance",
        "bias_interval",
        "bias_status",
        "model_discrepancy",
        "validity_domain",
    ),
    "gradient_permission": ("allow", "frozen", "forbid", "evaluation_only"),
    "split_policy": (
        "grouping_keys",
        "temporal_holdout",
        "stimulus_holdout",
        "notes",
        "leakage_barrier",
        "role_locked",
        "stimuli_shared_across_folds",
    ),
}

VALID_ROLES = frozenset(
    {
        "prior",
        "likelihood",
        "boundary_target",
        "distillation",
        "calibration",
        "negative_control",
        "evaluation_only",
    }
)

VALID_STATUS = frozenset({"live", "unavailable", "partial", "withdrawn"})

VALID_BIAS_STATUS = frozenset(
    {"design_estimable", "externally_bounded", "prior_specified_sensitivity", "unknown"}
)


class CardError(ValueError):
    """A source card is structurally invalid.  Refusal ``R01``/``R08`` family."""

    def __init__(self, card_id: str, message: str) -> None:
        super().__init__(f"[source-card:{card_id}] {message}")
        self.card_id = card_id
        self.message = message


def _get_path(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return KeyError
    return cur


def _is_unknown(value: Any) -> bool:
    """True when a card field declares itself unresolved.

    A field counts as unknown when its value is the literal ``unknown`` **or**
    a sentence that *begins* with ``unknown`` - cards are allowed (encouraged)
    to say ``unknown - the release does not document the reference``, and that
    must still disable the gradient paths that depend on it.  A field that
    merely mentions the word later in a sentence is not unknown.
    """
    if value is KeyError:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        if s == UNKNOWN:
            return True
        return s.startswith(UNKNOWN) and (
            len(s) == len(UNKNOWN) or not s[len(UNKNOWN)].isalnum()
        )
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(_is_unknown(v) for v in value)
    if isinstance(value, dict):
        return len(value) > 0 and all(_is_unknown(v) for v in value.values())
    return False


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class GradientDecision:
    """One entry of the compiled gradient mask ``A_k``."""

    target: str
    enabled: bool
    scales: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    reason: str = ""


@dataclass
class SourceCardDoc:
    """A loaded source card.

    ``data`` is the raw YAML mapping (the machine-readable artifact itself);
    the helpers below are the interpretation the compiler needs.
    """

    path: Path
    data: dict[str, Any]
    typed: Any = None  # scwbd.schema.SourceCard instance, when the projection validates
    typed_error: str = ""

    # -- identity -------------------------------------------------------
    @property
    def id(self) -> str:
        return str(self.data["identity"]["id"])

    @property
    def version(self) -> str:
        return str(self.data["identity"]["version"])

    @property
    def role(self) -> str:
        return str(self.data["role"])

    @property
    def status(self) -> str:
        return str(self.data["governance"]["status"])

    @property
    def available(self) -> bool:
        return self.status in ("live", "partial")

    @property
    def local_path(self) -> Path | None:
        p = self.data["identity"].get("local_path")
        if not p or _is_unknown(p):
            return None
        return Path(str(p))

    def content_hash(self) -> str:
        """Stable sha256 over canonical JSON of the card."""
        return hashlib.sha256(canonical_json(self.data).encode()).hexdigest()

    # -- unknown-field accounting --------------------------------------
    def unknown_fields(self) -> list[str]:
        """Dotted paths whose value is the literal ``unknown``."""
        out: list[str] = []

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))
            elif _is_unknown(node):
                out.append(prefix)

        walk(self.data, "")
        return sorted(out)

    # -- gradient contract ---------------------------------------------
    def gradient_decisions(self) -> list[GradientDecision]:
        gp = self.data.get("gradient_permission") or {}
        decisions: list[GradientDecision] = []
        if gp.get("evaluation_only"):
            for entry in gp.get("allow") or []:
                decisions.append(
                    GradientDecision(
                        target=str(entry.get("target")),
                        enabled=False,
                        scales=tuple(entry.get("scales") or ()),
                        requires=tuple(entry.get("requires") or ()),
                        reason="card declares evaluation_only: no training gradient ever",
                    )
                )
            return decisions
        if not self.available:
            for entry in gp.get("allow") or []:
                decisions.append(
                    GradientDecision(
                        target=str(entry.get("target")),
                        enabled=False,
                        scales=tuple(entry.get("scales") or ()),
                        requires=tuple(entry.get("requires") or ()),
                        reason=f"governance.status={self.status!r}: no bytes on disk",
                    )
                )
            return decisions
        for entry in gp.get("allow") or []:
            requires = tuple(entry.get("requires") or ())
            unresolved = tuple(r for r in requires if _is_unknown(_get_path(self.data, r)))
            decisions.append(
                GradientDecision(
                    target=str(entry.get("target")),
                    enabled=not unresolved,
                    scales=tuple(entry.get("scales") or ()),
                    requires=requires,
                    unresolved=unresolved,
                    reason=(
                        ""
                        if not unresolved
                        else "disabled: prerequisite field(s) unknown -> "
                        + ", ".join(unresolved)
                    ),
                )
            )
        return decisions

    def effective_gradient_permission(self) -> dict[str, Any]:
        """The compiled gradient mask: what this source may actually update."""
        decisions = self.gradient_decisions()
        gp = self.data.get("gradient_permission") or {}
        return {
            "source_id": self.id,
            "role": self.role,
            "enabled": [d.target for d in decisions if d.enabled],
            "disabled": {d.target: d.reason for d in decisions if not d.enabled},
            "frozen": list(gp.get("frozen") or []),
            "forbid": list(gp.get("forbid") or []),
            "evaluation_only": bool(gp.get("evaluation_only")),
        }

    # -- structural validation -----------------------------------------
    def validate(self) -> "SourceCardDoc":
        cid = str(self.data.get("identity", {}).get("id", self.path.stem))
        if self.data.get("schema_version") != CARD_SCHEMA_VERSION:
            raise CardError(cid, f"schema_version must be {CARD_SCHEMA_VERSION!r}")
        for section in REQUIRED_SECTIONS:
            if section not in self.data:
                raise CardError(cid, f"missing required section {section!r}")
            if self.data[section] is None and section not in NULLABLE_SECTIONS:
                raise CardError(cid, f"section {section!r} may not be null")
        for section, fields in REQUIRED_FIELDS.items():
            body = self.data.get(section)
            if body is None:
                continue
            if not isinstance(body, dict):
                raise CardError(cid, f"section {section!r} must be a mapping")
            for f in fields:
                if f not in body:
                    raise CardError(cid, f"{section}.{f} is required (use 'unknown' if unresolved)")
        if self.role not in VALID_ROLES:
            raise CardError(cid, f"role {self.role!r} not in {sorted(VALID_ROLES)}")
        if self.status not in VALID_STATUS:
            raise CardError(cid, f"governance.status {self.status!r} not in {sorted(VALID_STATUS)}")
        bstat = self.data["ledger"]["bias_status"]
        if bstat not in VALID_BIAS_STATUS:
            raise CardError(cid, f"ledger.bias_status {bstat!r} not in {sorted(VALID_BIAS_STATUS)}")
        bi = self.data["ledger"]["bias_interval"]
        if not (_is_unknown(bi) or (isinstance(bi, (list, tuple)) and len(bi) == 2)):
            raise CardError(cid, "ledger.bias_interval must be a [lo, hi] pair or 'unknown'")
        if not _is_unknown(bi) and float(bi[0]) > float(bi[1]):
            raise CardError(cid, "ledger.bias_interval lower bound exceeds upper bound")
        gp = self.data["gradient_permission"]
        for entry in gp.get("allow") or []:
            if "target" not in entry:
                raise CardError(cid, "gradient_permission.allow entry without a 'target'")
            if "requires" not in entry:
                raise CardError(
                    cid,
                    f"gradient_permission.allow[{entry['target']}] must declare 'requires' "
                    "(possibly []) so unknown fields can disable it",
                )
        # governance consistency: a live card must point at real bytes
        if self.status == "unavailable":
            if not self.data["governance"].get("unavailable_reason"):
                raise CardError(cid, "governance.status='unavailable' requires unavailable_reason")
            if self.data["identity"].get("local_path") not in (None, "", UNKNOWN):
                raise CardError(cid, "unavailable card must not claim a local_path")
        else:
            fm = self.data["identity"]["file_manifest"]
            for k in ("n_files", "total_bytes", "manifest_sha256", "exemplars"):
                if k not in fm:
                    raise CardError(cid, f"identity.file_manifest.{k} is required for a live card")
            if not fm["exemplars"]:
                raise CardError(cid, "a live card must record at least one exemplar file hash")
        # split policy must name grouping keys unless evaluation is impossible
        sp = self.data["split_policy"]
        if not sp.get("grouping_keys"):
            raise CardError(cid, "split_policy.grouping_keys must be non-empty (R10)")
        return self

    # -- typed schema bridge -------------------------------------------
    def to_typed_payload(self) -> dict[str, Any]:
        """Project this card onto the field shape of ``scwbd.schema.SourceCard``.

        The YAML card is the audit artifact: it carries prose, provenance and
        the honest ``unknown``s.  ``scwbd.schema.SourceCard`` (agent A) is the
        *compiler* view: narrow enums and floats.  The projection is therefore
        lossy **downwards only** - every typed field is either taken verbatim
        from an explicit machine-readable field of this card, or omitted.

        Two consequences are deliberate and must not be "fixed":

        * ``temporal.group_delay``/``jitter_sd`` that are ``unknown`` here are
          omitted, so the typed model falls back to its ``0.0`` default.  That
          default is *not* a claim: the corresponding gradient path
          (``transforms.clock_graph.*``) is disabled by
          :meth:`effective_gradient_permission`, and the reason travels with
          the mask.
        * ``ledger.variance`` entries that are ``unknown`` are dropped rather
          than zero-filled; a zero variance would be a false reliability claim
          (Appendix B, "Bias, variance and discrepancy").
        """
        d = self.data
        ident, gov, pop = d["identity"], d["governance"], d["population"]
        sig, spa, tem = d.get("signal") or {}, d["spatial"], d["temporal"]
        cal, mis, led = d["calibration"], d["missingness"], d["ledger"]
        gp, sp = d["gradient_permission"], d["split_policy"]
        fm = ident.get("file_manifest") or {}

        def clean(x: Any) -> Any:
            return None if _is_unknown(x) else x

        def num(x: Any) -> float | None:
            x = clean(x)
            try:
                return None if x is None else float(x)
            except (TypeError, ValueError):
                return None

        file_hashes = dict(fm.get("exemplars") or {})
        if fm.get("manifest_sha256"):
            file_hashes["__manifest__"] = str(fm["manifest_sha256"])

        payload: dict[str, Any] = {
            "label": str(ident.get("name", "")),
            "role": self.role,
            "identity": {
                "persistent_id": self.id,
                "version": self.version,
                "release": str(ident.get("release_date") or ""),
                "file_hashes": file_hashes,
                "parent_ids": tuple(x for x in [ident.get("parent_dataset")] if x),
                "software_hash": clean(ident.get("software_hash")),
                "container_hash": clean(ident.get("container_hash")),
                "preprocessing_hash": clean(ident.get("preprocessing_hash")),
                "mutable_download": bool(ident.get("mutable_download", False)),
            },
            "governance": {
                "license": str(gov["license"]),
                "dua": clean(gov.get("dua")),
                "purpose_limits": tuple(gov.get("purpose_limits_list") or ()),
                "consent_scope": str(gov.get("consent_scope", "")),
                "redistribution": gov.get("redistribution_class", "unknown"),
                "withdrawal_status": gov.get("withdrawal_status", "unknown"),
                "privacy_class": gov.get("privacy_class", "deidentified"),
                "retention": str(gov.get("retention", "")),
                "may_release_weights": bool(gov.get("may_release_weights", False)),
                "may_release_examples": bool(gov.get("may_release_examples", False)),
                "contains_biometric": bool(gov.get("contains_biometric", True)),
                "safeguards": tuple(gov.get("safeguards") or ()),
            },
            "population": {
                "levels": tuple(pop.get("grouping_keys") or ()),
                "n_participants": pop.get("n_participants")
                if isinstance(pop.get("n_participants"), int)
                else None,
                "selection_mechanism": str(pop.get("selection_mechanism", "")),
                "min_subgroup_n": pop.get("min_subgroup_n"),
            },
            "spatial": {
                "kind": spa["kind"],
                "frame": spa["frame"],
                "units": spa["units"],
                "extent": tuple(spa["extent"]) if isinstance(spa.get("extent"), list) else None,
                "n_elements": spa.get("n_elements") if isinstance(spa.get("n_elements"), int) else None,
                "label": str(spa.get("label", "")),
            },
            "temporal": {"clock": tem["clock"], "dt": float(tem["dt"])},
            "calibration": {"id": str(cal.get("id", f"{self.id}.calibration"))},
            "missingness": {
                "mechanism": mis.get("mechanism", "unknown"),
                "handling": mis.get("handling", "mask"),
                "artifact_rejection": str(mis.get("artifact_rejection", "")),
                "post_outcome_selection": bool(mis.get("post_outcome_selection", False)),
                "censoring_correlated_with_state": bool(
                    mis.get("censoring_correlated_with_state", False)
                ),
            },
            "ledger": {
                "variance": {
                    k: float(v)
                    for k, v in (led.get("variance") or {}).items()
                    if num(v) is not None
                },
                "bias_interval": tuple(led["bias_interval"])
                if not _is_unknown(led["bias_interval"])
                else (0.0, 0.0),
                "bias_status": led["bias_status"]
                if led["bias_status"] != UNKNOWN
                else "prior_specified_sensitivity",
                "model_discrepancy": num(led.get("model_discrepancy")),
                "validity_domain": led.get("validity_domain") or {},
                "external_bound_source": clean(led.get("bias_bound_justification")),
            },
            "gradient_permission": {
                "modules": tuple(self.effective_gradient_permission()["enabled"]),
                "scales": tuple(
                    sorted({s for dec in self.gradient_decisions() if dec.enabled for s in dec.scales})
                ),
                "frozen": tuple(gp.get("frozen") or ()),
                "evaluation_only": bool(gp.get("evaluation_only", False)),
                "notes": "; ".join(
                    f"{t}: {r}" for t, r in self.effective_gradient_permission()["disabled"].items()
                ),
            },
            "split_policy": {
                "grouping_keys": tuple(sp.get("grouping_keys") or ()),
                "leakage_barrier": sp.get("leakage_barrier", "parent_lineage"),
                "role_locked": bool(sp.get("role_locked", True)),
                "stimuli_shared_across_folds": bool(sp.get("stimuli_shared_across_folds", False)),
            },
        }
        for key, target in (("integration_window", "integration_window"),
                            ("group_delay", "group_delay"),
                            ("jitter_sd", "jitter_sd")):
            v = num(tem.get(key))
            if v is not None:
                payload["temporal"][target] = v

        if sig and not _is_unknown(sig.get("si_units")):
            payload["signal"] = {
                "name": str(sig.get("name", self.id)),
                "units": sig["si_units"],
                "reference": str(sig.get("reference", "")),
                "dynamic_range": tuple(sig["si_dynamic_range"])
                if isinstance(sig.get("si_dynamic_range"), list)
                else None,
                "saturation": num(sig.get("si_saturation")),
                "quantization": num(sig.get("si_quantization")),
                "n_channels": sig.get("n_channels") if isinstance(sig.get("n_channels"), int) else None,
                "irreversible_aggregation": bool(sig.get("irreversible_aggregation", False)),
            }
        obs = d.get("observation")
        if obs:
            payload["observation"] = {
                "name": str(obs.get("name", f"{self.id}.observation")),
                "forward_physics": str(obs.get("forward_physics", "unknown")),
                "observed_variables": tuple(obs.get("observed_variables") or ()),
                "preprocessing": tuple(obs.get("preprocessing_steps") or ()),
                "likelihood_kind": obs.get("likelihood_kind", "generative"),
                "calibration_status": obs.get("calibration_status", "uncalibrated"),
                "leakage_checked": bool(obs.get("leakage_checked", False)),
                "target_ports": tuple(obs.get("target_ports") or ()),
            }
        itv = d.get("intervention")
        if itv:
            payload["intervention"] = {
                "name": str(itv.get("name", f"{self.id}.intervention")),
                "modality": itv["modality"],
                "waveform": str(itv.get("waveform", "")),
                "dose": {k: float(v) for k, v in (itv.get("dose") or {}).items()
                         if num(v) is not None},
                "dose_units": dict(itv.get("dose_units") or {}),
                "pose_frame": clean(itv.get("pose_frame")),
                "field_solver": clean(itv.get("field_solver")),
                "target_ports": tuple(itv.get("target_ports") or ()),
                "sham": clean(itv.get("sham")),
                "randomized": bool(itv.get("randomized", False)),
                "dose_independently_calibrated": bool(
                    itv.get("dose_independently_calibrated", False)
                ),
            }
        th = sp.get("temporal_holdout_spec")
        if isinstance(th, dict):
            payload["split_policy"]["temporal_holdout"] = th
        return payload

    def bind_typed(self, *, strict: bool = False) -> Any:
        """Validate against ``scwbd.schema.SourceCard`` when that type exists.

        Never fatal by default: agent A owns the typed schema and it may move.
        The failure reason is kept in :attr:`typed_error` so tests and the
        inventory report can show it instead of pretending the card is bound.
        """
        self.typed_error = ""
        if self.status == "unavailable":
            # There is nothing to compile: no dt, no units, no hashes.  The
            # typed model requires real numbers and we will not invent them.
            self.typed = None
            self.typed_error = (
                "not projected: governance.status='unavailable' - the typed model requires "
                "measured values (dt, units, hashes) that do not exist for a source we did "
                "not download"
            )
            return None
        try:  # pragma: no cover - depends on agent A's module
            from scwbd.schema import SourceCard  # type: ignore
        except Exception as exc:
            self.typed = None
            self.typed_error = f"scwbd.schema.SourceCard not importable: {exc}"
            return None
        try:
            self.typed = SourceCard.model_validate(self.to_typed_payload())  # type: ignore[attr-defined]
        except Exception as exc:
            self.typed = None
            self.typed_error = f"{type(exc).__name__}: {exc}"
            if strict:
                raise
        return self.typed


def load_card(path: str | Path, *, validate: bool = True, typed: bool = True) -> SourceCardDoc:
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise CardError(path.stem, "card must be a YAML mapping")
    doc = SourceCardDoc(path=path, data=data)
    if validate:
        doc.validate()
    if typed:
        doc.bind_typed()
    return doc


def load_all_cards(card_dir: str | Path = CARD_DIR, **kw: Any) -> dict[str, SourceCardDoc]:
    out: dict[str, SourceCardDoc] = {}
    for p in sorted(Path(card_dir).glob("*.yaml")):
        doc = load_card(p, **kw)
        if doc.id in out:
            raise CardError(doc.id, f"duplicate card id (also in {out[doc.id].path})")
        out[doc.id] = doc
    return out


def iter_card_paths(card_dir: str | Path = CARD_DIR) -> Iterator[Path]:
    yield from sorted(Path(card_dir).glob("*.yaml"))


__all__: Sequence[str] = (
    "CARD_DIR",
    "CARD_SCHEMA_VERSION",
    "CardError",
    "GradientDecision",
    "SourceCardDoc",
    "load_card",
    "load_all_cards",
    "iter_card_paths",
)
