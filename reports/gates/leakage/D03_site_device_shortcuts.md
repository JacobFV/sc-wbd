# D03_site_device_shortcuts — COULD_NOT_RUN

**Claim.** Site/device shortcuts is controlled for. Primary metric: Domain calibration, worst-site error and residual site predictability.

**Falsified by (thesis).** The mandatory control (Leave-site/device/protocol-out evaluation; nuisance-only classifier and label permutation within site) shows the result survives only without it.

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 1996fba · 2026-08-06T08:36:42+00:00*

## Could not run

> This gate did **not** pass. It did not run. Nothing may be claimed on its basis.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| leave_site_out | yes | COULD_NOT_RUN | no per-site datasets or model factory supplied; pooled accuracy alone cannot detect a site shortcut |
| nuisance_only_classifier | yes | COULD_NOT_RUN | no nuisance features/labels supplied; residual site predictability is unmeasured |
| within_site_label_permutation | yes | COULD_NOT_RUN | no permutation scores supplied; without them, an apparent effect cannot be distinguished from site structure |

## Blocking reasons

- leave_site_out: could not run — no per-site datasets or model factory supplied; pooled accuracy alone cannot detect a site shortcut
- nuisance_only_classifier: could not run — no nuisance features/labels supplied; residual site predictability is unmeasured
- within_site_label_permutation: could not run — no permutation scores supplied; without them, an apparent effect cannot be distinguished from site structure

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `min_delta_log_score`: 0.0
- `max_coverage_error`: 0.05
- `max_overconfidence_increase`: 0.02
- `max_delay_rel_error`: 0.15
- `boundary_rel_tol`: 0.05
- `max_hallucination_index`: 1.25
- `min_uncertainty_inflation`: 1.05
- `min_fisher_eig_gain`: 1.1
- `capacity_tol`: 0.1
- `min_model_discrimination`: 0.05
- `n_boot`: 1000

## Explicit non-goals

- This gate does not claim a validated digital twin of any specific person.
- This gate does not claim that any admitted operator is neurally realized.
- No prospective human TMS/tFUS protocol is implemented or implied (build order stops at item 5; item 6 is out of scope: no IRB, no consent, no participants).
