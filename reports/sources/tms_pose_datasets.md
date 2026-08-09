# Public TMS releases that carry a per-pulse coil pose and a measured response

Searched 2026-08-09. Nothing was downloaded; every field below was read from a
repository API, a licence file, a data-description file, or a paper's data
availability statement, and each is cited with the URL it came from.

**Why this search exists.** `ds004024` is MRI-navigated and distributes no
per-pulse coil pose log, so the loader records `coil_pose:
Provenance.UNKNOWN` and notes that this "disables the E-field operator path
outright" (`scwbd/sources/perturbation/ds004024.py:486-496`). With no pose there
is no E-field, so the map from a computed field to a neural response has nothing
to be tested against, and `site/content/possibilities/index.html:145-153` keeps
that open. The cheapest way to close it is a public release that ships
`(pose, response)` per pulse — `reports/robotics_bridge_assessment.md` §6.

---

## 0. Headline

**Four public releases pair a recorded coil configuration with a measured
response, pulse by pulse. All four come from two laboratories. One of them
states a licence.**

| # | release | n | pose granularity | response | licence | size | fetch first |
|---|---|---|---|---|---|---|---|
| 1 | `github.com/david-schu/gp-tms-hsh` | 8 | per pulse, 3-parameter (x, y, θ) | measured MEP, right FDI | **CC BY-NC 4.0, stated in the repository** | 226.9 MB | **yes** |
| 2 | OSF `myrqn` / `subject_0` | 1 | per pulse, 6-DOF, Localite | measured MEP, per pulse | **none** | 0.99 GB measured, more unmeasured | no |
| 3 | GWDG GitLab `papers/pcd`, `data/pcd_e_mep.csv` | 9 | per pulse, collapsed to one scalar | measured MEP, normalised | **none** | ~0.5 MB | no |
| 4 | OSF `9f3bc` | 10 raw + 14 derived | absent from the release | measured MEP (validation blocks) | **none** | ~150 MB | no |

The single fact that decides the ranking: **only #1 states a licence anywhere in
the artifact.** #2, #3 and #4 are public, anonymously downloadable, and carry no
grant of any kind — OSF's API returns `node_license: null` for both OSF nodes,
and the GitLab project reports `license: null` with no `LICENSE` blob in its
tree. Under this repository's rule — a licence identifier or text **in this
repository**, not a pointer to where one might be found — none of the three can
reach `established`, regardless of how good the data is. And #2 is the best data.

**What does not exist.** No public release carries, for more than one
participant, all three of: a per-pulse 6-DOF coil pose, the individual MRI
needed to turn that pose into an E-field, and a measured response — under a
stated licence. That combination is available for exactly one participant
(#2, unlicensed). This is a real finding about the field, not a gap in the
search: §5 lists what was swept and what each candidate was rejected for.

---

## 1. `gp-tms-hsh` — Freiburg, 8 participants, licensed

**URL** https://github.com/david-schu/gp-tms-hsh
**Paper** Schultheiss, Turi, Boedecker, Vlachos (2026), *Efficient Gaussian
process-based motor hotspot hunting with concurrent optimization of TMS coil
location and orientation*, PLOS Comput Biol 22(2):e1013994,
[doi:10.1371/journal.pcbi.1013994](https://doi.org/10.1371/journal.pcbi.1013994)
**Source experiment** Schultheiss et al. (2025), Imaging Neuroscience,
[doi:10.1162/IMAG.a.1056](https://doi.org/10.1162/IMAG.a.1056)
([PMC12720167](https://pmc.ncbi.nlm.nih.gov/articles/PMC12720167/))

**n participants: 8.** The release holds one file per participant —
`result_002`, `003`, `004`, `006`, `007`, `008`, `009`, `010`. The source study
mapped eight of ten recruited; two were excluded.

**Pose: per pulse, three parameters.** `data/result_<subj>.npy` is a pickled
dict. `run_exp.py:53-64` reads `res['grid']` and `res['meps']`, treats
`grid[:,0]` as the coil rotation angle in degrees about 90 and `grid[:,1:]` as
the coil-centre offset inside a 30 mm search radius, and pairs each row with the
matching entry of `res['meps']`. PMC12720167 states that coil position and
orientation were prospectively selected and recorded for each pulse under
neuronavigated robotic-arm guidance, at roughly 300 configuration/MEP pairs per
participant.

**Response: measured MEP.** Peak-to-peak amplitude from surface EMG over the
right first dorsal interosseous, extracted 18–35 ms post-stimulus
(PMC12720167).

**Licence — verbatim, from `README.md` §2 of the repository:**

> The content of the repository is licensed under a Creative Commons
> Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0), which
> permits non-commercial use, sharing, adaptation, distribution and reproduction
> in any medium or format, as long as you give appropriate credit to the
> original author(s) and the source, provide a link to the Creative Commons
> license, and indicate if changes were made.

https://raw.githubusercontent.com/david-schu/gp-tms-hsh/main/README.md ·
licence text at https://creativecommons.org/licenses/by-nc/4.0/legalcode

**Download size: 226,932,868 B = 226.9 MB** across the eight `.npy` files
(27,586,540 / 30,286,540 / 25,734,288 / 30,262,540 / 27,387,340 / 28,949,740 /
26,307,340 / 30,418,540), plus about 50 kB of scripts. Sizes read from
`api.github.com/repos/david-schu/gp-tms-hsh/git/trees/HEAD?recursive=1`.

**Access: anonymous `git clone`. No agreement, no registration.**

Three caveats, each after what it qualifies.

- The licence is asserted in prose in `README.md`; there is no `LICENSE` file,
  and GitHub's own licence detection returns `null` for the repository. Using it
  means recording the README sentence as the licence text and vendoring
  CC BY-NC-4.0 into this tree.
- **NC.** `reports/licence_audit.md` is the record of how expensive an
  unnoticed non-commercial term is here. CC BY-NC-4.0 would be a second
  established NC source, and unlike `hansen_receptors` it would be read rather
  than merely carried. Whether a validation-only use avoids that is a decision,
  not a fact, and it belongs in the source card before any fetch.
- **No MRI, so no individual E-field.** The Imaging Neuroscience data statement
  is *"The raw datasets supporting this study are available from the
  corresponding author upon reasonable request"*, and the head models are part
  of the raw data. The sibling repository
  `github.com/david-schu/efield-informed-motor-mapping` publishes the SimNIBS
  pipeline that built them — including `create_instrument_markers.m`,
  `create_instrument_marker_mni.m` and `update_matsimnibs.m` — with two coil
  models (`MagVenture_Cool-B65`, `Magstim_70mm_Fig8`) and no participant
  anatomy. Testing the map on this release therefore means computing the field
  on a template head, which weakens the individual E-field and leaves the
  monotone field→response relation intact and testable.

---

## 2. OSF `myrqn` / `subject_0` — the complete triple, for one participant, unlicensed

**URL** https://osf.io/myrqn/ ·
[doi:10.17605/OSF.IO/MYRQN](https://doi.org/10.17605/OSF.IO/MYRQN)
**Paper** Weise, Numssen, Kalloch et al. (2023), *Precise motor mapping with
transcranial magnetic stimulation*, Nat Protoc 18:293–318,
[doi:10.1038/s41596-022-00776-6](https://doi.org/10.1038/s41596-022-00776-6)
**Scripts** https://gitlab.gwdg.de/tms-localization/papers/tmsloc_proto

**n participants: 1.** The OSF node's `owncloud` provider holds
`example_data/{coils,EMG,MRI,ROI,subject_0}`; `subject_0` is the only
participant.

**Pose: per pulse, 6-DOF, Localite TMS Navigator.**
`example_data/subject_0/exp/m1/tms_navigator/TriggerMarkers.xml`, 884,386 B.
`tmsloc_proto/README.md` describes script 05: *"This scripts combines coil
positions/orientations and EMG responses and is explicitly written to be used
with Localite TMS Navigator and CED Signal. EMG data is processed to calculate
one motor evoked potential (MEP) per TMS pulse."*

**Response: measured MEP, one per pulse.**
`example_data/subject_0/exp/m1/mep/` holds `mep.cfs` (125,114,396 B, raw CED
Signal), `mep.txt` (866,843,029 B) and `mep.mat`.

**This is the only release found that also ships the anatomy.**
`subject_0/mri`, `subject_0/mesh`, and a SimNIBS-4 charm mesh
(`exp/m1/mesh0_refinedM1_charm`) are present, so pose → individual E-field →
MEP is computable end to end without contacting anybody. It is one participant.

**Licence: none.** `GET https://api.osf.io/v2/nodes/myrqn/` returns
`"node_license": null`, and the node's `license` relationship is `null`. The
node is `"public": true`, category `data`, created 2021-11-22. The *scripts*
repository carries a licence — `LICENSE` in `tmsloc_proto` is the full
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public
License** text
(`https://gitlab.gwdg.de/api/v4/projects/tms-localization%2Fpapers%2Ftmsloc_proto/repository/files/LICENSE/raw?ref=master`)
— and that file governs the scripts repository, not the OSF data node. Treating
it as the data's licence would be exactly the substitution
`reports/licence_audit.md` finding 2 warns about.

**Download size: 993,307,516 B ≈ 0.99 GB** for `TriggerMarkers.xml` + `mep.cfs`
+ `mep.txt` + `subject_0.hdf5`, measured by HTTP `HEAD` against
`files.osf.io/v1/resources/myrqn/providers/owncloud/...`. The `mri`, `mesh`,
`opt` and `results` subtrees and the top-level `coils`/`MRI`/`ROI` folders are
additional and unmeasured — the owncloud provider reports `size: null` in the
listing API, so each file needs its own `HEAD`.

**Access: anonymous HTTP. No agreement.**

**What would make this the right choice:** a written licence grant from the
authors, or an OSF licence set on the node. It is one email, and it converts the
only complete public `(MRI, per-pulse pose, per-pulse MEP)` release from
unusable to usable. Worth sending regardless of which dataset is fetched first.

---

## 3. `pcd_e_mep.csv` — 6,488 per-pulse `(|E|, MEP)` pairs, 9 participants, unlicensed

**URL** https://gitlab.gwdg.de/tms-localization/papers/pcd, file
`data/pcd_e_mep.csv`
**Origin of the measurements** Numssen et al. (2021), *Efficient high-resolution
TMS mapping of the human motor cortex by nonlinear regression*, NeuroImage 245:118654,
[doi:10.1016/j.neuroimage.2021.118654](https://doi.org/10.1016/j.neuroimage.2021.118654)

This is the field-to-response map already extracted, one row per pulse.
`data/pcd_e_mep.md` states the columns: `sub`, `pcd` (pulsewise coil
displacement from the optimal placement), `e` (cortical |E| at the FDI muscle
representation), `mep` (motor response from FDI). Counted from the file:
**6,488 rows over 9 subjects** — Sub01 696, Sub02 820, Sub03 521, Sub04 815,
Sub05 815, Sub06 703, Sub07 738, Sub08 675, Sub09 705. Values are normalised.

**Pose: per pulse, collapsed to one scalar.** `pcd` is a displacement magnitude
relative to the optimal coil placement, not a pose. The full poses are not in
this repository.

**Licence: none.** `GET
https://gitlab.gwdg.de/api/v4/projects/tms-localization%2Fpapers%2Fpcd` returns
`"license": null` and `license_url: null`; the repository tree is
`data, import_from_neuronavigation, pics, scripts, .gitignore, README.md` with
no `LICENSE` blob.

**Size:** a few hundred kB. **Access:** anonymous clone, no agreement.

Worth knowing even though it cannot be used: it is the smallest object in
existence that directly exercises the claim under test — 6,488 measured
`(field magnitude at a cortical target, response amplitude)` pairs from
individually modelled E-fields.

---

## 4. OSF `9f3bc` — the 24-participant cohort, without the poses, unlicensed

**URL** https://osf.io/9f3bc/
**Paper** Jing, Numssen, Hartwigsen, Knösche, Weise, *Effects of electric field
direction on TMS-based motor cortex mapping*, Imaging Neuroscience 4,
[doi:10.1162/IMAG.a.1211](https://doi.org/10.1162/IMAG.a.1211)
([PMC13100673](https://pmc.ncbi.nlm.nih.gov/articles/PMC13100673/))

The paper's cohorts are the strongest described anywhere in this search. Dataset
1: 14 participants, 900–1100 single biphasic pulses each at 150% rMT, coil
position and orientation drawn at random within a 3 cm radius and ±60°, recorded
in real time per pulse by Localite with a Polaris Spectra camera, MEP from right
FDI at 18–35 ms. Dataset 2: 10 participants, 500 random pulses each, same rig.
Dataset 1 is Numssen et al. (2021), whose own methods section says **thirteen**
participants — the discrepancy with the paper's fourteen is unresolved here.

**The release does not contain the poses.** The node's `osfstorage` has exactly
two top-level folders: `validation_MEP` (per-subject raw CED `.cfs`, ~2.05 MB
each, 6–9 files for each of sub-01…sub-05 and a single `mep.hdf5` for sub-06…
sub-10) and `regression_data` (`r2_roi_data_sub01…14.hdf5` and
`r2_roi_geo_sub01…14.hdf5`, 89 kB–609 kB each). Those are fitted goodness-of-fit
maps and validation-block MEPs, not `(pose, MEP)` pairs.

**Licence: none.** `node_license: null`, `license` relationship `null`.
**Size:** roughly 150 MB. **Access:** anonymous, no agreement.

---

## 5. Rejected, and why

Recorded so the negative result is auditable rather than asserted.

**Public, licensed, no per-pulse pose.** These are good MEP releases whose coil
sat on one hotspot, so they carry no spatial variation to test a map with.

| release | licence | size | why rejected |
|---|---|---|---|
| Zenodo [10.5281/zenodo.18339892](https://doi.org/10.5281/zenodo.18339892) — Sarkar et al., NINDS | CC-BY-4.0 | 356.2 MB | 84 participants, 60 pulses each over 5–100% MSO. One hotspot per participant, coil at a fixed 45°. Intensity varies; pose does not. |
| Zenodo [10.5281/zenodo.6456830](https://doi.org/10.5281/zenodo.6456830) — ipsilateral MEP | CC-BY-4.0 | 34.0 MB | 4,244 EMG trials, Localite used for positioning, no per-pulse coordinates in the three released files. |
| Zenodo [10.5281/zenodo.15225065](https://doi.org/10.5281/zenodo.15225065) | CC-BY-4.0 | 550.8 MB | Trial-by-trial MEP with stimulation parameters; parameters are timing and intensity, not pose. |
| Zenodo [10.5281/zenodo.4010860](https://doi.org/10.5281/zenodo.4010860) | CC-BY-4.0 | 136.0 MB | Recruitment curves at one site. |
| Zenodo [10.5281/zenodo.15054450](https://doi.org/10.5281/zenodo.15054450) | CC-BY-NC-4.0 | 31.6 MB | "Raw MEP data used in the paper", poststroke; no pose. |
| [github.com/ShayOfir/MEP_IOcurve](https://github.com/ShayOfir/MEP_IOcurve) | BSD-3-Clause | 180 kB | 1,917 rows; columns are `coil` (h7 / fig-8 / T360°), `machine_setting`, `percent_threshold`, `side` — coil *type*, not coil pose. |

**WAND** — [doi:10.12751/g-node.5mv3bf](https://doi.gin.g-node.org/10.12751/g-node.5mv3bf/),
Creative Commons Attribution 4.0 International Public License, 1.8 TiB, GIN,
170 volunteers. The TMS session (`ses-08`, 40 participants) measured motor
threshold and short/long-interval intracortical inhibition with a Magstim
BiStim², coil held on a tripod and positioned with BrainSight — one hotspot, no
per-pulse pose in the release. The GIN README notes physiological data are
currently unavailable, and GIN requires registration and `gin login`. Licensed
and large and not the thing.

**OpenNeuro** — swept via the GraphQL API (`advancedSearch` on keywords TMS,
transcranial magnetic stimulation, MEP, motor evoked potential, electromyography,
neuronavigation; 60 hits each, names resolved individually). No dataset pairs a
per-pulse coil pose with a measured response. The TMS datasets that exist are
`ds002094` (Single-pulse open-loop TMS-EEG, 20 subjects, CC0), `ds005779`
(Real-time personalized brain state-dependent TMS, 19 subjects, CC0),
`ds005498` (Single-pulse TMS fMRI, 152 subjects, CC0), `ds008037` (123 subjects,
CC0), and `ds004024` itself, all with the coil parked on one target.

**PhysioNet** — 713 published projects enumerated through
`api/v1/project/published`. Zero TMS. The only string match was a transcranial
*Doppler* study.

**figshare and Dryad** — searched through their APIs for TMS motor mapping, MEP,
and coil position. Journal supplementary PDFs and unrelated datasets only.

**Deposited nowhere; "on request".** Each of these ran the right experiment and
released no data.

- Schultheiss et al. 2025, Imaging Neuroscience (8 participants, ~300
  configurations each, neuronavigated robotic arm): *"The raw datasets
  supporting this study are available from the corresponding author upon
  reasonable request."* The derived per-pulse pairs are #1 above.
- [PMC9263445](https://pmc.ncbi.nlm.nih.gov/articles/PMC9263445/) — 20
  participants, 360 stimulations per session over two sessions, eight muscles,
  Neural Navigator storing "the position and orientation of the coil with
  respect to the head". *"The raw data supporting the conclusions of this
  article will be made available by the authors, without undue reservation."*
  This is the largest per-pulse pose + multi-muscle MEP corpus located, and it
  is not deposited.
- [PMC10733534](https://pmc.ncbi.nlm.nih.gov/articles/PMC10733534/) — 12
  participants, 42-point grid, 6 pulses per point, orientation fixed. *"not
  publicly available due to the privacy of volunteers."*
- [PMC8452135](https://pmc.ncbi.nlm.nih.gov/articles/PMC8452135/) — Tervo et
  al., GP active-learning mapping, 5 subjects, 294 pulses; no data availability
  statement, orientation fixed at 45°.

**NIBS-BIDS** — [doi:10.5281/zenodo.19337642](https://doi.org/10.5281/zenodo.19337642),
BEP037, a BIDS extension with a `nibs/` datatype carrying stimulation
parameters and spatial targeting. The preprint says "Example datasets will be
made available"; none exist yet. Worth tracking, because it is the mechanism by
which a per-pulse pose would become a standard field rather than a vendor XML.

---

## 6. Recommendation

**Fetch `github.com/david-schu/gp-tms-hsh` first**, and nothing else yet.

It is the only release located that satisfies both halves of the requirement at
once: a coil configuration recorded per pulse, a measured MEP recorded per
pulse, and a licence stated in the artifact. Eight participants and roughly 300
pairs each is enough spatial spread to fit and falsify a monotone
field→response function, which is what the falsifier at
`site/content/possibilities/index.html:145-153` actually asks for. It is 226.9
MB, clones anonymously, needs no agreement, and the two arrays that matter are
named in twelve lines of `run_exp.py`.

Two things must happen before the fetch, not after.

1. **Vendor the CC BY-NC-4.0 text into this tree and decide the NC question in
   the source card.** The licence lives in a README sentence and GitHub does not
   detect it; recording a pointer would repeat the `diedrichsen2009` failure in
   `reports/licence_audit.md`. Whether an NC input may touch a checkpoint, or
   only a validation report, is an R12-shaped decision and is not settled by
   this report.
2. **State in the card that the head models are absent**, so the E-field must be
   computed on a template and the resulting test is of the field→response
   relation rather than of individual field accuracy.

**In parallel, ask the Nature Protocols authors to put a licence on OSF
`myrqn`.** `subject_0` is the only public object anywhere that contains an MRI,
a 6-DOF Localite pose per pulse, and an MEP per pulse — the complete chain, for
one participant, blocked by a null field in an API response. That is the
cheapest possible unblock in this whole report.

**Do not treat #3 or #4 as available.** They are unlicensed, and #3 in
particular is tempting precisely because it is small and already in the form the
test wants.
