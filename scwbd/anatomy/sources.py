"""Upstream source registry: URL, license, version, citation, known bias.

Every anatomical object this package produces traces back to one of these
entries.  ``license`` is load-bearing: ``CC-BY-NC-SA-4.0`` on the Hansen
receptor atlas means the receptor-derived priors may not be shipped in a
commercial artifact, and the manifest records that.

``bias`` is a prose statement of what the source *cannot* support.  It is
copied verbatim into ``reports/anatomy_prior.md`` and into the uncertainty
ledgers, so it should be written to be read by someone deciding whether to
trust a number.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SRC", "license_of", "citations_for"]

SRC: dict[str, dict[str, Any]] = {
    # -- parcellations ---------------------------------------------------
    "schaefer2018": {
        "name": "Schaefer 2018 local-global parcellation",
        "url": "https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal",
        "license": "MIT (CBIG); underlying GSP data under its own terms",
        "version": "2018 (CBIG release)",
        "citation": "Schaefer A. et al. (2018) Cereb Cortex 28:3095-3114.",
        "bias": (
            "Derived from resting-state functional connectivity in 1489 young "
            "healthy adults; parcel boundaries are a group-level functional "
            "compromise and do not mark cytoarchitectonic transitions. Parcel "
            "count is a modelling choice, not a discovered number."
        ),
    },
    "desikan2006": {
        "name": "Desikan-Killiany gyral atlas",
        "url": "https://surfer.nmr.mgh.harvard.edu/",
        "license": "FreeSurfer license (free for research use)",
        "version": "FreeSurfer aparc",
        "citation": "Desikan R.S. et al. (2006) NeuroImage 31:968-980.",
        "bias": (
            "Gyral/sulcal landmarks only. Large parcels average heterogeneous "
            "function; the atlas encodes macroanatomy, not areal identity."
        ),
    },
    "destrieux2010": {
        "name": "Destrieux sulco-gyral atlas (a2009s)",
        "url": "https://surfer.nmr.mgh.harvard.edu/",
        "license": "FreeSurfer license (free for research use)",
        "version": "FreeSurfer aparc.a2009s",
        "citation": "Destrieux C. et al. (2010) NeuroImage 53:1-15.",
        "bias": "Anatomical boundary definition; sulcal variability is high across individuals.",
    },
    "glasser2016": {
        "name": "HCP-MMP1.0 multimodal parcellation",
        "url": "https://balsa.wustl.edu/study/RVVG",
        "license": "HCP open-access data-use terms; redistribution of derived labels permitted with citation",
        "version": "1.0",
        "citation": "Glasser M.F. et al. (2016) Nature 536:171-178.",
        "bias": (
            "Semi-automated areal classifier trained on a 210-subject group "
            "average; 360 areas is the authors' confident subset, not a complete "
            "areal inventory. Individual areal boundaries deviate from the group."
        ),
    },
    "voneconomo": {
        "name": "von Economo-Koskinas cytoarchitectonic classes",
        "url": "https://github.com/netneurolab/netneurotools",
        "license": "Digitisation released with netneurotools (BSD-3)",
        "version": "Scholtens/Vertes digitisation",
        "citation": "von Economo C., Koskinas G.N. (1925); digitised by Scholtens L.H. et al. (2018) NeuroImage 170:412-423.",
        "bias": (
            "Historical single-atlas cytoarchitecture projected onto a modern "
            "surface template; class boundaries carry both original and "
            "registration error."
        ),
    },
    "tian2020": {
        "name": "Tian subcortical parcellation S1-S4",
        "url": "https://github.com/yetianmed/subcortex",
        "license": "See repository LICENSE (open, academic use)",
        "version": "3T Group-Parcellation",
        "citation": "Tian Y. et al. (2020) Nat Neurosci 23:1421-1432.",
        "bias": (
            "Functional-gradient subdivision from 3T rs-fMRI in 1080 HCP adults. "
            "Subdivisions are not cytoarchitectonic nuclei and have no tractography "
            "coverage in the connectomes shipped here."
        ),
    },
    "harvardoxford": {
        "name": "Harvard-Oxford subcortical structural atlas",
        "url": "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases",
        "license": "FSL license (free for non-commercial research)",
        "version": "maxprob-thr25",
        "citation": "Makris N. et al. (2006) Schizophr Res 83:155-171; FSL/FMRIB.",
        "bias": (
            "Maximum-probability labelling from 21-37 manually segmented brains; "
            "thresholding at 25% discards boundary uncertainty that the "
            "probabilistic version retains."
        ),
    },
    "buckner2011": {
        "name": "Buckner cerebellar functional atlas (7/17 networks)",
        "url": "https://github.com/DiedrichsenLab/cerebellar_atlases",
        "license": "See repository (open, academic use, citation required)",
        "version": "cerebellar_atlases release",
        "citation": "Buckner R.L. et al. (2011) J Neurophysiol 106:2322-2345.",
        "bias": (
            "Cerebellar voxels labelled by which cerebral network they correlate "
            "with. Functional correlation is not anatomical connectivity: "
            "cortico-cerebellar traffic is polysynaptic via pons and thalamus."
        ),
    },
    "diedrichsen2009": {
        "name": "SUIT probabilistic cerebellar lobular atlas",
        "url": "https://github.com/DiedrichsenLab/cerebellar_atlases",
        "license": "See repository (open, academic use, citation required)",
        "version": "Diedrichsen 2009 Anatom",
        "citation": "Diedrichsen J. et al. (2009) NeuroImage 46:39-46.",
        "bias": "Lobular anatomy from 20 brains; maximum-probability labels discard boundary probability.",
    },
    # -- surfaces and toolbox --------------------------------------------
    "enigmatoolbox": {
        "name": "ENIGMA Toolbox",
        "url": "https://github.com/MICA-MNI/ENIGMA",
        "license": "BSD-3-Clause",
        "version": "2.0.3",
        "citation": "Lariviere S. et al. (2021) Nat Methods 18:698-700.",
        "bias": "Redistribution layer; inherits the biases of each bundled source.",
    },
    "conte69": {
        "name": "Conte69 / fsLR-32k surface template",
        "url": "https://balsa.wustl.edu/",
        "license": "HCP open-access terms",
        "version": "fsLR 32k",
        "citation": "Van Essen D.C. et al. (2012) Cereb Cortex 22:2241-2262.",
        "bias": (
            "A 69-subject group-average midthickness surface. Coordinates are "
            "aligned to MNI152 but not identical to it; per-vertex mismatch of a "
            "few millimetres is expected and is not modelled here."
        ),
    },
    "fsaverage": {
        "name": "FreeSurfer fsaverage surface template",
        "url": "https://surfer.nmr.mgh.harvard.edu/",
        "license": "FreeSurfer license",
        "version": "fsaverage / fsaverage5 / fsaverage6",
        "citation": "Fischl B. et al. (1999) Hum Brain Mapp 8:272-284.",
        "bias": "40-subject average surface; downsampled densities lose sulcal detail.",
    },
    # -- connectomes ------------------------------------------------------
    "enigma_hcp_sc": {
        "name": "ENIGMA/HCP group-average structural connectome",
        "url": "https://enigma-toolbox.readthedocs.io/en/latest/pages/05.HCP/index.html",
        "license": "BSD-3-Clause code; HCP open-access data-use terms for the underlying scans",
        "version": "ENIGMA Toolbox 2.0.3 bundled matrices",
        "citation": (
            "Lariviere S. et al. (2021) Nat Methods 18:698-700; "
            "Van Essen D.C. et al. (2013) NeuroImage 80:62-79 (HCP)."
        ),
        "cohort": (
            "207 unrelated healthy young adults from the HCP S1200 release "
            "(age 22-37). NOTE: the count comes from the ENIGMA Toolbox preprint "
            "(doi:10.1101/2020.12.21.423838), not from the bundled data itself, "
            "which ships no per-subject metadata. Sex, handedness and ancestry "
            "composition are not recoverable from what is redistributed here."
        ),
        "pipeline": (
            "MRtrix3 anatomically-constrained tractography; multi-shell multi-tissue "
            "CSD; 40M seed streamlines; max tract length 250 mm; FA cutoff 0.06; "
            "SIFT2 cross-section weighting; distance-dependent group consistency "
            "thresholding; log transform of streamline density."
        ),
        "bias": (
            "Diffusion tractography is undirected and error-prone. It produces "
            "false positives that scale with tract length and with the number of "
            "fibre crossings, and false negatives for long, thin, or highly curved "
            "bundles (notably lateral temporal and inferior frontal projections). "
            "Group averaging removes exactly the individual variation the model "
            "later claims to individualise. The log transform means weights are "
            "on an arbitrary monotone scale, not in physical units."
        ),
    },
    "hansen_schaefer_sc": {
        "name": "Hansen 2022 group structural/functional connectome (Schaefer-100)",
        "url": "https://github.com/netneurolab/hansen_receptors",
        "license": "CC-BY-NC-SA-4.0",
        "version": "hansen_receptors repository data/schaefer",
        "citation": "Hansen J.Y. et al. (2022) Nat Neurosci 25:1569-1581.",
        "cohort": "HCP subset processed independently of the ENIGMA pipeline",
        "bias": (
            "Independent processing of overlapping HCP scans: an agreement between "
            "this and the ENIGMA matrix is evidence about algorithm stability, and "
            "only weak evidence about biology, because the underlying subjects "
            "partly coincide."
        ),
    },
    "hansen_lausanne_sc": {
        "name": "Hansen 2022 group structural connectome (Lausanne/DK-68)",
        "url": "https://github.com/netneurolab/hansen_receptors",
        "license": "CC-BY-NC-SA-4.0",
        "version": "hansen_receptors repository data/lausanne",
        "citation": "Hansen J.Y. et al. (2022) Nat Neurosci 25:1569-1581.",
        "bias": "As above; Lausanne scale-033 is the Desikan-Killiany partition.",
    },
    "netneuro_lausanne_sc": {
        "name": "netneurolab consensus structural connectomes (Cammoun/Lausanne scales)",
        "url": "https://github.com/netneurolab/netneurotools",
        "license": "BSD-3-Clause (code); data as released with the cited papers",
        "version": "netneurotools 0.3.0 fetch_famous_gmat",
        "citation": "Griffa A. et al. (2019) Zenodo; Betzel R.F., Bassett D.S. (2018) PNAS 115:E4880.",
        "cohort": "70 healthy adults, Lausanne cohort, deterministic tractography",
        "bias": (
            "A genuinely different cohort, scanner and tractography algorithm from "
            "HCP; agreement with the HCP matrices is therefore stronger evidence "
            "than agreement between two HCP-derived matrices. Deterministic "
            "streamline tracking under-detects crossing-fibre pathways."
        ),
    },
    "markov2014": {
        "name": "Markov et al. macaque retrograde tracer connectivity (M132)",
        "url": "https://core-nets.org/ (redistributed via netneurotools fetch_famous_gmat)",
        "license": "As released with the cited papers; redistributed by netneurolab",
        "version": "netneurotools macaque_markov",
        "citation": (
            "Markov N.T. et al. (2014) Cereb Cortex 24:17-36; "
            "Ercsey-Ravasz M. et al. (2013) Neuron 80:184-197."
        ),
        "bias": (
            "CROSS-SPECIES. Quantitative, directed, laminar tracer data in macaque. "
            "Nothing here licenses a claim about a specific human pathway: areal "
            "homology between macaque M132 and human parcellations is contested "
            "and incomplete. Used in this package only for the species-general "
            "exponential distance rule on edge existence, never to assert a human "
            "edge."
        ),
    },
    # -- maps -------------------------------------------------------------
    "hansen_receptors": {
        "name": "Hansen neurotransmitter receptor/transporter PET atlas",
        "url": "https://github.com/netneurolab/hansen_receptors",
        "license": "CC-BY-NC-SA-4.0",
        "version": "repository HEAD (see manifest)",
        "citation": (
            "Hansen J.Y. et al. (2022) Nat Neurosci 25:1569-1581, plus the primary "
            "PET study for each tracer (see Table S3 of that paper)."
        ),
        "bias": (
            "GROUP AVERAGES over small, separate donor cohorts (n between 3 and "
            "204 per tracer), each scanned on a different scanner with a different "
            "tracer, kinetic model and reference region. Values are on "
            "tracer-specific scales that are not mutually commensurate without "
            "z-scoring, and z-scoring destroys absolute density. Appendix A of the "
            "thesis is explicit: a subject's receptor density is NOT inferable "
            "from an atlas value. Non-commercial license."
        ),
    },
    "neuromaps": {
        "name": "neuromaps annotation collection",
        "url": "https://netneurolab.github.io/neuromaps/",
        "license": "BSD-3-Clause (toolbox); per-annotation source terms",
        "version": "0.0.7",
        "citation": "Markello R.D. et al. (2022) Nat Methods 19:1472-1479.",
        "bias": (
            "Each annotation carries the bias of its own study. Spatial "
            "autocorrelation makes naive across-map correlation tests "
            "anticonservative; use spin tests."
        ),
    },
    "margulies2016": {
        "name": "Principal functional connectivity gradient",
        "url": "https://netneurolab.github.io/neuromaps/",
        "license": "As distributed via neuromaps",
        "version": "fsLR 32k",
        "citation": "Margulies D.S. et al. (2016) PNAS 113:12574-12579.",
        "bias": (
            "A diffusion-embedding component of a group resting-state FC matrix. "
            "It is a low-dimensional summary of correlation structure, not a "
            "measured anatomical hierarchy."
        ),
    },
    "hcps1200_maps": {
        "name": "HCP S1200 group myelin (T1w/T2w), thickness and MEG maps",
        "url": "https://netneurolab.github.io/neuromaps/",
        "license": "HCP open-access data-use terms",
        "version": "S1200 group average via neuromaps",
        "citation": (
            "Glasser M.F., Van Essen D.C. (2011) J Neurosci 31:11597-11616; "
            "Shafiei G. et al. (2022) PLoS Biol 20:e3001735 (MEG)."
        ),
        "bias": (
            "T1w/T2w ratio is a proxy for myelin content confounded by iron and "
            "water; MEG source maps carry the ill-posedness of the inverse problem "
            "and are smoothed over centimetres."
        ),
    },
    "sydnor2021": {
        "name": "Sensorimotor-association cortical axis",
        "url": "https://netneurolab.github.io/neuromaps/",
        "license": "As distributed via neuromaps",
        "version": "fsLR 32k",
        "citation": "Sydnor V.J. et al. (2021) Neuron 109:2820-2846.",
        "bias": "A rank-average of ten heterogeneous maps; a coordinate, not a mechanism.",
    },
    "hill2010": {
        "name": "Evolutionary and developmental cortical expansion",
        "url": "https://netneurolab.github.io/neuromaps/",
        "license": "As distributed via neuromaps",
        "version": "fsLR 164k",
        "citation": "Hill J. et al. (2010) PNAS 107:13135-13140.",
        "bias": "Surface-registration-based expansion between species/ages; homology assumptions are strong.",
    },
    "raichle_metabolism": {
        "name": "Cerebral blood flow / volume / oxygen and glucose metabolism",
        "url": "https://netneurolab.github.io/neuromaps/",
        "license": "As distributed via neuromaps",
        "version": "fsLR 164k",
        "citation": "Vaishnavi S.N. et al. (2010) PNAS 107:17757-17762.",
        "bias": "PET group averages from 33 adults; regional metabolism is state dependent.",
    },
    "goulas_autoradiography": {
        "name": "Zilles/Palomero-Gallagher receptor autoradiography",
        "url": "https://github.com/AlGoulas/receptor_principles",
        "license": "As released with the cited papers",
        "version": "redistributed in hansen_receptors/data/autoradiography",
        "citation": (
            "Zilles K., Palomero-Gallagher N. (2017) Front Neuroanat 11:78; "
            "Goulas A. et al. (2021) PNAS 118:e2020574118."
        ),
        "bias": (
            "Post-mortem quantitative autoradiography in a handful of brains over "
            "44 areas: far more chemically specific than PET, far sparser in "
            "coverage, and not registered to any in-vivo surface."
        ),
    },
    # -- unavailable / gated ---------------------------------------------
    "julich_brain": {
        "name": "Julich-Brain probabilistic cytoarchitectonic maps",
        "url": "https://julich-brain-atlas.de/",
        "license": "EBRAINS terms; account required for programmatic access",
        "version": "v3.x",
        "citation": "Amunts K. et al. (2020) Science 369:988-992.",
        "bias": "Post-mortem sample of ~10 brains per area; probabilities must be retained, not thresholded.",
    },
    "bigbrain_layers": {
        "name": "BigBrain cortical layer thickness (Wagstyl segmentation)",
        "url": "https://github.com/kwagstyl/cortical_layers_tutorial",
        "license": "CC-BY-4.0 (BigBrain derived data)",
        "version": "Wagstyl 2020 layer surfaces",
        "citation": (
            "Amunts K. et al. (2013) Science 340:1472-1475; "
            "Wagstyl K. et al. (2020) PLoS Biol 18:e3000678."
        ),
        "bias": (
            "ONE post-mortem brain (65-year-old female), sectioned and stained; "
            "fixation and sectioning deformation are uncorrected in places. Layer "
            "thickness here is a single-subject observation, never a population mean."
        ),
    },
}


def license_of(key: str) -> str:
    return SRC[key]["license"]


def citations_for(keys: list[str]) -> list[str]:
    return [SRC[k]["citation"] for k in keys if k in SRC]
