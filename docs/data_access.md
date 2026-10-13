# Data access and redistribution

This study used **fastMRI brain multicoil** acquisitions: an AXT2 development/evaluation cohort and a patient-disjoint AXFLAIR cohort for a within-fastMRI sequence-shift analysis.

The official fastMRI access portal is <https://fastmri.med.nyu.edu/>. Request or obtain the data through the provider and comply with the applicable terms. Data access is **not** granted by this software repository.

## What is not redistributed

- Original fastMRI HDF5 k-space files and image reconstructions.
- MRI slice images, reference/error arrays, masks or derived arrays that reproduce individual examinations.
- Patient-, volume-, or slice-level records or manifest tables containing acquisition identifiers.
- `.npz` reliability caches, sampled-pixel evaluation tables, and per-subject inference records.
- Model checkpoints and private execution environment credentials.

Publication of a figure containing MRI anatomy must be assessed against the fastMRI terms separately. Inclusion of a figure in a manuscript does **not** automatically authorize hosting it in a public repository.

## Local working directories

The public configuration templates use paths relative to the repository root:

```text
local_data/       # restricted, separately obtained input data and private manifests
local_outputs/    # generated checkpoints, caches and analyses
```

Both directories are ignored by Git. **Do not remove their ignore rules or force-add their contents.** Source data should be stored locally under a researcher-selected subdirectory; command examples use `local_data/fastmri_brain_multicoil/` as a placeholder.

The `data/` and `results/` paths, where they still occur in historical scripts/configurations, are legacy local destinations and must also remain private. Review `git status` before each commit.

## Public material

This repository provides software, configuration templates, tests, high-level documentation, and reviewed **aggregate** performance tables. Aggregate reporting does not imply that every source file or imaging-derived figure is redistributable. No user-level data are necessary to read the published summary tables.
