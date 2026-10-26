# Data access and redistribution

This study used **fastMRI brain multicoil** acquisitions: an AXT2 development/evaluation cohort and a patient-disjoint AXFLAIR cohort for a within-fastMRI sequence-shift analysis.

The official fastMRI access portal is <https://fastmri.med.nyu.edu/>. Request or obtain the data through the provider and comply with the applicable terms. Data access is **not** granted by this software repository.





## Local working directories

The public configuration templates use paths relative to the repository root:

```text
local_data/       # restricted, separately obtained input data and private manifests
local_outputs/    # generated checkpoints, caches and analyses
```



The `data/` and `results/` paths, where they still occur in historical scripts/configurations, are legacy local destinations and must also remain private. Review `git status` before each commit.


