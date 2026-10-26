# Reproduction guide — Paper 1 R1

This guide describes the supplied entry points and frozen experiment configuration **without redistributing fastMRI data or private intermediate arrays**. The public code and unit tests have been checked for source-level consistency.

All paths in the commands below are relative to the **repository root**. The template configurations preserve frozen settings such as the seeds, ACS allocation, network dimensions, and thresholds.

## 1. Software environment

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

A CUDA-capable environment is strongly recommended for model training. Unit tests can use synthetic data and do not validate the frozen numerical endpoints.

## 2. Prepare restricted data locally

Obtain the fastMRI brain multicoil data via the [official fastMRI portal](https://fastmri.med.nyu.edu/), subject to its data-use conditions. Put the authorized `.h5` files in a directory of your choice; the example below uses `local_data/fastmri_brain_multicoil/`.

The original AXT2 selection retained 281 volumes and used 201 training, 40 calibration, and 40 test volumes. The AXFLAIR sequence-shift cohort used 26 patient-disjoint volumes
## 3. Generate local audit and mask artifacts

The retained manifest and ACS-protected mask indices are required by subsequent stages. The public `exp001` and `exp002` configurations use private `local_data/` and `local_outputs/` paths.

```bash
python scripts/exp001_data_audit_manifest.py   --config configs/exp001_data_audit_manifest.yaml   --data-root local_data/fastmri_brain_multicoil

python scripts/exp002_generate_fourway_masks.py   --config configs/exp002_fourway_masks.yaml
```
## 4. Reconstruction and risk-predictor development


```bash
python scripts/exp004_train_ssdu_v4_full_theta_norm_fix.py   --config configs/exp004_theta_norm_fix_full.yaml   --data-root local_data/fastmri_brain_multicoil

python scripts/exp006_build_reliability_cache_theta_norm_fix.py   --config configs/exp006_reliability_cache_traincal_theta_norm_fix.yaml   --data-root local_data/fastmri_brain_multicoil

python scripts/exp007_train_reliability_cnn_v2_traincal.py   --config configs/exp007_full_primary_traincal_corrected.yaml

python scripts/exp008_train_reliability_ablation_traincal.py   --config configs/exp008_a4_secondary_traincal_corrected.yaml   --variant A4_image_only
```

EXP004 uses the revised Θ-only normalization path. EXP007 is the primary Full-CNN; EXP008 `A4_image_only` is a **secondary** ablation with three zeroed channels in a six-channel network interface.
## 5. Reserved verification and inference

After the appropriate private calibration/test cache artifacts exist, the corrected verification entry point is:

```bash
python scripts/exp009_holdout_verification_corrected.py   --config configs/exp009_holdout_verification_corrected_final.yaml   --data-root local_data/fastmri_brain_multicoil   --cache-manifest local_data/evaluation/calibration_test_cache_manifest.csv
```

**Important limitation:** The public train/calibration cache builder in Section 4 does **not** itself generate the referenced calibration/test cache manifest. This prerequisite must be generated under the frozen evaluation protocol. The protected Λ_hold samples are operationally reserved, but are not statistically independent of the other portions of the same acquisition.

The volume-cluster bootstrap entry point is:

```bash
python scripts/exp010_bootstrap_metrics_corrected.py   --config configs/exp010_bootstrap_metrics_corrected.yaml
```

It requires `local_data/evaluation/sampled_test_pixels_corrected.csv.gz`, a **private, per-pixel** input; that file is neither supplied nor suitable for GitHub. The configuration retains the frozen input SHA-256 and calibration-derived high-risk threshold. The aggregate output tables are already supplied under `paper/supporting_materials/`.

## 6. Predictive-variance comparators

The R2.6 scripts provide `r2_6_train_uq_comparator.py`, `r2_6_infer_uq_test.py`, and `src/fourway_mri/r2_6_uq_model.py`. Their frozen MC-dropout and ensemble settings are in `configs/r2_6_mc_dropout.yaml` and `configs/r2_6_ensemble.yaml`.

The training entry point needs `--method`, `--config`, `--resolved-manifest`, and `--output-dir`, with `--member-seed` for individual ensemble members. Test inference additionally requires the frozen checkpoints, resolved test manifest, sampled-pixel artifact, and output destinations. These data-dependent inputs are **not** distributed. Consult each script's `--help` and the config files.

R2.9 full-pipeline alternative-partition seed retraining was **not completed** and should not be reported as reproduced.

## 7. Interpretation and available outputs

Refer to `paper/supporting_materials/` for frozen aggregate AXT2 metrics, bootstrap summaries, volume-level calibration, AXFLAIR results, verification-budget sensitivity, target-construction sensitivity, auxiliary reference-error correspondences.

All performance endpoints compare predictions with **measurement-derived residual-risk targets**, not clinical ground truth. AXFLAIR is a within-fastMRI sequence-shift stress test, not external-site validation. Fully sampled reference error is examined only in the auxiliary analysis.

## 8. Reproducibility scope

The historical `scripts/exp010_compute_conformal_bounds.py`
uses pooled pixel-level nonconformity scores. 

Aggregate results are provided for target-construction
sensitivity, verification-budget sensitivity, AXFLAIR
sequence-shift evaluation, fully sampled reference-error
correspondence.
Reproduction also requires
restricted data and intermediate artifacts.


