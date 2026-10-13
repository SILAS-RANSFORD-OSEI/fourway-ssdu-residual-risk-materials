# Measurement-derived residual-risk verification through four-way k-space partitioning in self-supervised accelerated brain MRI

Research software and selected supporting material for the revised manuscript submitted to *Magnetic Resonance Imaging*. This repository is a reproducibility resource, **not** a distribution of the fastMRI dataset, and its contents must not be interpreted as a clinically validated system.

## Research scope

The method assigns acquired k-space samples from each MRI acquisition to four **non-overlapping operational roles**:

1. **Θ** — reconstruction input;
2. **Λ_rec** — self-supervised reconstruction-loss evaluation;
3. **Λ_risk** — measurement-derived residual-risk learning target;
4. **Λ_hold** — reserved post-development verification target.

These roles are disjoint in sample use, **not statistically independent acquisitions**: they share anatomy, imaging physics, and noise/acquisition characteristics. The verification score measures consistency with reserved acquired measurements. It is **not** a direct estimate of image reconstruction error, clinical risk, or diagnostic uncertainty.

The primary predictor is **Full-CNN** (six active input channels; learned PSF/gain skip). **A4-CNN** is the secondary image-derived ablation (six-channel interface with the last three channels zeroed; no PSF/gain skip). The tested MC-dropout and ensemble predictive-variance measures are conceptually different comparators, not interchangeable estimates of residual risk.

## Reported R1 evaluation

| Setting or endpoint | Reported value |
| --- | --- |
| AXT2 training / calibration / test | 201 / 40 / 40 MRI volumes |
| AXT2 test evaluation | 636 slices; 999,792 sampled pixels |
| Full-CNN test AUPRC / AUROC | 0.8348 / 0.9818 |
| Full-CNN test MAE / Spearman | 0.0834 / 0.9389 |
| A4-CNN test AUPRC | 0.8452 |
| MRI-volume interval event coverage | 39/40 (0.975) for each predictor |
| Full-CNN AUPRC with 25% of reserved lines retained | 0.7890 |
| Full-CNN AXFLAIR AUPRC | 0.7854 |

AXFLAIR is a **patient-disjoint, within-fastMRI cross-sequence stress test**, not independent external or clinical validation. Fully sampled reference images were used only for an auxiliary post-development correspondence analysis. The comparison of residual risk with image error was positive but moderate. The requested full multi-seed retraining under alternative random role partitions was **not completed**; no results are claimed for it.

These are reported manuscript results. The published aggregate CSVs are under [`paper/supporting_materials/`](paper/supporting_materials/). Unit tests do not constitute end-to-end regeneration of these numbers.

## Repository layout

- [`configs/`](configs/) — configuration templates; data and artifact paths are local placeholders.
- [`src/fourway_mri/`](src/fourway_mri/) — reusable MRI operators, datasets, models, and uncertainty-comparator module.
- [`scripts/`](scripts/) — data-audit, partitioning, training, verification, statistical-analysis, and plotting entry points.
- [`tests/`](tests/) — unit tests and shape/integrity checks.
- [`paper/supporting_materials/`](paper/supporting_materials/) — approved aggregate statistical summaries.
- `paper/captions/` — archived development notes and earlier captions, not authoritative R1 manuscript assets. Formatted manuscript figures and tables are intentionally omitted.
- [`docs/`](docs/) — data access, reproduction steps, and expected outputs.

This is the dedicated public materials repository. Experimental development history is maintained separately in the [research repository](https://github.com/SILAS-RANSFORD-OSEI/fourway-ssdu-reliability-mri-v2). Neither repository substitutes for the fastMRI access procedure.

## Installation

From the repository root in a Python environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

See [`docs/data_access.md`](docs/data_access.md) to obtain the dataset lawfully and [`docs/reproduction.md`](docs/reproduction.md) for the experiment prerequisites, ordered commands, and current reproduction limitations.

## Redistribution and research integrity

Do **not** commit fastMRI `.h5` files, patient- or slice-level manifests, intermediate pixel arrays, `.npz` caches, generated MRI slices, private model checkpoints, or private system paths. Researchers must obtain source data independently, and any figure containing MRI anatomy needs a separate permissions review. The repository's `.gitignore` blocks common local inputs/outputs but is not a substitute for manual review.

## Citation and status

See [`CITATION.cff`](CITATION.cff). The journal manuscript is under editorial consideration; **no acceptance or final journal publication is claimed here**. See [`LICENSE`](LICENSE) for the software license; it does not grant rights to third-party fastMRI data.
