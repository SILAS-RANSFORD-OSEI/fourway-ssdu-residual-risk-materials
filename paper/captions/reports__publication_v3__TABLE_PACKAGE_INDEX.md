# Publication v3 Table Package Index

This file records the final main and supplementary table package for the publication v3 manuscript draft. Each table is linked to its generated files, build script, source basis, and audit status.

---

## Main Tables

### Table 1 — Independent \(\Lambda_{\mathrm{hold}}\) residual-risk verification performance

**Purpose:**  
Reports the primary independent holdout residual-risk verification metrics for learned CNN predictors, direct residual-based predictors, and deterministic heuristic baselines.

**Files:**

- `reports/publication_v3/tables/main/Table_1_independent_lambda_hold_performance.csv`
- `reports/publication_v3/tables/main/Table_1_independent_lambda_hold_performance.md`

**Build script:**

- `scripts/build_publication_v3_table1.py`

**Reported metrics:**

- Top-10% AUPRC
- Spearman correlation
- MAE
- Top-10% AUROC

**Source basis:**  
Independent \(\Lambda_{\mathrm{hold}}\) residual-risk verification results with volume-bootstrap confidence intervals.

**Audit status:**  
Passed. Table values are consistent with Figure 3.

---

### Table 2 — Paired volume-bootstrap comparison of A4 CNN against competing predictors

**Purpose:**  
Reports paired bootstrap differences between A4 CNN and each comparator.

**Files:**

- `reports/publication_v3/tables/main/Table_2_paired_bootstrap_against_A4.csv`
- `reports/publication_v3/tables/main/Table_2_paired_bootstrap_against_A4.md`

**Build script:**

- `scripts/build_publication_v3_table2.py`

**Reported comparisons:**

- \(\Delta\) top-10% AUPRC
- \(\Delta\) MAE
- \(\Delta\) Spearman

**Interpretation convention:**

- Positive \(\Delta\) AUPRC favors A4.
- Positive \(\Delta\) Spearman favors A4.
- Negative \(\Delta\) MAE favors A4.

**Source basis:**  
Paired volume-bootstrap comparison against A4 CNN.

**Audit status:**  
Passed. Caption explicitly states the direction of favorable differences.

---

### Table 3 — Split conformal residual-risk bounds

**Purpose:**  
Reports split conformal residual-risk coverage and interval width for each predictor.

**Files:**

- `reports/publication_v3/tables/main/Table_3_split_conformal_residual_risk_bounds.csv`
- `reports/publication_v3/tables/main/Table_3_split_conformal_residual_risk_bounds.md`

**Build script:**

- `scripts/build_publication_v3_table3.py`

**Reported quantities:**

- Target coverage
- Observed coverage
- \(\hat{q}\)
- Interval width \(2\hat{q}\)

**Source basis:**  
Split conformal residual-risk calibration and independent test evaluation.

**Audit status:**  
Passed. Table values are consistent with Figure 4.

---

## Supplementary Tables

### Table S1 — Dataset, acquisition subset, and evaluation split summary

**Purpose:**  
Documents the dataset scope, acquisition subset, independent test population, sampled pixel counts, and conformal calibration/test quantities.

**Files:**

- `reports/publication_v3/tables/supplementary/Table_S1_dataset_and_split_summary.csv`
- `reports/publication_v3/tables/supplementary/Table_S1_dataset_and_split_summary.md`

**Build script:**

- `scripts/build_publication_v3_tableS1.py`

**Source basis:**  
Dataset audit, split summary, and conformal analysis summary.

**Audit status:**  
Passed. Caption avoids internal experiment-log language and reports the dataset scope directly.

---

### Table S2 — Full independent \(\Lambda_{\mathrm{hold}}\) residual-risk verification metrics

**Purpose:**  
Provides full point-estimate metrics for residual-risk predictors evaluated against \(u_{\Lambda_{\mathrm{hold}}}\).

**Files:**

- `reports/publication_v3/tables/supplementary/Table_S2_full_lambda_hold_metrics.csv`
- `reports/publication_v3/tables/supplementary/Table_S2_full_lambda_hold_metrics.md`

**Build script:**

- `scripts/build_publication_v3_tableS2.py`

**Reported metrics:**

- MAE
- MSE
- Pearson correlation
- Spearman correlation
- Top-risk fraction
- Top-risk threshold
- Top-10% AUROC
- Top-10% AUPRC
- Evaluation pixels

**Source basis:**  
Full independent \(\Lambda_{\mathrm{hold}}\) metric audit.

**Audit status:**  
Passed. Table supports Table 1 by reporting the full metric set.

---

### Table S3 — CNN residual-risk predictor ablation summary

**Purpose:**  
Reports ablation results for CNN residual-risk predictor input variants.

**Files:**

- `reports/publication_v3/tables/supplementary/Table_S3_reliability_cnn_ablation_summary.csv`
- `reports/publication_v3/tables/supplementary/Table_S3_reliability_cnn_ablation_summary.md`

**Build script:**

- `scripts/build_publication_v3_tableS3.py`

**Source basis:**  
Experiment 008 reliability ablation summary.

**Main role in manuscript:**  
Supports the selection of the A4 image-derived variant for subsequent independent \(\Lambda_{\mathrm{hold}}\) verification.

**Audit status:**  
Passed. Caption states that ablation results are model-selection evidence and are separate from the final independent holdout verification.

---

### Table S4 — Residual-risk target construction parameters

**Purpose:**  
Documents fixed parameters used to construct \(u_{\Lambda}\) from held-out k-space inconsistency.

**Files:**

- `reports/publication_v3/tables/supplementary/Table_S4_residual_risk_target_parameters.csv`
- `reports/publication_v3/tables/supplementary/Table_S4_residual_risk_target_parameters.md`

**Build script:**

- `scripts/build_publication_v3_tableS4.py`

**Parameter categories:**

- Log scaling
- Residual-energy normalization
- Numerical stability
- Local pooling
- Density compensation
- PSF/gain estimation
- Soft anatomical/support mask

**Source basis:**  
Residual-risk target construction configuration.

**Audit status:**  
Passed. Caption states that these are experimental target-construction parameters, not clinically optimized values.

---

### Table S5 — Volume-level A4 CNN win rates

**Purpose:**  
Reports how often A4 outperformed each comparator across independent test volumes.

**Files:**

- `reports/publication_v3/tables/supplementary/Table_S5_volume_level_A4_win_rates.csv`
- `reports/publication_v3/tables/supplementary/Table_S5_volume_level_A4_win_rates.md`

**Build script:**

- `scripts/build_publication_v3_tableS5.py`

**Reported win types:**

- AUPRC wins
- Spearman wins
- MAE wins

**Source basis:**  
Volume-level comparison of A4 CNN against competing predictors across 40 independent test volumes.

**Audit status:**  
Passed. Table checks whether aggregate bootstrap results are consistent across volumes.

---

## Caption file

All final table captions are stored in:

- `reports/publication_v3/captions/table_captions.md`

---

## Final audit summary

The publication v3 table package contains three main tables and five supplementary tables.

Main tables:

1. Table 1 — independent \(\Lambda_{\mathrm{hold}}\) residual-risk verification performance  
2. Table 2 — paired bootstrap comparison against A4  
3. Table 3 — split conformal residual-risk bounds  

Supplementary tables:

1. Table S1 — dataset and evaluation split summary  
2. Table S2 — full independent \(\Lambda_{\mathrm{hold}}\) metrics  
3. Table S3 — CNN residual-risk predictor ablation summary  
4. Table S4 — residual-risk target construction parameters  
5. Table S5 — volume-level A4 win rates  

The table package is considered locked unless a numerical inconsistency, caption mismatch, or manuscript-structure issue is later identified.
