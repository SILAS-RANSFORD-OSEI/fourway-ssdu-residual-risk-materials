# Experiment 009 Milestone: Full Independent Lambda-Hold Verification

## Experiment ID

**Experiment 009 — Full Independent \(\Lambda_{\mathrm{hold}}\) Verification**

## Purpose

Experiment 009 tested whether frozen ReliabilityCNN predictions generalize to independent held-out k-space measurements.

Previous reliability training used residual-risk targets derived from:

\[
\Lambda_{\mathrm{risk}}
\]

Experiment 009 evaluated agreement with a separately held-out measurement subset:

\[
\Lambda_{\mathrm{hold}}
\]

No training, checkpoint selection, or hyperparameter tuning was performed using \(\Lambda_{\mathrm{hold}}\).

## Evaluation Setup

| Setting | Value |
|---|---:|
| Train samples | 3190 |
| Calibration samples | 636 |
| Test samples | 636 |
| Total samples | 4462 |
| Training performed | False |
| Checkpoint selection performed | False |
| \(\Lambda_{\mathrm{hold}}\) used | True |

## Predictors Evaluated

The following frozen predictors were compared against the \(\Lambda_{\mathrm{hold}}\)-derived residual-risk target:

1. `reliability_full`
2. `reliability_a4_image_only`
3. `xhat_magnitude`
4. `image_gradient`
5. `intervention_magnitude`
6. `psf_gain_channel`

## Full Test Results

| Predictor | MAE | MSE | Pearson | Spearman | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| `reliability_a4_image_only` | 0.0801 | 0.0271 | 0.9715 | 0.9515 | 0.9824 | 0.8413 |
| `reliability_full` | 0.0829 | 0.0277 | 0.9708 | 0.9403 | 0.9821 | 0.8397 |
| `image_gradient` | 0.4181 | 0.6166 | 0.6331 | 0.8056 | 0.9410 | 0.5802 |
| `xhat_magnitude` | 0.3403 | 0.4033 | 0.7487 | 0.8807 | 0.9476 | 0.5723 |
| `psf_gain_channel` | 0.2445 | 0.2023 | 0.9243 | 0.9207 | 0.9492 | 0.5315 |
| `intervention_magnitude` | 0.5424 | 0.5005 | 0.0998 | 0.3801 | 0.5061 | 0.1588 |

## Main Finding

The best frozen predictor on the test split was:

\[
\texttt{reliability\_a4\_image\_only}
\]

with:

\[
\mathrm{AUPRC}=0.8413,\quad
\mathrm{AUROC}=0.9824,\quad
\rho_{\mathrm{Spearman}}=0.9515
\]

This substantially exceeded all simple predictors.

## Interpretation

Experiment 009 supports the claim that the reliability model predicts independent held-out measurement inconsistency.

The model is not merely reproducing the \(\Lambda_{\mathrm{risk}}\)-derived training target. Its predictions also agree with residual-risk maps derived from the unused \(\Lambda_{\mathrm{hold}}\) subset.

The strongest predictor remained the A4 image-only model, consistent with Experiment 008. This reinforces the conclusion that residual-risk information is strongly encoded in reconstruction-domain features and is not uniquely dependent on explicit PSF channels.

## Decision

Experiment 009 passes.

The reliability component is now supported by:

1. full baseline comparison,
2. ablation study,
3. independent \(\Lambda_{\mathrm{hold}}\) verification.

## Remaining Work

The next stage is manuscript-level analysis:

1. confidence intervals,
2. bootstrap testing,
3. qualitative figures,
4. failure-case review,
5. final result tables.
