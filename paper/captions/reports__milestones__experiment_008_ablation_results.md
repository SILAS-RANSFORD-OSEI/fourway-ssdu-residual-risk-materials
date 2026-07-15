# Experiment 008 Milestone: ReliabilityCNN Ablation Study

## Experiment ID

**Experiment 008 — ReliabilityCNN Ablation Study**

## Purpose

Experiment 008 evaluated whether the performance gain of the ReliabilityCNN depends on specific gain-envelope-aware input channels or architectural components.

The full model from Experiment 007-v2 used a six-channel input tensor and a Gain-envelope skip connection:

\[
\hat{u}
=
\mathrm{UNet}(X)
+
wq_{\mathrm{psf}}
+
b
\]

with:

\[
X =
[
|\hat{x}|,
|x_0|,
|\hat{x}-x_0|,
M_{\mathrm{soft}},
\mathrm{PSF}_{\Lambda_{\mathrm{risk}}},
q_{\mathrm{psf}}
]
\]

The ablation study tested whether removing these components degrades residual-risk prediction.

## Full Baseline Gate

The full Experiment 006 baseline gate was:

| Metric | Strongest baseline | Value |
|---|---|---:|
| AUPRC | image gradient | 0.5872 |
| AUROC | linear 6-channel | 0.9546 |
| Spearman | linear 6-channel | 0.9306 |
| Pearson | linear 6-channel | 0.9269 |

## Ablation Variants

| Variant | Description |
|---|---|
| A0 full gain-envelope-skip | full six-channel model with gain-envelope skip |
| A1 no gain-envelope skip | all channels retained, no explicit \(q_{\mathrm{psf}}\) skip |
| A2 no \(q_{\mathrm{psf}}\) | \(q_{\mathrm{psf}}\) channel zeroed and skip removed |
| A3 no analytical PSF | analytical gain-envelope channel zeroed |
| A4 image-only | only \(|\hat{x}|\), \(|x_0|\), and \(|\hat{x}-x_0|\) retained |
| A5 no intervention | \(|\hat{x}-x_0|\) channel zeroed |

## Test Results

| Variant | MAE | MSE | Pearson | Spearman | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| A0 full gain-envelope-skip | 0.0793 | 0.0265 | 0.9727 | 0.9431 | 0.9829 | 0.8469 |
| A5 no intervention | 0.0791 | 0.0262 | 0.9728 | 0.9484 | 0.9830 | 0.8463 |
| A2 no \(q_{\mathrm{psf}}\) | 0.0790 | 0.0275 | 0.9716 | 0.9522 | 0.9825 | 0.8462 |
| A4 image-only | 0.0767 | 0.0262 | 0.9729 | 0.9561 | 0.9829 | 0.8459 |
| A3 no analytical PSF | 0.0777 | 0.0262 | 0.9729 | 0.9456 | 0.9827 | 0.8446 |
| A1 no gain-envelope skip | 0.0777 | 0.0268 | 0.9723 | 0.9564 | 0.9828 | 0.8429 |

## Interpretation

All ablation variants remained far above the deterministic and linear baseline gates.

The AUPRC spread across all CNN variants was narrow:

\[
0.8429 \leq \mathrm{AUPRC} \leq 0.8469
\]

This indicates that no single explicit PSF-related channel or architectural skip connection is uniquely necessary for high performance under the current residual-risk target.

The most surprising result was the image-only ablation. A4 used only:

\[
|\hat{x}|,\quad |x_0|,\quad |\hat{x}-x_0|
\]

and zeroed:

\[
M_{\mathrm{soft}},\quad
\mathrm{PSF}_{\Lambda_{\mathrm{risk}}},\quad
q_{\mathrm{psf}}
\]

yet achieved AUPRC 0.8459 and Spearman 0.9561.

## Scientific Conclusion

Experiment 008 does not support the claim that explicit PSF channels are essential.

The stronger and more defensible conclusion is:

**The residual-risk target is highly predictable by nonlinear CNNs, and the predictive information appears distributed across redundant reconstruction-domain and acquisition-derived representations.**

The full gain-envelope-aware model remains valid, but its advantage over reduced CNN variants is small. The main performance gain is due to nonlinear CNN modeling rather than a single explicit PSF channel.

## Implication for the Paper

The manuscript should not claim:

> Explicit PSF channels are required.

Instead, it should claim:

> Nonlinear CNN models substantially outperform deterministic and linear baselines for residual-risk prediction, while ablation shows that the learned risk signal is robust to removal of individual physics channels.

## Remaining Concern

Because the target is derived from \(\Lambda_{\mathrm{risk}}\), the ablation results do not yet prove that the predicted risk generalizes to independent held-out measurements.

The next decisive experiment is:

**Experiment 009 — Independent \(\Lambda_{\mathrm{hold}}\) Verification**

## Decision

Experiment 008 passes.

The ablation study is complete and changes the interpretation of the reliability model.

## Output Files

| File | Purpose |
|---|---|
| `results/exp008_reliability_ablation_full/ablation_summary.csv` | combined ablation table |
| `results/exp008_reliability_ablation_full/A1_no_psf_skip/summary.json` | A1 result |
| `results/exp008_reliability_ablation_full/A2_no_qpsf_channel/summary.json` | A2 result |
| `results/exp008_reliability_ablation_full/A3_no_analytical_psf/summary.json` | A3 result |
| `results/exp008_reliability_ablation_full/A4_image_only/summary.json` | A4 result |
| `results/exp008_reliability_ablation_full/A5_no_intervention/summary.json` | A5 result |

## Git Commits

| Commit | Description |
|---|---|
| `dc093f1` | Record Experiment 008 A1-A3 ablation results |
| `bce12ac` | Record Experiment 008 A4 image-only ablation result |
| `4e64be3` | Record Experiment 008 A5 no-intervention ablation result |
| `88ef370` | Summarize Experiment 008 ablation results |
