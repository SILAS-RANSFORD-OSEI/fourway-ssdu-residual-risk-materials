# Experiment 008 Design: Ablation Study for PSF-Aware ReliabilityCNN

## Experiment ID

**Experiment 008 — ReliabilityCNN Ablation Study**

## Purpose

Experiment 008 tests which components of the PSF-aware ReliabilityCNN are responsible for the performance gain observed in full Experiment 007-v2.

Full Experiment 007-v2 showed that the PSF-skip ReliabilityCNN outperformed deterministic and linear baselines across MAE, MSE, Pearson, Spearman, AUROC, and AUPRC.

However, a publishable paper must show that the model components are necessary. Experiment 008 therefore removes or disables key input channels and architectural components.

## Full Model Reference

The accepted full model is:

\[
\hat{u}
=
\mathrm{UNet}(X)
+
w q_{\mathrm{psf}}
+
b
\]

with input tensor:

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

The target is:

\[
y =
u_{\log,\mathrm{masked}}
\]

## Baseline Gate

The full Experiment 006 baseline gate is:

| Metric | Strongest baseline | Value |
|---|---|---:|
| AUPRC | image gradient | 0.5872 |
| AUROC | linear 6-channel | 0.9546 |
| Spearman | linear 6-channel | 0.9306 |
| Pearson | linear 6-channel | 0.9269 |

The full Experiment 007-v2 CNN achieved:

| Metric | Full CNN |
|---|---:|
| AUPRC | 0.8469 |
| AUROC | 0.9829 |
| Spearman | 0.9431 |
| Pearson | 0.9727 |

## Ablation Models

| ID | Model | Description |
|---|---|---|
| A0 | Full model | all 6 channels + PSF skip |
| A1 | No PSF skip | all 6 channels, but remove \(wq_{\mathrm{psf}}\) skip |
| A2 | No \(q_{\mathrm{psf}}\) channel | set channel 5 to zero and remove skip |
| A3 | No analytical PSF channel | set channel 4 to zero |
| A4 | Image-only | use only \(|\hat{x}|, \|x_0\|, |\hat{x}-x_0|\); zero other channels |
| A5 | No intervention channel | set \(|\hat{x}-x_0|\) channel to zero |

## Fixed Training Protocol

All ablations must use the same protocol:

| Setting | Value |
|---|---:|
| Train split | train |
| Validation split | calibration |
| Test split | final frozen evaluation only |
| Loss | Huber |
| Checkpoint metric | calibration MAE |
| Batch size | 1 |
| Epochs | 50 |
| Early stopping patience | 8 |
| Learning rate | \(1\times10^{-4}\) |

No ablation may use test metrics for model selection.

## Primary Questions

1. Does removing the PSF skip reduce Spearman/global rank fidelity?
2. Does removing \(q_{\mathrm{psf}}\) reduce both continuous fidelity and extreme-risk detection?
3. Does removing the analytical PSF channel reduce AUPRC?
4. Can image-only inputs explain residual risk without acquisition-geometry channels?
5. Does the intervention channel \(|\hat{x}-x_0|\) contribute to local risk localization?

## Success Criteria

Experiment 008 passes if it shows that the full model is not trivially replaceable by a reduced configuration.

The strongest evidence would be:

- full model has highest or near-highest AUPRC,
- full model has highest or near-highest Spearman/Pearson,
- removal of PSF-related information degrades performance,
- image-only model underperforms full PSF-aware model.

## Interpretation Rule

If an ablation matches the full model, the corresponding removed component is not necessary.

If the full model clearly outperforms reduced models, then the paper can claim that both anatomical and acquisition-physics channels contribute to reliability prediction.

## Status

Experiment 008 is open.

Next step:

Implement ablation training/evaluation using the full reliability cache and locked Experiment 007-v2 protocol.
