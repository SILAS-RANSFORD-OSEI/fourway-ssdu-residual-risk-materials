## Table S4. Training controls, selected comparator checkpoints, and observed computational cost

### A. Optimization and checkpoint controls

| Training stage | Optimization controls | Selection / stopping |
|---|---|---|
| Reconstruction | Initial learning rate 5 × 10⁻⁵; batch 1; accumulation 4; gradient clipping 10; maximum 30 epochs; relative reconstruction loss + ACS term (weight 1) | Mean calibration reconstruction loss; learning-rate factor 0.5, patience 3; stopping patience 6; minimum improvement 10⁻⁴ |
| Residual-risk predictors and comparators | Adam; learning rate 10⁻⁴; weight decay 10⁻⁶; Huber δ = 0.1; batch 1; maximum 50 epochs | Calibration MAE against learning-subset target; stopping patience 8; minimum improvement 5 × 10⁻⁴; dropout disabled during selection |

### B. Recorded comparator runs

| Model | Seed | Epochs run | Selected epoch | Selected calibration MAE | Training time (min) |
|---|---:|---:|---:|---:|---:|
| MC-dropout | 20260529 | 30 | 21 | 0.085799269 | 89.63 |
| Ensemble 1 | 20260529 | 18 | 9 | 0.083545617 | 53.81 |
| Ensemble 2 | 20260530 | 22 | 13 | 0.082183984 | 65.68 |
| Ensemble 3 | 20260531 | 18 | 9 | 0.083600737 | 53.49 |

**Note.** Checkpoint updates followed the minimum-improvement rule; the selected epoch need not have the smallest recorded MAE. Timings were measured on an NVIDIA A100-SXM4-80GB with the development cache in local runtime storage. Mean epoch time was 179.26 s for MC-dropout and 178.30–179.36 s across ensemble members. Reliable original Full-CNN and A4-CNN wall-clock measurements were unavailable. Combined inference over 636 test slices required 240.74 s (0.3785 s/slice) for 20 MC-dropout passes and three ensemble-member passes; these timings exclude MRI reconstruction.
