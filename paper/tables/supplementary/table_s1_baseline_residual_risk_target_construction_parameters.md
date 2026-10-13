## Table S1. Baseline residual-risk target-construction parameters

| Component | Setting | Value / implementation |
|---|---|---|
| Density compensation | Kernel width | 9 phase-encoding lines; applied before adjoint projection |
| Spatial averaging | Pooling | 16 × 16; stride 1; replicated padding; applied to residual energy and gain |
| Gain estimate | Complex probes | 4 per subset-specific envelope |
| Target scaling | Robust percentile | 99th percentile across all pixels of the support-weighted map |
| Log compression | Coefficient α | 10; natural logarithm |
| Numerical stability | ε | 1 × 10⁻⁸ |
| Soft support | Source and parameters | Θ-only zero-filled RSS; percentile 99; sigmoid threshold 0.05; softness 0.02; smoothing 15 × 15; clipping [0,1]; power 1 |
| Probe generation | Learning seed | 20260529 |
| Probe generation | Verification base seed | 20260530 |

**Note.** Parameters apply separately to residual-risk learning and reserved verification subsets. The target percentile scale is distinct from the calibration high-residual-risk threshold and the interval radius.
