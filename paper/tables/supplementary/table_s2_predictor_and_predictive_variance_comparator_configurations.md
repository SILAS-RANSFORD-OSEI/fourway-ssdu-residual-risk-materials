## Table S2. Predictor and predictive-variance comparator configurations

| Method / role | Input interface | Explicit gain skip | Trainable parameters | Inference and reported output |
|---|---|---|---:|---|
| Full-CNN — primary predictor | 6 active channels: 3 image-derived + support + analytical PSF + gain | Yes | 123,035 | 1 deterministic pass; predicted residual risk |
| A4-CNN — secondary predictor | 6-channel interface; first 3 channels active and remaining 3 zeroed | No | 123,033 | 1 deterministic pass; predicted residual risk |
| MC-dropout — variance comparator | A4-CNN-aligned interface | No | 123,033 | 20 stochastic passes; population predictive variance |
| Ensemble — variance comparator | A4-CNN-aligned interface for each member | No | 123,033 per member | 3 independently trained members; population between-member variance |

**Note.** A4-CNN is conceptually image-derived but retains the same six-channel convolutional interface as Full-CNN with channels 3–5 deterministically zeroed. The uncertainty comparators are residual-risk networks evaluated through predictive variance.
