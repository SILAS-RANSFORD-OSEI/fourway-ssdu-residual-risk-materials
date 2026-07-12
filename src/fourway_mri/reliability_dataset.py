from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReliabilityCacheDataset(Dataset):
    """
    Dataset for cached Experiment 006 reliability input-target samples.

    Each .npz file contains:
        x: (6, H, W) PSF-aware input tensor
        y: (H, W) masked log residual-risk target
        y_raw: optional raw masked target

    The model input must not contain the residual target itself.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: Optional[str] = None,
        target_key: str = "y",
        input_key: str = "x",
    ):
        self.manifest_csv = Path(manifest_csv)
        self.split = split
        self.target_key = target_key
        self.input_key = input_key

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        manifest = pd.read_csv(self.manifest_csv)

        if split is not None:
            manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        if len(manifest) == 0:
            raise ValueError(f"No samples found for split={split!r}")

        required_columns = {"npz_path", "sample_id", "split", "volume_id", "slice_idx"}
        missing = required_columns.difference(set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        npz_path = Path(row["npz_path"])

        if not npz_path.exists():
            raise FileNotFoundError(f"Cached sample not found: {npz_path}")

        data = np.load(npz_path)

        if self.input_key not in data:
            raise KeyError(f"{self.input_key!r} not found in {npz_path}")

        if self.target_key not in data:
            raise KeyError(f"{self.target_key!r} not found in {npz_path}")

        x = data[self.input_key].astype(np.float32)
        y = data[self.target_key].astype(np.float32)

        if x.ndim != 3:
            raise ValueError(f"x must have shape (C,H,W), got {x.shape}")

        if y.ndim != 2:
            raise ValueError(f"y must have shape (H,W), got {y.shape}")

        if x.shape[-2:] != y.shape:
            raise ValueError(
                f"Spatial mismatch: x has {x.shape[-2:]}, y has {y.shape}"
            )

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y)[None, :, :],
            "sample_id": row["sample_id"],
            "split": row["split"],
            "volume_id": row["volume_id"],
            "slice_idx": int(row["slice_idx"]),
            "npz_path": str(npz_path),
        }
