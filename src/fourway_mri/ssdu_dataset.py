from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from fourway_mri.sensitivity import indices_to_torch_mask


def load_mask_records(mask_indices_json: str | Path) -> dict:
    with open(mask_indices_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {rec["volume_id"]: rec for rec in obj["records"]}


def resolve_h5_path(row: pd.Series, data_root: Optional[str | Path] = None) -> Path:
    """
    Resolve HDF5 path.

    If data_root is provided, prefer data_root/filename over the manifest path.
    This allows training from a local SSD cache instead of repeatedly reading
    HDF5 files from Google Drive.
    """
    if data_root is not None:
        candidate = Path(data_root) / row["filename"]
        if candidate.exists():
            return candidate

    path = Path(row["path"])
    if path.exists():
        return path

    raise FileNotFoundError(f"Could not locate HDF5 file: {row['filename']}")


class SSDUSliceDataset(Dataset):
    """
    Slice-level dataset for complex multicoil SSDU training.

    Returns:
        - full complex k-space slice,
        - theta mask,
        - lambda_rec mask,
        - theta ACS indices,
        - R_omega,
        - R_theta,
        - metadata.

    It deliberately does not return lambda_risk or lambda_hold masks.
    """

    def __init__(
        self,
        split_csv: str | Path,
        mask_indices_json: str | Path,
        split: str,
        data_root: Optional[str | Path] = None,
        max_volumes: Optional[int] = None,
        max_slices_per_volume: Optional[int] = None,
    ):
        self.split_csv = Path(split_csv)
        self.mask_indices_json = Path(mask_indices_json)
        self.split = split
        self.data_root = data_root

        manifest = pd.read_csv(self.split_csv)
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)

        if max_volumes is not None:
            manifest = manifest.head(int(max_volumes)).reset_index(drop=True)

        self.manifest = manifest
        self.mask_records = load_mask_records(self.mask_indices_json)

        samples = []

        for _, row in manifest.iterrows():
            volume_id = str(row["volume_id"])
            num_slices = int(row["num_slices"])

            if volume_id not in self.mask_records:
                raise KeyError(f"Missing mask record for volume_id={volume_id}")

            if max_slices_per_volume is not None:
                n_slices = min(num_slices, int(max_slices_per_volume))
            else:
                n_slices = num_slices

            for slice_idx in range(n_slices):
                samples.append(
                    {
                        "volume_id": volume_id,
                        "slice_idx": int(slice_idx),
                        "row": row.to_dict(),
                    }
                )

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        row = pd.Series(sample["row"])
        volume_id = sample["volume_id"]
        slice_idx = int(sample["slice_idx"])

        h5_path = resolve_h5_path(row, self.data_root)
        mask_rec = self.mask_records[volume_id]

        with h5py.File(h5_path, "r") as hf:
            kspace_np = hf["kspace"][slice_idx]

        kspace = torch.from_numpy(kspace_np).to(torch.complex64)

        width = int(kspace.shape[-1])

        theta_mask = indices_to_torch_mask(width, mask_rec["theta"])
        lambda_rec_mask = indices_to_torch_mask(width, mask_rec["lambda_rec"])

        r_omega = width / len(mask_rec["omega"])
        r_theta = width / len(mask_rec["theta"])

        return {
            "kspace": kspace,
            "theta_mask": theta_mask,
            "lambda_rec_mask": lambda_rec_mask,
            "theta_acs_indices": torch.as_tensor(mask_rec["theta_acs"], dtype=torch.long),
            "r_omega": torch.tensor(r_omega, dtype=torch.float32),
            "r_theta": torch.tensor(r_theta, dtype=torch.float32),
            "volume_id": volume_id,
            "filename": str(row["filename"]),
            "slice_idx": torch.tensor(slice_idx, dtype=torch.long),
            "num_coils": torch.tensor(int(row["num_coils"]), dtype=torch.long),
            "height": torch.tensor(int(row["height"]), dtype=torch.long),
            "width": torch.tensor(width, dtype=torch.long),
        }
