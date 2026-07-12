import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fourway_mri.operators import apply_kspace_mask, multicoil_adjoint
from fourway_mri.reliability_inputs import build_reliability_input_tensor
from fourway_mri.residuals import (
    apply_support_mask_to_target,
    generate_residual_risk_target,
    soft_anatomy_mask_from_magnitude,
)
from fourway_mri.sensitivity import estimate_sensitivities_from_acs, indices_to_torch_mask
from fourway_mri.ssdu_dataset import SSDUSliceDataset, load_mask_records
from fourway_mri.ssdu_model_v4 import ScaleConstrainedSingleStepMoDL
from fourway_mri.torch_fft import ifft2c, rss_combine_torch


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_loader(cfg, split, data_root):
    max_volumes = cfg["data"]["max_volumes_per_split"].get(split, None)

    ds = SSDUSliceDataset(
        split_csv=cfg["inputs"]["split_csv"],
        mask_indices_json=cfg["inputs"]["mask_indices_json"],
        split=split,
        data_root=data_root,
        max_volumes=max_volumes,
        max_slices_per_volume=cfg["data"]["max_slices_per_volume"],
    )

    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


def compute_reference_scale(kspace_raw, percentile=99.0, eps=1e-8):
    with torch.no_grad():
        full_rss_raw = rss_combine_torch(ifft2c(kspace_raw), coil_dim=1)
        scale = float(np.percentile(full_rss_raw.detach().cpu().numpy(), percentile) + eps)

    return scale, full_rss_raw


def rss_from_sense_image(image, sensitivities):
    coil_images = sensitivities * image[:, None, :, :]
    return rss_combine_torch(coil_images, coil_dim=1)


def prepare_batch(batch, device, cfg):
    kspace_raw = batch["kspace"].to(device)

    reference_scale, full_rss_raw = compute_reference_scale(
        kspace_raw,
        percentile=float(cfg["normalization"]["reference_percentile"]),
        eps=float(cfg["normalization"]["epsilon"]),
    )

    kspace = kspace_raw / reference_scale
    full_rss = full_rss_raw / reference_scale

    theta_mask = batch["theta_mask"][0].to(device)
    theta_acs_indices = batch["theta_acs_indices"][0].tolist()
    r_theta = batch["r_theta"].to(device).float()

    y_theta = apply_kspace_mask(kspace, theta_mask)

    sensitivities = estimate_sensitivities_from_acs(
        kspace[0],
        theta_acs_indices,
        body_mask_threshold=cfg["reconstruction"].get("body_mask_threshold", None),
        body_mask_softness=float(cfg["reconstruction"].get("body_mask_softness", 0.02)),
    )[None, ...].to(device)

    x0 = multicoil_adjoint(y_theta, sensitivities, theta_mask)
    input_scale = torch.sqrt(r_theta)

    return {
        "kspace": kspace,
        "full_rss": full_rss,
        "y_theta": y_theta,
        "theta_mask": theta_mask,
        "sensitivities": sensitivities,
        "x0": x0,
        "input_scale": input_scale,
    }


def tensor_stats(x):
    arr = x.detach().cpu().numpy()
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "finite": bool(np.isfinite(arr).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp006_reliability_cache_pilot.yaml")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_dir = Path(cfg["outputs"]["output_dir"])
    cache_dir = Path(cfg["outputs"]["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_records = load_mask_records(cfg["inputs"]["mask_indices_json"])

    checkpoint = torch.load(cfg["inputs"]["checkpoint"], map_location=device)

    model = ScaleConstrainedSingleStepMoDL(
        features=int(cfg["model"]["features"]),
        init_step_size=float(cfg["model"]["init_step_size"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []

    save_dtype = np.float16 if cfg["cache"]["dtype"] == "float16" else np.float32

    for split in cfg["data"]["splits"]:
        loader = make_loader(cfg, split, args.data_root)

        for batch in tqdm(loader, desc=f"cache {split}"):
            volume_id = batch["volume_id"][0]
            filename = batch["filename"][0]
            slice_idx = int(batch["slice_idx"][0].item())
            width = int(batch["width"][0].item())

            mask_rec = mask_records[volume_id]
            if "lambda_risk" not in mask_rec:
                raise KeyError(f"Mask record for {volume_id} has no lambda_risk field.")

            risk_mask = indices_to_torch_mask(
                width=width,
                indices=mask_rec["lambda_risk"],
                device=device,
                dtype=torch.float32,
            )

            prepared = prepare_batch(batch, device, cfg)

            with torch.no_grad():
                x_hat = model(
                    x0=prepared["x0"],
                    y_theta=prepared["y_theta"],
                    sensitivities=prepared["sensitivities"],
                    theta_mask=prepared["theta_mask"],
                    input_scale=prepared["input_scale"],
                )

                x0_rss = rss_from_sense_image(prepared["x0"], prepared["sensitivities"])

                support_mask = soft_anatomy_mask_from_magnitude(
                    x0_rss,
                    threshold=float(cfg["support_mask"]["threshold"]),
                    softness=float(cfg["support_mask"]["softness"]),
                    smooth_kernel_size=int(cfg["support_mask"]["smooth_kernel_size"]),
                    percentile=float(cfg["support_mask"]["percentile"]),
                    eps=float(cfg["normalization"]["epsilon"]),
                )

                residual_target = generate_residual_risk_target(
                    prediction=x_hat,
                    target_kspace=prepared["kspace"],
                    sensitivities=prepared["sensitivities"],
                    risk_mask=risk_mask,
                    patch_size=int(cfg["residual_target"]["patch_size"]),
                    dcf_kernel_size=int(cfg["residual_target"]["dcf_kernel_size"]),
                    psf_num_probes=int(cfg["residual_target"]["psf_num_probes"]),
                    psf_seed=int(cfg["residual_target"]["psf_seed"]) + slice_idx,
                    normalize_percentile=float(cfg["residual_target"]["normalize_percentile"]),
                    log_alpha=float(cfg["residual_target"]["log_alpha"]),
                    eps=float(cfg["normalization"]["epsilon"]),
                )

                masked_target = apply_support_mask_to_target(
                    residual_target["target"],
                    support_mask,
                    power=float(cfg["support_mask"].get("power", 1.0)),
                )

                masked_p99 = torch.quantile(
                    masked_target.detach().flatten(),
                    float(cfg["residual_target"]["normalize_percentile"]) / 100.0,
                )

                masked_norm = masked_target / (
                    masked_p99 + float(cfg["normalization"]["epsilon"])
                )

                masked_log = torch.log1p(
                    float(cfg["residual_target"]["log_alpha"]) * masked_norm
                )

                input_dict = build_reliability_input_tensor(
                    x_hat=x_hat,
                    x0=prepared["x0"],
                    support_mask=support_mask,
                    risk_mask=risk_mask,
                    psf_gain=residual_target["psf_envelope"],
                    normalize_percentile=float(cfg["normalization"]["reference_percentile"]),
                    eps=float(cfg["normalization"]["epsilon"]),
                )

                x_np = input_dict["input"][0].detach().cpu().numpy().astype(save_dtype)
                y_np = masked_log[0].detach().cpu().numpy().astype(save_dtype)
                y_raw_np = masked_target[0].detach().cpu().numpy().astype(save_dtype)

                split_dir = cache_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)

                sample_id = f"{volume_id}_slice{slice_idx:03d}"
                npz_path = split_dir / f"{sample_id}.npz"

                np.savez_compressed(
                    npz_path,
                    x=x_np,
                    y=y_np,
                    y_raw=y_raw_np,
                )

                y_stats = tensor_stats(masked_log)
                mask_stats = tensor_stats(support_mask)

                rows.append(
                    {
                        "split": split,
                        "volume_id": volume_id,
                        "filename": filename,
                        "slice_idx": slice_idx,
                        "sample_id": sample_id,
                        "npz_path": str(npz_path),
                        "channels": int(x_np.shape[0]),
                        "height": int(x_np.shape[1]),
                        "width": int(x_np.shape[2]),
                        "target_finite": y_stats["finite"],
                        "target_mean": y_stats["mean"],
                        "target_std": y_stats["std"],
                        "target_p50": y_stats["p50"],
                        "target_p90": y_stats["p90"],
                        "target_p95": y_stats["p95"],
                        "target_p99": y_stats["p99"],
                        "target_max": y_stats["max"],
                        "support_mask_mean": mask_stats["mean"],
                        "lambda_risk_used": True,
                        "lambda_hold_used": False,
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "cache_manifest.csv", index=False)

    summary_by_split = (
        manifest.groupby("split", as_index=False)
        .agg(
            num_samples=("sample_id", "count"),
            target_finite_all=("target_finite", "all"),
            target_mean_mean=("target_mean", "mean"),
            target_std_mean=("target_std", "mean"),
            target_p99_mean=("target_p99", "mean"),
            support_mask_mean=("support_mask_mean", "mean"),
        )
    )

    summary_by_split.to_csv(output_dir / "cache_summary_by_split.csv", index=False)

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "num_samples": int(len(manifest)),
        "splits": cfg["data"]["splits"],
        "channels": 6,
        "target": "masked_log_residual_risk",
        "lambda_risk_used": True,
        "lambda_hold_used": False,
        "all_targets_finite": bool(manifest["target_finite"].all()),
        "outputs": {
            "cache_manifest": str(output_dir / "cache_manifest.csv"),
            "cache_summary_by_split": str(output_dir / "cache_summary_by_split.csv"),
            "cache_dir": str(cache_dir),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
