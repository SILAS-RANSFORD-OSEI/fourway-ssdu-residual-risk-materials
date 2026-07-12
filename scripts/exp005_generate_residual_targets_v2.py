import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fourway_mri.operators import apply_kspace_mask, multicoil_adjoint
from fourway_mri.residuals import generate_residual_risk_target, soft_anatomy_mask_from_magnitude, apply_support_mask_to_target
from fourway_mri.sensitivity import estimate_sensitivities_from_acs, indices_to_torch_mask
from fourway_mri.ssdu_dataset import SSDUSliceDataset, load_mask_records
from fourway_mri.ssdu_model_v4 import ScaleConstrainedSingleStepMoDL
from fourway_mri.torch_fft import ifft2c, rss_combine_torch


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_loader(cfg, split, data_root):
    max_volumes_cfg = cfg["data"]["max_volumes_per_split"]
    max_volumes = max_volumes_cfg.get(split, None)

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
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "nonzero_fraction": float(np.mean(arr > 0)),
        "finite": bool(np.isfinite(arr).all()),
    }


def save_example_panel(path, full_rss, x0_rss, out_rss, target8_log, target16_log):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [
        ("Full RSS", full_rss),
        ("Initial adjoint", x0_rss),
        ("v4 output", out_rss),
        ("Risk target log K=8", target8_log),
        ("Risk target log K=16", target16_log),
    ]

    fig, axes = plt.subplots(1, len(images), figsize=(18, 4))

    for ax, (title, img) in zip(axes, images):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp005_generate_residual_targets_v2.yaml")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

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

    example_saved = False
    example_payload = {}

    for split in cfg["data"]["splits"]:
        loader = make_loader(cfg, split, args.data_root)

        for batch in tqdm(loader, desc=f"targets {split}"):
            volume_id = batch["volume_id"][0]
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
                out = model(
                    x0=prepared["x0"],
                    y_theta=prepared["y_theta"],
                    sensitivities=prepared["sensitivities"],
                    theta_mask=prepared["theta_mask"],
                    input_scale=prepared["input_scale"],
                )

                full_rss = prepared["full_rss"]
                x0_rss = rss_from_sense_image(prepared["x0"], prepared["sensitivities"])
                out_rss = rss_from_sense_image(out, prepared["sensitivities"])

                target_logs_by_patch = {}

                for patch_size in cfg["residual_target"]["patch_sizes"]:
                    target = generate_residual_risk_target(
                        prediction=out,
                        target_kspace=prepared["kspace"],
                        sensitivities=prepared["sensitivities"],
                        risk_mask=risk_mask,
                        patch_size=int(patch_size),
                        dcf_kernel_size=int(cfg["residual_target"]["dcf_kernel_size"]),
                        psf_num_probes=int(cfg["residual_target"]["psf_num_probes"]),
                        psf_seed=int(cfg["residual_target"]["psf_seed"]) + slice_idx,
                        normalize_percentile=float(cfg["residual_target"]["normalize_percentile"]),
                        log_alpha=float(cfg["residual_target"]["log_alpha"]),
                        eps=float(cfg["normalization"]["epsilon"]),
                    )

                    support_mask = soft_anatomy_mask_from_magnitude(
                        x0_rss,
                        threshold=float(cfg["support_mask"]["threshold"]),
                        softness=float(cfg["support_mask"]["softness"]),
                        smooth_kernel_size=int(cfg["support_mask"]["smooth_kernel_size"]),
                        percentile=float(cfg["support_mask"]["percentile"]),
                        eps=float(cfg["normalization"]["epsilon"]),
                    )

                    masked_target = apply_support_mask_to_target(
                        target["target"],
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

                    raw_stats = tensor_stats(target["target"])
                    masked_stats = tensor_stats(masked_target)
                    masked_log_stats = tensor_stats(masked_log)
                    residual_energy_stats = tensor_stats(target["residual_energy"])
                    psf_stats = tensor_stats(target["psf_envelope"])
                    mask_stats = tensor_stats(support_mask)

                    rows.append(
                        {
                            "split": split,
                            "volume_id": volume_id,
                            "filename": batch["filename"][0],
                            "slice_idx": slice_idx,
                            "patch_size": int(patch_size),
                            "num_risk_lines": int(torch.sum(risk_mask).item()),
                            "lambda_hold_used": False,
                            "target_finite": raw_stats["finite"],
                            "masked_target_finite": masked_stats["finite"],
                            "masked_target_log_finite": masked_log_stats["finite"],
                            "target_mean": raw_stats["mean"],
                            "target_std": raw_stats["std"],
                            "target_p99": raw_stats["p99"],
                            "target_nonzero_fraction": raw_stats["nonzero_fraction"],
                            "masked_target_mean": masked_stats["mean"],
                            "masked_target_std": masked_stats["std"],
                            "masked_target_p50": masked_stats["p50"],
                            "masked_target_p90": masked_stats["p90"],
                            "masked_target_p95": masked_stats["p95"],
                            "masked_target_p99": masked_stats["p99"],
                            "masked_target_max": masked_stats["max"],
                            "masked_target_nonzero_fraction": masked_stats["nonzero_fraction"],
                            "masked_target_log_mean": masked_log_stats["mean"],
                            "masked_target_log_p99": masked_log_stats["p99"],
                            "support_mask_mean": mask_stats["mean"],
                            "support_mask_p50": mask_stats["p50"],
                            "support_mask_p99": mask_stats["p99"],
                            "residual_energy_mean": residual_energy_stats["mean"],
                            "residual_energy_p99": residual_energy_stats["p99"],
                            "psf_envelope_mean": psf_stats["mean"],
                            "psf_envelope_p99": psf_stats["p99"],
                        }
                    )

                    target_logs_by_patch[int(patch_size)] = masked_log

                if not example_saved:
                    example_payload = {
                        "full_rss": full_rss[0],
                        "x0_rss": x0_rss[0],
                        "out_rss": out_rss[0],
                        "target8_log": target_logs_by_patch[8][0],
                        "target16_log": target_logs_by_patch[16][0],
                    }
                    example_saved = True

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "target_statistics.csv", index=False)

    summary_by_patch = (
        stats_df.groupby(["split", "patch_size"], as_index=False)
        .agg(
            num_targets=("target_mean", "count"),
            target_finite_all=("target_finite", "all"),
            masked_target_finite_all=("masked_target_finite", "all"),
            masked_target_log_finite_all=("masked_target_log_finite", "all"),
            target_mean_mean=("target_mean", "mean"),
            target_p99_mean=("target_p99", "mean"),
            target_nonzero_fraction_mean=("target_nonzero_fraction", "mean"),
            masked_target_mean_mean=("masked_target_mean", "mean"),
            masked_target_p99_mean=("masked_target_p99", "mean"),
            masked_target_log_mean_mean=("masked_target_log_mean", "mean"),
            masked_target_log_p99_mean=("masked_target_log_p99", "mean"),
            masked_target_nonzero_fraction_mean=("masked_target_nonzero_fraction", "mean"),
            support_mask_mean_mean=("support_mask_mean", "mean"),
            support_mask_p99_mean=("support_mask_p99", "mean"),
            psf_envelope_mean_mean=("psf_envelope_mean", "mean"),
        )
    )

    summary_by_patch.to_csv(output_dir / "summary_by_patch.csv", index=False)

    if example_saved:
        save_example_panel(
            output_dir / "example_targets.png",
            example_payload["full_rss"],
            example_payload["x0_rss"],
            example_payload["out_rss"],
            example_payload["target8_log"],
            example_payload["target16_log"],
        )

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "num_rows": int(len(stats_df)),
        "splits": cfg["data"]["splits"],
        "patch_sizes": cfg["residual_target"]["patch_sizes"],
        "lambda_risk_used": True,
        "lambda_hold_used": False,
        "all_targets_finite": bool(stats_df["target_finite"].all()),
        "all_masked_targets_finite": bool(stats_df["masked_target_finite"].all()),
        "all_masked_log_targets_finite": bool(stats_df["masked_target_log_finite"].all()),
        "mean_nonzero_fraction": float(stats_df["target_nonzero_fraction"].mean()),
        "mean_masked_nonzero_fraction": float(stats_df["masked_target_nonzero_fraction"].mean()),
        "mean_support_mask": float(stats_df["support_mask_mean"].mean()),
        "outputs": {
            "target_statistics_csv": str(output_dir / "target_statistics.csv"),
            "summary_by_patch_csv": str(output_dir / "summary_by_patch.csv"),
            "example_targets_png": str(output_dir / "example_targets.png"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
