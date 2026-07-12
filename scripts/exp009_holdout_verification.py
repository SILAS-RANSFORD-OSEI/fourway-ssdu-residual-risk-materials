import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fourway_mri.operators import apply_kspace_mask, multicoil_adjoint
from fourway_mri.reliability_model import ReliabilityUNetSmall, ReliabilityUNetWithPSFSkip
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


def safe_pearson(y_true, y_pred):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()

    denom = np.sqrt(np.sum(yt**2) * np.sum(yp**2))
    if denom < 1e-12:
        return np.nan

    return float(np.sum(yt * yp) / denom)


def safe_spearman(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(spearmanr(y_true, y_pred).correlation)


def gradient_magnitude_2d(image):
    dx = np.zeros_like(image, dtype=np.float32)
    dy = np.zeros_like(image, dtype=np.float32)

    dx[:, 1:] = image[:, 1:] - image[:, :-1]
    dy[1:, :] = image[1:, :] - image[:-1, :]

    return np.sqrt(dx * dx + dy * dy + 1e-12).astype(np.float32)


def compute_reference_scale(kspace_raw, percentile=99.0, eps=1e-8):
    with torch.no_grad():
        full_rss_raw = rss_combine_torch(ifft2c(kspace_raw), coil_dim=1)
        scale = float(np.percentile(full_rss_raw.detach().cpu().numpy(), percentile) + eps)

    return scale, full_rss_raw


def rss_from_sense_image(image, sensitivities):
    coil_images = sensitivities * image[:, None, :, :]
    return rss_combine_torch(coil_images, coil_dim=1)


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

    x0_rss = rss_from_sense_image(x0, sensitivities)

    return {
        "kspace": kspace,
        "full_rss": full_rss,
        "y_theta": y_theta,
        "theta_mask": theta_mask,
        "sensitivities": sensitivities,
        "x0": x0,
        "x0_rss": x0_rss,
        "input_scale": input_scale,
    }


def load_ssdu_model(cfg, device):
    model = ScaleConstrainedSingleStepMoDL(
        features=int(cfg["ssdu_model"]["features"]),
        init_step_size=float(cfg["ssdu_model"]["init_step_size"]),
    ).to(device)

    checkpoint = torch.load(cfg["checkpoints"]["ssdu_v4"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_full_reliability_model(cfg, device):
    model = ReliabilityUNetWithPSFSkip(
        in_channels=int(cfg["reliability_model"]["in_channels"]),
        out_channels=int(cfg["reliability_model"]["out_channels"]),
        base_channels=int(cfg["reliability_model"]["base_channels"]),
        psf_channel_index=int(cfg["reliability_model"]["psf_channel_index"]),
        init_psf_weight=float(cfg["reliability_model"]["init_psf_weight"]),
        init_psf_bias=float(cfg["reliability_model"]["init_psf_bias"]),
    ).to(device)

    checkpoint = torch.load(cfg["checkpoints"]["reliability_full"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_a4_model(cfg, device):
    model = ReliabilityUNetSmall(
        in_channels=int(cfg["reliability_model"]["in_channels"]),
        out_channels=int(cfg["reliability_model"]["out_channels"]),
        base_channels=int(cfg["reliability_model"]["base_channels"]),
    ).to(device)

    checkpoint = torch.load(cfg["checkpoints"]["reliability_a4_image_only"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_cache_lookup(cache_manifest):
    df = pd.read_csv(cache_manifest)

    required = {"sample_id", "npz_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cache manifest missing columns: {missing}")

    return dict(zip(df["sample_id"], df["npz_path"]))


def load_cached_input(cache_lookup, sample_id):
    if sample_id not in cache_lookup:
        raise KeyError(f"sample_id not found in cache manifest: {sample_id}")

    path = Path(cache_lookup[sample_id])
    if not path.exists():
        raise FileNotFoundError(f"Cached input not found: {path}")

    data = np.load(path)
    x = data["x"].astype(np.float32)

    return torch.from_numpy(x)[None, ...]


def make_holdout_target(x_hat, prepared, hold_mask, cfg, slice_idx):
    support_mask = soft_anatomy_mask_from_magnitude(
        prepared["x0_rss"],
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
        risk_mask=hold_mask,
        patch_size=int(cfg["holdout_target"]["patch_size"]),
        dcf_kernel_size=int(cfg["holdout_target"]["dcf_kernel_size"]),
        psf_num_probes=int(cfg["holdout_target"]["psf_num_probes"]),
        psf_seed=int(cfg["holdout_target"]["psf_seed"]) + int(slice_idx),
        normalize_percentile=float(cfg["holdout_target"]["normalize_percentile"]),
        log_alpha=float(cfg["holdout_target"]["log_alpha"]),
        eps=float(cfg["normalization"]["epsilon"]),
    )

    masked_target = apply_support_mask_to_target(
        residual_target["target"],
        support_mask,
        power=float(cfg["support_mask"].get("power", 1.0)),
    )

    masked_p99 = torch.quantile(
        masked_target.detach().flatten(),
        float(cfg["holdout_target"]["normalize_percentile"]) / 100.0,
    )

    masked_norm = masked_target / (
        masked_p99 + float(cfg["normalization"]["epsilon"])
    )

    masked_log = torch.log1p(
        float(cfg["holdout_target"]["log_alpha"]) * masked_norm
    )

    return masked_log


def evaluate_scores(y_eval, p_eval, top_fraction):
    threshold = np.quantile(y_eval, 1.0 - top_fraction)
    labels = (y_eval >= threshold).astype(np.int32)

    try:
        auroc = float(roc_auc_score(labels, p_eval))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(labels, p_eval))
    except ValueError:
        auprc = np.nan

    diff = p_eval - y_eval

    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "pearson": safe_pearson(y_eval, p_eval),
        "spearman": safe_spearman(y_eval, p_eval),
        "top_threshold": float(threshold),
        "top_risk_auroc": auroc,
        "top_risk_auprc": auprc,
        "num_eval_pixels_sampled": int(y_eval.size),
    }


def run_split(cfg, split, loader, data_root, cache_lookup, mask_records, ssdu_model, full_model, a4_model, device):
    rng = np.random.default_rng(int(cfg["metrics"]["random_seed"]) + len(split))
    max_pixels = int(cfg["metrics"]["max_metric_pixels_per_split"])
    top_fraction = float(cfg["metrics"]["top_risk_fraction"])

    per_sample = max(1, int(max_pixels // max(1, len(loader.dataset))))

    y_samples = []
    score_samples = {
        "reliability_full": [],
        "reliability_a4_image_only": [],
        "xhat_magnitude": [],
        "image_gradient": [],
        "intervention_magnitude": [],
        "psf_gain_channel": [],
    }

    sample_rows = []

    for batch in tqdm(loader, desc=f"holdout {split}"):
        volume_id = batch["volume_id"][0]
        filename = batch["filename"][0]
        slice_idx = int(batch["slice_idx"][0].item())
        width = int(batch["width"][0].item())

        sample_id = f"{volume_id}_slice{slice_idx:03d}"

        if volume_id not in mask_records:
            raise KeyError(f"volume_id missing from mask records: {volume_id}")

        mask_rec = mask_records[volume_id]

        if "lambda_hold" not in mask_rec:
            raise KeyError(f"Mask record for {volume_id} has no lambda_hold field.")

        hold_mask = indices_to_torch_mask(
            width=width,
            indices=mask_rec["lambda_hold"],
            device=device,
            dtype=torch.float32,
        )

        prepared = prepare_batch(batch, device, cfg)

        with torch.no_grad():
            x_hat = ssdu_model(
                x0=prepared["x0"],
                y_theta=prepared["y_theta"],
                sensitivities=prepared["sensitivities"],
                theta_mask=prepared["theta_mask"],
                input_scale=prepared["input_scale"],
            )

            y_hold = make_holdout_target(
                x_hat=x_hat,
                prepared=prepared,
                hold_mask=hold_mask,
                cfg=cfg,
                slice_idx=slice_idx,
            )

            x_cached = load_cached_input(cache_lookup, sample_id).to(device)

            pred_full = full_model(x_cached)

            x_a4 = x_cached.clone()
            x_a4[:, 3, :, :] = 0.0
            x_a4[:, 4, :, :] = 0.0
            x_a4[:, 5, :, :] = 0.0
            pred_a4 = a4_model(x_a4)

        y_np = y_hold[0].detach().cpu().numpy().astype(np.float32)

        x_np = x_cached[0].detach().cpu().numpy().astype(np.float32)

        scores = {
            "reliability_full": pred_full[0, 0].detach().cpu().numpy().astype(np.float32),
            "reliability_a4_image_only": pred_a4[0, 0].detach().cpu().numpy().astype(np.float32),
            "xhat_magnitude": x_np[0],
            "image_gradient": gradient_magnitude_2d(x_np[0]),
            "intervention_magnitude": x_np[2],
            "psf_gain_channel": x_np[5],
        }

        y_flat = y_np.reshape(-1)
        n = y_flat.size
        k = min(per_sample, n)

        idx = rng.choice(n, size=k, replace=False)
        y_samples.append(y_flat[idx])

        for name, score in scores.items():
            score_samples[name].append(score.reshape(-1)[idx])

        row = {
            "split": split,
            "sample_id": sample_id,
            "volume_id": volume_id,
            "filename": filename,
            "slice_idx": slice_idx,
            "target_hold_mean": float(np.mean(y_flat)),
            "target_hold_p99": float(np.percentile(y_flat, 99)),
            "lambda_hold_used": True,
        }

        for name, score in scores.items():
            s_flat = score.reshape(-1)
            row[f"{name}_pearson_slice"] = safe_pearson(y_flat, s_flat)
            row[f"{name}_spearman_slice"] = safe_spearman(y_flat, s_flat)

        sample_rows.append(row)

    y_eval = np.concatenate(y_samples)

    split_rows = []
    for name, parts in score_samples.items():
        p_eval = np.concatenate(parts)

        metrics = evaluate_scores(
            y_eval=y_eval,
            p_eval=p_eval,
            top_fraction=top_fraction,
        )

        metrics.update({
            "split": split,
            "predictor": name,
            "top_fraction": top_fraction,
            "lambda_hold_used": True,
        })

        split_rows.append(metrics)

    return split_rows, sample_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp009_holdout_verification_pilot.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-manifest", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.cache_manifest is not None:
        cfg["inputs"]["reliability_cache_manifest"] = args.cache_manifest

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_records = load_mask_records(cfg["inputs"]["mask_indices_json"])
    cache_lookup = load_cache_lookup(cfg["inputs"]["reliability_cache_manifest"])

    ssdu_model = load_ssdu_model(cfg, device)
    full_model = load_full_reliability_model(cfg, device)
    a4_model = load_a4_model(cfg, device)

    all_split_rows = []
    all_sample_rows = []

    for split in cfg["data"]["splits"]:
        loader = make_loader(cfg, split, args.data_root)

        split_rows, sample_rows = run_split(
            cfg=cfg,
            split=split,
            loader=loader,
            data_root=args.data_root,
            cache_lookup=cache_lookup,
            mask_records=mask_records,
            ssdu_model=ssdu_model,
            full_model=full_model,
            a4_model=a4_model,
            device=device,
        )

        all_split_rows.extend(split_rows)
        all_sample_rows.extend(sample_rows)

    split_df = pd.DataFrame(all_split_rows)
    sample_df = pd.DataFrame(all_sample_rows)

    split_df = split_df[
        [
            "split",
            "predictor",
            "mae",
            "mse",
            "pearson",
            "spearman",
            "top_fraction",
            "top_threshold",
            "top_risk_auroc",
            "top_risk_auprc",
            "num_eval_pixels_sampled",
            "lambda_hold_used",
        ]
    ]

    split_df.to_csv(output_dir / "holdout_split_metrics.csv", index=False)
    sample_df.to_csv(output_dir / "holdout_sample_metrics.csv", index=False)

    test_df = split_df[split_df["split"] == "test"].copy()
    best_test = test_df.sort_values("top_risk_auprc", ascending=False).iloc[0].to_dict()

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "lambda_hold_used": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "splits": cfg["data"]["splits"],
        "num_sample_rows": int(len(sample_df)),
        "predictors": sorted(split_df["predictor"].unique().tolist()),
        "best_test_by_auprc": best_test,
        "outputs": {
            "holdout_split_metrics_csv": str(output_dir / "holdout_split_metrics.csv"),
            "holdout_sample_metrics_csv": str(output_dir / "holdout_sample_metrics.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
