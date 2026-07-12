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

from exp009_holdout_verification import (
    gradient_magnitude_2d,
    load_a4_model,
    load_cache_lookup,
    load_cached_input,
    load_full_reliability_model,
    load_ssdu_model,
    make_holdout_target,
    make_loader,
    prepare_batch,
)
from fourway_mri.sensitivity import indices_to_torch_mask
from fourway_mri.ssdu_dataset import load_mask_records


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


def fit_affine(z, y, eps=1e-8):
    z = z.astype(np.float64)
    y = y.astype(np.float64)

    z_mean = z.mean()
    y_mean = y.mean()

    var_z = np.mean((z - z_mean) ** 2)

    if var_z < eps:
        return 0.0, float(y_mean)

    cov = np.mean((z - z_mean) * (y - y_mean))

    a = cov / (var_z + eps)
    b = y_mean - a * z_mean

    return float(a), float(b)


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]

    if scores.size == 0:
        raise ValueError("No finite conformal scores.")

    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)

    return float(np.sort(scores)[rank - 1])


def evaluate_interval(y, pred_scaled, qhat, alpha):
    lower = pred_scaled - qhat
    upper = pred_scaled + qhat

    covered = (y >= lower) & (y <= upper)

    return {
        "target_coverage": float(1.0 - alpha),
        "coverage": float(np.mean(covered)),
        "interval_width": float(2.0 * qhat),
        "mae": float(np.mean(np.abs(pred_scaled - y))),
        "mse": float(np.mean((pred_scaled - y) ** 2)),
        "pearson": safe_pearson(y, pred_scaled),
        "spearman": safe_spearman(y, pred_scaled),
    }


def evaluate_top_risk(y, pred_scaled, top_fraction=0.10):
    threshold = np.quantile(y, 1.0 - top_fraction)
    labels = (y >= threshold).astype(np.int32)

    try:
        auroc = float(roc_auc_score(labels, pred_scaled))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(labels, pred_scaled))
    except ValueError:
        auprc = np.nan

    return {
        "top_fraction": float(top_fraction),
        "top_threshold": float(threshold),
        "top_risk_auroc": auroc,
        "top_risk_auprc": auprc,
    }


def collect_pixels_for_split(
    cfg,
    split,
    data_root,
    cache_lookup,
    mask_records,
    ssdu_model,
    full_model,
    a4_model,
    device,
):
    loader = make_loader(cfg, split, data_root)

    rng = np.random.default_rng(int(cfg["conformal"]["random_seed"]) + len(split))
    pixels_per_slice = int(cfg["conformal"]["pixels_per_slice"])

    rows = []

    for batch in tqdm(loader, desc=f"collect {split}"):
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

        score_maps = {
            "reliability_a4_image_only": pred_a4[0, 0].detach().cpu().numpy().astype(np.float32),
            "reliability_full": pred_full[0, 0].detach().cpu().numpy().astype(np.float32),
            "image_gradient": gradient_magnitude_2d(x_np[0]),
            "xhat_magnitude": x_np[0],
            "psf_gain_channel": x_np[5],
            "intervention_magnitude": x_np[2],
        }

        flat_y = y_np.reshape(-1)
        n = flat_y.size
        k = min(pixels_per_slice, n)
        idx = rng.choice(n, size=k, replace=False)

        for j in idx:
            row = {
                "split": split,
                "volume_id": volume_id,
                "filename": filename,
                "slice_idx": slice_idx,
                "target": float(flat_y[j]),
            }

            for name, smap in score_maps.items():
                row[name] = float(smap.reshape(-1)[j])

            rows.append(row)

    return pd.DataFrame(rows)


def split_calibration_volumes(cal_df, fit_fraction, seed):
    volumes = sorted(cal_df["volume_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(volumes)

    n_fit = int(round(len(volumes) * fit_fraction))
    n_fit = min(max(n_fit, 1), len(volumes) - 1)

    fit_vols = set(volumes[:n_fit])
    conf_vols = set(volumes[n_fit:])

    fit_df = cal_df[cal_df["volume_id"].isin(fit_vols)].copy()
    conf_df = cal_df[cal_df["volume_id"].isin(conf_vols)].copy()

    return fit_df, conf_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp010_conformal_bounds.yaml")
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

    cal_df = collect_pixels_for_split(
        cfg=cfg,
        split="calibration",
        data_root=args.data_root,
        cache_lookup=cache_lookup,
        mask_records=mask_records,
        ssdu_model=ssdu_model,
        full_model=full_model,
        a4_model=a4_model,
        device=device,
    )

    test_df = collect_pixels_for_split(
        cfg=cfg,
        split="test",
        data_root=args.data_root,
        cache_lookup=cache_lookup,
        mask_records=mask_records,
        ssdu_model=ssdu_model,
        full_model=full_model,
        a4_model=a4_model,
        device=device,
    )

    fit_df, conf_df = split_calibration_volumes(
        cal_df,
        fit_fraction=float(cfg["data"]["calibration_volume_fit_fraction"]),
        seed=int(cfg["conformal"]["random_seed"]),
    )

    alpha = float(cfg["conformal"]["alpha"])
    predictors = cfg["predictors"]

    rows = []
    volume_rows = []

    for predictor in predictors:
        a, b = fit_affine(
            fit_df[predictor].to_numpy(),
            fit_df["target"].to_numpy(),
            eps=float(cfg["normalization"]["epsilon"]),
        )

        conf_pred = a * conf_df[predictor].to_numpy() + b
        conf_y = conf_df["target"].to_numpy()
        scores = np.abs(conf_y - conf_pred)

        qhat = conformal_quantile(scores, alpha=alpha)

        test_pred = a * test_df[predictor].to_numpy() + b
        test_y = test_df["target"].to_numpy()

        interval_metrics = evaluate_interval(test_y, test_pred, qhat, alpha=alpha)
        top_metrics = evaluate_top_risk(test_y, test_pred, top_fraction=0.10)

        row = {
            "predictor": predictor,
            "affine_a": a,
            "affine_b": b,
            "qhat": qhat,
            "num_scale_pixels": int(len(fit_df)),
            "num_conformal_pixels": int(len(conf_df)),
            "num_test_pixels": int(len(test_df)),
        }
        row.update(interval_metrics)
        row.update(top_metrics)
        rows.append(row)

        tmp = test_df[["volume_id", "target", predictor]].copy()
        tmp["pred_scaled"] = a * tmp[predictor].to_numpy() + b
        tmp["covered"] = (
            (tmp["target"].to_numpy() >= tmp["pred_scaled"].to_numpy() - qhat)
            &
            (tmp["target"].to_numpy() <= tmp["pred_scaled"].to_numpy() + qhat)
        )

        for volume_id, g in tmp.groupby("volume_id"):
            volume_rows.append({
                "predictor": predictor,
                "volume_id": volume_id,
                "coverage": float(g["covered"].mean()),
                "mae": float(np.mean(np.abs(g["target"].to_numpy() - g["pred_scaled"].to_numpy()))),
                "num_pixels": int(len(g)),
            })

    result_df = pd.DataFrame(rows).sort_values("interval_width", ascending=True)
    volume_df = pd.DataFrame(volume_rows)

    result_df.to_csv(output_dir / "conformal_summary.csv", index=False)
    volume_df.to_csv(output_dir / "conformal_volume_metrics.csv", index=False)
    cal_df.to_csv(output_dir / "sampled_calibration_pixels.csv", index=False)
    test_df.to_csv(output_dir / "sampled_test_pixels.csv", index=False)

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "alpha": alpha,
        "target_coverage": float(1.0 - alpha),
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "lambda_hold_used": True,
        "predictors": predictors,
        "best_by_interval_width": result_df.iloc[0].to_dict(),
        "outputs": {
            "conformal_summary_csv": str(output_dir / "conformal_summary.csv"),
            "conformal_volume_metrics_csv": str(output_dir / "conformal_volume_metrics.csv"),
            "sampled_calibration_pixels_csv": str(output_dir / "sampled_calibration_pixels.csv"),
            "sampled_test_pixels_csv": str(output_dir / "sampled_test_pixels.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
