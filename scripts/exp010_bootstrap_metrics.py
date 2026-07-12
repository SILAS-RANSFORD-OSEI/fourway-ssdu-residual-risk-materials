import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


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


def compute_metrics(y, pred, top_fraction):
    y = y.astype(np.float64)
    pred = pred.astype(np.float64)

    threshold = np.quantile(y, 1.0 - top_fraction)
    labels = (y >= threshold).astype(np.int32)

    try:
        auroc = float(roc_auc_score(labels, pred))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(labels, pred))
    except ValueError:
        auprc = np.nan

    diff = pred - y

    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff**2)),
        "pearson": safe_pearson(y, pred),
        "spearman": safe_spearman(y, pred),
        "top_threshold": float(threshold),
        "top_risk_auroc": auroc,
        "top_risk_auprc": auprc,
    }


def percentile_ci(values, confidence_level):
    values = np.asarray(values, dtype=np.float64)
    alpha = 1.0 - confidence_level

    lo = 100.0 * alpha / 2.0
    hi = 100.0 * (1.0 - alpha / 2.0)

    return float(np.nanpercentile(values, lo)), float(np.nanpercentile(values, hi))


def summarize_bootstrap(boot_df, confidence_level):
    rows = []

    metrics = [
        "mae",
        "mse",
        "pearson",
        "spearman",
        "top_risk_auroc",
        "top_risk_auprc",
    ]

    for predictor, g in boot_df.groupby("predictor"):
        row = {
            "predictor": predictor,
            "n_bootstrap": int(g["bootstrap_iter"].nunique()),
        }

        for metric in metrics:
            vals = g[metric].to_numpy()
            lo, hi = percentile_ci(vals, confidence_level)

            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_median"] = float(np.nanmedian(vals))
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_differences(boot_df, reference_predictor, comparisons, confidence_level):
    rows = []

    ref = boot_df[boot_df["predictor"] == reference_predictor].copy()

    for other in comparisons:
        other_df = boot_df[boot_df["predictor"] == other].copy()

        merged = ref.merge(
            other_df,
            on="bootstrap_iter",
            suffixes=("_ref", "_other"),
        )

        for metric in ["spearman", "top_risk_auroc", "top_risk_auprc", "mae", "mse"]:
            diff = merged[f"{metric}_ref"].to_numpy() - merged[f"{metric}_other"].to_numpy()

            lo, hi = percentile_ci(diff, confidence_level)

            # Two-sided paired bootstrap p-value against zero difference.
            # Plus-one correction avoids zero p-values.
            p_left = (np.sum(diff <= 0.0) + 1.0) / (diff.size + 1.0)
            p_right = (np.sum(diff >= 0.0) + 1.0) / (diff.size + 1.0)
            p_two_sided = float(min(1.0, 2.0 * min(p_left, p_right)))

            rows.append(
                {
                    "reference": reference_predictor,
                    "comparison": other,
                    "metric": metric,
                    "diff_mean_ref_minus_comparison": float(np.nanmean(diff)),
                    "diff_median_ref_minus_comparison": float(np.nanmedian(diff)),
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "prob_ref_greater": float(np.mean(diff > 0.0)),
                    "p_two_sided_bootstrap": p_two_sided,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp010_bootstrap_metrics.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["inputs"]["sampled_test_pixels"])

    required = {"volume_id", "target"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    predictors = cfg["predictors"]
    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Predictor column missing: {p}")

    volumes = sorted(df["volume_id"].unique().tolist())
    grouped = {v: g.copy() for v, g in df.groupby("volume_id")}

    rng = np.random.default_rng(int(cfg["bootstrap"]["random_seed"]))
    n_iter = int(cfg["bootstrap"]["n_iterations"])
    top_fraction = float(cfg["bootstrap"]["top_risk_fraction"])
    confidence_level = float(cfg["bootstrap"]["confidence_level"])

    boot_rows = []

    for b in tqdm(range(n_iter), desc="bootstrap"):
        sampled_volumes = rng.choice(volumes, size=len(volumes), replace=True)
        boot_df = pd.concat([grouped[v] for v in sampled_volumes], ignore_index=True)

        y = boot_df["target"].to_numpy()

        for predictor in predictors:
            pred = boot_df[predictor].to_numpy()
            metrics = compute_metrics(y, pred, top_fraction=top_fraction)

            row = {
                "bootstrap_iter": int(b),
                "predictor": predictor,
                "num_volumes_sampled": int(len(sampled_volumes)),
                "num_unique_volumes": int(len(set(sampled_volumes))),
                "num_pixels": int(len(boot_df)),
            }
            row.update(metrics)
            boot_rows.append(row)

    boot_df = pd.DataFrame(boot_rows)
    summary_df = summarize_bootstrap(boot_df, confidence_level=confidence_level)

    diff_df = summarize_differences(
        boot_df=boot_df,
        reference_predictor=cfg["reference_predictor"],
        comparisons=cfg["comparisons"],
        confidence_level=confidence_level,
    )

    boot_df.to_csv(output_dir / "bootstrap_samples.csv", index=False)
    summary_df.to_csv(output_dir / "bootstrap_summary.csv", index=False)
    diff_df.to_csv(output_dir / "bootstrap_differences.csv", index=False)

    best_auprc = summary_df.sort_values("top_risk_auprc_mean", ascending=False).iloc[0].to_dict()
    best_spearman = summary_df.sort_values("spearman_mean", ascending=False).iloc[0].to_dict()

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "n_iterations": n_iter,
        "num_test_volumes": int(len(volumes)),
        "confidence_level": confidence_level,
        "top_risk_fraction": top_fraction,
        "reference_predictor": cfg["reference_predictor"],
        "predictors": predictors,
        "best_by_auprc_mean": best_auprc,
        "best_by_spearman_mean": best_spearman,
        "outputs": {
            "bootstrap_samples_csv": str(output_dir / "bootstrap_samples.csv"),
            "bootstrap_summary_csv": str(output_dir / "bootstrap_summary.csv"),
            "bootstrap_differences_csv": str(output_dir / "bootstrap_differences.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
