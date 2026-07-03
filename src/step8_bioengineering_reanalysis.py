#!/home/lxk/anaconda3/envs/ana/bin/python
"""Bioengineering revision analysis for Topic 9.

This script replaces the original densely sampled excursion analysis with a
patient-split, leakage-resistant design:

1. Fit one fixed scaler/PCA coupling score in a development half of cases.
2. Apply that frozen transform to a held-out validation half.
3. Detect only threshold-crossing onsets and enforce a refractory interval.
4. Compare strictly non-overlapping 10-minute pre/post windows.
5. Use the patient mean change as the primary unit of inference.

It does not overwrite any original Topic 9 output.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


ROOT = Path("/home/lxk/vitaldb/analysis/topic9")
FEATURE_DIR = ROOT / "outputs" / "features"
WINDOW_DIR = ROOT / "outputs" / "ncci"
OUT = ROOT / "outputs" / "bioengineering_v2"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20260629
STEP_SECONDS = 10
COUPLING_WINDOW_POINTS = 60       # 10 min at a 10-s feature step
HALF_WINDOW_POINTS = 30
BASE_FEATURE_WINDOW_POINTS = 3    # 30-s raw-signal window at a 10-s step
# A 60-row coupling window starts at center-30. Its last feature row starts
# at center+29 and uses 30 s of raw signal, so raw support ends at center+32.
RIGHT_SUPPORT_POINTS = HALF_WINDOW_POINTS + BASE_FEATURE_WINDOW_POINTS - 1
EVENT_LOOKBACK_POINTS = 9         # 90 s
PRIMARY_REFRACTORY_POINTS = 120   # 20 min
SCORE_COLUMNS = ["GC_e2h", "WC", "pearson_r"]


def case_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def balanced_rows(frame: pd.DataFrame, maximum: int = 200) -> np.ndarray:
    valid = frame[SCORE_COLUMNS].dropna()
    if len(valid) <= maximum:
        return valid.to_numpy(float)
    idx = np.linspace(0, len(valid) - 1, maximum).round().astype(int)
    return valid.iloc[idx].to_numpy(float)


def fit_frozen_score(development_ids):
    rows = []
    contributing_cases = 0
    for caseid in development_ids:
        path = WINDOW_DIR / f"case_{caseid:04d}_ncci.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        values = balanced_rows(frame)
        if len(values):
            rows.append(values)
            contributing_cases += 1
    matrix = np.vstack(rows)
    # GC F statistics have a severe right tail. Estimate the cap using only
    # development data, then freeze it together with the scaler/PCA.
    gc_upper = float(np.quantile(matrix[:, 0], 0.99))
    matrix[:, 0] = np.minimum(matrix[:, 0], gc_upper)
    scaler = StandardScaler().fit(matrix)
    pca = PCA(n_components=1, random_state=SEED).fit(scaler.transform(matrix))
    sign = 1.0 if pca.components_[0, 1] >= 0 else -1.0
    return scaler, pca, sign, gc_upper, matrix.shape[0], contributing_cases


def add_frozen_score(frame, scaler, pca, sign, gc_upper):
    out = frame.copy()
    out["GCS"] = np.nan
    valid = out[SCORE_COLUMNS].notna().all(axis=1)
    if valid.any():
        values = out.loc[valid, SCORE_COLUMNS].to_numpy(float)
        values[:, 0] = np.minimum(values[:, 0], gc_upper)
        z = scaler.transform(values)
        out.loc[valid, "GCS"] = sign * pca.transform(z).ravel()
    return out


def event_onsets(features, map_rise=20, bis_max=60,
                 refractory_points=PRIMARY_REFRACTORY_POINTS):
    # Use the 10-s numeric monitor MAP for causal event timing. A trailing
    # 30-s median suppresses isolated monitor artefacts without future data.
    map_values = features["MAP_monitor"].to_numpy(float)
    map_values[(map_values < 20) | (map_values > 300)] = np.nan
    map_values = pd.Series(map_values).rolling(3, min_periods=2).median().to_numpy()
    bis_values = features["BIS_value"].to_numpy(float)
    condition = np.zeros(len(features), dtype=bool)
    delta = np.full(len(features), np.nan)
    for i in range(EVENT_LOOKBACK_POINTS, len(features)):
        if np.isnan(map_values[i]) or np.isnan(map_values[i - EVENT_LOOKBACK_POINTS]):
            continue
        delta[i] = map_values[i] - map_values[i - EVENT_LOOKBACK_POINTS]
        condition[i] = delta[i] > map_rise and not np.isnan(bis_values[i]) and bis_values[i] < bis_max

    raw_onsets = np.flatnonzero(condition & ~np.r_[False, condition[:-1]])
    selected = []
    for idx in raw_onsets:
        if not selected or idx - selected[-1] >= refractory_points:
            selected.append(int(idx))
    return selected, delta


def score_events(caseid, scored_windows, features, map_rise=20, bis_max=60,
                 refractory_points=PRIMARY_REFRACTORY_POINTS):
    onsets, map_delta = event_onsets(features, map_rise, bis_max, refractory_points)
    rows = []
    for event_index, onset in enumerate(onsets, start=1):
        # Coupling uses 60 overlapping 30-s base features. These support-aware
        # rules ensure the underlying raw signal does not cross the event time.
        pre = scored_windows[
            scored_windows["window_center"] + RIGHT_SUPPORT_POINTS <= onset
        ].dropna(subset=["GCS"]).tail(1)
        post = scored_windows[
            scored_windows["window_center"] - HALF_WINDOW_POINTS >= onset
        ].dropna(subset=["GCS"]).head(1)
        if pre.empty or post.empty:
            continue
        pre_row = pre.iloc[0]
        post_row = post.iloc[0]
        rows.append({
            "caseid": caseid,
            "event_index": event_index,
            "onset_feature_index": onset,
            "onset_time_min": onset * STEP_SECONDS / 60,
            "map_rise_mmHg": float(map_delta[onset]),
            "bis_at_onset": float(features.iloc[onset]["BIS_value"]),
            "pre_center": int(pre_row["window_center"]),
            "post_center": int(post_row["window_center"]),
            "pre_GCS": float(pre_row["GCS"]),
            "post_GCS": float(post_row["GCS"]),
            "delta_GCS": float(post_row["GCS"] - pre_row["GCS"]),
        })
    return rows


def analyse_event_rows(rows, label):
    events = pd.DataFrame(rows)
    if events.empty:
        return None, None, None
    patient = events.groupby("caseid", as_index=False).agg(
        n_events=("delta_GCS", "size"),
        mean_pre_GCS=("pre_GCS", "mean"),
        mean_post_GCS=("post_GCS", "mean"),
        mean_delta_GCS=("delta_GCS", "mean"),
    )
    values = patient["mean_delta_GCS"].to_numpy(float)
    t_result = ttest_1samp(values, 0.0)
    try:
        w_result = wilcoxon(values)
        wilcoxon_p = float(w_result.pvalue)
    except ValueError:
        wilcoxon_p = np.nan

    rng = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(5000)
    ])
    first = events.sort_values(["caseid", "onset_feature_index"]).groupby("caseid").first()
    first_values = first["delta_GCS"].to_numpy(float)
    first_t = ttest_1samp(first_values, 0.0)

    gee = sm.GEE.from_formula(
        "delta_GCS ~ 1", groups="caseid", data=events,
        family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable()
    ).fit()

    summary = {
        "label": label,
        "n_validation_cases_with_events": int(len(patient)),
        "n_events": int(len(events)),
        "events_per_patient_median": float(patient["n_events"].median()),
        "events_per_patient_q1": float(patient["n_events"].quantile(0.25)),
        "events_per_patient_q3": float(patient["n_events"].quantile(0.75)),
        "mean_pre_GCS": float(patient["mean_pre_GCS"].mean()),
        "mean_post_GCS": float(patient["mean_post_GCS"].mean()),
        "mean_delta_GCS": float(values.mean()),
        "sd_delta_GCS": float(values.std(ddof=1)),
        "cohens_d": float(values.mean() / values.std(ddof=1)),
        "t_p": float(t_result.pvalue),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "wilcoxon_p": wilcoxon_p,
        "pct_patient_negative": float(100 * np.mean(values < 0)),
        "first_event_mean_delta": float(first_values.mean()),
        "first_event_t_p": float(first_t.pvalue),
        "gee_beta": float(gee.params.iloc[0]),
        "gee_ci_low": float(gee.conf_int().iloc[0, 0]),
        "gee_ci_high": float(gee.conf_int().iloc[0, 1]),
        "gee_p": float(gee.pvalues.iloc[0]),
    }
    return summary, events, patient


def collect_analysis(validation_ids, scaler, pca, sign, gc_upper,
                     map_rise=20, bis_max=60,
                     refractory_points=PRIMARY_REFRACTORY_POINTS):
    rows = []
    for caseid in validation_ids:
        window_path = WINDOW_DIR / f"case_{caseid:04d}_ncci.parquet"
        feature_path = FEATURE_DIR / f"case_{caseid:04d}_features.parquet"
        if not window_path.exists() or not feature_path.exists():
            continue
        windows = add_frozen_score(
            pd.read_parquet(window_path), scaler, pca, sign, gc_upper
        )
        features = pd.read_parquet(feature_path)
        rows.extend(score_events(
            caseid, windows, features, map_rise, bis_max, refractory_points
        ))
    label = f"MAP>{map_rise}, BIS<{bis_max}, refractory={refractory_points * STEP_SECONDS // 60}min"
    return analyse_event_rows(rows, label)


def make_figure(events, patient, sensitivity, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(patient["mean_pre_GCS"], patient["mean_post_GCS"],
                    s=12, alpha=0.35, color="#2878B5")
    low = min(patient["mean_pre_GCS"].min(), patient["mean_post_GCS"].min())
    high = max(patient["mean_pre_GCS"].max(), patient["mean_post_GCS"].max())
    axes[0].plot([low, high], [low, high], "--", color="0.35", linewidth=1)
    axes[0].set(xlabel="Patient mean pre-event EHCS", ylabel="Patient mean post-event EHCS",
                title="A  Strict non-overlapping windows")

    values = patient["mean_delta_GCS"].to_numpy(float)
    axes[1].hist(values, bins=40, color="#5AB4AC", edgecolor="white")
    axes[1].axvline(0, color="0.2", linestyle="--", linewidth=1)
    axes[1].axvline(values.mean(), color="#D73027", linewidth=1.5)
    axes[1].set(xlabel="Patient mean $\\Delta$EHCS", ylabel="Number of patients",
                title="B  Patient-level primary outcome")

    sens = sensitivity.copy().sort_values("order", ascending=False)
    y = np.arange(len(sens))
    axes[2].errorbar(
        sens["mean_delta_GCS"], y,
        xerr=[sens["mean_delta_GCS"] - sens["bootstrap_ci_low"],
              sens["bootstrap_ci_high"] - sens["mean_delta_GCS"]],
        fmt="o", color="#7B3294", capsize=3
    )
    axes[2].axvline(0, color="0.2", linestyle="--", linewidth=1)
    axes[2].set_yticks(y, sens["short_label"])
    axes[2].set(xlabel="Mean $\\Delta$EHCS (95% patient-bootstrap CI)",
                title="C  Sensitivity analyses")

    fig.suptitle("Held-Out Evaluation of Coupling-Score Response at De-Clustered Excursions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    caseids = sorted(case_id_from_path(p) for p in WINDOW_DIR.glob("case_*_ncci.parquet"))
    rng = np.random.default_rng(SEED)
    shuffled = np.array(caseids, dtype=int)
    rng.shuffle(shuffled)
    split = len(shuffled) // 2
    development_ids = sorted(shuffled[:split].tolist())
    validation_ids = sorted(shuffled[split:].tolist())

    scaler, pca, sign, gc_upper, n_fit_windows, n_fit_cases = fit_frozen_score(development_ids)
    signed_loadings = sign * pca.components_[0]

    primary, events, patient = collect_analysis(
        validation_ids, scaler, pca, sign, gc_upper,
        20, 60, PRIMARY_REFRACTORY_POINTS
    )
    if primary is None:
        raise RuntimeError("Primary analysis returned no eligible events")

    specifications = [
        (20, 60, 120, "Primary: MAP>20, BIS<60, 20 min", 0),
        (25, 60, 120, "MAP>25, BIS<60, 20 min", 1),
        (30, 60, 120, "MAP>30, BIS<60, 20 min", 2),
        (20, 55, 120, "MAP>20, BIS<55, 20 min", 3),
        (20, 60, 60, "MAP>20, BIS<60, 10 min", 4),
        (20, 60, 180, "MAP>20, BIS<60, 30 min", 5),
    ]
    sensitivity_rows = []
    for map_rise, bis_max, refractory, short_label, order in specifications:
        if map_rise == 20 and bis_max == 60 and refractory == 120:
            result = primary
        else:
            result, _, _ = collect_analysis(
                validation_ids, scaler, pca, sign, gc_upper,
                map_rise, bis_max, refractory
            )
        result["short_label"] = short_label
        result["order"] = order
        sensitivity_rows.append(result)
    sensitivity = pd.DataFrame(sensitivity_rows)

    split_frame = pd.DataFrame({
        "caseid": development_ids + validation_ids,
        "split": ["development"] * len(development_ids) + ["validation"] * len(validation_ids),
    }).sort_values("caseid")
    split_frame.to_csv(METRICS / "case_split.csv", index=False)
    events.to_csv(METRICS / "primary_events.csv", index=False)
    patient.to_csv(METRICS / "primary_patient_summary.csv", index=False)
    sensitivity.to_csv(METRICS / "sensitivity_analysis.csv", index=False)

    metadata = {
        "seed": SEED,
        "n_total_cases": len(caseids),
        "n_development_cases": len(development_ids),
        "n_validation_cases": len(validation_ids),
        "n_development_cases_contributing_windows": n_fit_cases,
        "n_balanced_development_windows": n_fit_windows,
        "pca_explained_variance": float(pca.explained_variance_ratio_[0]),
        "development_gc_99th_percentile_cap": gc_upper,
        "signed_loadings": dict(zip(SCORE_COLUMNS, map(float, signed_loadings))),
        "scaler_mean": dict(zip(SCORE_COLUMNS, map(float, scaler.mean_))),
        "scaler_scale": dict(zip(SCORE_COLUMNS, map(float, scaler.scale_))),
        "primary": primary,
    }
    (METRICS / "analysis_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_figure(events, patient, sensitivity, FIGURES / "fig5_validation_event_analysis.png")
    print(json.dumps(metadata, indent=2))
    print(sensitivity[[
        "short_label", "n_validation_cases_with_events", "n_events",
        "mean_delta_GCS", "bootstrap_ci_low", "bootstrap_ci_high", "t_p"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
