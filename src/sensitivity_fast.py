#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: sensitivity_fast.py | Topic: 9 | Purpose: Phase 7 sensitivity analyses (fast)
Strategy: skips expensive per-window GC; uses WC + Pearson only for sensitivity configs.
For EEG feature variants (7.1): replaces ADR with SEF95/alpha_rel/spectral_entropy.
For time resolution (7.2): re-aggregates existing 10s features to coarser windows.
For BIS value (7.3): uses BIS_value as EEG proxy.
All variants use the same MAP+20/BIS<60 event windows from windowed_ncci_events.csv.
"""
import logging, warnings, gc
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr, ttest_1samp
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pywt

warnings.filterwarnings('ignore')

RAW_DATA    = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK        = Path(".")  # repository root; override as needed
OUT_FEATURES = WORK / "outputs" / "features"
OUT_METRICS  = WORK / "outputs" / "metrics"
OUT_FIGS     = WORK / "outputs" / "figures"
OUT_LOGS     = WORK / "outputs" / "logs"

COLORS = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'gray': '#999999',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

GC_WIN          = 60
GC_STEP         = 3
EVENT_MAP_RISE  = 20
EVENT_LOOKBACK_W = 9
EVENT_BIS_MAX   = 60


def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)


def wc_scalar_fast(x_win, y_win, dt=10):
    """Fast wavelet coherence scalar — skips GC to save time."""
    n = min(len(x_win), len(y_win))
    if n < 10:
        return np.nan
    x = np.array(x_win[:n], float)
    y = np.array(y_win[:n], float)
    for arr in [x, y]:
        nans = np.isnan(arr)
        if nans.any() and (~nans).sum() > 3:
            arr[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], arr[~nans])
        elif nans.any():
            return np.nan
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)
    scales = np.arange(2, min(n // 2, 30), 2)  # fewer scales → faster
    if len(scales) < 3:
        return np.nan
    try:
        cx, _ = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=dt)
        cy, _ = pywt.cwt(y, scales, 'cmor1.5-1.0', sampling_period=dt)
        cross = cx * np.conj(cy)
        sm = (3, 5)
        sr = uniform_filter(cross.real, sm)
        si = uniform_filter(cross.imag, sm)
        sx = uniform_filter(np.abs(cx)**2, sm)
        sy = uniform_filter(np.abs(cy)**2, sm)
        coh = np.clip((sr**2 + si**2) / (sx * sy + 1e-10), 0, 1)
        return float(np.nanmean(coh))
    except Exception:
        return np.nan


def compute_case_sensitivity(args):
    """
    Compute NCCI (WC + Pearson only, fast) for a single case and EEG config.
    Returns case-level stats + event responses.
    """
    caseid, eeg_col, win_size, win_step = args
    feat_path = OUT_FEATURES / f"case_{caseid:04d}_features.parquet"
    if not feat_path.exists():
        return None
    try:
        df = pd.read_parquet(feat_path)
        n = len(df)
        if n < win_size + 10 or eeg_col not in df.columns:
            return None

        eeg_v = df[eeg_col].values.astype(float)
        map_v = df['MAP_mean'].values.astype(float)
        bis_v = df['BIS_value'].values.astype(float) if 'BIS_value' in df.columns else np.full(n, np.nan)

        # Build windowed WC + Pearson (no GC)
        win_rows = []
        for start in range(0, n - win_size, win_step):
            end = start + win_size
            ew = eeg_v[start:end]
            hw = map_v[start:end]
            wc = wc_scalar_fast(ew, hw, dt=10)
            valid = ~(np.isnan(ew) | np.isnan(hw))
            r = pearsonr(ew[valid], hw[valid])[0] if valid.sum() > 5 else np.nan
            win_rows.append({'window_center': start + win_size // 2, 'WC': wc, 'pearson_r': r})

        if not win_rows:
            return None

        win_df = pd.DataFrame(win_rows)
        cols = ['WC', 'pearson_r']
        valid_rows = win_df[cols].dropna()
        if len(valid_rows) < 5:
            return None

        X = StandardScaler().fit_transform(valid_rows)
        pc1 = PCA(n_components=1).fit_transform(X).flatten()
        ncci = MinMaxScaler().fit_transform(pc1.reshape(-1, 1)).flatten()
        if np.corrcoef(ncci, valid_rows['WC'].values)[0, 1] < 0:
            ncci = 1 - ncci
        win_df.loc[valid_rows.index, 'NCCI_window'] = ncci

        # Event detection (fixed thresholds)
        events = []
        for i in range(EVENT_LOOKBACK_W, n):
            if np.isnan(map_v[i]) or np.isnan(map_v[i - EVENT_LOOKBACK_W]):
                continue
            if map_v[i] - map_v[i - EVENT_LOOKBACK_W] > EVENT_MAP_RISE:
                bis_now = bis_v[i] if not np.isnan(bis_v[i]) else 99
                if bis_now < EVENT_BIS_MAX:
                    events.append(i)

        event_responses = []
        if events and 'NCCI_window' in win_df.columns:
            pre_w, post_w = 3, 3
            for t in events:
                pre_wins  = win_df[win_df['window_center'] <= t - win_size // 2].tail(pre_w)
                post_wins = win_df[win_df['window_center'] >  t + win_size // 4].head(post_w)
                if len(pre_wins) == 0 or len(post_wins) == 0:
                    continue
                pre_ncci  = pre_wins['NCCI_window'].mean() if 'NCCI_window' in pre_wins.columns else np.nan
                post_ncci = post_wins['NCCI_window'].mean() if 'NCCI_window' in post_wins.columns else np.nan
                if not np.isnan(pre_ncci) and not np.isnan(post_ncci):
                    event_responses.append({'pre_ncci': pre_ncci, 'post_ncci': post_ncci,
                                            'delta_ncci': post_ncci - pre_ncci})

        del df, win_df
        gc.collect()
        return {
            'caseid': caseid,
            'NCCI_mean': float(np.nanmean(ncci)),
            'WC_mean': float(valid_rows['WC'].mean()),
            'pearson_mean': float(valid_rows['pearson_r'].mean()),
            'n_events': len(events),
            'event_responses': event_responses,
        }
    except Exception:
        return None


def run_config(label, eeg_col, win_size, win_step, case_ids, n_workers=4, batch_size=50):
    logging.info(f"  Config '{label}': eeg={eeg_col}, win={win_size}, step={win_step}")
    args = [(cid, eeg_col, win_size, win_step) for cid in case_ids]
    all_results = []

    for b_start in range(0, len(args), batch_size):
        batch = args[b_start:b_start + batch_size]
        with Pool(processes=n_workers) as pool:
            results = pool.map(compute_case_sensitivity, batch)
        all_results.extend([r for r in results if r is not None])
        gc.collect()

    all_events = []
    for res in all_results:
        all_events.extend(res.get('event_responses', []))

    if not all_events:
        logging.warning(f"  No events found for '{label}'")
        return {'label': label, 'n_cases': len(all_results), 'n_events': 0,
                'NCCI_mean': np.nan, 'WC_mean': np.nan, 'pearson_mean': np.nan,
                'delta_ncci_mean': np.nan, 'delta_ncci_std': np.nan,
                'pct_decrease': np.nan, 'pval': np.nan, 't_stat': np.nan, 'effect_d': np.nan}

    ev_df = pd.DataFrame(all_events)
    pre_v   = ev_df['pre_ncci'].dropna().values
    post_v  = ev_df['post_ncci'].dropna().values
    delta_v = ev_df['delta_ncci'].dropna().values
    n_paired = min(len(pre_v), len(post_v))
    t, p = (np.nan, np.nan)
    if n_paired > 5:
        t, p = ttest_rel(pre_v[:n_paired], post_v[:n_paired])

    effect_d = delta_v.mean() / (delta_v.std() + 1e-10)
    ncci_means = [r['NCCI_mean'] for r in all_results]
    wc_means   = [r['WC_mean']   for r in all_results]
    pr_means   = [r['pearson_mean'] for r in all_results]

    logging.info(f"    Cases: {len(all_results)}, Events: {len(all_events)}, "
                 f"ΔNCCI={delta_v.mean():.4f}±{delta_v.std():.4f}, p={p:.4f}")

    return {
        'label': label,
        'eeg_col': eeg_col,
        'win_size': win_size,
        'n_cases': len(all_results),
        'n_events': len(all_events),
        'NCCI_mean': float(np.nanmean(ncci_means)),
        'WC_mean': float(np.nanmean(wc_means)),
        'pearson_mean': float(np.nanmean(pr_means)),
        'delta_ncci_mean': float(delta_v.mean()),
        'delta_ncci_std': float(delta_v.std()),
        'pct_decrease': float((delta_v < 0).mean() * 100),
        'pval': float(p) if not np.isnan(p) else np.nan,
        't_stat': float(t) if not np.isnan(t) else np.nan,
        'effect_d': float(effect_d),
    }


def plot_sensitivity_figure(results_df, save_path):
    """Supp Fig 2: Sensitivity analysis comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels   = results_df['label'].tolist()
    means    = results_df['delta_ncci_mean'].values
    stds     = results_df['delta_ncci_std'].values
    pvals    = results_df['pval'].values
    n_ev     = results_df['n_events'].values
    y_pos    = np.arange(len(labels))
    primary  = 0

    # Panel A: ΔNCCI forest plot
    ax = axes[0]
    for i, (m, s, p, n, lbl) in enumerate(zip(means, stds, pvals, n_ev, labels)):
        color = COLORS['blue'] if i == primary else COLORS['gray']
        lw = 2 if i == primary else 1
        se = s / np.sqrt(max(n, 1))
        ax.errorbar(m, i, xerr=se, fmt='o', color=color,
                    ms=8 if i == primary else 5, lw=lw, capsize=3)
        p_str = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(0.97, i, p_str, va='center', ha='right', fontsize=8, transform=ax.get_yaxis_transform(),
                color='red' if p < 0.05 else COLORS['gray'])
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('ΔNCCI (Post − Pre events)', fontsize=9)
    ax.set_title('ΔNCCI: Sensitivity\nAcross Configurations', fontweight='bold')
    ax.text(0.02, 0.02, 'Primary in blue', transform=ax.transAxes, fontsize=7, color='gray')

    # Panel B: % NCCI decrease
    ax2 = axes[1]
    pct = results_df['pct_decrease'].values
    colors_b = [COLORS['blue'] if i == primary else COLORS['gray'] for i in range(len(labels))]
    ax2.barh(y_pos, pct, color=colors_b, alpha=0.75, edgecolor='white', height=0.6)
    ax2.axvline(50, color='black', ls='--', lw=1, label='50% line')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_xlabel('% Events with NCCI↓', fontsize=9)
    ax2.set_title('Proportion of Events\nShowing NCCI Decrease', fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(fontsize=7)

    # Panel C: NCCI mean ± SD across cases
    ax3 = axes[2]
    ncci_m = results_df['NCCI_mean'].values
    wc_m   = results_df['WC_mean'].values
    pr_m   = results_df['pearson_mean'].values
    x = np.arange(len(labels))
    ax3.scatter(x, ncci_m, label='NCCI', color=COLORS['blue'], s=60, zorder=3)
    ax3.scatter(x, wc_m,   label='WC',   color=COLORS['orange'], marker='s', s=50, zorder=3)
    ax3.scatter(x, pr_m,   label='Pearson r', color=COLORS['green'], marker='^', s=50, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Mean value (cohort)', fontsize=9)
    ax3.set_title('Coupling Metrics Across\nSensitivity Configurations', fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.set_ylim(0, 1)

    plt.suptitle('Sensitivity Analysis: NCCI Robustness Across EEG Features & Window Sizes',
                 fontweight='bold', y=1.01, fontsize=10)
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Supp Fig 2 → {Path(save_path).name}")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"sensitivity_fast_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info("=== Phase 7: Sensitivity Analysis (Fast, WC+Pearson NCCI) ===")

    feat_files = sorted(OUT_FEATURES.glob("case_*.parquet"))
    all_case_ids = [int(f.stem.split('_')[1]) for f in feat_files]
    logging.info(f"Total cases: {len(all_case_ids)}")

    # Fixed sample of 200 cases (sufficient for sensitivity analysis)
    np.random.seed(42)
    sample_ids = np.random.choice(all_case_ids, min(200, len(all_case_ids)), replace=False).tolist()
    logging.info(f"Sensitivity sample: {len(sample_ids)} cases")

    # Configurations:
    # Primary: ADR, 60-window (10min), step 3 (30s)
    # 7.1 EEG variants: same window sizes
    # 7.2 Time resolution: 12-window (2min) & 180-window (30min)
    # 7.3 BIS value
    configs = [
        # label,                        eeg_col,            win, step
        ('Primary (ADR)',               'ADR',              60,  3),
        ('7.1a: SEF95',                'SEF95',            60,  3),
        ('7.1b: Alpha power (rel)',     'alpha_rel',        60,  3),
        ('7.1c: Spectral entropy',      'spectral_entropy', 60,  3),
        ('7.2a: Short win (2 min)',     'ADR',              12,  1),
        ('7.2b: Long win (30 min)',     'ADR',              180, 9),
        ('7.3: BIS value',             'BIS_value',        60,  3),
    ]

    # Verify columns
    sample_df = pd.read_parquet(OUT_FEATURES / f"case_{sample_ids[0]:04d}_features.parquet")
    avail = sample_df.columns.tolist()
    del sample_df

    valid_configs = [(lbl, col, win, stp) for lbl, col, win, stp in configs if col in avail]
    skipped = [lbl for lbl, col, win, stp in configs if col not in avail]
    if skipped:
        logging.warning(f"Skipping configs (column not found): {skipped}")
    logging.info(f"Running {len(valid_configs)} configs")

    results = []
    for label, eeg_col, win_size, win_step in valid_configs:
        res = run_config(label, eeg_col, win_size, win_step, sample_ids, n_workers=4, batch_size=50)
        results.append(res)

    results_df = pd.DataFrame(results)

    out_table = OUT_METRICS / "supp_table1_sensitivity.csv"
    assert not str(out_table.resolve()).startswith(str(RAW_DATA.resolve()))
    results_df.to_csv(out_table, index=False)
    logging.info(f"\nSaved Supp Table 1 → supp_table1_sensitivity.csv")

    logging.info("\n=== Sensitivity Summary ===")
    for _, row in results_df.iterrows():
        p_str = f"p={row['pval']:.4f}" if not np.isnan(row['pval']) else "p=N/A"
        logging.info(f"  {row['label']:38s}  ΔNCCI={row['delta_ncci_mean']:+.4f}  "
                     f"pct_dec={row['pct_decrease']:.0f}%  {p_str}  n_ev={row['n_events']}")

    if len(results_df) > 1:
        plot_sensitivity_figure(results_df, OUT_FIGS / "supp_fig2_sensitivity.png")

    logging.info("=== Phase 7 Complete ===")


if __name__ == "__main__":
    main()
