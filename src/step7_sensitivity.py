#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step7_sensitivity.py | Topic: 9 | Purpose: Phase 7 sensitivity analyses
Runs three sensitivity checks:
  7.1 Different EEG features (SEF95, alpha_rel, spectral_entropy) replacing ADR
  7.2 Different time resolutions (60s/30s and 10s/5s windows)
  7.3 BIS value vs raw EEG-derived features
Outputs comparison tables and a supplementary figure (Supp Fig 2).
"""
import sys, logging, warnings, gc
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
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
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

# Default config (primary analysis)
DEFAULT_EEG_COL = 'ADR'
DEFAULT_GC_WIN  = 60
DEFAULT_GC_STEP = 3
MAX_LAG         = 5
EVENT_MAP_RISE  = 20
EVENT_LOOKBACK_W = 9
EVENT_BIS_MAX   = 60


def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)


def is_stationary(x):
    if len(x) < 20 or np.std(x) < 1e-10:
        return True
    try:
        return adfuller(x, autolag='AIC')[1] < 0.05
    except Exception:
        return True


def granger_f(eeg_win, hemo_win, max_lag=MAX_LAG):
    valid = ~(np.isnan(eeg_win) | np.isnan(hemo_win))
    if valid.sum() < max_lag * 4:
        return np.nan, np.nan
    e = eeg_win[valid]
    h = hemo_win[valid]
    e = np.diff(e) if not is_stationary(e) else e
    h = np.diff(h) if not is_stationary(h) else h
    n = min(len(e), len(h))
    if n < max_lag * 3:
        return np.nan, np.nan
    try:
        gc_eh = grangercausalitytests(np.column_stack([h[:n], e[:n]]), maxlag=max_lag, verbose=False)
        beh = min(gc_eh, key=lambda k: gc_eh[k][0]['ssr_ftest'][1])
        return float(gc_eh[beh][0]['ssr_ftest'][0]), np.nan
    except Exception:
        return np.nan, np.nan


def wc_scalar(x_win, y_win, dt=10):
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
    scales = np.arange(2, min(n // 2, 50), 2)
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


def run_sensitivity_case(args):
    """
    Process a single case with given sensitivity config.
    args = (caseid, eeg_col, gc_win, gc_step, step_sec)
    Returns dict with NCCI_mean, n_events, event_responses, etc.
    """
    caseid, eeg_col, gc_win, gc_step, step_sec = args
    feat_path = OUT_FEATURES / f"case_{caseid:04d}_features.parquet"
    if not feat_path.exists():
        return None
    try:
        df = pd.read_parquet(feat_path)
        n = len(df)
        if n < gc_win + 10:
            return None

        if eeg_col not in df.columns:
            return None

        eeg_v = df[eeg_col].values.astype(float)
        map_v = df['MAP_mean'].values.astype(float)
        bis_v = df['BIS_value'].values.astype(float) if 'BIS_value' in df.columns else np.full(n, np.nan)

        win_rows = []
        for start in range(0, n - gc_win, gc_step):
            end = start + gc_win
            ew = eeg_v[start:end]
            hw = map_v[start:end]
            gc_e2h, _ = granger_f(ew, hw)
            wc = wc_scalar(ew, hw, dt=step_sec)
            valid = ~(np.isnan(ew) | np.isnan(hw))
            r = pearsonr(ew[valid], hw[valid])[0] if valid.sum() > 5 else np.nan
            win_rows.append({'window_center': start + gc_win // 2, 'GC_e2h': gc_e2h, 'WC': wc, 'pearson_r': r})

        if not win_rows:
            return None

        win_df = pd.DataFrame(win_rows)
        cols = ['GC_e2h', 'WC', 'pearson_r']
        valid_rows = win_df[cols].dropna()
        if len(valid_rows) < 5:
            return None

        X = StandardScaler().fit_transform(valid_rows)
        pc1 = PCA(n_components=1).fit_transform(X).flatten()
        ncci = MinMaxScaler().fit_transform(pc1.reshape(-1, 1)).flatten()
        if np.corrcoef(ncci, valid_rows['WC'].values)[0, 1] < 0:
            ncci = 1 - ncci
        win_df.loc[valid_rows.index, 'NCCI_window'] = ncci

        # Event detection (fixed thresholds regardless of sensitivity variant)
        events = []
        for i in range(EVENT_LOOKBACK_W, n):
            if np.isnan(map_v[i]) or np.isnan(map_v[i - EVENT_LOOKBACK_W]):
                continue
            map_delta = map_v[i] - map_v[i - EVENT_LOOKBACK_W]
            if map_delta > EVENT_MAP_RISE:
                bis_now = bis_v[i] if not np.isnan(bis_v[i]) else 99
                if bis_now < EVENT_BIS_MAX:
                    events.append({'time_idx': i, 'map_delta': float(map_delta)})

        event_responses = []
        if events and 'NCCI_window' in win_df.columns:
            pre_w = 3
            post_w = 3
            for ev in events:
                t = ev['time_idx']
                pre_wins  = win_df[win_df['window_center'] <= t - gc_win // 2].tail(pre_w)
                post_wins = win_df[win_df['window_center'] >  t + gc_win // 4].head(post_w)
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
            'n_events': len(events),
            'event_responses': event_responses,
        }
    except Exception:
        return None


def run_sensitivity_analysis(label, eeg_col, gc_win, gc_step, step_sec,
                              case_ids, n_workers=4, batch_size=40):
    """Run one sensitivity configuration on a sample of cases."""
    logging.info(f"  Config '{label}': eeg={eeg_col}, gc_win={gc_win}, step={step_sec}s")
    args = [(cid, eeg_col, gc_win, gc_step, step_sec) for cid in case_ids]
    all_results = []

    for b_start in range(0, len(args), batch_size):
        batch = args[b_start:b_start + batch_size]
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_sensitivity_case, batch)
        all_results.extend([r for r in results if r is not None])
        gc.collect()

    all_events = []
    for res in all_results:
        for ev in res.get('event_responses', []):
            all_events.append(ev)

    if not all_events:
        logging.warning(f"  No events found for '{label}'")
        return {'label': label, 'n_cases': len(all_results), 'n_events': 0,
                'NCCI_mean': np.nan, 'delta_ncci_mean': np.nan,
                'pct_decrease': np.nan, 'pval': np.nan, 'effect_d': np.nan}

    ev_df = pd.DataFrame(all_events)
    pre_v  = ev_df['pre_ncci'].dropna().values
    post_v = ev_df['post_ncci'].dropna().values
    delta_v = ev_df['delta_ncci'].dropna().values
    n_paired = min(len(pre_v), len(post_v))

    t, p = (np.nan, np.nan)
    if n_paired > 5:
        t, p = ttest_rel(pre_v[:n_paired], post_v[:n_paired])

    effect_d = delta_v.mean() / (delta_v.std() + 1e-10)  # Cohen's d vs 0
    ncci_means = [r['NCCI_mean'] for r in all_results if not np.isnan(r.get('NCCI_mean', np.nan))]

    logging.info(f"    Cases: {len(all_results)}, Events: {len(all_events)}, "
                 f"ΔNCCI={delta_v.mean():.4f}±{delta_v.std():.4f}, p={p:.4f}")

    return {
        'label': label,
        'eeg_col': eeg_col,
        'gc_win': gc_win,
        'step_sec': step_sec,
        'n_cases': len(all_results),
        'n_events': len(all_events),
        'NCCI_mean': np.mean(ncci_means) if ncci_means else np.nan,
        'delta_ncci_mean': float(delta_v.mean()),
        'delta_ncci_std': float(delta_v.std()),
        'pct_decrease': float((delta_v < 0).mean() * 100),
        'pval': float(p) if not np.isnan(p) else np.nan,
        't_stat': float(t) if not np.isnan(t) else np.nan,
        'effect_d': float(effect_d),
    }


def plot_sensitivity_figure(results_df, save_path):
    """Supp Fig 2: Sensitivity analysis comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel A: ΔNCCI by config (forest plot style)
    ax = axes[0]
    labels = results_df['label'].tolist()
    means  = results_df['delta_ncci_mean'].values
    stds   = results_df['delta_ncci_std'].values
    pvals  = results_df['pval'].values
    n_ev   = results_df['n_events'].values

    primary_idx = 0  # First row is primary analysis
    y_pos = np.arange(len(labels))

    for i, (m, s, p, n, lbl) in enumerate(zip(means, stds, pvals, n_ev, labels)):
        color = COLORS['blue'] if i == primary_idx else COLORS['gray']
        lw = 2 if i == primary_idx else 1
        ax.errorbar(m, i, xerr=s / np.sqrt(max(n, 1)), fmt='o', color=color,
                    ms=7 if i == primary_idx else 5, lw=lw, capsize=3)
        p_str = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(max(means) + max(stds) * 0.2 + 0.01, i, p_str, va='center', fontsize=8,
                color='red' if p < 0.05 else 'gray')

    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('ΔNCCI (Post − Pre)', fontsize=9)
    ax.set_title('ΔNCCI Across Sensitivity Configs\n(mean ± SE)', fontweight='bold')
    ax.text(0.02, 0.02, 'Primary analysis in blue', transform=ax.transAxes, fontsize=7, color='gray')

    # Panel B: % events showing NCCI decrease
    ax2 = axes[1]
    pct = results_df['pct_decrease'].values
    colors_b = [COLORS['blue'] if i == primary_idx else COLORS['gray'] for i in range(len(labels))]
    bars = ax2.barh(y_pos, pct, color=colors_b, alpha=0.75, edgecolor='white')
    ax2.axvline(50, color='black', ls='--', lw=1, label='50%')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel('% Events with NCCI decrease', fontsize=9)
    ax2.set_title('Proportion of Events\nShowing NCCI Decrease', fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(fontsize=7)

    # Panel C: Effect size (Cohen's d)
    ax3 = axes[2]
    d = results_df['effect_d'].values
    colors_c = [COLORS['red'] if i == primary_idx else COLORS['gray'] for i in range(len(labels))]
    ax3.barh(y_pos, np.abs(d), color=colors_c, alpha=0.75, edgecolor='white')
    ax3.axvline(0.2, color='orange', ls=':', lw=1, label='Small (d=0.2)')
    ax3.axvline(0.5, color='red',    ls=':', lw=1, label='Medium (d=0.5)')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_xlabel("|Cohen's d| (effect size)", fontsize=9)
    ax3.set_title("Effect Size Across\nSensitivity Configs", fontweight='bold')
    ax3.legend(fontsize=7)

    plt.suptitle('Sensitivity Analysis: NCCI Robustness Across EEG Features & Window Sizes',
                 fontweight='bold', y=1.01, fontsize=10)
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Supp Fig 2 → {Path(save_path).name}")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"step7_sensitivity_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info("=== Phase 7: Sensitivity Analysis ===")

    # Load feature files
    feat_files = sorted(OUT_FEATURES.glob("case_*.parquet"))
    all_case_ids = [int(f.stem.split('_')[1]) for f in feat_files]
    logging.info(f"Total cases available: {len(all_case_ids)}")

    # Use a 300-case sample for speed (random seed fixed for reproducibility)
    np.random.seed(42)
    sample_ids = np.random.choice(all_case_ids, min(300, len(all_case_ids)), replace=False).tolist()
    logging.info(f"Sensitivity sample: {len(sample_ids)} cases")

    # ── Sensitivity configurations ──────────────────────────────────────────
    # 7.1: EEG feature variants (primary + 3 alternatives)
    # 7.2: Time resolution variants (2 alternatives)
    # 7.3: BIS value instead of raw EEG (1 alternative)
    configs = [
        # label,                    eeg_col,            gc_win, gc_step, step_sec
        ('Primary (ADR)',            'ADR',              60,     3,       10),
        ('7.1a: SEF95',             'SEF95',            60,     3,       10),
        ('7.1b: Alpha power (rel)', 'alpha_rel',        60,     3,       10),
        ('7.1c: Spectral entropy',  'spectral_entropy', 60,     3,       10),
        ('7.2a: Coarse (60s/30s)',  'ADR',              180,    9,       30),  # 60s real = 6 steps of 10s
        ('7.2b: Fine (10s/5s)',     'ADR',              18,     1,       10),  # smaller windows
        ('7.3: BIS value',          'BIS_value',        60,     3,       10),
    ]

    # Check which EEG columns exist in a sample file
    sample_feat = pd.read_parquet(OUT_FEATURES / f"case_{sample_ids[0]:04d}_features.parquet")
    avail_cols = sample_feat.columns.tolist()
    logging.info(f"Available EEG columns: {[c for c in avail_cols if c not in ['time_sec', 'MAP_mean', 'HR_est', 'PPV', 'SDNN', 'RMSSD', 'dPdt_max', 'SBP_mean', 'DBP_mean', 'PP_mean', 'MAP_std', 'PP_std', 'BIS_value', 'GC_e2h_F', 'GC_h2e_F', 'WC_overall', 'pearson_r']]}")
    del sample_feat

    # Filter configs to only those with available columns
    valid_configs = []
    for cfg in configs:
        col = cfg[1]
        if col in avail_cols:
            valid_configs.append(cfg)
        else:
            logging.warning(f"  Skipping '{cfg[0]}': column '{col}' not in features")

    logging.info(f"Running {len(valid_configs)} sensitivity configurations...")

    results = []
    for label, eeg_col, gc_win, gc_step, step_sec in valid_configs:
        res = run_sensitivity_analysis(
            label=label,
            eeg_col=eeg_col,
            gc_win=gc_win,
            gc_step=gc_step,
            step_sec=step_sec,
            case_ids=sample_ids,
            n_workers=4,
            batch_size=40,
        )
        results.append(res)

    results_df = pd.DataFrame(results)

    # Save table
    out_table = OUT_METRICS / "supp_table1_sensitivity.csv"
    assert not str(out_table.resolve()).startswith(str(RAW_DATA.resolve()))
    results_df.to_csv(out_table, index=False)
    logging.info(f"\nSaved Supp Table 1 → supp_table1_sensitivity.csv")

    # Print summary
    logging.info("\n=== Sensitivity Analysis Summary ===")
    for _, row in results_df.iterrows():
        p_str = f"p={row['pval']:.4f}" if not np.isnan(row['pval']) else "p=N/A"
        logging.info(f"  {row['label']:35s} ΔNCCI={row['delta_ncci_mean']:+.4f}, "
                     f"pct_dec={row['pct_decrease']:.0f}%, {p_str}, n_ev={row['n_events']}")

    if len(results_df) > 1:
        plot_sensitivity_figure(results_df, OUT_FIGS / "supp_fig2_sensitivity.png")

    logging.info("=== Phase 7 Complete ===")


if __name__ == "__main__":
    main()
