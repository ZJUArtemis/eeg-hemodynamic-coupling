#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step4b_ncci_event_figure.py | Topic: 9
Generates Fig 4b: Window-level NCCI at nociceptive events (strict thresholds).
Also generates the updated Table 2 with all three coupling methods summarized.
"""
import sys, logging, warnings, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_1samp

warnings.filterwarnings('ignore')

RAW_DATA  = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK      = Path(".")  # repository root; override as needed
OUT_METRICS = WORK / "outputs" / "metrics"
OUT_FIGS    = WORK / "outputs" / "figures"
OUT_LOGS    = WORK / "outputs" / "logs"

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


def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)


def plot_ncci_event_figure(ev_df, coupling_df, save_path):
    """
    Fig 4b: Window-level NCCI response at nociceptive events.
    4-panel figure:
      A: Paired pre/post NCCI (scatter + mean)
      B: ΔNCCI distribution
      C: NCCI vs MAP-delta correlation
      D: Event rate by NCCI quartile
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    # Panel A: Paired pre/post
    ax = axes[0]
    pre_v  = ev_df['pre_ncci'].dropna().values
    post_v = ev_df['post_ncci'].dropna().values
    n = min(len(pre_v), len(post_v))
    n_paired = n

    idx = np.random.choice(n, min(n, 60), replace=False)
    for i in idx:
        ax.plot([0, 1], [pre_v[i], post_v[i]], 'o-',
                color=COLORS['gray'], alpha=0.2, lw=0.5, ms=2)

    ax.errorbar([0], [pre_v.mean()], yerr=[pre_v.std()/np.sqrt(n)],
                fmt='o', color=COLORS['blue'], ms=9, lw=2, capsize=5, label='Pre-event')
    ax.errorbar([1], [post_v.mean()], yerr=[post_v.std()/np.sqrt(n)],
                fmt='o', color=COLORS['red'], ms=9, lw=2, capsize=5, label='Post-event')
    t, p = ttest_rel(pre_v[:n], post_v[:n])
    p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
    y_max = max(pre_v.mean(), post_v.mean()) + max(pre_v.std(), post_v.std()) * 0.4
    ax.text(0.5, y_max * 1.05, p_str, ha='center', fontweight='bold', fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-event\n(~1.5 min)', 'Post-event\n(~1.5 min)'])
    ax.set_ylabel('Window-level NCCI', fontsize=9)
    ax.set_title('NCCI at Nociceptive Events', fontweight='bold')
    ax.legend(fontsize=8)
    ax.text(0.05, 0.02, f'n={n:,} events', transform=ax.transAxes, fontsize=7, color='gray')

    # Panel B: ΔNCCI distribution
    ax2 = axes[1]
    delta = ev_df['delta_ncci'].dropna().values
    pct_dec = (delta < 0).mean() * 100
    ax2.hist(delta, bins=40, color=COLORS['blue'], alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='black', ls='--', lw=1.2, label='No change')
    ax2.axvline(delta.mean(), color=COLORS['red'], lw=1.5,
                label=f'Mean={delta.mean():.3f}')
    t1, p1 = ttest_1samp(delta, 0)
    ax2.set_xlabel('ΔNCCI (Post − Pre)', fontsize=9)
    ax2.set_ylabel('Event count', fontsize=9)
    ax2.set_title('Distribution of NCCI Change', fontweight='bold')
    ax2.text(0.05, 0.92, f'{pct_dec:.0f}% events show NCCI↓\n'
              f'One-sample t: p={p1:.4f}\n(n={len(delta):,} events)',
              transform=ax2.transAxes, va='top', fontsize=8,
              bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
    ax2.legend(fontsize=7)

    # Panel C: NCCI vs MAP delta
    ax3 = axes[2]
    map_d = ev_df['map_delta'].dropna().values
    delta_c = ev_df.loc[ev_df['map_delta'].notna(), 'delta_ncci'].values
    n_c = min(len(map_d), len(delta_c))
    if n_c > 100:
        idx_s = np.random.choice(n_c, 500, replace=(n_c < 500))
        ax3.scatter(map_d[idx_s], delta_c[idx_s], alpha=0.2, s=5, color=COLORS['blue'])
    ax3.axhline(0, color='black', ls='--', lw=0.8)
    ax3.axvline(0, color='black', ls='--', lw=0.8)
    # Bin means
    bins = np.percentile(map_d, np.linspace(0, 100, 11))
    bin_x, bin_y, bin_e = [], [], []
    for i in range(len(bins)-1):
        mask = (map_d >= bins[i]) & (map_d < bins[i+1])
        if mask.sum() > 5:
            bin_x.append(np.mean(map_d[mask]))
            bin_y.append(np.mean(delta_c[mask]))
            bin_e.append(np.std(delta_c[mask]) / np.sqrt(mask.sum()))
    ax3.errorbar(bin_x, bin_y, yerr=bin_e, fmt='o-', color=COLORS['red'],
                 ms=6, lw=1.5, capsize=3, label='Bin mean ± SE')
    ax3.set_xlabel('MAP rise (mmHg)', fontsize=9)
    ax3.set_ylabel('ΔNCCI', fontsize=9)
    ax3.set_title('MAP Rise vs NCCI Change', fontweight='bold')
    ax3.legend(fontsize=7)

    # Panel D: Event rate by NCCI quartile (case-level)
    ax4 = axes[3]
    if 'NCCI' in coupling_df.columns and 'n_events' in coupling_df.columns:
        df = coupling_df.dropna(subset=['NCCI']).copy()
        df['Q'] = pd.qcut(df['NCCI'], 4, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
        event_rate = df.groupby('Q', observed=True)['n_events'].mean()
        n_cases = df.groupby('Q', observed=True).size()
        err = df.groupby('Q', observed=True)['n_events'].sem()

        ax4.bar(range(4), event_rate.values, yerr=err.values,
                capsize=4, color=COLORS['orange'], alpha=0.8, edgecolor='white',
                error_kw={'elinewidth': 1.5})
        for i, (n, ev) in enumerate(zip(n_cases.values, event_rate.values)):
            ax4.text(i, ev + err.values[i] + 0.3, f'n={n}', ha='center', fontsize=7)
        ax4.set_xticks(range(4))
        ax4.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
        ax4.set_xlabel('NCCI Quartile', fontsize=9)
        ax4.set_ylabel('Events per case (mean)', fontsize=9)
        ax4.set_title('Nociceptive Event Rate\nby NCCI Quartile', fontweight='bold')

    plt.suptitle('Window-Level NCCI Response to Nociceptive Hemodynamic Events\n'
                  f'(MAP+20 mmHg in 90s, BIS<60; strict thresholds)',
                  fontweight='bold', y=1.01, fontsize=10)
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Fig 4b → {save_path.name}")


def make_table2(coupling_df, ev_df, save_path):
    """Table 2: Coupling methods summary."""
    rows = []
    def add(cat, metric, val):
        rows.append({'Category': cat, 'Metric': metric, 'Value': val})

    n = len(coupling_df)

    # Granger Causality
    gc = coupling_df['GC_e2h_F_mean'].dropna()
    add('Granger Causality', 'EEG→Hemo F-statistic (mean±SD)', f"{gc.mean():.2f} ± {gc.std():.2f}")
    gc2 = coupling_df['GC_h2e_F_mean'].dropna()
    add('Granger Causality', 'Hemo→EEG F-statistic (mean±SD)', f"{gc2.mean():.2f} ± {gc2.std():.2f}")
    gd = coupling_df['GC_directionality'].dropna()
    pct_pos = (gd > 0).mean() * 100
    add('Granger Causality', 'GC directionality (EEG→Hemo dominant)', f"{pct_pos:.0f}% of cases")

    # Wavelet Coherence
    wc = coupling_df['WC_overall'].dropna()
    add('Wavelet Coherence', 'Overall WC (mean±SD)', f"{wc.mean():.3f} ± {wc.std():.3f}")
    wl = coupling_df['WC_low'].dropna()
    add('Wavelet Coherence', 'Low-freq WC (<0.01 Hz, mean±SD)', f"{wl.mean():.3f} ± {wl.std():.3f}")
    wm = coupling_df['WC_mid'].dropna()
    add('Wavelet Coherence', 'Mid-freq WC (0.01–0.03 Hz, mean±SD)', f"{wm.mean():.3f} ± {wm.std():.3f}")

    # NCCI
    ncci = coupling_df['NCCI'].dropna()
    add('NCCI', 'NCCI (mean±SD)', f"{ncci.mean():.3f} ± {ncci.std():.3f}")
    add('NCCI', 'NCCI range', f"[{ncci.min():.3f}, {ncci.max():.3f}]")

    # Event analysis
    pre = ev_df['pre_ncci'].dropna()
    post = ev_df['post_ncci'].dropna()
    n_ev = min(len(pre), len(post))
    t, p = ttest_rel(pre.values[:n_ev], post.values[:n_ev])
    add('Event Analysis (strict)', 'Total events detected', f"{len(ev_df):,}")
    add('Event Analysis (strict)', 'Cases with events', f"{(coupling_df['n_events']>0).sum()} ({(coupling_df['n_events']>0).mean()*100:.1f}%)")
    add('Event Analysis (strict)', 'NCCI pre-event (mean±SD)', f"{pre.mean():.3f} ± {pre.std():.3f}")
    add('Event Analysis (strict)', 'NCCI post-event (mean±SD)', f"{post.mean():.3f} ± {post.std():.3f}")
    delta = ev_df['delta_ncci'].dropna()
    add('Event Analysis (strict)', 'ΔNCCI (mean±SD)', f"{delta.mean():.4f} ± {delta.std():.4f}")
    add('Event Analysis (strict)', '% events with NCCI decrease', f"{(delta<0).mean()*100:.1f}%")
    p_str = 'p<0.001' if p < 0.001 else f'p={p:.4f}'
    add('Event Analysis (strict)', 'Paired t-test (pre vs post)', f"t={t:.3f}, {p_str}")

    table_df = pd.DataFrame(rows)
    assert not str(Path(save_path).resolve()).startswith(str(RAW_DATA.resolve()))
    table_df.to_csv(save_path, index=False)
    logging.info(f"Saved Table 2 → {Path(save_path).name}")
    return table_df


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"step4b_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info("=== Generating Window-NCCI Event Figure & Table 2 ===")

    coupling_df = pd.read_csv(OUT_METRICS / "coupling_summary.csv")
    ev_path = OUT_METRICS / "windowed_ncci_events.csv"

    if not ev_path.exists():
        logging.error("windowed_ncci_events.csv not found. Run step3b first.")
        return

    ev_df = pd.read_csv(ev_path)
    logging.info(f"Coupling cases: {len(coupling_df)}")
    logging.info(f"Event responses: {len(ev_df)}")

    # Print key stats
    pre = ev_df['pre_ncci'].dropna()
    post = ev_df['post_ncci'].dropna()
    delta = ev_df['delta_ncci'].dropna()
    n = min(len(pre), len(post))
    t, p = ttest_rel(pre.values[:n], post.values[:n])
    t1, p1 = ttest_1samp(delta.values, 0)

    logging.info(f"\nKey Results:")
    logging.info(f"  Events: {len(ev_df):,} from {ev_df['caseid'].nunique()} cases")
    logging.info(f"  NCCI pre:  {pre.mean():.3f} ± {pre.std():.3f}")
    logging.info(f"  NCCI post: {post.mean():.3f} ± {post.std():.3f}")
    logging.info(f"  ΔNCCI:     {delta.mean():.4f} ± {delta.std():.4f}")
    logging.info(f"  % decrease: {(delta<0).mean()*100:.1f}%")
    logging.info(f"  Paired t-test: t={t:.3f}, p={p:.4f}")
    logging.info(f"  One-sample t (vs 0): t={t1:.3f}, p={p1:.4f}")

    plot_ncci_event_figure(ev_df, coupling_df, OUT_FIGS / "fig4b_windowed_ncci_events.png")
    make_table2(coupling_df, ev_df, OUT_METRICS / "table2_coupling_methods.csv")

    logging.info("=== Done ===")


if __name__ == "__main__":
    main()
