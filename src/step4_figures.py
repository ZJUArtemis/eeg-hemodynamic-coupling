#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step4_figures.py | Topic: 9 | Purpose: Publication-quality figures
Generates all 6 main figures + supplementary figures for the paper.
"""
import sys, logging, warnings, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter
import pywt

warnings.filterwarnings('ignore')

# === PATHS ===
RAW_DATA = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK = Path(".")  # repository root; override as needed
OUT_FEATURES  = WORK / "outputs" / "features"
OUT_COUPLING  = WORK / "outputs" / "coupling"
OUT_METRICS   = WORK / "outputs" / "metrics"
OUT_FIGS      = WORK / "outputs" / "figures"
OUT_LOGS      = WORK / "outputs" / "logs"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_LOGS.mkdir(parents=True, exist_ok=True)

# === PLOT STYLE ===
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette
COLORS = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'gray':   '#999999',
    'yellow': '#F0E442',
}

EEG_FS = 128
STEP_SEC = 10


# ─────────────────────── HELPER: Wavelet Coherence ───────────────────────

def wavelet_coherence_matrix(x, y, dt=STEP_SEC):
    """Compute coherence matrix for plotting."""
    n = min(len(x), len(y))
    x, y = np.array(x[:n], float), np.array(y[:n], float)
    for arr in [x, y]:
        nans = np.isnan(arr)
        if nans.any():
            idx = np.arange(len(arr))
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)

    scales = np.arange(2, min(n // 4, 150), 2)
    cx, freqs = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=dt)
    cy, _     = pywt.cwt(y, scales, 'cmor1.5-1.0', sampling_period=dt)
    cross = cx * np.conj(cy)
    sm = (5, 11)
    sr = uniform_filter(cross.real, sm)
    si = uniform_filter(cross.imag, sm)
    sx = uniform_filter(np.abs(cx) ** 2, sm)
    sy = uniform_filter(np.abs(cy) ** 2, sm)
    coh = np.clip((sr**2 + si**2) / (sx * sy + 1e-10), 0, 1)
    return coh, freqs, scales


def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)


# ─────────────────────── FIG 1: Cohort Flowchart ───────────────────────

def plot_flowchart(flowchart_path, save_path):
    """Fig 1: CONSORT-style patient-selection flowchart.

    Fixed-size rounded boxes; arrows connect the bottom edge of each box to the
    top edge of the next (no arrowheads landing inside boxes). Exclusion counts
    branch off each connector into right-side boxes.
    """
    from matplotlib.patches import FancyBboxPatch

    with open(flowchart_path) as f:
        fc = json.load(f)
    total  = fc.get('total', 6388)
    basic  = fc.get('after_basic_filters', 3825)
    eegabp = fc.get('eeg_abp_cohort', 2076)
    final  = fc.get('final_cohort', 2070)

    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # --- main column geometry ---
    cx = 3.5                      # centre x of the main boxes
    bw, bh = 5.2, 1.2             # box width / height
    cy = [8.7, 6.5, 4.3, 2.1]     # box centres (equal spacing)
    main_txt = [
        f"VitalDB cases\nn = {total:,}",
        f"After basic clinical filters\n(age ≥18, general anaesthesia,\n"
        f"non-cardiac/non-thoracic, op ≥60 min)\nn = {basic:,}",
        f"EEG + ABP cohort\n(both waveforms present)\nn = {eegabp:,}",
        f"Final analysis cohort\n(≥30 min concurrent valid EEG + ABP)\nn = {final:,}",
    ]

    def box(x, y, w, h, txt, fcol, ecol, fs, lw):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     linewidth=lw, edgecolor=ecol, facecolor=fcol, zorder=2))
        ax.text(x, y, txt, ha='center', va='center', fontsize=fs, zorder=3)

    for y, t in zip(cy, main_txt):
        box(cx, y, bw, bh, t, '#eaf1f8', 'black', 8.3, 1.3)

    # --- vertical arrows: bottom edge of upper box -> top edge of lower box ---
    for i in range(len(cy) - 1):
        y0 = cy[i]     - bh / 2   # bottom of upper box
        y1 = cy[i + 1] + bh / 2   # top of lower box
        ax.annotate('', xy=(cx, y1), xytext=(cx, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.4, color='black',
                                    shrinkA=0, shrinkB=0, mutation_scale=16))

    # --- right-side exclusion boxes branching off each connector ---
    excl = [total - basic, basic - eegabp, eegabp - final]
    excl_txt = [
        f"Excluded (n = {excl[0]:,}):\nage <18, non-general\n"
        f"anaesthesia, cardiac/\nthoracic, op <60 min",
        f"Excluded (n = {excl[1]:,}):\nEEG and/or ABP\nwaveform absent",
        f"Excluded (n = {excl[2]:,}):\n<30 min concurrent\nvalid EEG + ABP",
    ]
    ex_cx, ex_w, ex_h = 8.0, 3.4, 1.05
    for i in range(3):
        my = (cy[i] - bh / 2 + cy[i + 1] + bh / 2) / 2   # midpoint of the gap
        ax.annotate('', xy=(ex_cx - ex_w / 2, my), xytext=(cx, my),
                    arrowprops=dict(arrowstyle='-|>', lw=1.0, color='0.45',
                                    shrinkA=0, shrinkB=0, mutation_scale=12))
        box(ex_cx, my, ex_w, ex_h, excl_txt[i], '#f6efe9', '0.45', 7.0, 1.0)

    ax.set_title('Patient selection flowchart', fontweight='bold', fontsize=11)
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─────────────────────── FIG 2: Case-level Panel ───────────────────────

def plot_case_panel(caseid, df, save_path, event_times=None):
    """
    Fig 2: Multi-panel case visualization
    A: EEG spectrogram
    B: BIS value
    C: MAP + HR
    D: Wavelet coherence (ADR ↔ MAP)
    E: NCCI proxy (ADR time course)
    """
    time_min = df['time_sec'].values / 60

    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.1,
                            height_ratios=[2.5, 1, 1.2, 2.5, 1.5])

    # ── Panel A: EEG Spectrogram (using band powers) ──
    ax_a = fig.add_subplot(gs[0])
    band_cols = ['delta_rel', 'theta_power', 'alpha_rel', 'beta_rel']
    band_labels = ['Delta', 'Theta', 'Alpha', 'Beta']
    band_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]

    for col, label, color in zip(band_cols, band_labels, band_colors):
        if col in df.columns:
            vals = df[col].rolling(3, center=True, min_periods=1).mean()
            ax_a.fill_between(time_min, vals * 100, alpha=0.5, color=color, label=label)
    ax_a.set_ylabel('Rel. Power (%)', fontsize=8)
    ax_a.set_xlim(time_min[0], time_min[-1])
    ax_a.legend(loc='upper right', ncol=4, fontsize=7)
    ax_a.set_title(f'Case {caseid}: EEG–Hemodynamic Coupling', fontweight='bold')
    ax_a.set_xticklabels([])

    # ── Panel B: BIS ──
    ax_b = fig.add_subplot(gs[1])
    if 'BIS_value' in df.columns:
        bis = df['BIS_value'].copy()
        bis[bis <= 0] = np.nan
        ax_b.plot(time_min, bis, color=COLORS['purple'], lw=1.2, label='BIS')
        ax_b.axhline(40, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax_b.axhline(60, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax_b.fill_between(time_min, 40, 60, alpha=0.07, color='green')
    ax_b.set_ylabel('BIS', fontsize=8)
    ax_b.set_ylim(0, 100)
    ax_b.set_xlim(time_min[0], time_min[-1])
    ax_b.set_xticklabels([])

    # ── Panel C: MAP + HR ──
    ax_c = fig.add_subplot(gs[2])
    if 'MAP_mean' in df.columns:
        map_v = df['MAP_mean'].rolling(3, center=True, min_periods=1).mean()
        ax_c.plot(time_min, map_v, color=COLORS['red'], lw=1.2, label='MAP (ABP)', zorder=3)
    if 'MAP_monitor' in df.columns:
        map_mon = df['MAP_monitor'].copy()
        map_mon[map_mon < 20] = np.nan
        ax_c.plot(time_min, map_mon, color=COLORS['orange'], lw=0.8,
                  ls='--', alpha=0.7, label='MAP (monitor)')
    ax_c_r = ax_c.twinx()
    if 'HR_est' in df.columns:
        hr = df['HR_est'].rolling(3, center=True, min_periods=1).mean()
        ax_c_r.plot(time_min, hr, color=COLORS['blue'], lw=0.8, alpha=0.7, label='HR')
        ax_c_r.set_ylabel('HR (bpm)', fontsize=8, color=COLORS['blue'])
        ax_c_r.tick_params(axis='y', labelcolor=COLORS['blue'])
    ax_c.set_ylabel('MAP (mmHg)', fontsize=8)
    ax_c.set_xlim(time_min[0], time_min[-1])
    ax_c.legend(loc='upper left', fontsize=7)
    ax_c.set_xticklabels([])

    # ── Panel D: Wavelet Coherence ──
    ax_d = fig.add_subplot(gs[3])
    try:
        adr = df['ADR'].values
        map_series = df['MAP_mean'].values
        valid = ~(np.isnan(adr) | np.isnan(map_series))
        if valid.sum() > 50:
            coh, freqs, scales = wavelet_coherence_matrix(adr, map_series)
            t_ax = np.arange(coh.shape[1]) * STEP_SEC / 60

            im = ax_d.pcolormesh(t_ax, freqs, coh,
                                   cmap='RdYlBu_r', vmin=0, vmax=1, shading='auto')
            ax_d.set_yscale('log')
            ax_d.set_ylabel('Frequency (Hz)', fontsize=8)
            ax_d.set_ylim(freqs[-1], freqs[0])
            plt.colorbar(im, ax=ax_d, label='Coherence', shrink=0.8)
            ax_d.set_xlim(time_min[0], time_min[-1])
            ax_d.set_xticklabels([])
    except Exception as e:
        ax_d.text(0.5, 0.5, f'Coherence unavailable\n{e}',
                   ha='center', va='center', transform=ax_d.transAxes, fontsize=8)

    # ── Panel E: ADR (NCCI proxy) ──
    ax_e = fig.add_subplot(gs[4])
    if 'ADR' in df.columns:
        adr_smooth = df['ADR'].rolling(6, center=True, min_periods=1).mean()
        ax_e.plot(time_min, adr_smooth, color=COLORS['green'], lw=1.5, label='ADR (NCCI proxy)')
        ax_e.fill_between(time_min, adr_smooth, alpha=0.2, color=COLORS['green'])
    ax_e.set_ylabel('ADR', fontsize=8)
    ax_e.set_xlabel('Time (min)', fontsize=9)
    ax_e.set_xlim(time_min[0], time_min[-1])
    ax_e.legend(loc='upper right', fontsize=7)

    # Mark nociceptive events
    if event_times:
        for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
            for et in event_times:
                ax.axvline(et / 60, color='red', ls=':', lw=1.2, alpha=0.7)

    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Fig 2 → {save_path.name}")
    return save_path


# ─────────────────────── FIG 3: Wavelet Heatmaps ───────────────────────

def plot_wavelet_examples(case_dfs, save_path):
    """
    Fig 3: Wavelet coherence for 3 representative cases
    (high coupling / moderate coupling / decoupled)
    """
    n_cases = len(case_dfs)
    fig, axes = plt.subplots(n_cases, 1, figsize=(9, 3.5 * n_cases))
    if n_cases == 1:
        axes = [axes]

    labels = ['High coupling', 'Moderate coupling', 'Low coupling (surrogate event)']

    for i, (df, label) in enumerate(zip(case_dfs, labels)):
        ax = axes[i]
        try:
            adr = df['ADR'].values
            map_v = df['MAP_mean'].values
            valid = ~(np.isnan(adr) | np.isnan(map_v))
            if valid.sum() > 50:
                coh, freqs, _ = wavelet_coherence_matrix(adr, map_v)
                t_ax = np.arange(coh.shape[1]) * STEP_SEC / 60

                im = ax.pcolormesh(t_ax, freqs, coh,
                                    cmap='RdYlBu_r', vmin=0, vmax=1, shading='auto')
                ax.set_yscale('log')
                ax.set_ylabel('Frequency (Hz)', fontsize=8)
                plt.colorbar(im, ax=ax, label='Wavelet coherence', shrink=0.9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                     transform=ax.transAxes, fontsize=8)

        ax.set_title(f'Case {i+1}: {label}', fontweight='bold')
        if i == n_cases - 1:
            ax.set_xlabel('Time (min)', fontsize=9)
        else:
            ax.set_xticklabels([])

    fig.suptitle('EEG–Hemodynamic Wavelet Coherence (ADR ↔ MAP)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Fig 3 → {save_path.name}")
    return save_path


# ─────────────────────── FIG 4: Event Response ───────────────────────

def plot_event_response(coupling_df, save_path):
    """Fig 4: ADR response at nociceptive events (mean ± SEM)."""
    event_cases = coupling_df[coupling_df['n_events'] > 0].dropna(
        subset=['event_adr_pre', 'event_adr_post'])

    if len(event_cases) < 5:
        logging.warning(f"Too few event cases ({len(event_cases)}) for Fig 4")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Panel A: Paired dots
    ax = axes[0]
    pre_v  = event_cases['event_adr_pre'].values
    post_v = event_cases['event_adr_post'].values
    n = len(pre_v)

    # Sample to avoid overplotting
    idx_sample = np.random.choice(n, min(n, 60), replace=False) if n > 60 else np.arange(n)
    for i in idx_sample:
        ax.plot([0, 1], [pre_v[i], post_v[i]], 'o-', color=COLORS['gray'],
                alpha=0.25, lw=0.5, ms=2)

    # Means + SEM
    ax.errorbar([0], [pre_v.mean()], yerr=[pre_v.std() / np.sqrt(n)],
                fmt='o', color=COLORS['blue'], ms=9, lw=2, capsize=5,
                label='Pre-excursion', zorder=5)
    ax.errorbar([1], [post_v.mean()], yerr=[post_v.std() / np.sqrt(n)],
                fmt='o', color=COLORS['red'], ms=9, lw=2, capsize=5,
                label='Post-excursion', zorder=5)

    # Robust y-limits so outliers do not crush the means at the bottom
    y_hi = float(np.nanpercentile(np.concatenate([pre_v, post_v]), 99)) * 1.08
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, y_hi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre\n(5 min)', 'Post\n(5 min)'])
    ax.set_ylabel('ADR (Alpha/Delta Ratio)', fontsize=9)
    ax.set_title('ADR at surrogate haemodynamic excursions', fontweight='bold', fontsize=9.5)

    from scipy.stats import ttest_rel
    t, p = ttest_rel(pre_v, post_v)
    p_str = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
    ax.text(0.5, 0.95, p_str, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # Panel B: Distribution of deltas (robust x-limits)
    ax2 = axes[1]
    delta = event_cases['event_adr_delta'].values
    x_lo, x_hi = -0.6, 0.6
    ax2.hist(np.clip(delta, x_lo, x_hi), bins=40, range=(x_lo, x_hi),
             color=COLORS['blue'], alpha=0.75, edgecolor='white')
    ax2.axvline(0, color='black', ls='--', lw=1.2)
    ax2.axvline(delta.mean(), color=COLORS['red'], ls='-', lw=1.6,
                label=f'Mean = {delta.mean():.3f}')
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xlabel('ΔADR (Post − Pre)', fontsize=9)
    ax2.set_ylabel('Number of cases', fontsize=9)
    ax2.set_title('Distribution of ADR change at excursions', fontweight='bold', fontsize=9.5)
    ax2.legend(fontsize=8, loc='upper right')

    pct_decrease = (delta < 0).mean() * 100
    ax2.text(0.04, 0.95, f'{pct_decrease:.0f}% show ADR decrease\n(n = {n:,} cases)',
              transform=ax2.transAxes, va='top', fontsize=8,
              bbox=dict(fc='white', ec='gray', alpha=0.85))

    plt.suptitle('Alpha/Delta Ratio (ADR) Response at Surrogate Haemodynamic Excursions',
                  fontweight='bold', y=1.02, fontsize=11)
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Fig 4 → {save_path.name}")
    return save_path


# ─────────────────────── FIG 5: NCCI vs BIS Comparison ───────────────────────

def plot_ncci_vs_bis(coupling_df, save_path):
    """Fig 5: NCCI distribution and relationship with BIS."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Panel A: NCCI distribution
    ax = axes[0]
    if 'NCCI' in coupling_df.columns:
        ncci = coupling_df['NCCI'].dropna()
        ax.hist(ncci, bins=30, color=COLORS['blue'], alpha=0.7, edgecolor='white')
        ax.axvline(ncci.mean(), color=COLORS['red'], lw=1.5, label=f'Mean={ncci.mean():.2f}')
        ax.set_xlabel('NCCI (0=decoupled, 1=coupled)', fontsize=9)
        ax.set_ylabel('Number of cases', fontsize=9)
        ax.set_title('NCCI Distribution', fontweight='bold')
        ax.legend(fontsize=8)

    # Panel B: Wavelet coherence distribution
    ax2 = axes[1]
    if 'WC_overall' in coupling_df.columns:
        wc = coupling_df['WC_overall'].dropna()
        ax2.hist(wc, bins=30, color=COLORS['green'], alpha=0.7, edgecolor='white')
        ax2.axvline(wc.mean(), color=COLORS['red'], lw=1.5, label=f'Mean={wc.mean():.2f}')
        ax2.set_xlabel('Wavelet Coherence (overall)', fontsize=9)
        ax2.set_ylabel('Number of cases', fontsize=9)
        ax2.set_title('Wavelet Coherence Distribution', fontweight='bold')
        ax2.legend(fontsize=8)

    # Panel C: Granger directionality (use winsorized column if available)
    ax3 = axes[2]
    if 'GC_directionality_win' in coupling_df.columns or 'GC_directionality' in coupling_df.columns:
        col = 'GC_directionality_win' if 'GC_directionality_win' in coupling_df.columns else 'GC_directionality'
        gc_dir = coupling_df[col].dropna()
        gc_dir_clipped = gc_dir.clip(-10, 10)
        # no histogram label (the bars are the distribution, not a single category)
        ax3.hist(gc_dir_clipped, bins=30, color=COLORS['orange'], alpha=0.7, edgecolor='white')
        ax3.axvline(0, color='black', ls='--', lw=1.2)
        ax3.axvline(gc_dir_clipped.mean(), color=COLORS['red'], lw=1.5,
                     label=f'Mean = {gc_dir_clipped.mean():.2f}')
        ax3.set_xlim(-10, 10)
        pct_pos = (gc_dir > 0).mean() * 100
        # annotation in the empty LEFT corner; legend in the empty RIGHT corner,
        # so neither covers the central peak of the distribution
        ax3.text(0.03, 0.97, f'{pct_pos:.0f}% EEG→Hemo\ndominant',
                  transform=ax3.transAxes, va='top', ha='left', fontsize=8,
                  bbox=dict(fc='white', ec='gray', alpha=0.9))
        ax3.set_xlabel('Granger Directionality (EEG→Hemo F − Hemo→EEG F)', fontsize=8)
        ax3.set_ylabel('Number of cases', fontsize=9)
        ax3.set_title('Information Flow Direction', fontweight='bold')
        ax3.legend(fontsize=8, loc='upper right', framealpha=0.9)

    plt.suptitle('Neurocardiac Coupling Metrics Across Cohort', fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Fig 5 → {save_path.name}")
    return save_path


# ─────────────────────── FIG 6: Granger Over Surgery Phases ───────────────────────

def plot_coupling_by_quartile(coupling_df, save_path):
    """Fig 6: Coupling metrics stratified by NCCI quartiles."""
    if 'NCCI' not in coupling_df.columns:
        logging.warning("NCCI not available for Fig 6")
        return None

    df = coupling_df.dropna(subset=['NCCI']).copy()
    df['NCCI_quartile'] = pd.qcut(df['NCCI'], 4, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])

    gc_dir_col = 'GC_directionality_win' if 'GC_directionality_win' in df.columns else 'GC_directionality'
    metrics = ['WC_overall', 'GC_e2h_F_mean', gc_dir_col, 'pearson_r']
    metric_labels = ['Wavelet Coherence', 'GC Strength (EEG→Hemo)', 'GC Directionality', 'Pearson r']
    colors_list = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    for ax, metric, label, color in zip(axes, metrics, metric_labels, colors_list):
        if metric not in df.columns:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue
        groups = [df[df['NCCI_quartile'] == q][metric].dropna().values
                  for q in ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']]
        means = [g.mean() for g in groups]
        sems  = [g.std() / np.sqrt(len(g)) if len(g) > 0 else 0 for g in groups]

        ax.bar(range(4), means, yerr=sems, capsize=4, color=color, alpha=0.7,
               edgecolor='white', error_kw={'elinewidth': 1.5})
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'], fontsize=8)
        ax.set_xlabel('NCCI Quartile', fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontweight='bold', fontsize=9)

    plt.suptitle('Coupling Metrics by NCCI Quartile', fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Fig 6 → {save_path.name}")
    return save_path


# ─────────────────────── SUPPLEMENTARY: Granger Matrix ───────────────────────

def plot_granger_matrix(coupling_df, save_path):
    """Supp Fig 1: Granger causality F-statistics heatmap."""
    metrics = {
        'ADR→MAP': coupling_df['GC_e2h_F_mean'].dropna(),
        'MAP→ADR': coupling_df['GC_h2e_F_mean'].dropna(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (name, vals) in zip(axes, metrics.items()):
        ax.hist(vals, bins=30, color=COLORS['blue'], alpha=0.7, edgecolor='white')
        ax.axvline(vals.mean(), color=COLORS['red'], lw=1.5,
                    label=f'Mean={vals.mean():.2f}')
        ax.set_title(f'Granger: {name}', fontweight='bold')
        ax.set_xlabel('F-statistic', fontsize=9)
        ax.set_ylabel('N cases', fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(safe_path(save_path)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved Supp Fig 1 → {save_path.name}")
    return save_path


# ─────────────────────── TABLE 1 ───────────────────────

def make_table1(cohort_path, coupling_path, save_path):
    """Table 1: Baseline characteristics."""
    cohort = pd.read_csv(cohort_path)
    coupling = pd.read_csv(coupling_path)

    cohort['age'] = pd.to_numeric(cohort['age'], errors='coerce')
    cohort['bmi'] = pd.to_numeric(cohort['bmi'], errors='coerce')
    cohort['opstart'] = pd.to_numeric(cohort['opstart'], errors='coerce')
    cohort['opend'] = pd.to_numeric(cohort['opend'], errors='coerce')
    cohort['op_duration_min'] = (cohort['opend'] - cohort['opstart']) / 60

    rows = []
    def add(label, val):
        rows.append({'Variable': label, 'Value': val})

    add('N', len(cohort))
    add('Age, yr', f"{cohort['age'].mean():.1f} ± {cohort['age'].std():.1f}")
    add('Sex (male), n (%)', f"{(cohort['sex']=='M').sum()} ({(cohort['sex']=='M').mean()*100:.1f}%)")
    add('BMI, kg/m²', f"{cohort['bmi'].mean():.1f} ± {cohort['bmi'].std():.1f}")
    add('ASA I', f"{(cohort['asa']==1).sum()} ({(cohort['asa']==1).mean()*100:.1f}%)")
    add('ASA II', f"{(cohort['asa']==2).sum()} ({(cohort['asa']==2).mean()*100:.1f}%)")
    add('ASA III–IV', f"{(cohort['asa']>=3).sum()} ({(cohort['asa']>=3).mean()*100:.1f}%)")
    add('Operation duration, min', f"{cohort['op_duration_min'].mean():.0f} ± {cohort['op_duration_min'].std():.0f}")

    if 'WC_overall' in coupling.columns:
        wc = coupling['WC_overall'].dropna()
        add('WC overall (mean ± SD)', f"{wc.mean():.3f} ± {wc.std():.3f}")
    if 'GC_e2h_F_mean' in coupling.columns:
        gc = coupling['GC_e2h_F_mean'].dropna()
        add('GC EEG→Hemo F (mean ± SD)', f"{gc.mean():.2f} ± {gc.std():.2f}")
    if 'NCCI' in coupling.columns:
        ncci = coupling['NCCI'].dropna()
        add('NCCI (mean ± SD)', f"{ncci.mean():.3f} ± {ncci.std():.3f}")
    if 'n_events' in coupling.columns:
        ev = coupling['n_events']
        add('Cases with events, n (%)', f"{(ev>0).sum()} ({(ev>0).mean()*100:.1f}%)")
        add('Events per case (median [IQR])',
             f"{ev[ev>0].median():.0f} [{ev[ev>0].quantile(0.25):.0f}–{ev[ev>0].quantile(0.75):.0f}]")

    table_df = pd.DataFrame(rows)
    assert not str(Path(save_path).resolve()).startswith(str(RAW_DATA.resolve()))
    table_df.to_csv(save_path, index=False)
    logging.info(f"Saved Table 1 → {Path(save_path).name}")
    return table_df


# ─────────────────────── MAIN ───────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"step4_figures_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info("=== Topic 9 Phase 8: Figure Generation ===")

    # Check required files
    coupling_path = OUT_METRICS / "coupling_summary.csv"
    cohort_path   = OUT_METRICS / "cohort_clinical_eeg_abp.csv"
    flow_path     = OUT_METRICS / "cohort_flowchart_numbers.json"

    if not coupling_path.exists():
        logging.error(f"coupling_summary.csv not found. Run step3 first.")
        sys.exit(1)

    coupling_df = pd.read_csv(coupling_path)
    logging.info(f"Loaded coupling data: {len(coupling_df)} cases")

    # ── Fig 1: Flowchart ──
    if flow_path.exists():
        plot_flowchart(flow_path, OUT_FIGS / "fig1_flowchart.png")

    # ── Table 1 ──
    if cohort_path.exists():
        make_table1(cohort_path, coupling_path,
                     OUT_METRICS / "table1_baseline.csv")

    # ── Fig 2: Best case panel ──
    # Select case with best NCCI (high coupling + events)
    if 'NCCI' in coupling_df.columns and 'n_events' in coupling_df.columns:
        event_cases = coupling_df[(coupling_df['n_events'] > 0) &
                                    coupling_df['NCCI'].notna()].copy()
        if len(event_cases) > 0:
            # Pick median NCCI case with events for illustration
            mid_idx = event_cases['NCCI'].sort_values().index[len(event_cases) // 2]
            demo_caseid = int(event_cases.loc[mid_idx, 'caseid'])
            feat_path = OUT_FEATURES / f"case_{demo_caseid:04d}_features.parquet"
            if feat_path.exists():
                df_case = pd.read_parquet(feat_path)
                from step3_coupling_analysis import detect_nociceptive_events
                events = detect_nociceptive_events(
                    df_case['MAP_mean'].values,
                    df_case.get('BIS_value', pd.Series(np.full(len(df_case), np.nan))).values
                )
                event_times = [e['time_sec'] for e in events[:3]]
                plot_case_panel(demo_caseid, df_case,
                                 OUT_FIGS / "fig2_case_panel.png",
                                 event_times=event_times)

    # ── Fig 3: Wavelet examples (3 cases) ──
    # Select 3 cases by NCCI tertile
    if 'NCCI' in coupling_df.columns:
        df_sorted = coupling_df.dropna(subset=['NCCI']).sort_values('NCCI')
        n_s = len(df_sorted)
        tertile_ids = [
            int(df_sorted.iloc[n_s // 6]['caseid']),          # low NCCI
            int(df_sorted.iloc[n_s // 2]['caseid']),           # medium NCCI
            int(df_sorted.iloc[int(n_s * 5 / 6)]['caseid']),   # high NCCI
        ]
        case_dfs = []
        for cid in tertile_ids:
            fp = OUT_FEATURES / f"case_{cid:04d}_features.parquet"
            if fp.exists():
                case_dfs.append(pd.read_parquet(fp))
        if len(case_dfs) >= 2:
            plot_wavelet_examples(case_dfs[:3], OUT_FIGS / "fig3_wavelet_coherence.png")

    # ── Fig 4: Event response ──
    plot_event_response(coupling_df, OUT_FIGS / "fig4_event_response.png")

    # ── Fig 5: NCCI distribution ──
    plot_ncci_vs_bis(coupling_df, OUT_FIGS / "fig5_coupling_distributions.png")

    # ── Fig 6: NCCI quartile stratification ──
    plot_coupling_by_quartile(coupling_df, OUT_FIGS / "fig6_ncci_quartiles.png")

    # ── Supp Fig 1: Granger matrix ──
    plot_granger_matrix(coupling_df, OUT_FIGS / "supp_fig1_granger.png")

    logging.info("\n=== All figures saved to: outputs/figures/ ===")
    logging.info("=== Phase 8 Complete ===")


if __name__ == "__main__":
    main()
