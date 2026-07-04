#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: coupling_features.py | Topic: 9 | Purpose: EEG-Hemodynamic coupling analysis
Computes: Granger Causality + Cross-Wavelet Coherence + NCCI construction
Input: per-case feature parquet files from signal_processing
Output: coupling metrics per case, aggregate statistics, NCCI
"""
import os, sys, json, logging, gc, warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr, ttest_rel
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pywt

warnings.filterwarnings('ignore')

# === PATHS ===
RAW_DATA = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK = Path(".")  # repository root; override as needed
OUT_FEATURES = WORK / "outputs" / "features"
OUT_COUPLING = WORK / "outputs" / "coupling"
OUT_METRICS  = WORK / "outputs" / "metrics"
OUT_LOGS     = WORK / "outputs" / "logs"
for d in [OUT_COUPLING, OUT_METRICS, OUT_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
STEP_SEC = 10        # seconds per window
GC_WINDOW = 60       # windows for Granger (60 * 10s = 10 min)
GC_STEP   = 6        # step (6 * 10s = 1 min)
MAX_LAG   = 5        # max Granger lag
DT = STEP_SEC        # time step in seconds


# ─────────────────────── GRANGER CAUSALITY ───────────────────────

def is_stationary(x, alpha=0.05):
    """ADF test for stationarity."""
    if len(x) < 20 or np.std(x) < 1e-10:
        return True
    try:
        result = adfuller(x, autolag='AIC')
        return result[1] < alpha
    except Exception:
        return True


def compute_granger_window(eeg_series, hemo_series, max_lag=MAX_LAG):
    """Granger causality for a single window pair."""
    # Remove NaN
    valid = ~(np.isnan(eeg_series) | np.isnan(hemo_series))
    if valid.sum() < max_lag * 4 + 10:
        return {k: np.nan for k in ['gc_e2h_F', 'gc_h2e_F', 'gc_e2h_p', 'gc_h2e_p',
                                     'gc_directionality']}

    e = eeg_series[valid]
    h = hemo_series[valid]

    # Stationarize
    e_in = np.diff(e) if not is_stationary(e) else e
    h_in = np.diff(h) if not is_stationary(h) else h
    n = min(len(e_in), len(h_in))
    e_in, h_in = e_in[:n], h_in[:n]

    if n < max_lag * 3:
        return {k: np.nan for k in ['gc_e2h_F', 'gc_h2e_F', 'gc_e2h_p', 'gc_h2e_p',
                                     'gc_directionality']}
    try:
        # EEG → Hemo
        data_eh = np.column_stack([h_in, e_in])
        gc_eh = grangercausalitytests(data_eh, maxlag=max_lag, verbose=False)
        best_eh = min(gc_eh, key=lambda k: gc_eh[k][0]['ssr_ftest'][1])
        F_eh = gc_eh[best_eh][0]['ssr_ftest'][0]
        p_eh = gc_eh[best_eh][0]['ssr_ftest'][1]

        # Hemo → EEG
        data_he = np.column_stack([e_in, h_in])
        gc_he = grangercausalitytests(data_he, maxlag=max_lag, verbose=False)
        best_he = min(gc_he, key=lambda k: gc_he[k][0]['ssr_ftest'][1])
        F_he = gc_he[best_he][0]['ssr_ftest'][0]
        p_he = gc_he[best_he][0]['ssr_ftest'][1]

        return {
            'gc_e2h_F': float(F_eh), 'gc_h2e_F': float(F_he),
            'gc_e2h_p': float(p_eh), 'gc_h2e_p': float(p_he),
            'gc_directionality': float(F_eh - F_he),
        }
    except Exception:
        return {k: np.nan for k in ['gc_e2h_F', 'gc_h2e_F', 'gc_e2h_p', 'gc_h2e_p',
                                     'gc_directionality']}


def windowed_granger(eeg_series, hemo_series, win=GC_WINDOW, step=GC_STEP):
    """Sliding window Granger causality."""
    n = min(len(eeg_series), len(hemo_series))
    results = []
    for start in range(0, n - win, step):
        res = compute_granger_window(eeg_series[start:start + win],
                                      hemo_series[start:start + win])
        res['window_start_idx'] = start
        results.append(res)
    return pd.DataFrame(results)


# ─────────────────────── CROSS-WAVELET COHERENCE ───────────────────────

def wavelet_coherence(signal_x, signal_y, dt=DT):
    """
    Continuous wavelet coherence using Morlet wavelet.
    Returns coherence matrix (scales × time), freqs, time.
    """
    n = min(len(signal_x), len(signal_y))
    x = np.array(signal_x[:n], dtype=np.float64)
    y = np.array(signal_y[:n], dtype=np.float64)

    # Handle NaN by linear interpolation
    for arr in [x, y]:
        nans = np.isnan(arr)
        if nans.any():
            idx = np.arange(len(arr))
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

    # Normalize
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-10)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-10)

    # Scales: periods from 2*dt to n*dt/2, log-spaced
    min_scale = 2
    max_scale = min(n // 4, 200)
    scales = np.arange(min_scale, max_scale, 2)

    # CWT
    coef_x, freqs = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=dt)
    coef_y, _     = pywt.cwt(y, scales, 'cmor1.5-1.0', sampling_period=dt)

    # Cross-wavelet spectrum
    cross = coef_x * np.conj(coef_y)

    # Smooth
    sm = (5, 11)
    smooth_cross_r = uniform_filter(cross.real, sm)
    smooth_cross_i = uniform_filter(cross.imag, sm)
    smooth_x = uniform_filter(np.abs(coef_x) ** 2, sm)
    smooth_y = uniform_filter(np.abs(coef_y) ** 2, sm)

    coherence = (smooth_cross_r ** 2 + smooth_cross_i ** 2) / (smooth_x * smooth_y + 1e-10)
    coherence = np.clip(coherence, 0, 1)

    return coherence, freqs, scales


def wavelet_coherence_summary(coh, freqs):
    """Summarize coherence into frequency bands."""
    # Frequency bands (in Hz, dt=10s so max freq = 0.05 Hz = Nyquist)
    low_mask   = freqs < 0.01   # > 100s period
    mid_mask   = (freqs >= 0.01) & (freqs < 0.03)
    high_mask  = freqs >= 0.03

    def safe_mean(mask):
        if mask.sum() == 0:
            return np.nan
        return float(np.nanmean(coh[mask, :]))

    return {
        'WC_low':  safe_mean(low_mask),
        'WC_mid':  safe_mean(mid_mask),
        'WC_high': safe_mean(high_mask),
        'WC_overall': float(np.nanmean(coh)),
    }


# ─────────────────────── NOCI EVENT DETECTION ───────────────────────

def detect_nociceptive_events(map_series, bis_series, step_sec=STEP_SEC):
    """
    Detect nociceptive hemodynamic responses:
    MAP rises > 15 mmHg in 2 min while BIS < 70
    """
    events = []
    lookback = int(120 / step_sec)  # 2 min

    for i in range(lookback, len(map_series)):
        if np.isnan(map_series[i]) or np.isnan(map_series[i - lookback]):
            continue

        map_delta = map_series[i] - map_series[i - lookback]
        if map_delta > 15:
            bis_now = bis_series[i] if not np.isnan(bis_series[i]) else 50
            if bis_now < 70:  # BIS didn't rise much → dissociation
                bis_delta = (bis_series[i] - bis_series[i - lookback]
                             if not np.isnan(bis_series[i - lookback]) else 0)
                events.append({
                    'time_idx': i,
                    'time_sec': i * step_sec,
                    'map_delta': float(map_delta),
                    'bis_at_event': float(bis_now),
                    'bis_delta': float(bis_delta),
                })
    return events


# ─────────────────────── PER-CASE COUPLING ───────────────────────

def compute_case_coupling(caseid):
    """Compute full coupling analysis for one case."""
    feat_path = OUT_FEATURES / f"case_{caseid:04d}_features.parquet"
    if not feat_path.exists():
        return caseid, None, 'no_features'

    try:
        df = pd.read_parquet(feat_path)
        n = len(df)
        if n < 30:
            return caseid, None, 'too_short'

        # Primary signal pair: ADR (EEG) ↔ MAP_mean (Hemo)
        adr   = df['ADR'].values.astype(float)
        map_v = df['MAP_mean'].values.astype(float)
        bis_v = df.get('BIS_value', pd.Series(np.full(n, np.nan))).values.astype(float)

        # Valid overlap
        valid = ~(np.isnan(adr) | np.isnan(map_v))
        if valid.sum() < 30:
            return caseid, None, 'insufficient_overlap'

        # ── 1. Windowed Granger Causality ──
        gc_df = windowed_granger(adr, map_v)

        # ── 2. Wavelet Coherence (on full valid series) ──
        adr_full = adr[valid]
        map_full = map_v[valid]

        coh, freqs, scales = wavelet_coherence(adr_full, map_full)
        wc_summary = wavelet_coherence_summary(coh, freqs)

        # ── 3. Aggregate coupling metrics ──
        coupling = {
            'caseid': caseid,
            'n_windows': int(n),
            'n_valid': int(valid.sum()),

            # Granger (mean across sliding windows)
            'GC_e2h_F_mean': float(np.nanmean(gc_df['gc_e2h_F'])),
            'GC_h2e_F_mean': float(np.nanmean(gc_df['gc_h2e_F'])),
            'GC_directionality': float(np.nanmean(gc_df['gc_directionality'])),
            'GC_e2h_sig': float((gc_df['gc_e2h_p'] < 0.05).mean()),  # fraction significant

            # Wavelet coherence
            **wc_summary,

            # Basic correlation
            'pearson_r': float(pearsonr(adr_full, map_full)[0]),
        }

        # ── 4. Nociceptive event analysis ──
        events = detect_nociceptive_events(map_v, bis_v)
        coupling['n_events'] = len(events)

        if events:
            event_ncci_pre  = []
            event_ncci_post = []
            pre_w = 30   # 5 min pre
            post_w = 30  # 5 min post

            # NCCI proxy: WC_overall per window (simplified for per-case)
            # Use ADR as NCCI proxy at this stage
            for ev in events:
                t = ev['time_idx']
                if t - pre_w < 0 or t + post_w >= len(adr):
                    continue
                pre  = np.nanmean(adr[t - pre_w:t])
                post = np.nanmean(adr[t:t + post_w])
                event_ncci_pre.append(pre)
                event_ncci_post.append(post)

            if len(event_ncci_pre) > 0:
                coupling['event_adr_pre']  = float(np.nanmean(event_ncci_pre))
                coupling['event_adr_post'] = float(np.nanmean(event_ncci_post))
                coupling['event_adr_delta'] = float(np.nanmean(event_ncci_post) -
                                                      np.nanmean(event_ncci_pre))
            else:
                coupling.update({'event_adr_pre': np.nan, 'event_adr_post': np.nan,
                                  'event_adr_delta': np.nan})
        else:
            coupling.update({'event_adr_pre': np.nan, 'event_adr_post': np.nan,
                              'event_adr_delta': np.nan})

        # Save per-case GC details
        gc_out = OUT_COUPLING / f"case_{caseid:04d}_gc.parquet"
        assert not str(gc_out.resolve()).startswith(str(RAW_DATA.resolve()))
        gc_df['caseid'] = caseid
        gc_df.to_parquet(gc_out, index=False)

        # Save coherence summary (not full matrix — too large)
        coupling['WC_high_frac'] = float(np.nanmean(coh[freqs >= 0.03, :]))

        del df, coh
        gc.collect()

        return caseid, coupling, 'ok'

    except Exception as e:
        return caseid, None, f'error_{str(e)[:80]}'


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"coupling_features_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info("=== Topic 9 Phase 4-5: Coupling Analysis ===")

    # Find processed feature files
    feat_files = sorted(OUT_FEATURES.glob("case_*.parquet"))
    case_ids = [int(f.stem.split('_')[1]) for f in feat_files]
    logging.info(f"Found {len(case_ids)} processed cases")

    if len(case_ids) == 0:
        logging.error("No feature files found. Run signal_processing first.")
        sys.exit(1)

    # Process in parallel
    n_workers = min(6, cpu_count())
    logging.info(f"Computing coupling with {n_workers} workers...")

    all_coupling = []
    batch_size = 50

    for batch_start in range(0, len(case_ids), batch_size):
        batch = case_ids[batch_start:batch_start + batch_size]
        logging.info(f"Batch {batch_start//batch_size + 1}: cases {batch[0]}-{batch[-1]}")

        with Pool(processes=n_workers) as pool:
            results = pool.map(compute_case_coupling, batch)

        ok_count = 0
        for cid, coupling, status in results:
            if coupling is not None:
                all_coupling.append(coupling)
                ok_count += 1
            elif status != 'ok':
                logging.debug(f"  Case {cid}: {status}")

        logging.info(f"  OK: {ok_count}/{len(batch)}")
        gc.collect()

    if not all_coupling:
        logging.error("No coupling data computed!")
        sys.exit(1)

    coupling_df = pd.DataFrame(all_coupling)
    logging.info(f"\nCoupling computed for {len(coupling_df)} cases")

    # ─── NCCI Construction (Phase 5) ───
    logging.info("\n=== Building NCCI ===")

    index_inputs = ['GC_e2h_F_mean', 'WC_overall', 'pearson_r']
    valid_rows = coupling_df[index_inputs].dropna()
    logging.info(f"Cases with all NCCI inputs: {len(valid_rows)}")

    if len(valid_rows) >= 30:
        X_scaled = StandardScaler().fit_transform(valid_rows)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled).flatten()

        ncci = MinMaxScaler().fit_transform(pc1.reshape(-1, 1)).flatten()

        # Ensure NCCI is positively correlated with WC_overall
        if np.corrcoef(ncci, valid_rows['WC_overall'].values)[0, 1] < 0:
            ncci = 1 - ncci

        coupling_df.loc[valid_rows.index, 'NCCI'] = ncci

        logging.info(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.2%}")
        logging.info(f"NCCI: mean={np.nanmean(ncci):.3f}, std={np.nanstd(ncci):.3f}")
        logging.info(f"NCCI range: [{np.nanmin(ncci):.3f}, {np.nanmax(ncci):.3f}]")

    # ─── Statistics ───
    logging.info("\n=== Summary Statistics ===")
    for col in ['GC_e2h_F_mean', 'GC_h2e_F_mean', 'GC_directionality',
                 'WC_overall', 'WC_low', 'WC_mid', 'pearson_r', 'NCCI']:
        if col in coupling_df:
            v = coupling_df[col].dropna()
            logging.info(f"  {col}: {v.mean():.3f} ± {v.std():.3f} (n={len(v)})")

    # Event analysis
    events_df = coupling_df[coupling_df['n_events'] > 0].copy()
    logging.info(f"\nCases with nociceptive events: {len(events_df)}")
    if len(events_df) > 0:
        delta = events_df['event_adr_delta'].dropna()
        logging.info(f"ADR change at events: {delta.mean():.4f} ± {delta.std():.4f}")
        pre_v = events_df['event_adr_pre'].dropna().values
        post_v = events_df['event_adr_post'].dropna().values
        n_paired = min(len(pre_v), len(post_v))
        if n_paired > 5:
            from scipy.stats import ttest_rel
            t, p = ttest_rel(pre_v[:n_paired], post_v[:n_paired])
            logging.info(f"Paired t-test ADR pre vs post: t={t:.3f}, p={p:.4f}")

    # ─── Save ───
    assert not str((OUT_METRICS / "coupling_summary.csv").resolve()).startswith(str(RAW_DATA.resolve()))
    coupling_df.to_csv(OUT_METRICS / "coupling_summary.csv", index=False)
    coupling_df.to_parquet(OUT_METRICS / "coupling_summary.parquet", index=False)

    logging.info(f"\nSaved coupling summary ({len(coupling_df)} cases)")
    logging.info("=== Phase 4-5 Complete ===")


if __name__ == "__main__":
    main()
