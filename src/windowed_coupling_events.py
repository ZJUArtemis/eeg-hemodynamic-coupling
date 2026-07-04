#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: windowed_coupling_events.py | Topic: 9 | Purpose: Window-level NCCI + refined event analysis
Key fix from EXP-TOPIC9-001:
  - NCCI computed per time window (not per case)
  - Stricter event definition: MAP+20 in 90s, BIS<60
  - Per-window NCCI used in event response analysis
"""
import sys, json, logging, gc, warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.stats import ttest_rel, pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import pywt

warnings.filterwarnings('ignore')

RAW_DATA = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK = Path(".")  # repository root; override as needed
OUT_FEATURES = WORK / "outputs" / "features"
OUT_NCCI     = WORK / "outputs" / "ncci"
OUT_METRICS  = WORK / "outputs" / "metrics"
OUT_LOGS     = WORK / "outputs" / "logs"
for d in [OUT_NCCI, OUT_METRICS, OUT_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

STEP_SEC = 10
GC_WIN   = 60   # windows (~10min)
GC_STEP  = 3    # step (~30s)
MAX_LAG  = 5

# Stricter event thresholds
EVENT_MAP_RISE   = 20    # mmHg (increased from 15)
EVENT_LOOKBACK_W = 9     # 90s lookback (was 12=120s)
EVENT_BIS_MAX    = 60    # stricter (was 70)


def is_stationary(x):
    if len(x) < 20 or np.std(x) < 1e-10:
        return True
    try:
        return adfuller(x, autolag='AIC')[1] < 0.05
    except Exception:
        return True


def granger_f(eeg_win, hemo_win, max_lag=MAX_LAG):
    """One-shot GC for a single window pair. Returns (F_e2h, F_h2e)."""
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
        gc_he = grangercausalitytests(np.column_stack([e[:n], h[:n]]), maxlag=max_lag, verbose=False)
        beh = min(gc_eh, key=lambda k: gc_eh[k][0]['ssr_ftest'][1])
        bhe = min(gc_he, key=lambda k: gc_he[k][0]['ssr_ftest'][1])
        return float(gc_eh[beh][0]['ssr_ftest'][0]), float(gc_he[bhe][0]['ssr_ftest'][0])
    except Exception:
        return np.nan, np.nan


def wc_scalar(x_win, y_win, dt=STEP_SEC):
    """Scalar wavelet coherence for a short window."""
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


def compute_windowed_ncci(caseid):
    """
    Compute per-window NCCI for a single case.
    Each window: GC_e2h_F, WC_scalar, pearson_r → combined score
    """
    feat_path = OUT_FEATURES / f"case_{caseid:04d}_features.parquet"
    if not feat_path.exists():
        return caseid, None

    try:
        df = pd.read_parquet(feat_path)
        n = len(df)
        if n < GC_WIN + 10:
            return caseid, None

        adr   = df['ADR'].values.astype(float)
        map_v = df['MAP_mean'].values.astype(float)
        bis_v = df['BIS_value'].values.astype(float) if 'BIS_value' in df.columns else np.full(n, np.nan)

        # Compute per-GC-window metrics
        win_rows = []
        for start in range(0, n - GC_WIN, GC_STEP):
            end = start + GC_WIN
            ew = adr[start:end]
            hw = map_v[start:end]

            gc_e2h, _ = granger_f(ew, hw)
            wc = wc_scalar(ew, hw)

            valid = ~(np.isnan(ew) | np.isnan(hw))
            r = pearsonr(ew[valid], hw[valid])[0] if valid.sum() > 5 else np.nan

            win_rows.append({
                'window_center': start + GC_WIN // 2,
                'time_sec': (start + GC_WIN // 2) * STEP_SEC,
                'GC_e2h': gc_e2h,
                'WC': wc,
                'pearson_r': r,
            })

        if not win_rows:
            return caseid, None

        win_df = pd.DataFrame(win_rows)

        # Build NCCI per window (requires global scaler from cohort — skip for now,
        # use local z-score normalization as proxy)
        cols = ['GC_e2h', 'WC', 'pearson_r']
        valid_rows = win_df[cols].dropna()
        if len(valid_rows) < 5:
            return caseid, None

        X = StandardScaler().fit_transform(valid_rows)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X).flatten()
        ncci = MinMaxScaler().fit_transform(pc1.reshape(-1,1)).flatten()

        # Ensure positive correlation with WC
        if np.corrcoef(ncci, valid_rows['WC'].values)[0,1] < 0:
            ncci = 1 - ncci

        win_df.loc[valid_rows.index, 'NCCI_window'] = ncci

        # ── Nociceptive event analysis with stricter thresholds ──
        events = []
        for i in range(EVENT_LOOKBACK_W, n):
            if np.isnan(map_v[i]) or np.isnan(map_v[i - EVENT_LOOKBACK_W]):
                continue
            map_delta = map_v[i] - map_v[i - EVENT_LOOKBACK_W]
            if map_delta > EVENT_MAP_RISE:
                bis_now = bis_v[i] if not np.isnan(bis_v[i]) else 99
                if bis_now < EVENT_BIS_MAX:
                    events.append({'time_idx': i, 'map_delta': float(map_delta),
                                   'bis': float(bis_now)})

        # For each event, find the NCCI window closest to it
        event_responses = []
        if events and 'NCCI_window' in win_df.columns:
            ncci_series = win_df.set_index('window_center')['NCCI_window']
            pre_w = 3   # 3 GC windows before = ~1.5min
            post_w = 3

            for ev in events:
                t = ev['time_idx']
                # Find GC windows near event
                nearby = win_df[np.abs(win_df['window_center'] - t) < GC_WIN].copy()
                pre_wins  = win_df[win_df['window_center'] <= t - GC_WIN // 2].tail(pre_w)
                post_wins = win_df[win_df['window_center'] >  t + GC_WIN // 4].head(post_w)

                if len(pre_wins) == 0 or len(post_wins) == 0:
                    continue
                pre_ncci  = pre_wins['NCCI_window'].mean() if 'NCCI_window' in pre_wins.columns else np.nan
                post_ncci = post_wins['NCCI_window'].mean() if 'NCCI_window' in post_wins.columns else np.nan
                if not np.isnan(pre_ncci) and not np.isnan(post_ncci):
                    event_responses.append({
                        'pre_ncci': pre_ncci,
                        'post_ncci': post_ncci,
                        'delta_ncci': post_ncci - pre_ncci,
                        'map_delta': ev['map_delta'],
                        'bis': ev['bis'],
                    })

        result = {
            'caseid': caseid,
            'n_gc_windows': len(win_df),
            'NCCI_window_mean': float(np.nanmean(ncci)) if len(ncci) > 0 else np.nan,
            'NCCI_window_std': float(np.nanstd(ncci)) if len(ncci) > 0 else np.nan,
            'n_events_strict': len(events),
            'event_responses': event_responses,
        }

        # Save per-case NCCI windows
        out_path = OUT_NCCI / f"case_{caseid:04d}_ncci.parquet"
        assert not str(out_path.resolve()).startswith(str(RAW_DATA.resolve()))
        win_df.to_parquet(out_path, index=False)

        del df, win_df
        gc.collect()
        return caseid, result

    except Exception as e:
        return caseid, None


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"windowed_coupling_events_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info("=== Topic 9: Window-level NCCI + Refined Event Analysis ===")
    logging.info(f"Event thresholds: MAP+{EVENT_MAP_RISE}mmHg in {EVENT_LOOKBACK_W*STEP_SEC}s, BIS<{EVENT_BIS_MAX}")

    feat_files = sorted(OUT_FEATURES.glob("case_*.parquet"))
    case_ids = [int(f.stem.split('_')[1]) for f in feat_files]
    logging.info(f"Cases to process: {len(case_ids)}")

    n_workers = 4
    batch_size = 30
    all_results = []

    for batch_start in range(0, len(case_ids), batch_size):
        batch = case_ids[batch_start:batch_start + batch_size]
        logging.info(f"Batch {batch_start//batch_size+1}: cases {batch[0]}-{batch[-1]}")
        with Pool(processes=n_workers) as pool:
            results = pool.map(compute_windowed_ncci, batch)
        for cid, res in results:
            if res is not None:
                all_results.append(res)
        logging.info(f"  Done: {sum(1 for _, r in results if r is not None)}/{len(batch)}")
        gc.collect()

    # Aggregate
    logging.info(f"\nTotal cases with window NCCI: {len(all_results)}")

    # Flatten event responses
    all_events = []
    for res in all_results:
        for ev in res.get('event_responses', []):
            ev['caseid'] = res['caseid']
            all_events.append(ev)

    logging.info(f"Total event responses (strict threshold): {len(all_events)}")

    # Cases with strict events
    n_event_cases = sum(1 for r in all_results if r['n_events_strict'] > 0)
    logging.info(f"Cases with strict events: {n_event_cases}/{len(all_results)} "
                  f"({100*n_event_cases/max(len(all_results),1):.1f}%)")

    if all_events:
        ev_df = pd.DataFrame(all_events)
        pre_v  = ev_df['pre_ncci'].dropna().values
        post_v = ev_df['post_ncci'].dropna().values
        delta_v = ev_df['delta_ncci'].dropna().values
        logging.info(f"NCCI pre-event:  {pre_v.mean():.3f} ± {pre_v.std():.3f}")
        logging.info(f"NCCI post-event: {post_v.mean():.3f} ± {post_v.std():.3f}")
        logging.info(f"ΔNCCI:           {delta_v.mean():.4f} ± {delta_v.std():.4f}")
        logging.info(f"% NCCI decrease: {(delta_v < 0).mean()*100:.1f}%")

        n_paired = min(len(pre_v), len(post_v))
        if n_paired > 5:
            t, p = ttest_rel(pre_v[:n_paired], post_v[:n_paired])
            logging.info(f"Paired t-test: t={t:.3f}, p={p:.4f}")

        assert not str((OUT_METRICS / "windowed_ncci_events.csv").resolve()).startswith(str(RAW_DATA.resolve()))
        ev_df.to_csv(OUT_METRICS / "windowed_ncci_events.csv", index=False)

    # Case-level summary
    summary_rows = [{k: v for k, v in r.items() if k != 'event_responses'}
                     for r in all_results]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_METRICS / "windowed_ncci_summary.csv", index=False)
    logging.info(f"\nSaved windowed NCCI summary ({len(summary_df)} cases)")
    logging.info("=== Window-level NCCI Complete ===")


if __name__ == "__main__":
    main()
