#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step2_signal_processing.py | Topic: 9 | Purpose: EEG + ABP feature extraction
Processes each case: EEG band powers + ABP beat features → windowed features
Saves per-case feature parquet files for coupling analysis.
"""
import os, sys, json, logging, gc, warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
import antropy

warnings.filterwarnings('ignore')

# === PATHS ===
RAW_DATA = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK = Path(".")  # repository root; override as needed
DATA_DIR = RAW_DATA / "files" / "vitaldb" / "1.0.0"
VITAL_DIR = DATA_DIR / "vital_files"
OUT_FEATURES = WORK / "outputs" / "features"
OUT_LOGS = WORK / "outputs" / "logs"
OUT_FEATURES.mkdir(parents=True, exist_ok=True)
OUT_LOGS.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
EEG_FS = 128      # Hz
ABP_FS = 500      # Hz
WINDOW_SEC = 30   # seconds
STEP_SEC = 10     # seconds


def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)


# ─────────────────────── EEG PROCESSING ───────────────────────

def preprocess_eeg(eeg_raw, fs=EEG_FS):
    """Bandpass + notch filter + artifact masking."""
    if len(eeg_raw) < fs * 10:  # need at least 10s
        return None, None

    # Replace NaN/Inf
    eeg = np.array(eeg_raw, dtype=np.float32)
    eeg = np.where(np.isfinite(eeg), eeg, 0.0)

    # Bandpass 0.5-45 Hz
    b, a = scipy_signal.butter(4, [0.5, 45], btype='bandpass', fs=fs)
    eeg_filtered = scipy_signal.filtfilt(b, a, eeg)

    # 50 Hz notch (Korean power grid)
    b_n, a_n = scipy_signal.iirnotch(50, Q=30, fs=fs)
    eeg_filtered = scipy_signal.filtfilt(b_n, a_n, eeg_filtered)

    # Artifact mask: amplitude > 200 μV
    artifact_mask = np.abs(eeg_filtered) > 200
    eeg_clean = eeg_filtered.copy()
    eeg_clean[artifact_mask] = np.nan

    return eeg_clean, artifact_mask


def compute_sef95(freqs, psd):
    """95% Spectral Edge Frequency."""
    cum = np.cumsum(psd)
    total = cum[-1]
    if total <= 0:
        return np.nan
    idx = np.searchsorted(cum, 0.95 * total)
    return freqs[min(idx, len(freqs) - 1)]


def extract_eeg_band_powers(eeg_clean, fs=EEG_FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    """Sliding window EEG band power extraction."""
    w = int(window_sec * fs)
    s = int(step_sec * fs)
    n_win = max(0, (len(eeg_clean) - w) // s + 1)

    rows = []
    for i in range(n_win):
        start = i * s
        seg = eeg_clean[start:start + w]

        nan_frac = np.isnan(seg).mean()
        if nan_frac > 0.3:
            rows.append({'window_idx': i, 'nan_frac': nan_frac,
                         **{k: np.nan for k in ['delta_power', 'theta_power', 'alpha_power',
                                                  'beta_power', 'total_power', 'delta_rel',
                                                  'alpha_rel', 'beta_rel', 'ADR', 'SEF95',
                                                  'spectral_entropy', 'suppression_ratio']}})
            continue

        seg_valid = seg[~np.isnan(seg)]
        if len(seg_valid) < fs * 5:
            rows.append({'window_idx': i, 'nan_frac': nan_frac,
                         **{k: np.nan for k in ['delta_power', 'theta_power', 'alpha_power',
                                                  'beta_power', 'total_power', 'delta_rel',
                                                  'alpha_rel', 'beta_rel', 'ADR', 'SEF95',
                                                  'spectral_entropy', 'suppression_ratio']}})
            continue

        nperseg = min(256, len(seg_valid))
        freqs, psd = scipy_signal.welch(seg_valid, fs=fs, nperseg=nperseg,
                                         noverlap=nperseg // 2)

        def band_power(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(np.trapz(psd[mask], freqs[mask])) if mask.sum() > 0 else np.nan

        delta = band_power(0.5, 4)
        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta  = band_power(13, 30)
        total = (delta or 0) + (theta or 0) + (alpha or 0) + (beta or 0)

        # Suppression ratio: fraction of samples < 5 μV
        sr = float(np.nanmean(np.abs(seg) < 5)) * 100

        try:
            sp_ent = float(antropy.spectral_entropy(seg_valid, sf=fs, normalize=True))
        except Exception:
            sp_ent = np.nan

        rows.append({
            'window_idx': i,
            'nan_frac': float(nan_frac),
            'delta_power': delta,
            'theta_power': theta,
            'alpha_power': alpha,
            'beta_power': beta,
            'total_power': float(total),
            'delta_rel': delta / total if total > 0 else np.nan,
            'alpha_rel': alpha / total if total > 0 else np.nan,
            'beta_rel': beta / total if total > 0 else np.nan,
            'ADR': alpha / delta if (delta and delta > 0) else np.nan,
            'SEF95': compute_sef95(freqs, psd),
            'spectral_entropy': sp_ent,
            'suppression_ratio': sr,
        })

    return pd.DataFrame(rows)


# ─────────────────────── ABP PROCESSING ───────────────────────

def extract_abp_windowed(abp_raw, fs=ABP_FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    """Windowed ABP features aligned to EEG windows."""
    w = int(window_sec * fs)
    s = int(step_sec * fs)
    n_win = max(0, (len(abp_raw) - w) // s + 1)

    rows = []
    for i in range(n_win):
        start = i * s
        seg = np.array(abp_raw[start:start + w], dtype=np.float32)

        # Artifact removal: ABP outside [20, 300] mmHg
        valid = (seg >= 20) & (seg <= 300) & np.isfinite(seg)
        valid_frac = valid.mean()

        if valid_frac < 0.7:
            rows.append({'window_idx': i,
                         **{k: np.nan for k in ['MAP_mean', 'SBP_mean', 'DBP_mean',
                                                  'PP_mean', 'MAP_std', 'PP_std',
                                                  'PPV', 'HR_est', 'RMSSD', 'SDNN',
                                                  'dPdt_max_mean']}})
            continue

        seg_v = seg[valid]

        # Find systolic peaks
        try:
            min_height = np.percentile(seg_v, 30)
            distance = int(fs * 0.4)  # min 400ms between beats
            peaks, props = scipy_signal.find_peaks(
                seg, height=min_height, distance=distance,
                prominence=5
            )
        except Exception:
            peaks = np.array([])

        # Compute MAP from waveform directly (1/3 SBP + 2/3 DBP approximation)
        MAP_mean = float(np.nanmean(seg_v))

        if len(peaks) < 3:
            rows.append({'window_idx': i,
                         'MAP_mean': MAP_mean,
                         **{k: np.nan for k in ['SBP_mean', 'DBP_mean', 'PP_mean',
                                                  'MAP_std', 'PP_std', 'PPV', 'HR_est',
                                                  'RMSSD', 'SDNN', 'dPdt_max_mean']}})
            continue

        sbps = seg[peaks]
        # Diastolic = valley between consecutive peaks
        dbps = []
        for j in range(len(peaks) - 1):
            valley_seg = seg[peaks[j]:peaks[j + 1]]
            dbps.append(float(np.min(valley_seg)))
        dbps = np.array(dbps)
        sbps_paired = sbps[:len(dbps)]

        pp = sbps_paired - dbps
        rr_intervals = np.diff(peaks) / fs * 1000  # ms
        hr_est = 60000 / np.nanmean(rr_intervals) if len(rr_intervals) > 0 else np.nan

        # dP/dt max (peak pressure rise rate)
        dpdt_list = []
        for pk in peaks[:min(20, len(peaks))]:
            lo = max(0, pk - int(0.15 * fs))
            hi = pk
            if hi > lo + 2:
                diff_seg = np.diff(seg[lo:hi])
                dpdt_list.append(float(np.max(diff_seg)) * fs)
        dpdt_max_mean = float(np.mean(dpdt_list)) if dpdt_list else np.nan

        rows.append({
            'window_idx': i,
            'MAP_mean': float(MAP_mean),
            'SBP_mean': float(np.nanmean(sbps_paired)),
            'DBP_mean': float(np.nanmean(dbps)),
            'PP_mean': float(np.nanmean(pp)),
            'MAP_std': float(np.nanstd(seg_v)),
            'PP_std': float(np.nanstd(pp)),
            'PPV': float((np.nanmax(pp) - np.nanmin(pp)) / np.nanmean(pp) * 100) if np.nanmean(pp) > 0 else np.nan,
            'HR_est': float(hr_est),
            'SDNN': float(np.nanstd(rr_intervals)),
            'RMSSD': float(np.sqrt(np.nanmean(np.diff(rr_intervals) ** 2))) if len(rr_intervals) > 1 else np.nan,
            'dPdt_max_mean': dpdt_max_mean,
        })

    return pd.DataFrame(rows)


# ─────────────────────── CASE PROCESSOR ───────────────────────

def process_single_case(caseid):
    """Process one case: extract EEG + ABP features, save to parquet."""
    import vitaldb

    out_path = OUT_FEATURES / f"case_{caseid:04d}_features.parquet"
    if out_path.exists():
        return caseid, 'cached'

    vital_path = VITAL_DIR / f"{caseid:04d}.vital"
    if not vital_path.exists():
        return caseid, 'no_file'

    try:
        # Load EEG + ABP waveforms
        vf = vitaldb.VitalFile(str(vital_path),
                                track_names=['BIS/EEG1_WAV', 'BIS/EEG2_WAV',
                                             'SNUADC/ART', 'BIS/BIS', 'BIS/SQI',
                                             'Solar8000/ART_MBP', 'Solar8000/HR'])

        # EEG: prefer EEG1, fall back to EEG2
        eeg_track = None
        eeg_channel = None
        for ch in ['BIS/EEG1_WAV', 'BIS/EEG2_WAV']:
            arr = vf.to_numpy([ch], interval=1/EEG_FS)
            if arr is not None and arr.shape[0] > 0:
                eeg_data = arr[:, 0]
                if not np.all(np.isnan(eeg_data)):
                    eeg_track = eeg_data
                    eeg_channel = ch
                    break

        if eeg_track is None:
            return caseid, 'no_eeg'

        # ABP
        abp_arr = vf.to_numpy(['SNUADC/ART'], interval=1/ABP_FS)
        if abp_arr is None or abp_arr.shape[0] == 0:
            return caseid, 'no_abp'
        abp_data = abp_arr[:, 0]

        # BIS and numeric signals at 1/STEP_SEC resolution
        bis_arr = vf.to_numpy(['BIS/BIS'], interval=STEP_SEC)
        map_arr = vf.to_numpy(['Solar8000/ART_MBP'], interval=STEP_SEC)
        hr_arr  = vf.to_numpy(['Solar8000/HR'], interval=STEP_SEC)

        # Determine overlap duration
        eeg_dur = len(eeg_track) / EEG_FS
        abp_dur = len(abp_data) / ABP_FS
        overlap_sec = min(eeg_dur, abp_dur)

        if overlap_sec < 30 * 60:  # need at least 30 min
            return caseid, f'too_short_{overlap_sec:.0f}s'

        # Trim to overlap
        eeg_trim = eeg_track[:int(overlap_sec * EEG_FS)]
        abp_trim = abp_data[:int(overlap_sec * ABP_FS)]

        # Process EEG
        eeg_clean, artifact_mask = preprocess_eeg(eeg_trim)
        if eeg_clean is None:
            return caseid, 'eeg_preproc_failed'

        eeg_features = extract_eeg_band_powers(eeg_clean)
        abp_features = extract_abp_windowed(abp_trim)

        # Align by window index
        n_windows = min(len(eeg_features), len(abp_features))
        if n_windows < 18:  # need at least 18 windows = 3 minutes effective
            return caseid, 'too_few_windows'

        eeg_f = eeg_features.iloc[:n_windows].reset_index(drop=True)
        abp_f = abp_features.iloc[:n_windows].reset_index(drop=True)

        # Merge
        combined = pd.concat([eeg_f, abp_f.drop(columns=['window_idx'])], axis=1)
        combined.insert(0, 'caseid', caseid)
        combined.insert(1, 'time_sec', combined['window_idx'] * STEP_SEC)
        combined.insert(2, 'eeg_channel', eeg_channel)

        # Add numeric BIS and MAP from monitor (for validation)
        n_num = min(len(combined), len(bis_arr) if bis_arr is not None else 0)
        if n_num > 0 and bis_arr is not None:
            bis_col = bis_arr[:n_num, 0] if bis_arr.shape[0] >= n_num else np.full(n_num, np.nan)
            combined.loc[:n_num-1, 'BIS_value'] = bis_col
        if map_arr is not None:
            n_map = min(len(combined), map_arr.shape[0])
            combined.loc[:n_map-1, 'MAP_monitor'] = map_arr[:n_map, 0]
        if hr_arr is not None:
            n_hr = min(len(combined), hr_arr.shape[0])
            combined.loc[:n_hr-1, 'HR_monitor'] = hr_arr[:n_hr, 0]

        # Save
        assert not str(out_path.resolve()).startswith(str(RAW_DATA.resolve()))
        combined.to_parquet(out_path, index=False, engine='pyarrow')

        del eeg_track, abp_data, eeg_clean, artifact_mask
        gc.collect()

        return caseid, f'ok_{n_windows}w'

    except Exception as e:
        return caseid, f'error_{str(e)[:60]}'


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"step2_signal_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info("=== Topic 9 Phase 2-3: Signal Processing ===")

    # Load cohort
    cohort_file = WORK / "outputs" / "metrics" / "eligible_caseids_eeg_abp.csv"
    if not cohort_file.exists():
        logging.error("Run step1_cohort_selection.py first!")
        sys.exit(1)

    cohort = pd.read_csv(cohort_file)
    case_ids = cohort['caseid'].tolist()
    logging.info(f"Processing {len(case_ids)} cases...")

    # Process in parallel (4 workers to avoid memory pressure)
    n_workers = 4
    batch_size = 100

    all_results = []
    for batch_start in range(0, len(case_ids), batch_size):
        batch = case_ids[batch_start:batch_start + batch_size]
        logging.info(f"Batch {batch_start//batch_size + 1}: cases {batch_start}-{batch_start+len(batch)}")

        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_case, batch)

        all_results.extend(results)

        # Log batch summary
        ok = sum(1 for _, s in results if s.startswith('ok'))
        cached = sum(1 for _, s in results if s == 'cached')
        errors = sum(1 for _, s in results if s.startswith('error'))
        logging.info(f"  OK: {ok}, Cached: {cached}, Errors: {errors}")
        gc.collect()

    # Summary
    results_df = pd.DataFrame(all_results, columns=['caseid', 'status'])
    ok_mask = results_df['status'].str.startswith('ok') | (results_df['status'] == 'cached')
    n_ok = ok_mask.sum()
    logging.info(f"\nProcessed: {n_ok}/{len(case_ids)} cases successfully")

    # Status breakdown
    for status, cnt in results_df['status'].value_counts().head(15).items():
        logging.info(f"  {status}: {cnt}")

    results_df.to_csv(WORK / "outputs" / "metrics" / "step2_processing_status.csv", index=False)
    logging.info("=== Phase 2-3 Complete ===")


if __name__ == "__main__":
    main()
