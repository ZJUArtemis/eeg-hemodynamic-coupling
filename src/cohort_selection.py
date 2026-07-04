#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: cohort_selection.py | Topic: 9 | Purpose: EEG+ABP cohort selection
Uses multiprocessing (8 cores) to scan vital file headers efficiently.
"""
import os, sys, json, logging, gc
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import vitaldb

# === PATH SAFETY ===
RAW_DATA = Path("data/physionet.org")  # set to your local VitalDB/PhysioNet root
WORK = Path(".")  # repository root; override as needed
OUT_METRICS = WORK / "outputs" / "metrics"
OUT_LOGS = WORK / "outputs" / "logs"
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_LOGS.mkdir(parents=True, exist_ok=True)

DATA_DIR = RAW_DATA / "files" / "vitaldb" / "1.0.0"
VITAL_DIR = DATA_DIR / "vital_files"

TRACKS_OF_INTEREST = {
    'BIS/EEG1_WAV', 'BIS/EEG2_WAV', 'BIS/BIS', 'BIS/SQI', 'BIS/SR',
    'SNUADC/ART', 'Solar8000/ART_MBP', 'Solar8000/HR',
    'Solar8000/ART_SBP', 'Solar8000/ART_DBP',
    'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
    'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE',
}


def check_case_tracks(caseid):
    """Worker: check a single vital file's tracks. Returns dict."""
    vital_path = VITAL_DIR / f"{caseid:04d}.vital"
    if not vital_path.exists():
        return None
    try:
        vf = vitaldb.VitalFile(str(vital_path), header_only=True)
        track_set = set(vf.trks.keys())
        return {
            'caseid': caseid,
            'has_eeg1': 'BIS/EEG1_WAV' in track_set,
            'has_eeg2': 'BIS/EEG2_WAV' in track_set,
            'has_eeg': ('BIS/EEG1_WAV' in track_set) or ('BIS/EEG2_WAV' in track_set),
            'has_abp': 'SNUADC/ART' in track_set,
            'has_bis': 'BIS/BIS' in track_set,
            'has_sqi': 'BIS/SQI' in track_set,
            'has_sr': 'BIS/SR' in track_set,
            'has_map': 'Solar8000/ART_MBP' in track_set,
            'has_hr': 'Solar8000/HR' in track_set,
            'has_ppf': 'Orchestra/PPF20_RATE' in track_set,
            'has_rft': 'Orchestra/RFTN20_RATE' in track_set,
        }
    except Exception:
        return None


def main():
    # === LOGGING ===
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUT_LOGS / f"cohort_selection_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info("=== Topic 9 Phase 1: Cohort Selection (Parallel) ===")

    # Load clinical data
    clinical = pd.read_csv(DATA_DIR / "clinical_data.csv")
    logging.info(f"Total cases: {len(clinical)}")

    # Convert numeric columns
    for col in ['age', 'opstart', 'opend']:
        clinical[col] = pd.to_numeric(clinical[col], errors='coerce')

    exclusion_log = {'total': len(clinical)}

    # Basic clinical filters
    eligible = clinical[clinical['age'] >= 18].copy()
    logging.info(f"After age>=18: {len(eligible)}")

    eligible = eligible[eligible['ane_type'].str.contains('General', na=False)]
    logging.info(f"After General anesthesia: {len(eligible)}")

    eligible = eligible[~eligible['department'].str.contains('Cardiac|Thoracic', na=False, case=False)]
    logging.info(f"After excluding cardiac/thoracic: {len(eligible)}")

    eligible = eligible.dropna(subset=['opstart', 'opend'])
    eligible['op_duration_min'] = (eligible['opend'] - eligible['opstart']) / 60
    eligible = eligible[eligible['op_duration_min'] >= 60]
    logging.info(f"After op duration >= 60 min: {len(eligible)}")

    exclusion_log['after_basic_filters'] = len(eligible)
    eligible_ids = eligible['caseid'].tolist()

    # Parallel track scanning
    n_workers = min(cpu_count(), 8)
    logging.info(f"Scanning {len(eligible_ids)} vital files using {n_workers} workers...")

    t_start = datetime.now()
    with Pool(processes=n_workers) as pool:
        results = pool.map(check_case_tracks, eligible_ids)
    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()
    logging.info(f"Track scan complete in {elapsed:.0f}s ({elapsed/len(eligible_ids):.2f}s/case)")

    # Build dataframe from results
    valid_results = [r for r in results if r is not None]
    track_df = pd.DataFrame(valid_results)
    logging.info(f"Cases scanned: {len(track_df)}")

    # Track availability summary
    for col in ['has_eeg', 'has_abp', 'has_bis', 'has_sqi', 'has_ppf']:
        logging.info(f"  {col}: {track_df[col].sum()}")

    # EEG + ABP cohort
    eeg_abp_mask = track_df['has_eeg'] & track_df['has_abp']
    eeg_abp_ids = track_df[eeg_abp_mask]['caseid'].tolist()
    logging.info(f"\nEEG+ABP cohort: {len(eeg_abp_ids)} cases")

    # BIS + ABP (sensitivity analysis / larger cohort)
    bis_abp_mask = track_df['has_bis'] & track_df['has_abp']
    bis_abp_ids = track_df[bis_abp_mask]['caseid'].tolist()
    logging.info(f"BIS+ABP cohort: {len(bis_abp_ids)} cases")

    # Merge with clinical
    track_df_eeg = track_df[eeg_abp_mask].copy()
    final_cohort = eligible.merge(track_df_eeg, on='caseid', how='inner')

    track_df_bis = track_df[bis_abp_mask].copy()
    bis_cohort = eligible.merge(track_df_bis, on='caseid', how='inner')

    logging.info(f"\nFinal EEG+ABP cohort: {len(final_cohort)}")
    logging.info(f"Final BIS+ABP cohort: {len(bis_cohort)}")

    # Demographics
    if len(final_cohort) > 0:
        logging.info("\n=== EEG+ABP Cohort Demographics ===")
        logging.info(f"Age: {final_cohort['age'].mean():.1f} ± {final_cohort['age'].std():.1f} yr")
        logging.info(f"Sex M%: {(final_cohort['sex']=='M').mean()*100:.1f}%")
        logging.info(f"Op duration: {final_cohort['op_duration_min'].mean():.0f} ± {final_cohort['op_duration_min'].std():.0f} min")
        logging.info(f"ASA distribution:\n{final_cohort['asa'].value_counts().to_string()}")
        logging.info(f"\nTop departments:\n{final_cohort['department'].value_counts().head(8).to_string()}")

    exclusion_log['eeg_abp_cohort'] = len(final_cohort)
    exclusion_log['bis_abp_cohort'] = len(bis_cohort)

    # Save outputs
    assert not str((OUT_METRICS / "eligible_caseids_eeg_abp.csv").resolve()).startswith(str(RAW_DATA.resolve()))

    final_cohort[['caseid']].to_csv(OUT_METRICS / "eligible_caseids_eeg_abp.csv", index=False)
    bis_cohort[['caseid']].to_csv(OUT_METRICS / "eligible_caseids_bis_abp.csv", index=False)
    final_cohort.to_csv(OUT_METRICS / "cohort_clinical_eeg_abp.csv", index=False)
    track_df.to_csv(OUT_METRICS / "track_availability.csv", index=False)

    with open(OUT_METRICS / "cohort_flowchart_numbers.json", 'w') as f:
        json.dump(exclusion_log, f, indent=2)

    logging.info("\nSaved all outputs to: outputs/metrics/")
    logging.info("=== Phase 1 Step 1 Complete ===")

    return final_cohort, bis_cohort


if __name__ == "__main__":
    main()
