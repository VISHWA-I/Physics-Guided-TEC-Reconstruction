import os
import sys
import json
import yaml
import math
import pickle
import logging
import hashlib
import calendar
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from codes.other_downloader import fetch_iono_data
from codes.omniweb import fetch_omni_data

def setup_logging():
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'dataset_generation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config():
    config_path = os.path.join(BASE_DIR, 'config', 'dataset_config.yaml')
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_stations():
    stations_path = os.path.join(BASE_DIR, 'config', 'stations.csv')
    if not os.path.exists(stations_path):
        logging.error(f"Stations database not found: {stations_path}")
        sys.exit(1)
    df = pd.read_csv(stations_path)
    return df.set_index('URSI').to_dict('index')

def get_user_inputs():
    print("="*50)
    print(" AUTOMATED DATASET GENERATION PIPELINE ")
    print("="*50)
    
    station = input("Enter Station URSI Code\nExample:\nLL721\n\n").strip().upper()
    start_date = input("Enter Start Date\n\nFormat\n\nYYYY-MM-DD\n\nExample\n\n2024-01-01\n\n").strip()
    end_date = input("Enter End Date\n\nFormat\n\nYYYY-MM-DD\n\nExample\n\n2024-12-31\n\n").strip()
    
    try:
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
    except Exception as e:
        logging.error(f"Invalid dates provided: {e}")
        sys.exit(1)
        
    return station, sd, ed

def fetch_monthly_data(year, month, station, force_redownload):
    cache_dir = os.path.join(BASE_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    giro_cache = os.path.join(cache_dir, f"GIRO_{station}_{year}_{month:02d}.pkl")
    omni_cache = os.path.join(cache_dir, f"OMNI_{year}_{month:02d}.pkl")
    
    # GIRO
    if not force_redownload and os.path.exists(giro_cache):
        df_giro = pd.read_pickle(giro_cache)
        giro_cached = True
    else:
        df_giro = fetch_iono_data(year=year, month=month, station=station, save_excel=False)
        if df_giro is not None and not df_giro.empty:
            df_giro.to_pickle(giro_cache)
        giro_cached = False
        
    # OMNI
    if not force_redownload and os.path.exists(omni_cache):
        df_omni = pd.read_pickle(omni_cache)
        omni_cached = True
    else:
        df_omni = fetch_omni_data(year=year, month=month, save_excel=False)
        if df_omni is not None and not df_omni.empty:
            df_omni.to_pickle(omni_cache)
        omni_cached = False
        
    return (year, month, df_giro, giro_cached, df_omni, omni_cached)

def download_data(station, start_date, end_date, config):
    giro_dfs = []
    omni_dfs = []
    
    months_to_download = pd.date_range(start=start_date.replace(day=1), end=end_date, freq='MS')
    logging.info(f"Determined {len(months_to_download)} month(s) to process.")
    
    max_workers = config.get('max_download_workers', 4)
    force = config.get('force_redownload', False)
    
    cache_hits = 0
    cache_misses = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_monthly_data, dt.year, dt.month, station, force): dt for dt in months_to_download}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading / Reading Cache"):
            try:
                y, m, df_g, g_cached, df_o, o_cached = future.result()
                if df_g is not None and not df_g.empty:
                    giro_dfs.append(df_g)
                if df_o is not None and not df_o.empty:
                    omni_dfs.append(df_o)
                    
                cache_hits += sum([g_cached, o_cached])
                cache_misses += sum([not g_cached, not o_cached])
            except Exception as e:
                logging.error(f"Error fetching data for month: {e}")
                
    logging.info(f"Cache Hits: {cache_hits}, Cache Misses: {cache_misses}")
    return giro_dfs, omni_dfs

def physics_and_quality_validation(df, config, source='GIRO'):
    report = {'source': source, 'initial_rows': len(df), 'dropped_qf_ff': 0, 'dropped_physics': 0}
    initial_rows = len(df)
    
    if source == 'GIRO':
        # Quality Flags
        min_qf = config['quality'].get('min_QF', 0)
        min_ff = config['quality'].get('min_FF', 0.0)
        
        if 'QF' in df.columns:
            df['QF'] = pd.to_numeric(df['QF'], errors='coerce')
            df = df[df['QF'].isna() | (df['QF'] >= min_qf)]
        if 'FF' in df.columns:
            df['FF'] = pd.to_numeric(df['FF'], errors='coerce')
            df = df[df['FF'].isna() | (df['FF'] >= min_ff)]
            
        qf_ff_rows = initial_rows - len(df)
        report['dropped_qf_ff'] = qf_ff_rows
        logging.info(f"Removed {qf_ff_rows} rows due to QF/FF thresholds.")
        
        # Physics validation
        initial_before_physics = len(df)
        phys = config['physics']
        
        for col, min_key, max_key in [('foF2', 'foF2_min', 'foF2_max'), 
                                      ('hmF2', 'hmF2_min', 'hmF2_max'), 
                                      ('TEC', 'TEC_min', 'TEC_max')]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[(df[col] < phys[min_key]) | (df[col] > phys[max_key]), col] = np.nan
        
        if 'B0' in df.columns:
            df['B0'] = pd.to_numeric(df['B0'], errors='coerce')
            df.loc[df['B0'] < phys['B0_min'], 'B0'] = np.nan
            
        report['dropped_physics'] = int(df[['foF2', 'hmF2', 'TEC', 'B0']].isna().sum().sum()) if 'foF2' in df.columns else 0
            
    elif source == 'OMNI':
        phys = config['physics']
        for col, min_key, max_key in [('Dst', 'Dst_min', 'Dst_max'), 
                                      ('Kp', 'Kp_min', 'Kp_max')]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[(df[col] < phys[min_key]) | (df[col] > phys[max_key]), col] = np.nan
                
        if 'F10.7' in df.columns:
            df['F10.7'] = pd.to_numeric(df['F10.7'], errors='coerce')
            df.loc[df['F10.7'] < phys['F107_min'], 'F10.7'] = np.nan
            
        report['dropped_physics'] = int(df[['Dst', 'Kp', 'F10.7']].isna().sum().sum()) if 'Dst' in df.columns else 0
            
    return df, report

def process_and_merge(giro_dfs, omni_dfs, station_info, config):
    if not giro_dfs or not omni_dfs:
        logging.error("Failed to download sufficient data.")
        sys.exit(1)
        
    logging.info("Processing and Merging Datasets using merge_asof...")
    
    physics_reports = []
    
    # Combine and parse GIRO
    df_giro = pd.concat(giro_dfs, ignore_index=True)
    df_giro['Datetime'] = pd.to_datetime(df_giro['DATE'] + ' ' + df_giro['Time']).dt.tz_localize(None)
    df_giro = df_giro.sort_values('Datetime').reset_index(drop=True)
    df_giro, rep_g = physics_and_quality_validation(df_giro, config, 'GIRO')
    physics_reports.append(rep_g)
    
    # Combine and parse OMNI
    df_omni = pd.concat(omni_dfs, ignore_index=True)
    df_omni['Datetime'] = pd.to_datetime(df_omni['YEAR'].astype(str) + df_omni['DOY'].astype(str), format='%Y%j').dt.tz_localize(None)
    df_omni['Datetime'] += pd.to_timedelta(df_omni['Hour'], unit='h')
    df_omni = df_omni.sort_values('Datetime').reset_index(drop=True)
    
    col_mapping = {
        'Kp index': 'Kp', 'ap_index, nT': 'Ap', 'Dst-index, nT': 'Dst',
        'R (Sunspot No.)': 'SSN', 'f10.7_index': 'F10.7'
    }
    df_omni.rename(columns=col_mapping, inplace=True)
    
    # Remove OMNI fill values (999/99.9) and fix Kp scale
    for col in ['Kp', 'Ap', 'Dst', 'SSN', 'F10.7']:
        if col in df_omni.columns:
            df_omni[col] = pd.to_numeric(df_omni[col], errors='coerce')
            df_omni.loc[df_omni[col] >= 999, col] = np.nan
            
    if 'Kp' in df_omni.columns:
        df_omni['Kp'] = df_omni['Kp'] / 10.0
            
    df_omni, rep_o = physics_and_quality_validation(df_omni, config, 'OMNI')
    physics_reports.append(rep_o)

    # Merge asof
    tolerance = pd.Timedelta(minutes=config.get('merge_tolerance_minutes', 15))
    df_merged = pd.merge_asof(
        df_giro, 
        df_omni.drop(columns=[c for c in ['YEAR', 'DOY', 'Hour'] if c in df_omni.columns]), 
        on='Datetime', 
        direction='nearest', 
        tolerance=tolerance
    )
    
    # Merge Stats
    total_giro = len(df_giro)
    total_omni = len(df_omni)
    matched = df_merged['Kp'].notna().sum() if 'Kp' in df_merged.columns else 0
    unmatched_giro = total_giro - matched
    merge_pct = (matched / total_giro) * 100 if total_giro > 0 else 0
    
    logging.info(f"Merge Stats: Matched rows: {matched}, Unmatched GIRO rows: {unmatched_giro}, Merge %: {merge_pct:.2f}%")
    
    # Static Features
    for k, v in station_info.items():
        df_merged[k] = v
        
    return df_merged, physics_reports

def generate_features(df):
    logging.info("Generating Temporal & Flag Features...")
    
    dt_col = df['Datetime']
    df['DayOfYear'] = dt_col.dt.dayofyear
    df['Month'] = dt_col.dt.month
    df['Week'] = dt_col.dt.isocalendar().week
    df['Hour'] = dt_col.dt.hour
    df['Minute'] = dt_col.dt.minute
    
    if 'Longitude' in df.columns and not df['Longitude'].isna().all():
        lon = pd.to_numeric(df['Longitude'].iloc[0], errors='coerce')
        if pd.notna(lon):
            offset = int(lon / 15.0)
            df['LocalTime'] = (df['Hour'] + offset) % 24
        else:
            df['LocalTime'] = df['Hour']
    else:
        df['LocalTime'] = df['Hour']
        
    df['Season'] = (df['Month'] % 12 + 3) // 3
    df['Sin(DayOfYear)'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Cos(DayOfYear)'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Sin(LocalTime)'] = np.sin(2 * np.pi * df['LocalTime'] / 24.0)
    df['Cos(LocalTime)'] = np.cos(2 * np.pi * df['LocalTime'] / 24.0)
            
    df['StormFlag'] = 0
    if 'Dst' in df.columns and 'Kp' in df.columns:
        df.loc[(df['Dst'] < -50) | (df['Kp'] >= 50), 'StormFlag'] = 1
        
    df['QuietFlag'] = 0
    if 'Kp' in df.columns:
        df.loc[(df['Kp'] <= 30), 'QuietFlag'] = 1
        
    df['WeekendFlag'] = dt_col.dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    return df

def clean_data(df, start_date, end_date, config):
    logging.info("Cleaning Data and Handling Missing Values...")
    
    df = df[(df['Datetime'] >= start_date) & 
            (df['Datetime'] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    df = df.sort_values('Datetime').drop_duplicates(subset=['Datetime'])
    
    # Advanced Missing Value Handling
    thresh = config.get('interpolation_threshold_hours', 3)
    # If using 15-minute intervals, 3 hours = 12 steps
    # We infer frequency by taking mode of time diff
    diffs = df['Datetime'].diff().dt.total_seconds()
    if len(diffs) > 1:
        mode_diff = diffs.mode()[0]
        limit = int((thresh * 3600) / mode_diff) if mode_diff > 0 else 12
    else:
        limit = 12
        
    missing_before = df.isna().sum().sum()
    
    df = df.interpolate(method='linear', limit=limit)
    
    missing_after = df.isna().sum().sum()
    interpolated = missing_before - missing_after
    
    logging.info(f"Interpolation Stats: Interpolated: {interpolated}, Remaining Missing: {missing_after}")
    
    # Forward fill OMNI features because they are hourly, so GIRO high-res might have Nans
    omni_cols = ['Kp', 'Ap', 'Dst', 'F10.7', 'SSN']
    exist_omni = [c for c in omni_cols if c in df.columns]
    df[exist_omni] = df[exist_omni].ffill(limit=4)
    
    # Drop rows that still have NaNs in essential features (since we shouldn't train on long gaps)
    # df = df.dropna(subset=config['targets']) 
    return df

def validate_features(df):
    logging.info("Validating Features...")
    issues = []
    
    # Check duplicates
    if df.columns.duplicated().any():
        issues.append("Duplicated columns found.")
        
    # Check constants and NaNs
    for col in df.columns:
        if df[col].isna().all():
            issues.append(f"Column {col} is entirely NaN.")
        elif df[col].nunique() <= 1 and col not in ['Station_Name', 'Country', 'Timezone', 'Latitude', 'Longitude', 'Geomagnetic_Latitude', 'Geomagnetic_Longitude']:
            pass # Constant features are okay for metadata, but worth tracking if dynamic
            
    if issues:
        for i in issues:
            logging.warning(i)
    else:
        logging.info("Feature validation passed.")

def get_next_dataset_version(processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith('dataset_v')]
    if not dirs:
        return 'dataset_v1'
    nums = [int(d.replace('dataset_v', '')) for d in dirs]
    return f"dataset_v{max(nums) + 1}"

def prepare_sliding_windows(df, window_size, horizon, features, targets):
    X, y = [], []
    df = df.dropna(subset=features + targets) # Drop NaNs before windowing
    
    data_x = df[features].values
    data_y = df[targets].values
    
    for i in range(len(df) - window_size - horizon + 1):
        X.append(data_x[i : i + window_size])
        y.append(data_y[i + window_size + horizon - 1])
        
    return np.array(X), np.array(y), df

def generate_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def storm_balancing_report(df):
    if 'Dst' not in df.columns:
        return {}
    quiet = len(df[df['Dst'] > -30])
    moderate = len(df[(df['Dst'] <= -30) & (df['Dst'] > -50)])
    strong = len(df[(df['Dst'] <= -50) & (df['Dst'] > -100)])
    extreme = len(df[df['Dst'] <= -100])
    stats = {'Quiet Days': quiet, 'Moderate Storm': moderate, 'Strong Storm': strong, 'Extreme Storm': extreme}
    logging.info(f"Storm Stats: {stats}")
    return stats

def main():
    setup_logging()
    logging.info("Starting Dataset Generation Pipeline...")
    config = load_config()
    stations_db = load_stations()
    
    station, start_date, end_date = get_user_inputs()
    
    if station not in stations_db:
        logging.error(f"Station {station} not found in stations.csv")
        sys.exit(1)
        
    giro_dfs, omni_dfs = download_data(station, start_date, end_date, config)
    df_merged, physics_reports = process_and_merge(giro_dfs, omni_dfs, stations_db[station], config)
    df_features = generate_features(df_merged)
    df_clean = clean_data(df_features, start_date, end_date, config)
    validate_features(df_clean)
    
    # Save Versioned Dataset
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    version = get_next_dataset_version(processed_dir)
    ver_dir = os.path.join(processed_dir, version)
    os.makedirs(ver_dir, exist_ok=True)
    
    with open(os.path.join(ver_dir, 'physics_validation_report.json'), 'w') as f:
        json.dump(physics_reports, f, indent=4)
    
    master_csv_path = os.path.join(ver_dir, 'master_dataset.csv')
    df_clean.to_csv(master_csv_path, index=False)
    
    dataset_hash = generate_hash(master_csv_path)
    
    # Splitting
    n = len(df_clean)
    train_end = int(n * config.get('train_split', 0.7))
    val_end = int(n * (config.get('train_split', 0.7) + config.get('validation_split', 0.15)))
    
    train_df = df_clean.iloc[:train_end].copy()
    val_df = df_clean.iloc[train_end:val_end].copy()
    test_df = df_clean.iloc[val_end:].copy()
    
    targets = config.get('targets', ['foF2', 'TEC'])
    # All numeric cols except static, Date, and Quality Flags
    feature_cols = [c for c in df_clean.columns if c not in ['DATE', 'Time', 'Datetime', 'Station_Name', 'Country', 'Timezone', 'QF', 'FF'] and pd.api.types.is_numeric_dtype(df_clean[c])]
    
    # Drop columns that are entirely NaN from features to prevent dropping all rows
    empty_cols = [c for c in feature_cols if df_clean[c].isna().all()]
    if empty_cols:
        logging.warning(f"Dropping entirely NaN columns from features: {empty_cols}")
        feature_cols = [c for c in feature_cols if c not in empty_cols]
        
    targets = [t for t in targets if t in df_clean.columns and t not in empty_cols]
    
    scaler_cls = StandardScaler if config.get('scaling_method') == 'StandardScaler' else MinMaxScaler
    f_scaler = scaler_cls()
    t_scaler = scaler_cls()
    
    train_df[feature_cols] = f_scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = f_scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = f_scaler.transform(test_df[feature_cols])
    t_scaler.fit(train_df[targets])
    
    with open(os.path.join(ver_dir, 'feature_scaler.pkl'), 'wb') as f: pickle.dump(f_scaler, f)
    with open(os.path.join(ver_dir, 'target_scaler.pkl'), 'wb') as f: pickle.dump(t_scaler, f)
        
    win = config.get('window_size', 96)
    horiz = config.get('prediction_horizon', 1)
    
    X_train, y_train, _ = prepare_sliding_windows(train_df, win, horiz, feature_cols, targets)
    X_val, y_val, _ = prepare_sliding_windows(val_df, win, horiz, feature_cols, targets)
    X_test, y_test, _ = prepare_sliding_windows(test_df, win, horiz, feature_cols, targets)
    
    with open(os.path.join(ver_dir, 'train_dataset.pkl'), 'wb') as f: pickle.dump({'X': X_train, 'y': y_train}, f)
    with open(os.path.join(ver_dir, 'validation_dataset.pkl'), 'wb') as f: pickle.dump({'X': X_val, 'y': y_val}, f)
    with open(os.path.join(ver_dir, 'test_dataset.pkl'), 'wb') as f: pickle.dump({'X': X_test, 'y': y_test}, f)
        
    train_df.to_csv(os.path.join(ver_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(ver_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(ver_dir, 'test.csv'), index=False)
        
    storm_stats = storm_balancing_report(df_clean)
    
    metadata = {
        'Dataset Version': version,
        'Creation Time': datetime.datetime.now().isoformat(),
        'Start Date': str(start_date.date()),
        'End Date': str(end_date.date()),
        'Station': station,
        'Latitude': stations_db[station].get('Latitude'),
        'Longitude': stations_db[station].get('Longitude'),
        'Features': feature_cols,
        'Targets': targets,
        'Window Size': win,
        'Prediction Horizon': horiz,
        'Training Samples': len(X_train),
        'Validation Samples': len(X_val),
        'Testing Samples': len(X_test),
        'Missing Values': int(df_clean.isna().sum().sum()),
        'Python Version': sys.version,
        'SHA256 Hash': dataset_hash
    }
    with open(os.path.join(ver_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Calculate Dataset Quality Score
    score = 100
    missing = metadata['Missing Values']
    total_cells = df_clean.size
    missing_ratio = missing / total_cells if total_cells > 0 else 1
    score -= (missing_ratio * 30) # Penalize up to 30 points for missing values
    if len(df_clean) < 100: score -= 20 # Penalize for small size
    if physics_reports and 'dropped_qf_ff' in physics_reports[0]:
        qf_ratio = physics_reports[0]['dropped_qf_ff'] / max(1, physics_reports[0]['initial_rows'])
        score -= (qf_ratio * 10)
        
    quality_metrics = {
        'overall_score': max(0, round(score, 2)),
        'completeness_ratio': 1.0 - missing_ratio,
        'missing_values': int(missing),
        'storm_coverage': storm_stats
    }
    with open(os.path.join(ver_dir, 'dataset_quality.json'), 'w') as f:
        json.dump(quality_metrics, f, indent=4)
        
    # Copy files directly to data/processed as a "latest" alias for backwards compatibility
    import shutil
    for fname in ['master_dataset.csv', 'train_dataset.pkl', 'validation_dataset.pkl', 'test_dataset.pkl', 'feature_scaler.pkl', 'target_scaler.pkl', 'train.csv', 'val.csv', 'test.csv']:
        src = os.path.join(ver_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(processed_dir, fname))

    logging.info(f"Dataset successfully created in {ver_dir}")
    print(f"\n[+] Pipeline completed successfully. Data saved to {ver_dir}")
    
    # Generate Plot
    report_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'data_quality_report_{version}.pdf')
    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 11))
        ax.axis('off')
        info_text = (
            "==========================================\n"
            f"   DATA QUALITY REPORT ({version})\n"
            "==========================================\n\n"
            f"Station: {station}\n"
            f"Date Range: {start_date.date()} to {end_date.date()}\n"
            f"Hash: {dataset_hash}\n\n"
            f"Total Master Records: {n}\n"
            f"Training Samples: {len(X_train)}\n"
            f"Validation Samples: {len(X_val)}\n"
            f"Testing Samples: {len(X_test)}\n\n"
            f"Window Size: {win}\n"
            f"Prediction Horizon: {horiz}\n"
            f"Total Missing Values: {metadata['Missing Values']}\n\n"
            f"Storm Stats:\n"
            f"  Quiet Days: {storm_stats.get('Quiet Days', 0)}\n"
            f"  Moderate Storm: {storm_stats.get('Moderate Storm', 0)}\n"
            f"  Strong Storm: {storm_stats.get('Strong Storm', 0)}\n"
            f"  Extreme Storm: {storm_stats.get('Extreme Storm', 0)}\n\n"
            f"Dataset Score: {max(0, round(score, 2))}/100\n"
        )
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', family='monospace')
        pdf.savefig(fig)
        plt.close()

if __name__ == "__main__":
    main()
