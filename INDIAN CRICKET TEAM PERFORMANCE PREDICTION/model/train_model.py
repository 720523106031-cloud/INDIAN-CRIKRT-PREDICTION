"""
India Cricket Team Performance Prediction System - Model Training Script
========================================================================
Algorithm: Random Forest Regressor (Scikit-Learn)
Domain: Cricket Analytics / Sports Performance Modeling
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'cricket_rf_model.joblib')
METRICS_FILE = os.path.join(MODEL_DIR, 'model_metrics.json')
TRAINING_CSV = os.path.join(DATA_DIR, 'cricket_training_dataset.csv')

# Feature Order (MUST be preserved during Flask inference)
FEATURE_COLUMNS = [
    'role_encoded',
    'format_encoded',
    'matches',
    'runs',
    'batting_avg',
    'strike_rate',
    'fours',
    'sixes',
    'wickets',
    'bowling_avg',
    'bowling_econ'
]

ROLE_MAP = {
    'Top-order Batter': 0,
    'Middle-order Batter': 1,
    'Wicketkeeper Batter': 2,
    'Batting All-rounder': 3,
    'Bowling All-rounder': 4,
    'Pace Bowler': 5,
    'Spin Bowler': 6
}

FORMAT_MAP = {
    'Test': 0,
    'ODI': 1,
    'T20I': 2
}


def calculate_cricket_impact_score(role, fmt, matches, runs, bat_avg, sr, fours, sixes, wickets, bowl_avg, bowl_econ):
    """
    Computes an empirical Cricket Performance Index (0-100) based on
    sports analytics principles (ICC rating formulations & batting/bowling impact weights).
    """
    # 1. BATTING COMPONENT
    bat_score = 0.0
    if fmt == 'Test':
        if bat_avg >= 50.0:
            avg_sub = 88.0 + min(12.0, (bat_avg - 50.0) * 1.5)
        elif bat_avg >= 40.0:
            avg_sub = 75.0 + (bat_avg - 40.0) * 1.3
        elif bat_avg >= 30.0:
            avg_sub = 55.0 + (bat_avg - 30.0) * 2.0
        else:
            avg_sub = max(5.0, bat_avg * 1.8)
        sr_sub = min(100.0, (sr / 65.0) * 80.0) if sr > 0 else 0.0
        bat_score = 0.75 * avg_sub + 0.25 * sr_sub

    elif fmt == 'ODI':
        if bat_avg >= 50.0:
            avg_sub = 88.0 + min(12.0, (bat_avg - 50.0) * 1.4)
        elif bat_avg >= 40.0:
            avg_sub = 72.0 + (bat_avg - 40.0) * 1.6
        elif bat_avg >= 30.0:
            avg_sub = 55.0 + (bat_avg - 30.0) * 1.7
        else:
            avg_sub = max(5.0, bat_avg * 1.8)

        if sr >= 100.0:
            sr_sub = 90.0 + min(10.0, (sr - 100.0) * 0.8)
        elif sr >= 85.0:
            sr_sub = 75.0 + (sr - 85.0) * 1.0
        else:
            sr_sub = max(10.0, (sr / 85.0) * 75.0)
        bat_score = 0.55 * avg_sub + 0.45 * sr_sub

    else:  # T20I
        if bat_avg >= 40.0:
            avg_sub = 90.0 + min(10.0, (bat_avg - 40.0) * 1.2)
        elif bat_avg >= 30.0:
            avg_sub = 75.0 + (bat_avg - 30.0) * 1.5
        elif bat_avg >= 20.0:
            avg_sub = 55.0 + (bat_avg - 20.0) * 2.0
        else:
            avg_sub = max(5.0, bat_avg * 2.7)

        if sr >= 160.0:
            sr_sub = 95.0 + min(5.0, (sr - 160.0) * 0.5)
        elif sr >= 140.0:
            sr_sub = 82.0 + (sr - 140.0) * 0.65
        elif sr >= 120.0:
            sr_sub = 65.0 + (sr - 120.0) * 0.85
        else:
            sr_sub = max(10.0, (sr / 120.0) * 65.0)
        bat_score = 0.40 * avg_sub + 0.60 * sr_sub

    # 2. BOWLING COMPONENT
    bowl_score = 0.0
    if wickets > 0:
        wkts_per_match = wickets / max(matches, 1)

        if fmt == 'Test':
            if bowl_avg <= 20.0:
                avg_sub = 94.0 + (20.0 - bowl_avg) * 0.8
            elif bowl_avg <= 25.0:
                avg_sub = 84.0 + (25.0 - bowl_avg) * 2.0
            elif bowl_avg <= 30.0:
                avg_sub = 70.0 + (30.0 - bowl_avg) * 2.8
            else:
                avg_sub = max(10.0, 70.0 - (bowl_avg - 30.0) * 2.0)

            if bowl_econ <= 2.8:
                econ_sub = 92.0 + (2.8 - bowl_econ) * 10.0
            elif bowl_econ <= 3.3:
                econ_sub = 78.0 + (3.3 - bowl_econ) * 28.0
            else:
                econ_sub = max(10.0, 78.0 - (bowl_econ - 3.3) * 25.0)

            wkt_sub = min(100.0, (wkts_per_match / 4.0) * 95.0)
            bowl_score = 0.45 * avg_sub + 0.30 * econ_sub + 0.25 * wkt_sub

        elif fmt == 'ODI':
            if bowl_avg <= 24.0:
                avg_sub = 92.0 + (24.0 - bowl_avg) * 1.5
            elif bowl_avg <= 28.0:
                avg_sub = 80.0 + (28.0 - bowl_avg) * 3.0
            elif bowl_avg <= 35.0:
                avg_sub = 65.0 + (35.0 - bowl_avg) * 2.1
            else:
                avg_sub = max(10.0, 65.0 - (bowl_avg - 35.0) * 1.8)

            if bowl_econ <= 4.8:
                econ_sub = 92.0 + (4.8 - bowl_econ) * 12.0
            elif bowl_econ <= 5.4:
                econ_sub = 78.0 + (5.4 - bowl_econ) * 23.0
            else:
                econ_sub = max(10.0, 78.0 - (bowl_econ - 5.4) * 18.0)

            wkt_sub = min(100.0, (wkts_per_match / 1.8) * 95.0)
            bowl_score = 0.45 * avg_sub + 0.35 * econ_sub + 0.20 * wkt_sub

        else:  # T20I
            if bowl_avg <= 18.0:
                avg_sub = 94.0 + (18.0 - bowl_avg) * 1.0
            elif bowl_avg <= 24.0:
                avg_sub = 80.0 + (24.0 - bowl_avg) * 2.3
            elif bowl_avg <= 30.0:
                avg_sub = 65.0 + (30.0 - bowl_avg) * 2.5
            else:
                avg_sub = max(10.0, 65.0 - (bowl_avg - 30.0) * 1.8)

            if bowl_econ <= 7.0:
                econ_sub = 94.0 + (7.0 - bowl_econ) * 12.0
            elif bowl_econ <= 8.0:
                econ_sub = 80.0 + (8.0 - bowl_econ) * 14.0
            else:
                econ_sub = max(10.0, 80.0 - (bowl_econ - 8.0) * 15.0)

            wkt_sub = min(100.0, (wkts_per_match / 1.4) * 95.0)
            bowl_score = 0.40 * avg_sub + 0.40 * econ_sub + 0.20 * wkt_sub

    # 3. ROLE COMPOSITION
    if role in ['Top-order Batter', 'Middle-order Batter']:
        final_score = bat_score * 0.96 + (min(4.0, bowl_score * 0.04) if wickets > 0 else 0)
    elif role == 'Wicketkeeper Batter':
        final_score = bat_score * 0.90 + 10.0  # Wicketkeeper specialisation value
    elif role == 'Batting All-rounder':
        final_score = bat_score * 0.58 + bowl_score * 0.42
    elif role == 'Bowling All-rounder':
        final_score = bat_score * 0.38 + bowl_score * 0.62
    elif role in ['Pace Bowler', 'Spin Bowler']:
        final_score = bowl_score * 0.95 + min(5.0, bat_score * 0.05)
    else:
        final_score = (bat_score + bowl_score) / 2.0

    # Sample-size confidence factor
    if matches < 3:
        final_score *= 0.88
    elif matches < 10:
        final_score *= 0.95

    return float(np.clip(final_score, 15.0, 99.0))


def build_cricket_dataset():
    records = []

    # 1. Real Indian Squad Data
    players_json_path = os.path.join(DATA_DIR, 'players.json')
    if os.path.exists(players_json_path):
        with open(players_json_path, 'r', encoding='utf-8') as f:
            indian_players = json.load(f)

        for p in indian_players:
            p_role = p.get('role', 'Top-order Batter')
            role_enc = ROLE_MAP.get(p_role, 0)

            for fmt_name, st in p.get('stats', {}).items():
                if st.get('matches', 0) > 0:
                    fmt_enc = FORMAT_MAP.get(fmt_name, 0)
                    m = st.get('matches', 1)
                    r = st.get('runs', 0)
                    b_avg = st.get('batting_avg', 0.0)
                    sr = st.get('strike_rate', 0.0)
                    fours = st.get('fours', 0)
                    sixes = st.get('sixes', 0)
                    w = st.get('wickets', 0)
                    bw_avg = st.get('bowling_avg', 0.0)
                    bw_ec = st.get('bowling_econ', 0.0)

                    score = calculate_cricket_impact_score(
                        p_role, fmt_name, m, r, b_avg, sr, fours, sixes, w, bw_avg, bw_ec
                    )

                    records.append({
                        'player_name': p['name'],
                        'role_encoded': role_enc,
                        'format_encoded': fmt_enc,
                        'matches': m,
                        'runs': r,
                        'batting_avg': b_avg,
                        'strike_rate': sr,
                        'fours': fours,
                        'sixes': sixes,
                        'wickets': w,
                        'bowling_avg': bw_avg,
                        'bowling_econ': bw_ec,
                        'performance_score': score,
                        'data_type': 'verified_real'
                    })

    # 2. International Benchmarks
    intl_benchmarks = [
        ("Steve Smith", "Top-order Batter", "Test", 109, 9685, 56.97, 53.50, 1020, 54, 19, 56.40, 3.40),
        ("Kane Williamson", "Top-order Batter", "Test", 100, 8743, 54.98, 51.40, 940, 24, 30, 40.20, 3.10),
        ("Pat Cummins", "Pace Bowler", "Test", 62, 1280, 16.41, 48.00, 110, 22, 269, 22.53, 2.76),
        ("Mitchell Starc", "Pace Bowler", "ODI", 121, 550, 11.20, 85.00, 42, 14, 236, 22.96, 5.15),
        ("Rashid Khan", "Spin Bowler", "T20I", 92, 420, 14.00, 128.00, 31, 22, 150, 14.15, 6.08),
        ("Ben Stokes", "Batting All-rounder", "Test", 105, 6500, 35.32, 59.20, 750, 131, 203, 32.07, 3.31),
        ("Glenn Maxwell", "Batting All-rounder", "T20I", 106, 2600, 29.50, 155.50, 210, 134, 43, 27.50, 7.50),
        ("Babar Azam", "Top-order Batter", "ODI", 117, 5729, 56.72, 88.75, 520, 60, 0, 0.0, 0.0),
        ("Travis Head", "Top-order Batter", "T20I", 38, 1090, 34.06, 159.35, 115, 52, 1, 35.0, 7.80),
        ("Kagiso Rabada", "Pace Bowler", "Test", 65, 950, 12.30, 52.00, 85, 12, 313, 22.05, 3.40)
    ]

    for name, role, fmt, m, r, b_avg, sr, f4, f6, w, bw_avg, bw_ec in intl_benchmarks:
        score = calculate_cricket_impact_score(role, fmt, m, r, b_avg, sr, f4, f6, w, bw_avg, bw_ec)
        records.append({
            'player_name': name,
            'role_encoded': ROLE_MAP[role],
            'format_encoded': FORMAT_MAP[fmt],
            'matches': m,
            'runs': r,
            'batting_avg': b_avg,
            'strike_rate': sr,
            'fours': f4,
            'sixes': f6,
            'wickets': w,
            'bowling_avg': bw_avg,
            'bowling_econ': bw_ec,
            'performance_score': score,
            'data_type': 'intl_verified'
        })

    # 3. Labelled Statistical Synthetic Augmentation
    np.random.seed(42)
    n_synthetic = 2400

    roles = list(ROLE_MAP.keys())
    formats = list(FORMAT_MAP.keys())

    for i in range(n_synthetic):
        role = np.random.choice(roles)
        fmt = np.random.choice(formats)
        role_enc = ROLE_MAP[role]
        fmt_enc = FORMAT_MAP[fmt]

        m = int(np.random.randint(5, 180))

        if role in ['Top-order Batter', 'Middle-order Batter', 'Wicketkeeper Batter']:
            if fmt == 'Test':
                b_avg = float(np.random.uniform(18.0, 62.0))
                sr = float(np.random.uniform(42.0, 78.0))
            elif fmt == 'ODI':
                b_avg = float(np.random.uniform(20.0, 58.0))
                sr = float(np.random.uniform(72.0, 115.0))
            else:  # T20I
                b_avg = float(np.random.uniform(15.0, 48.0))
                sr = float(np.random.uniform(115.0, 175.0))

            runs = int(m * (b_avg * np.random.uniform(0.75, 1.05)))
            fours = int(runs * np.random.uniform(0.08, 0.12))
            sixes = int(runs * (np.random.uniform(0.01, 0.03) if fmt == 'Test' else np.random.uniform(0.02, 0.06)))

            if np.random.rand() > 0.85:
                w = int(np.random.randint(1, 10))
                bw_avg = float(np.random.uniform(45.0, 95.0))
                bw_ec = float(np.random.uniform(4.5, 8.5))
            else:
                w, bw_avg, bw_ec = 0, 0.0, 0.0

        elif role in ['Batting All-rounder', 'Bowling All-rounder']:
            if fmt == 'Test':
                b_avg = float(np.random.uniform(22.0, 44.0))
                sr = float(np.random.uniform(48.0, 72.0))
                bw_avg = float(np.random.uniform(20.0, 40.0))
                bw_ec = float(np.random.uniform(2.5, 3.8))
                w_per_m = np.random.uniform(1.2, 4.0)
            elif fmt == 'ODI':
                b_avg = float(np.random.uniform(22.0, 42.0))
                sr = float(np.random.uniform(80.0, 120.0))
                bw_avg = float(np.random.uniform(24.0, 44.0))
                bw_ec = float(np.random.uniform(4.5, 6.2))
                w_per_m = np.random.uniform(0.8, 1.8)
            else:  # T20I
                b_avg = float(np.random.uniform(18.0, 36.0))
                sr = float(np.random.uniform(125.0, 160.0))
                bw_avg = float(np.random.uniform(18.0, 34.0))
                bw_ec = float(np.random.uniform(6.8, 9.2))
                w_per_m = np.random.uniform(0.7, 1.5)

            runs = int(m * (b_avg * np.random.uniform(0.60, 0.95)))
            fours = int(runs * np.random.uniform(0.07, 0.11))
            sixes = int(runs * np.random.uniform(0.02, 0.06))
            w = int(m * w_per_m)

        else:  # Bowlers
            if fmt == 'Test':
                b_avg = float(np.random.uniform(4.0, 16.0))
                sr = float(np.random.uniform(30.0, 60.0))
                bw_avg = float(np.random.uniform(18.0, 38.0))
                bw_ec = float(np.random.uniform(2.3, 3.6))
                w_per_m = np.random.uniform(2.5, 5.2)
            elif fmt == 'ODI':
                b_avg = float(np.random.uniform(4.0, 14.0))
                sr = float(np.random.uniform(40.0, 85.0))
                bw_avg = float(np.random.uniform(20.0, 38.0))
                bw_ec = float(np.random.uniform(4.2, 6.0))
                w_per_m = np.random.uniform(1.2, 2.3)
            else:  # T20I
                b_avg = float(np.random.uniform(2.0, 12.0))
                sr = float(np.random.uniform(45.0, 110.0))
                bw_avg = float(np.random.uniform(14.0, 32.0))
                bw_ec = float(np.random.uniform(6.0, 9.5))
                w_per_m = np.random.uniform(0.9, 1.9)

            runs = int(m * (b_avg * np.random.uniform(0.3, 0.7)))
            fours = int(runs * 0.08)
            sixes = int(runs * 0.02)
            w = int(m * w_per_m)

        base_score = calculate_cricket_impact_score(
            role, fmt, m, runs, b_avg, sr, fours, sixes, w, bw_avg, bw_ec
        )
        noise = np.random.normal(0, 1.4)
        score = float(np.clip(base_score + noise, 10.0, 99.0))

        records.append({
            'player_name': f"Synthetic_Sample_{i+1}",
            'role_encoded': role_enc,
            'format_encoded': fmt_enc,
            'matches': m,
            'runs': runs,
            'batting_avg': round(b_avg, 2),
            'strike_rate': round(sr, 2),
            'fours': fours,
            'sixes': sixes,
            'wickets': w,
            'bowling_avg': round(bw_avg, 2),
            'bowling_econ': round(bw_ec, 2),
            'performance_score': round(score, 2),
            'data_type': 'synthetic_benchmark'
        })

    df = pd.DataFrame(records)
    df.to_csv(TRAINING_CSV, index=False)
    return df


def train_and_evaluate():
    print("=" * 70)
    print("  INDIA CRICKET PERFORMANCE PREDICTION SYSTEM - MODEL TRAINING")
    print("=" * 70)

    df = build_cricket_dataset()

    X = df[FEATURE_COLUMNS]
    y = df['performance_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n--- OVERALL MODEL EVALUATION METRICS ---")
    print(f"Algorithm:       Random Forest Regressor")
    print(f"Train MAE:       {train_mae:.4f}")
    print(f"Test MAE:        {test_mae:.4f}")
    print(f"Train RMSE:      {train_rmse:.4f}")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print(f"Train R² Score:  {train_r2:.4f}")
    print(f"Test R² Score:   {test_r2:.4f}")

    format_eval = {}
    for fmt_name, fmt_val in FORMAT_MAP.items():
        fmt_mask = (X_test['format_encoded'] == fmt_val)
        if fmt_mask.sum() > 0:
            sub_y_true = y_test[fmt_mask]
            sub_y_pred = y_pred_test[fmt_mask.values]
            sub_mae = mean_absolute_error(sub_y_true, sub_y_pred)
            sub_rmse = np.sqrt(mean_squared_error(sub_y_true, sub_y_pred))
            sub_r2 = r2_score(sub_y_true, sub_y_pred)
            format_eval[fmt_name] = {
                'samples': int(fmt_mask.sum()),
                'mae': round(float(sub_mae), 3),
                'rmse': round(float(sub_rmse), 3),
                'r2_score': round(float(sub_r2), 3)
            }
            print(f"  Format [{fmt_name}]: Samples={fmt_mask.sum()}, MAE={sub_mae:.3f}, RMSE={sub_rmse:.3f}, R²={sub_r2:.3f}")

    importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_model, MODEL_FILE, compress=3)

    metrics_payload = {
        'algorithm': 'Random Forest Regressor',
        'n_estimators': 150,
        'max_depth': 14,
        'random_state': 42,
        'total_dataset_size': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_mae': round(float(test_mae), 3),
        'test_rmse': round(float(test_rmse), 3),
        'test_r2_score': round(float(test_r2), 3),
        'train_r2_score': round(float(train_r2), 3),
        'format_evaluation': format_eval,
        'feature_columns': FEATURE_COLUMNS,
        'feature_importances': sorted_importances,
        'role_mapping': ROLE_MAP,
        'format_mapping': FORMAT_MAP,
        'synthetic_data_limitation_notice': (
            "Synthetic training augmentations are statistical interpolations based on standard cricket performance distributions. "
            "They calibrate non-linear regression boundaries across rare feature combinations and do NOT represent actual upcoming match results."
        )
    }

    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nModel artifact saved successfully to: {MODEL_FILE}")
    print(f"Metrics metadata saved to: {METRICS_FILE}")


if __name__ == '__main__':
    train_and_evaluate()
