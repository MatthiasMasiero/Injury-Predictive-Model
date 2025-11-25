"""
live_risk_monitor.py - Real-time injury risk monitoring

Simulates live risk assessment during/after a training session.
Shows moment-by-moment risk percentage as session progresses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

def load_model():
    """Load trained injury prediction model."""
    base_dir = Path(__file__).parent.parent
    model_file = base_dir / 'models' / 'injury_risk_model.pkl'

    if not model_file.exists():
        print(f"Error: Model not found at {model_file}")
        print("Run: python3 scripts/train_model.py first")
        return None

    return joblib.load(model_file)


def predict_live_risk(session_file, model):
    """Simulate live risk prediction for a session."""

    print(f"\n{'='*80}")
    print(f"LIVE INJURY RISK MONITOR")
    print(f"{'='*80}\n")

    # Load rolling window features for this session
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'

    features_df = pd.read_csv(config_dir / 'rolling_window_features.csv')

    # Extract athlete info from filename
    filename = Path(session_file).name
    parts = filename.replace('.json', '').split('_')
    if len(parts) != 3:
        print(f"Invalid filename format: {filename}")
        return

    date, athlete_id, activity_id = parts

    # Get features for this specific session
    session_features = features_df[
        (features_df['date'] == date) &
        (features_df['athlete_id'] == athlete_id) &
        (features_df['activity_id'] == activity_id)
    ].copy()

    if len(session_features) == 0:
        print(f"No features found for session: {filename}")
        return

    athlete_name = session_features['athlete_name'].iloc[0]
    session_duration = session_features['session_duration'].iloc[0]

    print(f"Athlete: {athlete_name}")
    print(f"Date: {date}")
    print(f"Session Duration: {session_duration:.1f} minutes")
    print(f"\nMonitoring risk at {len(session_features)} time points...\n")

    # Feature columns
    feature_cols = [
        'window_minutes', 'data_points',
        'avg_velocity', 'max_velocity', 'std_velocity', 'p90_velocity',
        'avg_acceleration', 'max_acceleration', 'std_acceleration', 'p90_acceleration',
        'cumulative_player_load', 'avg_player_load_rate',
        'avg_metabolic_power', 'max_metabolic_power', 'p90_metabolic_power',
        'recent_avg_velocity', 'recent_max_acceleration',
        'avg_smooth_load'
    ]

    # Prepare features
    X = session_features[feature_cols].fillna(0)

    # Predict risk at each time point
    risk_probabilities = model.predict_proba(X)[:, 1] * 100  # Convert to percentage

    session_features['risk_percentage'] = risk_probabilities

    # Display moment-by-moment risk
    print(f"{'Time':>6} | {'Risk %':>7} | {'Load':>8} | {'Velocity':>9} | {'Status':>15}")
    print(f"{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*15}")

    for _, row in session_features.iterrows():
        time_min = row['window_minutes']
        risk = row['risk_percentage']
        load = row['cumulative_player_load']
        velocity = row['avg_velocity']

        # Risk status
        if risk < 10:
            status = "✓ Low Risk"
        elif risk < 30:
            status = "⚠ Moderate Risk"
        elif risk < 60:
            status = "⚠⚠ High Risk"
        else:
            status = "🚨 CRITICAL"

        print(f"{time_min:4.0f} min | {risk:6.1f}% | {load:7.1f} | {velocity:7.2f} m/s | {status}")

    # Summary
    max_risk = risk_probabilities.max()
    avg_risk = risk_probabilities.mean()

    print(f"\n{'='*80}")
    print(f"SESSION SUMMARY")
    print(f"{'='*80}")
    print(f"Average Risk: {avg_risk:.1f}%")
    print(f"Peak Risk:    {max_risk:.1f}%")
    print(f"Final Risk:   {risk_probabilities[-1]:.1f}%")

    if max_risk > 60:
        print(f"\n🚨 HIGH RISK DETECTED - Consider early exit or rest")
    elif max_risk > 30:
        print(f"\n⚠️  MODERATE RISK - Monitor closely")
    else:
        print(f"\n✓ Session completed safely")

    print(f"{'='*80}\n")

    return session_features


def monitor_all_athletes(date, model):
    """Monitor all athletes for a specific date."""

    print(f"\n{'='*80}")
    print(f"TEAM RISK REPORT - {date}")
    print(f"{'='*80}\n")

    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'
    features_df = pd.read_csv(config_dir / 'rolling_window_features.csv')

    # Get all athletes for this date
    date_features = features_df[features_df['date'] == date].copy()

    if len(date_features) == 0:
        print(f"No data found for {date}")
        return

    # Feature columns
    feature_cols = [
        'window_minutes', 'data_points',
        'avg_velocity', 'max_velocity', 'std_velocity', 'p90_velocity',
        'avg_acceleration', 'max_acceleration', 'std_acceleration', 'p90_acceleration',
        'cumulative_player_load', 'avg_player_load_rate',
        'avg_metabolic_power', 'max_metabolic_power', 'p90_metabolic_power',
        'recent_avg_velocity', 'recent_max_acceleration',
        'avg_smooth_load'
    ]

    # Predict for all
    X = date_features[feature_cols].fillna(0)
    date_features['risk_percentage'] = model.predict_proba(X)[:, 1] * 100

    # Aggregate per athlete (use peak risk)
    athlete_risk = date_features.groupby('athlete_name').agg({
        'risk_percentage': ['mean', 'max'],
        'session_duration': 'first'
    }).reset_index()

    athlete_risk.columns = ['athlete_name', 'avg_risk', 'peak_risk', 'duration']
    athlete_risk = athlete_risk.sort_values('peak_risk', ascending=False)

    print(f"{'Athlete':25s} | {'Avg Risk':>9} | {'Peak Risk':>10} | {'Duration':>9}")
    print(f"{'-'*25}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}")

    for _, row in athlete_risk.iterrows():
        status = "🚨" if row['peak_risk'] > 60 else "⚠️ " if row['peak_risk'] > 30 else "✓ "
        print(f"{status} {row['athlete_name']:22s} | {row['avg_risk']:8.1f}% | {row['peak_risk']:9.1f}% | {row['duration']:6.1f} min")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Load model
    model = load_model()
    if model is None:
        sys.exit(1)

    if len(sys.argv) > 1:
        # Monitor specific session file
        session_file = sys.argv[1]
        predict_live_risk(session_file, model)
    else:
        # Monitor all athletes on a date
        print("\nShowing team risk report for 2025-01-09...\n")
        monitor_all_athletes('2025-01-09', model)
