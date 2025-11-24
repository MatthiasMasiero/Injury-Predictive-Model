"""
rolling_window_features.py - Calculate moment-by-moment injury risk

This creates features at multiple time points during a session:
- Every 5 minutes: Calculate features from data up to that point
- Enables live prediction during practice
- Shows how risk evolves throughout session
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_session_data(json_file):
    """Load and prepare session data."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    athlete_id = data.get('athlete_id', '')
    athlete_name = f"{data.get('athlete_first_name', '')} {data.get('athlete_last_name', '')}".strip()
    data_points = data.get('data', [])

    if len(data_points) == 0:
        return None, None, None

    df = pd.DataFrame(data_points)
    return df, athlete_id, athlete_name


def calculate_window_features(df, start_idx, end_idx, window_minutes):
    """Calculate features for a time window."""

    window_df = df.iloc[start_idx:end_idx]

    if len(window_df) == 0:
        return None

    # Calculate features for this window
    features = {
        'window_minutes': window_minutes,
        'data_points': len(window_df),

        # Velocity features
        'avg_velocity': window_df['v'].mean(),
        'max_velocity': window_df['v'].max(),
        'std_velocity': window_df['v'].std(),
        'p90_velocity': window_df['v'].quantile(0.9),

        # Acceleration features
        'avg_acceleration': window_df['a'].mean(),
        'max_acceleration': window_df['a'].max(),
        'std_acceleration': window_df['a'].std(),
        'p90_acceleration': window_df['a'].quantile(0.9),

        # Load features
        'cumulative_player_load': window_df['pl'].iloc[-1] if len(window_df) > 0 else 0,
        'avg_player_load_rate': (window_df['pl'].iloc[-1] - window_df['pl'].iloc[0]) / window_minutes if window_minutes > 0 else 0,

        # Metabolic power
        'avg_metabolic_power': window_df['mp'].mean(),
        'max_metabolic_power': window_df['mp'].max(),
        'p90_metabolic_power': window_df['mp'].quantile(0.9),

        # Recent intensity (last 2 minutes of window)
        'recent_avg_velocity': window_df['v'].tail(1200).mean(),  # Last 2 min at 10Hz
        'recent_max_acceleration': window_df['a'].tail(1200).max(),

        # Smooth load
        'avg_smooth_load': window_df['sl'].mean(),
    }

    return features


def create_rolling_features(json_file, window_intervals=[5, 10, 15, 20, 30, 45, 60]):
    """
    Create features at multiple time points during session.

    window_intervals: List of minutes at which to calculate features
                     [5, 10, 15] = features at 5min, 10min, 15min into session
    """

    df, athlete_id, athlete_name = load_session_data(json_file)

    if df is None:
        return None

    # Calculate session duration
    start_ts = df['ts'].iloc[0]
    df['seconds_from_start'] = df['ts'] - start_ts
    df['minutes_from_start'] = df['seconds_from_start'] / 60

    total_duration = df['minutes_from_start'].max()

    # Get metadata
    filename = Path(json_file).name
    parts = filename.replace('.json', '').split('_')
    date, _, activity_id = parts

    rolling_features = []

    # Calculate features at each time interval
    for window_min in window_intervals:
        if window_min > total_duration:
            break  # Session ended before this window

        # Get all data up to this point
        window_mask = df['minutes_from_start'] <= window_min
        end_idx = window_mask.sum()

        if end_idx < 100:  # Need at least 10 seconds of data (100 points at 10Hz)
            continue

        features = calculate_window_features(df, 0, end_idx, window_min)

        if features:
            features.update({
                'date': date,
                'athlete_id': athlete_id,
                'athlete_name': athlete_name,
                'activity_id': activity_id,
                'session_duration': total_duration,
            })
            rolling_features.append(features)

    return rolling_features


def process_all_sessions(data_dir, output_csv, window_intervals=[5, 10, 15, 20, 30]):
    """Process all sessions to create rolling window dataset."""

    print(f"\n{'='*80}")
    print(f"CREATING ROLLING WINDOW FEATURES")
    print(f"{'='*80}")
    print(f"Time intervals: {window_intervals} minutes")
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_csv}\n")

    all_features = []
    processed = 0
    skipped = 0

    # Find all JSON files
    json_files = []
    for date_folder in sorted(data_dir.iterdir()):
        if date_folder.is_dir() and date_folder.name.startswith('202'):
            json_files.extend(date_folder.glob('*.json'))

    print(f"Found {len(json_files)} session files\n")

    for i, json_file in enumerate(json_files, 1):
        if i % 20 == 0:
            print(f"Processed {i}/{len(json_files)} files...")

        try:
            session_features = create_rolling_features(json_file, window_intervals)

            if session_features:
                all_features.extend(session_features)
                processed += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            skipped += 1

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Save
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*80}")
    print(f"ROLLING FEATURES CREATED")
    print(f"{'='*80}")
    print(f"Sessions processed: {processed}")
    print(f"Sessions skipped: {skipped}")
    print(f"Total feature rows: {len(df)}")
    print(f"Average windows per session: {len(df) / processed if processed > 0 else 0:.1f}")
    print(f"Saved to: {output_csv}")
    print(f"{'='*80}\n")

    return df


def simulate_live_prediction(json_file, model=None):
    """
    Simulate live prediction during a session.
    Shows how risk evolves minute-by-minute.
    """

    print(f"\n{'='*80}")
    print(f"SIMULATING LIVE RISK PREDICTION")
    print(f"{'='*80}\n")

    # Create features every 1 minute
    features = create_rolling_features(json_file, window_intervals=range(1, 121))  # 1-120 minutes

    if not features:
        print("No data to process")
        return

    athlete_name = features[0]['athlete_name']
    print(f"Athlete: {athlete_name}")
    print(f"Session duration: {features[-1]['session_duration']:.1f} minutes\n")

    print(f"{'Time':>6} | {'Risk':>5} | {'Velocity':>8} | {'Accel':>7} | {'Load':>7}")
    print(f"{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")

    for f in features:
        # If you have a trained model, use it here:
        # risk = model.predict_proba([feature_vector])[0][1]

        # For now, use a simple heuristic
        risk_score = (
            f['max_velocity'] * 0.2 +
            f['max_acceleration'] * 0.3 +
            f['avg_player_load_rate'] * 0.3 +
            f['p90_metabolic_power'] * 0.2
        ) / 100  # Normalize

        risk_pct = min(100, max(0, risk_score * 100))

        print(f"{f['window_minutes']:4.0f} min | {risk_pct:5.1f}% | "
              f"{f['avg_velocity']:6.2f} m/s | "
              f"{f['max_acceleration']:5.2f} m/s² | "
              f"{f['cumulative_player_load']:6.1f}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    output_csv = base_dir / 'config' / 'rolling_window_features.csv'

    if len(sys.argv) > 1 and sys.argv[1] == 'simulate':
        # Simulate live prediction for one session
        json_files = list((data_dir / '2025-01-09').glob('*.json'))
        if json_files:
            # Pick first file with data
            for f in json_files:
                if f.stat().st_size > 10000:  # Skip tiny files
                    simulate_live_prediction(f)
                    break
    else:
        # Process all sessions
        process_all_sessions(data_dir, output_csv, window_intervals=[5, 10, 15, 20, 30, 45, 60])
