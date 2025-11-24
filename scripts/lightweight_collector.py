"""
lightweight_collector.py - Collect and immediately process to save space

Instead of storing raw JSON (large), we:
1. Fetch from API
2. Extract features immediately
3. Save only the features (CSV) - 100x smaller
4. Optionally keep raw data for specific sessions
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import dotenv

# Load environment
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(env_path)

TOKEN = os.getenv("MSOC_API_KEY")
REGION = "us"
BASE = f"https://connect-{REGION}.catapultsports.com/api/v6"
HEADERS = {"accept": "application/json", "authorization": f"Bearer {TOKEN}"}


def get_activities(date):
    """Get activities for a date."""
    params = {"start_date": date, "end_date": date}
    r = requests.get(f"{BASE}/activities", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("items", [])


def get_roster(activity_id):
    """Get roster for activity."""
    r = requests.get(f"{BASE}/activities/{activity_id}/athletes", headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("items", [])


def get_sensor(activity_id, athlete_id):
    """Get sensor data."""
    url = f"{BASE}/activities/{activity_id}/athletes/{athlete_id}/sensor"
    r = requests.get(url, headers=HEADERS, params={"stream_type": "gps"}, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_features(data, date, activity_id):
    """Extract features from raw data instead of storing entire JSON."""

    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    elif isinstance(data, list):
        return None

    athlete_id = data.get('athlete_id', '')
    athlete_name = f"{data.get('athlete_first_name', '')} {data.get('athlete_last_name', '')}".strip()
    data_points = data.get('data', [])

    if len(data_points) == 0:
        return {
            'date': date,
            'athlete_id': athlete_id,
            'athlete_name': athlete_name,
            'activity_id': activity_id,
            'data_points': 0,
            'duration_minutes': 0,
        }

    # Convert to DataFrame for analysis
    df = pd.DataFrame(data_points)

    # Calculate duration
    duration_seconds = df['ts'].max() - df['ts'].min()
    duration_minutes = duration_seconds / 60

    # Session-level features
    features = {
        'date': date,
        'athlete_id': athlete_id,
        'athlete_name': athlete_name,
        'activity_id': activity_id,
        'data_points': len(df),
        'duration_minutes': duration_minutes,

        # Velocity features
        'avg_velocity': df['v'].mean(),
        'max_velocity': df['v'].max(),
        'std_velocity': df['v'].std(),
        'p90_velocity': df['v'].quantile(0.9),

        # Acceleration features
        'avg_acceleration': df['a'].mean(),
        'max_acceleration': df['a'].max(),
        'std_acceleration': df['a'].std(),
        'p90_acceleration': df['a'].quantile(0.9),

        # Load features
        'avg_player_load': df['pl'].mean(),
        'max_player_load': df['pl'].max(),
        'total_player_load': df['pl'].max() - df['pl'].min(),

        # Metabolic power
        'avg_metabolic_power': df['mp'].mean(),
        'max_metabolic_power': df['mp'].max(),
        'p90_metabolic_power': df['mp'].quantile(0.9),

        # Smooth load
        'avg_smooth_load': df['sl'].mean(),

        # Heart rate (if available)
        'avg_heart_rate': df['hr'].mean() if df['hr'].max() > 0 else None,
        'max_heart_rate': df['hr'].max() if df['hr'].max() > 0 else None,
    }

    return features


def collect_date_lightweight(date, output_csv, save_raw=False):
    """Collect data for one date and save features only."""

    print(f"\n{'='*60}")
    print(f"Processing {date}")
    print(f"{'='*60}")

    activities = get_activities(date)
    if not activities:
        print(f"  No activities found")
        return []

    print(f"  Found {len(activities)} activities")

    all_features = []

    for act in activities:
        act_id = act["id"]
        roster = get_roster(act_id)

        if not roster:
            continue

        print(f"  Activity {act_id}: {len(roster)} athletes")

        for i, ath in enumerate(roster, 1):
            ath_id = ath["id"]

            try:
                raw_data = get_sensor(act_id, ath_id)
                features = extract_features(raw_data, date, act_id)

                if features:
                    all_features.append(features)
                    print(f"    [{i}/{len(roster)}] {features['athlete_name']}: {features['data_points']} points")

                # Optionally save raw data for injured sessions
                if save_raw and features and features['data_points'] > 0:
                    raw_dir = Path(output_csv).parent.parent / 'data' / 'raw' / date
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_file = raw_dir / f"{date}_{ath_id}_{act_id}.json"
                    with open(raw_file, 'w') as f:
                        json.dump(raw_data, f)

            except Exception as e:
                print(f"    [{i}/{len(roster)}] Error: {e}")

            time.sleep(0.5)  # Rate limiting

    # Append to CSV (not storing JSON!)
    if all_features:
        df = pd.DataFrame(all_features)

        # Append to existing or create new
        if Path(output_csv).exists():
            df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv, index=False)

        print(f"\n  ✓ Saved {len(all_features)} sessions to {output_csv}")

    return all_features


def collect_multiple_dates(dates, output_csv):
    """Collect multiple dates efficiently."""

    print(f"\n{'='*60}")
    print(f"LIGHTWEIGHT DATA COLLECTION")
    print(f"{'='*60}")
    print(f"Dates to collect: {len(dates)}")
    print(f"Output: {output_csv}")
    print(f"Storage method: Features only (CSV)")
    print(f"{'='*60}\n")

    total_sessions = 0
    start_time = time.time()

    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {date}")
        sessions = collect_date_lightweight(date, output_csv)
        total_sessions += len(sessions)

        # Progress update
        elapsed = time.time() - start_time
        rate = i / elapsed * 60  # dates per hour
        remaining = (len(dates) - i) / rate * 60 if rate > 0 else 0

        print(f"\nProgress: {i}/{len(dates)} dates ({i/len(dates)*100:.1f}%)")
        print(f"Total sessions: {total_sessions}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Estimated remaining: {remaining:.1f} minutes")

    # Final summary
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total sessions: {total_sessions}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Check file size
    file_size = Path(output_csv).stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB (vs ~{total_sessions * 15:.0f} MB raw)")
    print(f"Space saved: {(total_sessions * 15 - file_size) / 1024:.1f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single date:   python3 lightweight_collector.py 2025-01-09")
        print("  Multiple dates: python3 lightweight_collector.py 2025-01-09 2025-01-10 2025-01-11")
        sys.exit(1)

    dates = sys.argv[1:]
    output_csv = Path(__file__).parent.parent / 'config' / 'lightweight_features.csv'

    collect_multiple_dates(dates, output_csv)
