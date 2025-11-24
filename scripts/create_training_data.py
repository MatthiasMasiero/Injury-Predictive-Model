"""
create_training_data.py - Generate cleaned training data with injury labels

Creates two output files:
1. training_data.csv - Full processed GPS data with injury labels
2. summary_data.csv - Aggregated features per athlete/session
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_injury_labels(config_dir):
    """Load detected injuries from CSV."""
    injury_file = Path(config_dir) / 'detected_injuries.csv'
    if not injury_file.exists():
        print(f"Warning: {injury_file} not found. Run detect_injuries.py first.")
        return {}

    df = pd.read_csv(injury_file)

    # Create lookup: (athlete_id, activity_id, date) -> injury info
    injury_lookup = {}
    for _, row in df.iterrows():
        key = (row['athlete_id'], row['activity_id'], row['date'])
        injury_lookup[key] = {
            'injured': True,
            'reason': row['reason'],
            'file_size': row['file_size']
        }

    print(f"Loaded {len(df)} injury records")
    print(f"Unique injured athletes: {df['athlete_name'].nunique()}")
    return injury_lookup


def process_session(file_path, injury_lookup):
    """Process a single session file and return cleaned data."""

    filename = file_path.name
    parts = filename.replace('.json', '').split('_')
    if len(parts) != 3:
        return None

    date, athlete_id, activity_id = parts

    # Load JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        elif isinstance(data, list):
            return None

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

    # Check if this session is marked as injured
    key = (athlete_id, activity_id, date)
    injury_info = injury_lookup.get(key, {'injured': False, 'reason': '', 'file_size': file_path.stat().st_size})

    # Extract athlete info
    athlete_name = f"{data.get('athlete_first_name', 'Unknown')} {data.get('athlete_last_name', '')}".strip()
    data_points = data.get('data', [])

    if len(data_points) == 0:
        # No data - definitely injured or didn't participate
        return {
            'date': date,
            'athlete_id': athlete_id,
            'athlete_name': athlete_name,
            'activity_id': activity_id,
            'injured': True,
            'reason': injury_info['reason'] or 'NO DATA',
            'data_points': 0,
            'duration_minutes': 0,
            'avg_velocity': 0,
            'max_velocity': 0,
            'avg_acceleration': 0,
            'max_acceleration': 0,
            'avg_player_load': 0,
            'max_player_load': 0,
            'avg_metabolic_power': 0,
            'max_metabolic_power': 0,
        }

    # Process GPS data
    df = pd.DataFrame(data_points)

    # Calculate duration
    if len(df) > 0:
        duration_seconds = df['ts'].max() - df['ts'].min()
        duration_minutes = duration_seconds / 60
    else:
        duration_minutes = 0

    # Calculate summary statistics
    summary = {
        'date': date,
        'athlete_id': athlete_id,
        'athlete_name': athlete_name,
        'activity_id': activity_id,
        'injured': injury_info['injured'],
        'reason': injury_info['reason'],
        'data_points': len(df),
        'duration_minutes': duration_minutes,
        'avg_velocity': df['v'].mean() if 'v' in df.columns else 0,
        'max_velocity': df['v'].max() if 'v' in df.columns else 0,
        'std_velocity': df['v'].std() if 'v' in df.columns else 0,
        'avg_acceleration': df['a'].mean() if 'a' in df.columns else 0,
        'max_acceleration': df['a'].max() if 'a' in df.columns else 0,
        'std_acceleration': df['a'].std() if 'a' in df.columns else 0,
        'avg_player_load': df['pl'].mean() if 'pl' in df.columns else 0,
        'max_player_load': df['pl'].max() if 'pl' in df.columns else 0,
        'avg_metabolic_power': df['mp'].mean() if 'mp' in df.columns else 0,
        'max_metabolic_power': df['mp'].max() if 'mp' in df.columns else 0,
        'avg_smooth_load': df['sl'].mean() if 'sl' in df.columns else 0,
    }

    return summary


def create_training_data():
    """Main function to create training datasets."""

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    config_dir = base_dir / 'config'
    output_dir = base_dir / 'config'

    print(f"\n{'='*80}")
    print(f"CREATING TRAINING DATA")
    print(f"{'='*80}\n")

    # Load injury labels
    injury_lookup = load_injury_labels(config_dir)

    # Process all sessions
    files = list(data_dir.glob("*.json"))
    print(f"\nProcessing {len(files)} session files...")

    session_summaries = []
    for i, file_path in enumerate(files, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(files)} files...")

        summary = process_session(file_path, injury_lookup)
        if summary:
            session_summaries.append(summary)

    # Create DataFrame
    df = pd.DataFrame(session_summaries)

    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total sessions: {len(df)}")
    print(f"Injured sessions: {df['injured'].sum()}")
    print(f"Normal sessions: {(~df['injured']).sum()}")
    print(f"Unique athletes: {df['athlete_name'].nunique()}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"\nInjured athletes:")
    injured_athletes = df[df['injured']]['athlete_name'].value_counts()
    for name, count in injured_athletes.items():
        print(f"  {name}: {count} sessions")

    # Save summary data
    output_file = output_dir / 'training_data_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Training data saved to: {output_file}")
    print(f"{'='*80}\n")

    # Create a clean version with only athletes who have full data
    df_clean = df[df['data_points'] > 0].copy()
    print(f"\nClean dataset (with GPS data): {len(df_clean)} sessions")

    output_clean = output_dir / 'training_data_clean.csv'
    df_clean.to_csv(output_clean, index=False)
    print(f"Clean data saved to: {output_clean}\n")

    return df


if __name__ == "__main__":
    create_training_data()
