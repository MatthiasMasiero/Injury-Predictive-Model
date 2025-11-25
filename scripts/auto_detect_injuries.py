"""
auto_detect_injuries.py - Automatically detect injured athletes from collected data

For each injury date provided, finds athletes with:
1. No data (didn't participate)
2. Early session exit
3. Minimal data points

Updates injury_dates.csv with athlete names automatically.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import json


def load_lightweight_features(features_csv):
    """Load collected features."""
    if not Path(features_csv).exists():
        print(f"Error: {features_csv} not found!")
        print("Make sure data collection has finished.")
        return None

    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} sessions from {features_csv}")
    return df


def detect_injuries_on_date(df, target_date):
    """Find athletes who likely got injured on a specific date."""

    # Get sessions on target date
    date_sessions = df[df['date'] == target_date].copy()

    if len(date_sessions) == 0:
        print(f"  No data found for {target_date}")
        return []

    # Get all sessions on this date to calculate averages
    avg_duration = date_sessions['duration_minutes'].mean()
    avg_data_points = date_sessions['data_points'].mean()

    injured_athletes = []

    for _, session in date_sessions.iterrows():
        athlete_name = session['athlete_name']
        data_points = session['data_points']
        duration = session['duration_minutes']

        # Criteria for injury detection
        if data_points == 0:
            # No data at all - didn't participate
            injured_athletes.append({
                'athlete_name': athlete_name,
                'athlete_id': session['athlete_id'],
                'reason': 'NO DATA',
                'data_points': 0,
                'duration_minutes': 0
            })
        elif data_points < 1000:  # Less than ~100 seconds
            # Minimal data
            injured_athletes.append({
                'athlete_name': athlete_name,
                'athlete_id': session['athlete_id'],
                'reason': 'MINIMAL DATA',
                'data_points': data_points,
                'duration_minutes': duration
            })
        elif duration < avg_duration * 0.5 and avg_duration > 10:
            # Early exit (less than 50% of average)
            injured_athletes.append({
                'athlete_name': athlete_name,
                'athlete_id': session['athlete_id'],
                'reason': f'EARLY EXIT ({duration:.1f} min vs avg {avg_duration:.1f} min)',
                'data_points': data_points,
                'duration_minutes': duration
            })

    return injured_athletes


def update_injury_dates_csv(injury_dates_csv, detected_injuries):
    """Update injury_dates.csv with detected athlete names."""

    # Load current injury dates
    df_injuries = pd.read_csv(injury_dates_csv)

    print(f"\n{'='*80}")
    print(f"UPDATING INJURY DATES WITH DETECTED ATHLETES")
    print(f"{'='*80}\n")

    updated_count = 0

    for idx, row in df_injuries.iterrows():
        injury_date = row['injury_date']
        current_name = row['athlete_name']

        # Skip if already has a real name
        if 'Unknown' not in current_name:
            continue

        # Find detected injuries for this date
        date_injuries = detected_injuries.get(injury_date, [])

        if not date_injuries:
            print(f"{injury_date}: No injuries detected (might be outside training)")
            continue

        # Handle multiple injuries on same date
        if len(date_injuries) == 1:
            detected = date_injuries[0]
            df_injuries.at[idx, 'athlete_name'] = detected['athlete_name']
            df_injuries.at[idx, 'notes'] = detected['reason']
            print(f"{injury_date}: {detected['athlete_name']} - {detected['reason']}")
            updated_count += 1

        elif len(date_injuries) > 1:
            # Multiple injuries on this date
            # Try to match by position in CSV (Unknown A, Unknown B, etc.)

            if 'A' in current_name and len(date_injuries) > 0:
                detected = date_injuries[0]
                df_injuries.at[idx, 'athlete_name'] = detected['athlete_name']
                df_injuries.at[idx, 'notes'] = detected['reason']
                print(f"{injury_date}: {detected['athlete_name']} (first) - {detected['reason']}")
                updated_count += 1

            elif 'B' in current_name and len(date_injuries) > 1:
                detected = date_injuries[1]
                df_injuries.at[idx, 'athlete_name'] = detected['athlete_name']
                df_injuries.at[idx, 'notes'] = detected['reason']
                print(f"{injury_date}: {detected['athlete_name']} (second) - {detected['reason']}")
                updated_count += 1

            elif 'Unknown' in current_name and not ('A' in current_name or 'B' in current_name):
                # Single Unknown entry but multiple injuries detected
                detected = date_injuries[0]
                df_injuries.at[idx, 'athlete_name'] = detected['athlete_name']
                df_injuries.at[idx, 'notes'] = f"{detected['reason']} (WARNING: {len(date_injuries)} injuries detected on this date)"
                print(f"{injury_date}: {detected['athlete_name']} - {detected['reason']}")
                print(f"  ⚠️  WARNING: {len(date_injuries)} injuries detected, but only 1 entry in CSV")
                updated_count += 1

    # Save updated CSV
    df_injuries.to_csv(injury_dates_csv, index=False)

    print(f"\n{'='*80}")
    print(f"Updated {updated_count} injury records")
    print(f"Saved to: {injury_dates_csv}")
    print(f"{'='*80}\n")

    return df_injuries


def main():
    """Main detection workflow."""

    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'

    features_csv = config_dir / 'lightweight_features.csv'
    injury_dates_csv = config_dir / 'injury_dates.csv'

    print(f"\n{'='*80}")
    print(f"AUTO-DETECTING INJURED ATHLETES")
    print(f"{'='*80}\n")

    # Load features
    df = load_lightweight_features(features_csv)
    if df is None:
        return

    # Load injury dates to check
    df_injuries = pd.read_csv(injury_dates_csv)
    injury_dates = df_injuries['injury_date'].unique()

    print(f"\nChecking {len(injury_dates)} injury dates for injured athletes...\n")

    # Detect injuries for each date
    all_detected = {}

    for injury_date in sorted(injury_dates):
        print(f"\n{injury_date}:")
        detected = detect_injuries_on_date(df, injury_date)

        if detected:
            all_detected[injury_date] = detected
            for athlete in detected:
                print(f"  ✓ {athlete['athlete_name']}: {athlete['reason']}")
        else:
            print(f"  No injuries detected (normal training day or no data)")

    # Update CSV
    if all_detected:
        update_injury_dates_csv(injury_dates_csv, all_detected)
    else:
        print("\n⚠️  No injuries auto-detected. Possible reasons:")
        print("  - Injuries occurred outside of training sessions")
        print("  - Athletes played through injuries (no early exit)")
        print("  - Data collection incomplete")

    # Show final summary
    print(f"\n{'='*80}")
    print(f"INJURY DETECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Review updated file: {injury_dates_csv}")
    print(f"2. Manually correct any mismatches")
    print(f"3. Run: python3 scripts/process_injury_dates.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
