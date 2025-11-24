"""
process_injury_dates.py - Process 50 injury dates and create labeled dataset

This script:
1. Reads your injury dates CSV
2. For each injury, labels sessions in the days leading up to it
3. Creates training data with risk levels based on proximity to injury
4. Identifies which dates need data collection
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

def parse_injury_dates(injury_file):
    """Load and parse injury dates."""
    df = pd.read_csv(injury_file)

    # Convert date strings to datetime
    df['injury_date'] = pd.to_datetime(df['injury_date'])

    print(f"\n{'='*80}")
    print(f"LOADED {len(df)} INJURY RECORDS")
    print(f"{'='*80}")
    print(f"Date range: {df['injury_date'].min()} to {df['injury_date'].max()}")
    print(f"Unique athletes: {df['athlete_name'].nunique()}")

    return df


def get_available_dates(data_dir):
    """Find what dates we have data for."""
    available = set()

    for date_folder in data_dir.iterdir():
        if date_folder.is_dir() and date_folder.name.startswith('202'):
            available.add(date_folder.name)

    return sorted(available)


def create_date_ranges(injury_df, lookback_days=7):
    """Create date ranges around each injury."""

    date_ranges = []

    for _, injury in injury_df.iterrows():
        injury_date = injury['injury_date']
        athlete_name = injury['athlete_name']

        # Create range: lookback days before injury through injury date
        for i in range(lookback_days, -1, -1):
            date = injury_date - timedelta(days=i)

            # Determine risk level based on proximity
            if i == 0:
                risk_level = 'injury_day'
            elif i <= 3:
                risk_level = 'high_risk'  # 1-3 days before
            else:
                risk_level = 'moderate_risk'  # 4-7 days before

            date_ranges.append({
                'athlete_name': athlete_name,
                'date': date.strftime('%Y-%m-%d'),
                'injury_date': injury_date.strftime('%Y-%m-%d'),
                'days_until_injury': i,
                'risk_level': risk_level,
                'injury_type': injury.get('injury_type', 'unknown')
            })

    return pd.DataFrame(date_ranges)


def identify_missing_dates(date_ranges_df, available_dates):
    """Identify which dates we need to collect data for."""

    needed_dates = set(date_ranges_df['date'].unique())
    available_dates = set(available_dates)

    missing_dates = sorted(needed_dates - available_dates)

    print(f"\n{'='*80}")
    print(f"DATA AVAILABILITY")
    print(f"{'='*80}")
    print(f"Dates needed: {len(needed_dates)}")
    print(f"Dates available: {len(available_dates)}")
    print(f"Dates missing: {len(missing_dates)}")

    if missing_dates:
        print(f"\nMissing dates (need to collect):")
        for date in missing_dates[:20]:  # Show first 20
            count = len(date_ranges_df[date_ranges_df['date'] == date])
            print(f"  {date} ({count} athletes)")
        if len(missing_dates) > 20:
            print(f"  ... and {len(missing_dates) - 20} more dates")

    return missing_dates


def match_sessions_to_labels(data_dir, date_ranges_df):
    """Match existing session data to injury labels."""

    # Get all available session files
    sessions = []

    for date_folder in sorted(data_dir.iterdir()):
        if not date_folder.is_dir() or not date_folder.name.startswith('202'):
            continue

        date_str = date_folder.name

        for json_file in date_folder.glob('*.json'):
            # Parse filename: DATE_athleteID_activityID.json
            parts = json_file.stem.split('_')
            if len(parts) != 3:
                continue

            _, athlete_id, activity_id = parts

            # Load minimal data to get athlete name
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]

                athlete_name = f"{data.get('athlete_first_name', '')} {data.get('athlete_last_name', '')}".strip()
                data_points = len(data.get('data', []))

            except:
                continue

            sessions.append({
                'date': date_str,
                'athlete_id': athlete_id,
                'athlete_name': athlete_name,
                'activity_id': activity_id,
                'data_points': data_points,
                'file_path': str(json_file)
            })

    sessions_df = pd.DataFrame(sessions)

    # Merge with injury labels
    labeled_df = sessions_df.merge(
        date_ranges_df,
        on=['date', 'athlete_name'],
        how='left'
    )

    # Fill non-injury sessions
    labeled_df['risk_level'] = labeled_df['risk_level'].fillna('normal')
    labeled_df['days_until_injury'] = labeled_df['days_until_injury'].fillna(-1)

    print(f"\n{'='*80}")
    print(f"LABELED SESSION STATISTICS")
    print(f"{'='*80}")
    print(f"Total sessions: {len(labeled_df)}")
    print(f"\nRisk distribution:")
    print(labeled_df['risk_level'].value_counts())

    print(f"\nSessions with GPS data (> 1000 points):")
    df_with_data = labeled_df[labeled_df['data_points'] > 1000]
    print(df_with_data['risk_level'].value_counts())

    return labeled_df


def generate_collection_script(missing_dates, output_file):
    """Generate bash script to collect missing dates."""

    if not missing_dates:
        print("\n✓ All dates already collected!")
        return

    script_lines = [
        "#!/bin/bash",
        "# Auto-generated script to collect missing injury date data",
        "",
        "cd \"$(dirname \"$0\")/..\""
        "",
        "echo 'Collecting data for injury analysis...'",
        "echo ''"
    ]

    for date in missing_dates:
        script_lines.append(f"echo 'Collecting {date}...'")
        script_lines.append(f"python3 scripts/data_collector.py --date {date}")
        script_lines.append("")

    script_lines.append("echo 'Data collection complete!'")
    script_lines.append("echo 'Run: python3 scripts/process_injury_dates.py to update labels'")

    with open(output_file, 'w') as f:
        f.write('\n'.join(script_lines))

    # Make executable
    import os
    os.chmod(output_file, 0o755)

    print(f"\n{'='*80}")
    print(f"COLLECTION SCRIPT GENERATED")
    print(f"{'='*80}")
    print(f"Run this to collect missing dates:")
    print(f"  {output_file}")
    print(f"{'='*80}\n")


def main():
    """Main processing pipeline."""

    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'
    data_dir = base_dir / 'data'

    injury_file = config_dir / 'injury_dates.csv'

    if not injury_file.exists():
        print(f"ERROR: {injury_file} not found!")
        print("Please create this file with your 50 injury dates.")
        print("Format: athlete_name,injury_date,injury_type,notes")
        return

    # Load injury dates
    injury_df = parse_injury_dates(injury_file)

    # Create date ranges (7 days before each injury)
    print(f"\nCreating labels for 7 days before each injury...")
    date_ranges_df = create_date_ranges(injury_df, lookback_days=7)

    print(f"Created {len(date_ranges_df)} labeled date-athlete pairs")
    print(f"  - Injury day: {(date_ranges_df['risk_level'] == 'injury_day').sum()}")
    print(f"  - High risk (1-3 days before): {(date_ranges_df['risk_level'] == 'high_risk').sum()}")
    print(f"  - Moderate risk (4-7 days before): {(date_ranges_df['risk_level'] == 'moderate_risk').sum()}")

    # Check what data we have
    available_dates = get_available_dates(data_dir)
    print(f"\nAvailable dates: {available_dates}")

    # Identify missing dates
    missing_dates = identify_missing_dates(date_ranges_df, available_dates)

    # Match existing sessions to labels
    if available_dates:
        labeled_df = match_sessions_to_labels(data_dir, date_ranges_df)

        # Save labeled dataset
        output_file = config_dir / 'labeled_sessions.csv'
        labeled_df.to_csv(output_file, index=False)
        print(f"\nLabeled sessions saved to: {output_file}")

    # Generate collection script
    if missing_dates:
        script_file = base_dir / 'collect_missing_dates.sh'
        generate_collection_script(missing_dates, script_file)

    # Save date ranges for reference
    date_ranges_file = config_dir / 'injury_date_ranges.csv'
    date_ranges_df.to_csv(date_ranges_file, index=False)
    print(f"Injury date ranges saved to: {date_ranges_file}")


if __name__ == "__main__":
    main()
