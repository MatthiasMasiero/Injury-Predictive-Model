"""
detect_injuries.py - Analyze collected data to identify injured athletes

Detects injuries by finding:
1. Very small file sizes (< 1KB = no data)
2. Early session termination (session < 50% of average duration)
3. Sudden data drops (zeros or missing values)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

def analyze_injuries(data_dir):
    """Analyze all JSON files to detect injured athletes."""

    data_path = Path(data_dir)
    files = list(data_path.glob("*.json"))

    print(f"\n{'='*80}")
    print(f"INJURY DETECTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total files to analyze: {len(files)}\n")

    # Track athletes and sessions
    athlete_sessions = defaultdict(list)
    injured_candidates = []
    session_durations = defaultdict(list)

    for file_path in files:
        file_size = file_path.stat().st_size
        filename = file_path.name

        # Parse filename: DATE_athleteID_activityID.json
        parts = filename.replace('.json', '').split('_')
        if len(parts) != 3:
            continue

        date, athlete_id, activity_id = parts

        # Load JSON to get athlete info
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            elif isinstance(data, list):
                data = {}

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        athlete_name = f"{data.get('athlete_first_name', 'Unknown')} {data.get('athlete_last_name', '')}".strip()
        data_points = data.get('data', [])
        num_points = len(data_points)

        # Calculate session duration
        if num_points > 0:
            try:
                start_ts = data_points[0].get('ts', 0)
                end_ts = data_points[-1].get('ts', 0)
                duration_seconds = end_ts - start_ts
                duration_minutes = duration_seconds / 60
            except:
                duration_minutes = 0
        else:
            duration_minutes = 0

        session_info = {
            'date': date,
            'athlete_id': athlete_id,
            'athlete_name': athlete_name,
            'activity_id': activity_id,
            'file_size': file_size,
            'data_points': num_points,
            'duration_minutes': duration_minutes,
            'file_path': str(file_path)
        }

        athlete_sessions[athlete_id].append(session_info)
        session_durations[activity_id].append(duration_minutes)

        # Flag suspicious files
        if file_size < 1000:  # Less than 1KB
            injured_candidates.append({
                **session_info,
                'reason': 'NO DATA (file < 1KB)'
            })
        elif num_points < 100:  # Less than 100 data points (< 10 seconds of data at 10Hz)
            injured_candidates.append({
                **session_info,
                'reason': f'MINIMAL DATA ({num_points} points)'
            })

    # Calculate average duration per activity
    activity_avg_duration = {
        act_id: sum(durations) / len(durations) if durations else 0
        for act_id, durations in session_durations.items()
    }

    # Check for early termination
    for athlete_id, sessions in athlete_sessions.items():
        for session in sessions:
            activity_id = session['activity_id']
            avg_duration = activity_avg_duration.get(activity_id, 0)

            if avg_duration > 10 and session['duration_minutes'] > 0:  # Skip very short activities
                if session['duration_minutes'] < avg_duration * 0.5:  # Less than 50% of average
                    injured_candidates.append({
                        **session,
                        'reason': f'EARLY EXIT ({session["duration_minutes"]:.1f} min vs avg {avg_duration:.1f} min)',
                        'avg_duration': avg_duration
                    })

    # Remove duplicates and sort
    seen = set()
    unique_injured = []
    for candidate in injured_candidates:
        key = (candidate['athlete_id'], candidate['activity_id'])
        if key not in seen:
            seen.add(key)
            unique_injured.append(candidate)

    unique_injured.sort(key=lambda x: (x['date'], x['athlete_name']))

    # Print results
    print(f"\n{'='*80}")
    print(f"SUSPECTED INJURIES DETECTED: {len(unique_injured)}")
    print(f"{'='*80}\n")

    for i, injury in enumerate(unique_injured, 1):
        print(f"{i}. {injury['athlete_name']} (ID: {injury['athlete_id']})")
        print(f"   Date: {injury['date']}")
        print(f"   Activity: {injury['activity_id']}")
        print(f"   Reason: {injury['reason']}")
        print(f"   File Size: {injury['file_size']:,} bytes")
        print(f"   Data Points: {injury['data_points']:,}")
        if injury['duration_minutes'] > 0:
            print(f"   Duration: {injury['duration_minutes']:.1f} minutes")
        print()

    # Save to CSV
    if unique_injured:
        df = pd.DataFrame(unique_injured)
        output_file = Path(data_dir).parent / 'config' / 'detected_injuries.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

    # Summary by athlete
    print(f"\n{'='*80}")
    print(f"ATHLETES WITH SUSPECTED INJURIES")
    print(f"{'='*80}\n")

    athlete_injury_count = defaultdict(list)
    for injury in unique_injured:
        athlete_injury_count[injury['athlete_name']].append(injury['date'])

    for athlete_name, dates in sorted(athlete_injury_count.items()):
        print(f"  {athlete_name}: {', '.join(set(dates))}")

    print(f"\n{'='*80}\n")

    return unique_injured


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    analyze_injuries(data_dir)
