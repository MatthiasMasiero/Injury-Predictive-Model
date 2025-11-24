"""
data_collector.py - Fetch athlete activity data from Catapult Sports API

Usage:
    python data_collector.py --start-date 2024-08-01 --end-date 2024-08-31
    python data_collector.py --date 2024-08-16  # Single day
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import requests
import dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(env_path)

TOKEN = os.getenv("MSOC_API_KEY")
REGION = "us"
BASE = f"https://connect-{REGION}.catapultsports.com/api/v6"
HEADERS = {"accept": "application/json", "authorization": f"Bearer {TOKEN}"}


def get_activities(date: str):
    """Return list of activity dicts that fall on a specific calendar day."""
    params = {"start_date": date, "end_date": date}
    r = requests.get(f"{BASE}/activities", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("items", [])


def get_roster(activity_id: int):
    """Return list of athlete dicts for a given activity."""
    r = requests.get(
        f"{BASE}/activities/{activity_id}/athletes", headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("items", [])


def get_sensor(activity_id: int, athlete_id: int, stream="gps"):
    """Return the raw sensor JSON for one athlete in one session."""
    url = f"{BASE}/activities/{activity_id}/athletes/{athlete_id}/sensor"
    r = requests.get(url, headers=HEADERS, params={"stream_type": stream}, timeout=60)
    r.raise_for_status()
    return r.json()


def collect_data_for_date(date: str, output_dir: Path):
    """Collect all athlete data for a specific date."""
    print(f"\n{'='*60}")
    print(f"Collecting data for {date}")
    print(f"{'='*60}")

    activities = get_activities(date)
    if not activities:
        print(f"  No activities found on {date}")
        return 0

    print(f"  Found {len(activities)} activity/activities")
    total_files = 0

    for act in activities:
        act_id = act["id"]
        act_name = act.get("name", "Unknown")
        print(f"\n  Activity {act_id}: {act_name}")

        roster = get_roster(act_id)
        if not roster:
            print(f"    No athletes returned")
            continue

        print(f"    Processing {len(roster)} athletes...")

        for i, ath in enumerate(roster, 1):
            ath_id = ath["id"]
            ath_name = f"{ath.get('first_name', '')} {ath.get('last_name', '')}".strip()

            try:
                payload = get_sensor(act_id, ath_id)

                # Save to file
                outfile = output_dir / f"{date}_{ath_id}_{act_id}.json"
                with outfile.open("w") as f:
                    json.dump(payload, f, indent=2)

                print(f"    [{i}/{len(roster)}] {ath_name} (ID: {ath_id}) -> {outfile.name}")
                total_files += 1

            except Exception as exc:
                print(f"    [{i}/{len(roster)}] {ath_name} (ID: {ath_id}) -> ERROR: {exc}")
                continue

            # Rate limiting: stay under 60 req/min
            time.sleep(0.5)

    return total_files


def date_range(start_date: str, end_date: str):
    """Generate list of dates between start and end (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Collect athlete activity data from Catapult Sports API"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Single date to collect (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data",
        help="Output directory for JSON files (default: ../data)",
    )

    args = parser.parse_args()

    # Validate API key
    if not TOKEN:
        raise RuntimeError(
            "MSOC_API_KEY is not set. Please check your .env file."
        )

    # Setup output directory
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # Determine dates to collect
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        dates = date_range(args.start_date, args.end_date)
    else:
        parser.error("Must specify either --date or both --start-date and --end-date")

    # Collect data
    print(f"\nCollecting data for {len(dates)} date(s)")
    total_files = 0

    start_time = time.time()
    for date in dates:
        files_collected = collect_data_for_date(date, output_dir)
        total_files += files_collected

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total files collected: {total_files}")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
