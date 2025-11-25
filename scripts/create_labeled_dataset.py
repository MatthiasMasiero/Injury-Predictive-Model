"""
create_labeled_dataset.py - Create labeled dataset for training

Combines:
- Rolling window features
- Injury labels
- Ready for model training
"""

import pandas as pd
from pathlib import Path

def create_labeled_dataset():
    """Create training dataset with labels."""

    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'

    print(f"\n{'='*80}")
    print(f"CREATING LABELED TRAINING DATASET")
    print(f"{'='*80}\n")

    # Load rolling window features
    features_df = pd.read_csv(config_dir / 'rolling_window_features.csv')
    print(f"Loaded {len(features_df)} feature rows")

    # Load injury dates
    injuries_df = pd.read_csv(config_dir / 'injury_dates.csv')
    print(f"Loaded {len(injuries_df)} injury records")

    # Create injury lookup
    injury_lookup = set()
    for _, row in injuries_df.iterrows():
        key = (row['athlete_name'], row['injury_date'])
        injury_lookup.add(key)

    # Label the features
    features_df['injured'] = features_df.apply(
        lambda row: (row['athlete_name'], row['date']) in injury_lookup,
        axis=1
    )

    # Create risk level based on injury status
    features_df['risk_level'] = features_df['injured'].map({
        True: 'injured',
        False: 'normal'
    })

    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total feature rows: {len(features_df)}")
    print(f"Injured: {features_df['injured'].sum()}")
    print(f"Normal: {(~features_df['injured']).sum()}")
    print(f"\nUnique athletes: {features_df['athlete_name'].nunique()}")
    print(f"Unique sessions: {features_df.groupby(['date', 'athlete_id', 'activity_id']).ngroups}")

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

    print(f"\nFeatures available: {len(feature_cols)}")

    # Save
    output_file = config_dir / 'labeled_training_data.csv'
    features_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"Labeled dataset saved to:")
    print(f"  {output_file}")
    print(f"{'='*80}\n")

    # Show sample
    print("Sample injured rows:")
    print(features_df[features_df['injured']][['athlete_name', 'date', 'window_minutes', 'avg_velocity', 'cumulative_player_load']].head(3))

    return features_df

if __name__ == "__main__":
    create_labeled_dataset()
