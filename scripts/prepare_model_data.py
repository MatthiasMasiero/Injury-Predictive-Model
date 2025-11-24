"""
prepare_model_data.py - Prepare final dataset for injury prediction model

Creates dataset suitable for rolling window feature engineering:
- Only includes sessions with actual GPS data (> 1000 data points)
- Flags sessions that ended early as "injured"
- Excludes no-participation sessions
"""

import pandas as pd
from pathlib import Path

def prepare_model_data():
    """Prepare clean dataset for model training."""

    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'

    print(f"\n{'='*80}")
    print(f"PREPARING MODEL TRAINING DATA")
    print(f"{'='*80}\n")

    # Load the full dataset
    df = pd.read_csv(config_dir / 'training_data_summary.csv')

    print(f"Original dataset: {len(df)} sessions")
    print(f"  - Injured/No-data: {df['injured'].sum()}")
    print(f"  - Normal: {(~df['injured']).sum()}")

    # Filter: Only sessions with sufficient GPS data
    # Require at least 1000 data points (~100 seconds at 10Hz)
    df_model = df[df['data_points'] >= 1000].copy()

    print(f"\nFiltered dataset (>= 1000 data points): {len(df_model)} sessions")
    print(f"  - Early exits (injured during session): {df_model['injured'].sum()}")
    print(f"  - Normal completions: {(~df_model['injured']).sum()}")

    # Summary stats
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total athletes: {df_model['athlete_name'].nunique()}")
    print(f"Training sessions: {len(df_model)}")
    print(f"Injury rate: {df_model['injured'].mean()*100:.1f}%")

    print(f"\nDuration statistics:")
    print(f"  Mean: {df_model['duration_minutes'].mean():.1f} minutes")
    print(f"  Median: {df_model['duration_minutes'].median():.1f} minutes")
    print(f"  Min: {df_model['duration_minutes'].min():.1f} minutes")
    print(f"  Max: {df_model['duration_minutes'].max():.1f} minutes")

    if df_model['injured'].sum() > 0:
        print(f"\nInjured sessions:")
        injured_df = df_model[df_model['injured']]
        for _, row in injured_df.iterrows():
            print(f"  {row['athlete_name']} on {row['date']}: {row['reason']}")

    # Feature columns for modeling
    feature_cols = [
        'date',
        'athlete_id',
        'athlete_name',
        'activity_id',
        'data_points',
        'duration_minutes',
        'avg_velocity',
        'max_velocity',
        'std_velocity',
        'avg_acceleration',
        'max_acceleration',
        'std_acceleration',
        'avg_player_load',
        'max_player_load',
        'avg_metabolic_power',
        'max_metabolic_power',
        'avg_smooth_load',
        'injured'  # TARGET VARIABLE
    ]

    df_model = df_model[feature_cols].copy()

    # Save
    output_file = config_dir / 'model_training_data.csv'
    df_model.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"Model training data saved to:")
    print(f"  {output_file}")
    print(f"{'='*80}\n")

    # Create train/test split suggestion
    print("RECOMMENDED APPROACH:")
    print("=" * 80)
    print("Since you have 2 dates, use temporal split:")
    print("  - TRAIN: 2025-01-09 data")
    print("  - TEST:  2025-01-17 data")
    print("\nThis simulates real-world scenario: train on past, predict future.")
    print("=" * 80)

    train_df = df_model[df_model['date'] == '2025-01-09']
    test_df = df_model[df_model['date'] == '2025-01-17']

    print(f"\nTrain set (2025-01-09): {len(train_df)} sessions, {train_df['injured'].sum()} injured")
    print(f"Test set (2025-01-17):  {len(test_df)} sessions, {test_df['injured'].sum()} injured")

    return df_model

if __name__ == "__main__":
    prepare_model_data()
