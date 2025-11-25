"""
train_model.py - Train injury prediction model

Uses rolling window features to predict injury risk at each time point.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def train_injury_model():
    """Train model to predict injury risk."""

    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'config'

    print(f"\n{'='*80}")
    print(f"TRAINING INJURY PREDICTION MODEL")
    print(f"{'='*80}\n")

    # Load data
    df = pd.read_csv(config_dir / 'labeled_training_data.csv')
    print(f"Loaded {len(df)} samples")
    print(f"  Injured: {df['injured'].sum()}")
    print(f"  Normal: {(~df['injured']).sum()}")

    # Features
    feature_cols = [
        'window_minutes', 'data_points',
        'avg_velocity', 'max_velocity', 'std_velocity', 'p90_velocity',
        'avg_acceleration', 'max_acceleration', 'std_acceleration', 'p90_acceleration',
        'cumulative_player_load', 'avg_player_load_rate',
        'avg_metabolic_power', 'max_metabolic_power', 'p90_metabolic_power',
        'recent_avg_velocity', 'recent_max_acceleration',
        'avg_smooth_load'
    ]

    X = df[feature_cols].fillna(0)
    y = df['injured']

    # Train/test split - use temporal split (date-based)
    train_mask = df['date'] == '2025-01-09'
    test_mask = df['date'] == '2025-01-17'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} injured)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} injured)")

    # Train model
    print(f"\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    print(f"\n{'='*80}")
    print(f"MODEL EVALUATION")
    print(f"{'='*80}\n")

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Injured']))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Neg: {cm[0,0]:4d}  |  False Pos: {cm[0,1]:4d}")
    print(f"  False Neg: {cm[1,0]:4d}  |  True Pos:  {cm[1,1]:4d}")

    if y_test.sum() > 0:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {auc:.3f}")

    # Feature importance
    print(f"\n{'='*80}")
    print(f"TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*80}")

    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importances.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.3f}")

    # Save model
    model_file = config_dir.parent / 'models' / 'injury_risk_model.pkl'
    model_file.parent.mkdir(exist_ok=True)

    joblib.dump(model, model_file)

    print(f"\n{'='*80}")
    print(f"Model saved to: {model_file}")
    print(f"{'='*80}\n")

    return model

if __name__ == "__main__":
    train_injury_model()
