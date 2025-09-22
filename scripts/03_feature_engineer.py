"""
scripts/03_feature_engineer.py

Runs feature engineering on aggregated time series data.
Should be run AFTER aggregate_data.py.

Feature engineering includes:
    - temporal features (e.g., rush hour, time of day) from datetime index
    - weather feature transformations
    - interaction effects between weather and temporal features
    - lagged features for time series analysis
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from argparse import ArgumentParser
from tqdm import tqdm
from src.features.temporal import add_temporal_features, create_lagged_features
from src.features.engineering import (
    engineer_weather_features,
    create_interaction_features,
    winsorize_and_normalize_all_features
)
from src.features.spatial import add_spatial_features
from src.data.validation import validate_aggregated_time_series_data, print_validation_report
from src.utils.config_loader import InteractionConfig, create_interactions_from_config


def parse_args():
    parser = ArgumentParser(description="Feature engineering for aggregated time series data")
    parser.add_argument("--load", default="./data/processed/02_aggregated.csv",
                       help="Path to aggregated time series dataset")
    parser.add_argument("--save", default="./data/processed/03_features.csv",
                       help="Path to save engineered features")
    parser.add_argument("--include-interactions", action="store_true",
                       help="Include interaction features")
    parser.add_argument("--include-lags", action="store_true",
                       help="Include lagged features for time series analysis")
    parser.add_argument("--config", type=str,
                       help="Path to YAML configuration file for interaction effects")
    parser.add_argument("--lag-hours", nargs="+", type=int, default=[1, 2, 3],
                       help="List of lag periods in hours for time series features")

    return parser.parse_args()


def add_features_to_timeseries(df: pd.DataFrame, args, pbar: tqdm) -> pd.DataFrame:
    """
    Runs feature engineering on aggregated time series data

    Args:
        df (pd.DataFrame): Aggregated time series dataframe with datetime index
        args: Command line arguments
        pbar (tqdm): Progress bar

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """

    # Validate input data has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Input data doesn't have datetime index. Attempting to parse...")
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        else:
            raise ValueError("Input data must have datetime index or 'datetime' column")

    # Calculate total steps
    total_steps = 4  # Spatial, temporal, weather engineering, validation
    if args.include_interactions or args.config:
        total_steps += 1
    if args.include_lags:
        total_steps += 1

    # Update progress bar total
    pbar.total = pbar.total - 1 + total_steps
    pbar.refresh()

    result = df.copy()

    # Step 1: Add spatial features
    pbar.set_description("Adding spatial features")
    result = add_spatial_features(result)
    pbar.update(1)

    # Step 2: Add temporal features from datetime index
    pbar.set_description("Adding temporal features")
    result = add_temporal_features(result)
    pbar.update(1)

    # Step 3: Weather feature engineering (encoding only)
    pbar.set_description("Engineering weather features")
    result = engineer_weather_features(result)
    pbar.update(1)

    # Step 4: Interaction features (optional)
    if args.include_interactions or args.config:
        if args.config:
            # Use config-based interaction features
            pbar.set_description("Config-based interaction features")
            try:
                config = InteractionConfig(args.config)
                result = create_interactions_from_config(result, config)
                print(f"Created interaction features from config: {args.config}")
            except Exception as e:
                print(f"Error loading config {args.config}: {e}")
                print("Falling back to default interaction features...")
                with tqdm(total=2, desc="Default interaction features", leave=False) as sub_pbar:
                    result = create_interaction_features(result, pbar=sub_pbar)
        else:
            # Use default interaction features
            with tqdm(total=2, desc="Default interaction features", leave=False) as sub_pbar:
                result = create_interaction_features(result, pbar=sub_pbar)
        pbar.update(1)

    # Step 5: Lagged features for time series (optional)
    if args.include_lags:
        with tqdm(total=len(args.lag_hours), desc="Lagged features", leave=False) as sub_pbar:
            result = create_lagged_features(
                result,
                lag_hours=args.lag_hours,
                pbar=sub_pbar
            )
        pbar.update(1)

    # Step 6: Winsorize and normalize all features (weather + interactions)
    pbar.set_description("Winsorizing and normalizing features")
    result = winsorize_and_normalize_all_features(result, pbar=None)
    pbar.update(1)

    # Final validation
    pbar.set_description("Validating engineered features")
    validation_results = validate_aggregated_time_series_data(result)

    # Print validation results
    print_validation_report(validation_results, "Engineered Time Series Features")

    # Print feature summary
    print(f"\nFeature Engineering Summary:")
    print(f"Original features: {len(df.columns)}")
    print(f"Final features: {len(result.columns)}")
    print(f"Features added: {len(result.columns) - len(df.columns)}")
    print(f"Final dataset shape: {result.shape}")
    print(f"Time range: {result.index.min()} to {result.index.max()}")

    return result


def main():
    args = parse_args()

    print("Starting time series feature engineering pipeline...")
    print(f"Input file: {args.load}")
    print(f"Output file: {args.save}")
    if args.config:
        print(f"Config file: {args.config}")
    if args.include_lags:
        print(f"Lag periods: {args.lag_hours} hours")

    with tqdm(total=3, desc="Overall Progress") as pbar:

        # Step 1: Load aggregated time series dataset
        pbar.set_description("Loading aggregated time series")
        try:
            # Load with datetime index
            df = pd.read_csv(args.load, index_col=0, parse_dates=True)
            print(f"Loaded aggregated dataset with shape: {df.shape}")
            print(f"Time range: {df.index.min()} to {df.index.max()}")
            print(f"Locations: {df['location_id'].nunique()}")
        except FileNotFoundError:
            print(f"Error: File {args.load} not found!")
            print("Make sure to run aggregate_data.py first to create aggregated time series data.")
            return
        except Exception as e:
            print(f"Error loading file: {e}")
            return
        pbar.update(1)

        # Step 2: Feature engineering on time series data
        pbar.set_description("Feature engineering")
        df = add_features_to_timeseries(df, args, pbar)

        # Step 3: Save dataset
        pbar.set_description("Saving dataset")
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            # Save with datetime index preserved
            df.to_csv(args.save, index=True)
            print(f"Saved engineered time series features to: {args.save}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return
        pbar.update(1)

        pbar.set_description("Completed!")

    print("Time series feature engineering pipeline completed successfully!")
    print(f"\nTo train models on this data:")
    print(f"  python scripts/train_model.py --load {args.save}")



if __name__ == "__main__":
    main()