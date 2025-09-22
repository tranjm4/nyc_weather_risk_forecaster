"""
scripts/02_aggregate_data.py

Aggregates individual accident records into time series format.
This should be run BEFORE feature engineering.

Pipeline:
1. Load raw merged data (accidents + basic weather merge)
2. Create spatial grid
3. Aggregate to hourly counts per location
4. Merge with weather data
5. Save aggregated time series data

Usage:
    # Basic usage (uses default cleaned weather data)
    python scripts/02_aggregate_data.py

    # With custom parameters
    python scripts/02_aggregate_data.py --min-accidents 150 --weather-file ./custom/weather.csv
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from argparse import ArgumentParser
from tqdm import tqdm
from src.features.aggregation import aggregate_accidents_to_timeseries
from src.data.validation import validate_aggregated_time_series_data, print_validation_report


def parse_args():
    parser = ArgumentParser(description="Aggregate accident data into time series format")
    parser.add_argument("--load", default="./data/processed/01_merged.csv",
                       help="Path to merged dataset (accidents + weather)")
    parser.add_argument("--save", default="./data/processed/02_aggregated.csv",
                       help="Path to save aggregated time series data")
    parser.add_argument("--weather-file", default="./data/processed/00_weather.csv",
                       help="Path to weather data file (default: cleaned weather from process_raw_data.py)")
    parser.add_argument("--min-accidents", type=int, default=200,
                       help="Minimum accidents per location for inclusion")
    parser.add_argument("--time-freq", default="h",
                       help="Time frequency for aggregation (h=hourly, D=daily)")
    parser.add_argument("--datetime-col", default="dt",
                       help="Name of datetime column")
    parser.add_argument("--bin-size", type=float, default=0.05,
                       help="Spatial grid bin size in degrees (default: 0.05 ≈ 5km)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting data aggregation pipeline...")
    print(f"Input file: {args.load}")
    print(f"Weather file: {args.weather_file}")
    print(f"Output file: {args.save}")
    print(f"Options: min_accidents={args.min_accidents}, time_freq={args.time_freq}, bin_size={args.bin_size}°")

    with tqdm(total=5, desc="Overall Progress") as pbar:

        # Step 1: Load accident dataset
        pbar.set_description("Loading accident dataset")
        try:
            df = pd.read_csv(args.load)
            df[args.datetime_col] = pd.to_datetime(df[args.datetime_col])
            print(f"Loaded accident dataset with shape: {df.shape}")
        except FileNotFoundError:
            print(f"Error: File {args.load} not found!")
            return
        except Exception as e:
            print(f"Error loading accident file: {e}")
            return
        pbar.update(1)

        # Step 2: Load weather dataset
        pbar.set_description("Loading weather dataset")
        try:
            weather_df = pd.read_csv(args.weather_file)
            weather_df[args.datetime_col] = pd.to_datetime(weather_df[args.datetime_col])
            weather_df = weather_df.set_index(args.datetime_col)
            print(f"Loaded weather dataset with shape: {weather_df.shape}")
        except FileNotFoundError:
            print(f"Error: Weather file {args.weather_file} not found!")
            return
        except Exception as e:
            print(f"Error loading weather file: {e}")
            return
        pbar.update(1)

        # Step 3: Aggregate to time series
        pbar.set_description("Aggregating to time series")
        try:
            aggregated_df = aggregate_accidents_to_timeseries(
                df, weather_df,
                time_freq=args.time_freq,
                min_accidents=args.min_accidents,
                datetime_col=args.datetime_col
            )
        except Exception as e:
            print(f"Error during aggregation: {e}")
            return
        pbar.update(1)

        # Step 4: Complete aggregation (spatial features moved to feature engineering)
        pbar.set_description("Finalizing aggregation")
        pbar.update(1)

        # Step 5: Validation and save
        pbar.set_description("Validating and saving")

        # Validate aggregated data
        print("Validating aggregated time series data...")
        validation_results = validate_aggregated_time_series_data(aggregated_df)
        print_validation_report(validation_results, "Aggregated Time Series")

        # Save dataset
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            aggregated_df.to_csv(args.save, index=True)  # Preserve datetime index
            print(f"Saved aggregated time series data to: {args.save}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return
        pbar.update(1)

        pbar.set_description("Completed!")

    print("Data aggregation pipeline completed successfully!")
    print(f"\nFinal dataset summary:")
    print(f"  Shape: {aggregated_df.shape}")
    print(f"  Locations: {aggregated_df['location_id'].nunique()}")
    print(f"  Time range: {aggregated_df.index.min()} to {aggregated_df.index.max()}")
    print(f"  Total accidents: {aggregated_df['accident_count'].sum():,}")


if __name__ == "__main__":
    main()