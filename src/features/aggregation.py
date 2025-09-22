"""
src/features/aggregation.py

Time series aggregation functions for converting individual accident records
into aggregated count data by spatial location and time period.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from .spatial import create_spatial_grid


def aggregate_accidents_to_timeseries(df: pd.DataFrame, weather_df: pd.DataFrame,
                                    time_freq: str = 'h', min_accidents: int = 100,
                                    datetime_col: str = 'dt') -> pd.DataFrame:
    """
    Complete pipeline to aggregate individual accident records into time series format

    This function performs minimal preprocessing (spatial grid only) then aggregates,
    leaving feature engineering for post-aggregation steps.

    Args:
        df (pd.DataFrame): Raw accident dataframe with coordinates
        weather_df (pd.DataFrame): Weather dataframe with datetime index
        time_freq (str): Pandas frequency string ('H' for hourly)
        min_accidents (int): Minimum total accidents per location to include
        datetime_col (str): DateTime column name

    Returns:
        pd.DataFrame: Aggregated time series data with datetime index and accident counts per location
    """
    print(f"Starting accident aggregation pipeline with min_accidents={min_accidents}")

    # Step 1: Create coordinate bins
    print("Creating spatial grid...")
    df_with_grid = create_spatial_grid(df, bin_size=0.05,
                                     lat_col='latitude', lon_col='longitude')

    print(f"Created {df_with_grid['location_id'].nunique()} unique locations")

    # Ensure weather_df has datetime index
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        if datetime_col in weather_df.columns:
            weather_df = weather_df.set_index(datetime_col)
        else:
            raise ValueError(f"Weather dataframe must have datetime index or '{datetime_col}' column")

    # Step 2: Aggregate to time series using existing functions
    ts_data_list, valid_locations = create_time_series_data(
        df_with_grid, weather_df,
        time_freq=time_freq,
        min_accidents=min_accidents,
        datetime_col=datetime_col
    )

    print(f"Created time series for {len(ts_data_list)} locations")

    # Step 3: Merge into single DataFrame (preserves datetime index)
    aggregated_df = merge_ts_data(ts_data_list, valid_locations)

    print(f"Final aggregated dataset shape: {aggregated_df.shape}")
    print(f"Index type: {type(aggregated_df.index)}")
    print(f"Unique timestamps: {aggregated_df.index.nunique():,}")
    print(f"Number of locations: {aggregated_df['location_id'].nunique()}")
    print(f"Date range: {aggregated_df.index.min()} to {aggregated_df.index.max()}")

    # Summary statistics
    print(f"\nAccident count statistics:")
    print(f"  Total accidents: {aggregated_df['accident_count'].sum():,}")
    print(f"  Mean per hour: {aggregated_df['accident_count'].mean():.2f}")
    print(f"  Max per hour: {aggregated_df['accident_count'].max()}")
    print(f"  Hours with zero accidents: {(aggregated_df['accident_count'] == 0).sum():,}")

    return aggregated_df


def create_time_series_data(df: pd.DataFrame, weather_df: pd.DataFrame,
                          time_freq: str = 'h', min_accidents: int = 50,
                          datetime_col: str = 'dt') -> tuple:
    location_counts = df['location_id'].value_counts()
    valid_locations = location_counts[location_counts >= min_accidents].index
    df_filtered = df[df['location_id'].isin(valid_locations)].copy()

    print(f"Locations with >= {min_accidents} accidents: {len(valid_locations)}")

    time_series_data = []

    for location in tqdm(valid_locations, desc="Aggregating by location"):
        location_data = df_filtered[df_filtered['location_id'] == location].copy()
        location_data[datetime_col] = pd.to_datetime(location_data[datetime_col])
        location_data = location_data.set_index(datetime_col)

        ts = location_data.resample(time_freq).agg({
            'latitude': 'count',
            'grid_lat': 'first',
            'grid_lon': 'first',
        }).rename(columns={'latitude': 'accident_count'})

        if isinstance(weather_df.index, pd.DatetimeIndex):
            weather_df_indexed = weather_df
        else:
            weather_df_indexed = weather_df.set_index(datetime_col)
        ts = ts.merge(weather_df_indexed, left_index=True, right_index=True, how='left')

        ts['grid_lon'] = ts['grid_lon'].ffill()
        ts['grid_lat'] = ts['grid_lat'].ffill()
        ts['location_id'] = location

        time_series_data.append(ts)

    return time_series_data, valid_locations


def merge_ts_data(df_list: List[pd.DataFrame], valid_locations: List[str]) -> pd.DataFrame:
    if not df_list:
        raise ValueError("No time series data to merge")

    merged_df = pd.concat(df_list, ignore_index=False, sort=False)
    merged_df = merged_df.sort_values(['location_id', merged_df.index])

    print(f"Merged {len(df_list)} location time series into single DataFrame")
    return merged_df