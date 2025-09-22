"""
src/features/temporal.py

Temporal feature engineering functions
"""

from .engineering import winsorize_features, normalize_features

import pandas as pd
import numpy as np

from typing import Optional

from tqdm import tqdm


def add_temporal_features(df: pd.DataFrame, datetime_col: str = None) -> pd.DataFrame:
    """
    Add temporal features to the dataframe

    Args:
        df (pd.DataFrame): Input dataframe with datetime column or index
        datetime_col (str): Name of datetime column (if None, uses index)

    Returns:
        pd.DataFrame: DataFrame with additional temporal features
    """
    df = df.copy()

    # Use datetime index or specified column
    if datetime_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        else:
            raise ValueError("No datetime column specified and index is not datetime")
    else:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        dt = df[datetime_col].dt

    # Basic time components
    hour = dt.hour
    day_of_week = dt.dayofweek
    day_of_year = dt.dayofyear

    # Weekend indicator
    df['is_weekend'] = (dt.dayofweek >= 5).astype(int)

    # Rush hour indicators
    is_morning_rush = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_rush = ((hour >= 17) & (hour <= 19)).astype(int)
    df['is_rush_hour'] = (is_morning_rush | is_evening_rush).astype(int)

    # Late night indicator
    df['is_late_night'] = ((hour >= 22) | (hour <= 5)).astype(int)

    # Seasonal features (cyclical encoding)
    df['sin_hour'] = np.sin(2 * np.pi * hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * hour / 24)
    df['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365.25)
    df['sin_day_of_week'] = np.sin(2 * np.pi * day_of_week / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * day_of_week / 7)

    # School hours (affecting traffic patterns)
    df['is_school_hours'] = ((hour >= 8) & (hour <= 15) &
                            (day_of_week < 5)).astype(int)

    return df


def create_lagged_features(df: pd.DataFrame, weather_file: str = "./data/processed/00_weather.csv",
                          lag_hours: list = [1, 2, 3], pbar: Optional[tqdm] = None) -> pd.DataFrame:
    """
    Create lagged weather features using complete weather dataset

    Loads the complete weather data to create proper lagged features,
    then merges back with the sparse accident time series data.

    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        weather_file (str): Path to complete weather dataset
        lag_hours (list): List of lag periods in hours
        pbar (Optional[tqdm]): Progress bar to update

    Returns:
        pd.DataFrame: DataFrame with lagged weather features
    """
    # Load complete weather dataset
    try:
        weather_df = pd.read_csv(weather_file)
        weather_df['dt'] = pd.to_datetime(weather_df['dt'])
        weather_df = weather_df.set_index('dt')
        print(f"Loaded complete weather data: {weather_df.shape}")
    except Exception as e:
        print(f"Error loading weather file {weather_file}: {e}")
        if pbar:
            pbar.update(len(lag_hours))
        return df.copy()
    
    # Normalize the weather dataset
    weather_df = winsorize_features(weather_df, columns=["rain", "snow"], lower_pct=0, upper_pct=0.95)
    weather_df = normalize_features(weather_df, columns=["rain", "snow"])

    # Create lagged features on complete weather data
    weather_cols = []
    if 'rain' in weather_df.columns:
        weather_cols.append('rain')
    if 'snow' in weather_df.columns:
        weather_cols.append('snow')

    if not weather_cols:
        print("No rain/snow columns found in weather data")
        if pbar:
            pbar.update(len(lag_hours))
        return df.copy()

    for lag in lag_hours:
        if pbar:
            pbar.set_description(f"Creating {lag}h lagged weather features")

        for col in weather_cols:
            # Lagged weather values
            weather_df[f'{col}_lag_{lag}h'] = weather_df[col].shift(lag)

            # Rolling max for cumulative effects
            weather_df[f'{col}_max_{lag}h'] = (
                weather_df[col].rolling(window=lag, min_periods=1).max()
            )

        if pbar:
            pbar.update(1)

    # Merge lagged weather features back to original data
    lag_cols = [col for col in weather_df.columns if any(f'_lag_{lag}h' in col or f'_max_{lag}h' in col for lag in lag_hours)]
    
    result = df.merge(weather_df[lag_cols], left_index=True, right_index=True, how='left')

    print(f"Added {len(lag_cols)} lagged weather features: {weather_cols}")
    return result