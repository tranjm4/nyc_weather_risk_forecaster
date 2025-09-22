"""
src/features/spatial.py

Spatial grid creation and coordinate binning for traffic data
"""

import pandas as pd
import numpy as np
from typing import List

from sklearn.preprocessing import StandardScaler

def create_spatial_grid(df: pd.DataFrame, bin_size: float = 0.05,
                       lat_col: str = 'latitude', lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Create coordinate bins for accident locations

    Args:
        df (pd.DataFrame): DataFrame with lat/lon coordinates
        bin_size (float): Grid bin size in degrees (default: 0.05 ≈ 5km)
        lat_col (str): Latitude column name
        lon_col (str): Longitude column name

    Returns:
        pd.DataFrame: DataFrame with grid coordinates and location_id
    """
    df = df.copy()

    # Remove rows with missing coordinates
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()

    # Create grid coordinates by rounding to nearest grid point
    df_clean['grid_lat'] = ((df_clean[lat_col] / bin_size).round() * bin_size).round(6)
    df_clean['grid_lon'] = ((df_clean[lon_col] / bin_size).round() * bin_size).round(6)

    # Create location identifier
    df_clean['location_id'] = (df_clean['grid_lat'].astype(str) + '_' +
                              df_clean['grid_lon'].astype(str))

    # Add bin ID for faster processing
    unique_locations = df_clean['location_id'].unique()
    location_to_bin = {loc: i for i, loc in enumerate(unique_locations)}
    df_clean['bin_id'] = df_clean['location_id'].map(location_to_bin)

    print(f"Created spatial grid with {len(unique_locations)} unique locations")

    return df_clean


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial features to aggregated data (distance from center, etc.)

    Args:
        df (pd.DataFrame): Aggregated time series dataframe

    Returns:
        pd.DataFrame: DataFrame with spatial features added
    """
    result = df.copy()

    # Distance from city center (approximate NYC center: 40.7589, -73.9851)
    nyc_center_lat, nyc_center_lon = 40.7589, -73.9851

    if 'grid_lat' in result.columns and 'grid_lon' in result.columns:
        # Haversine distance approximation (for small distances)
        lat_diff = result['grid_lat'] - nyc_center_lat
        lon_diff = result['grid_lon'] - nyc_center_lon
        result['distance_from_center'] = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion

        print(f"Added distance_from_center feature (range: {result['distance_from_center'].min():.1f} - {result['distance_from_center'].max():.1f} km)")

    # Grid density (total accidents per location across all time)
    if 'location_id' in result.columns and 'accident_count' in result.columns:
        location_totals = result.groupby('location_id')['accident_count'].sum()
        result['location_total_accidents'] = result['location_id'].map(location_totals)
        print(f"Added location_total_accidents feature")
    
    # Normalize location_total_accidents
    scaler = StandardScaler()
    result[["location_total_accidents"]] = scaler.fit_transform(result[["location_total_accidents"]])

    return result