"""
src/data/validation.py

Data validation functions to ensure data quality and consistency
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def validate_datetime_format(df: pd.DataFrame, datetime_col: str = 'dt',
                           expected_format: str = '%Y-%m-%d %H:00') -> bool:
    """
    Validate that datetime column follows expected format

    Args:
        df (pd.DataFrame): DataFrame to validate
        datetime_col (str): Name of datetime column
        expected_format (str): Expected datetime format

    Returns:
        bool: True if all datetime values are valid
    """
    try:
        # Try to parse all datetime values
        parsed_dates = pd.to_datetime(df[datetime_col], format=expected_format, errors='coerce')
        invalid_count = parsed_dates.isna().sum()

        if invalid_count > 0:
            warnings.warn(f"Found {invalid_count} invalid datetime values in {datetime_col}")
            return False
        return True
    except Exception as e:
        warnings.warn(f"Datetime validation failed: {e}")
        return False


def validate_coordinate_bounds(df: pd.DataFrame, lat_col: str = 'latitude',
                             lon_col: str = 'longitude',
                             lat_bounds: Tuple[float, float] = (40.0, 41.0),
                             lon_bounds: Tuple[float, float] = (-74.7, -73.7)) -> bool:
    """
    Validate that coordinates are within NYC bounds

    Args:
        df (pd.DataFrame): DataFrame to validate
        lat_col (str): Latitude column name
        lon_col (str): Longitude column name
        lat_bounds (Tuple[float, float]): (min_lat, max_lat)
        lon_bounds (Tuple[float, float]): (min_lon, max_lon)

    Returns:
        bool: True if all coordinates are within bounds
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        warnings.warn(f"Missing coordinate columns: {lat_col}, {lon_col}")
        return False

    lat_out_of_bounds = ((df[lat_col] < lat_bounds[0]) | (df[lat_col] > lat_bounds[1])).sum()
    lon_out_of_bounds = ((df[lon_col] < lon_bounds[0]) | (df[lon_col] > lon_bounds[1])).sum()

    if lat_out_of_bounds > 0:
        warnings.warn(f"Found {lat_out_of_bounds} latitude values outside NYC bounds {lat_bounds}")
    if lon_out_of_bounds > 0:
        warnings.warn(f"Found {lon_out_of_bounds} longitude values outside NYC bounds {lon_bounds}")

    return lat_out_of_bounds == 0 and lon_out_of_bounds == 0


def validate_weather_ranges(df: pd.DataFrame) -> bool:
    """
    Validate that weather values are within reasonable ranges

    Args:
        df (pd.DataFrame): Weather DataFrame

    Returns:
        bool: True if all weather values are reasonable
    """
    validation_rules = {
        'visibility': (0, 20000),  # meters
        'wind_gust': (0, 50),      # m/s
        'rain': (0, 100),          # mm/h (if rain column exists)
        'snow': (0, 100),          # mm/h (if snow column exists)
        'rain_1h': (0, 100),       # mm/h (alternative name)
        'snow_1h': (0, 100),       # mm/h (alternative name)
    }

    all_valid = True

    for col, (min_val, max_val) in validation_rules.items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                warnings.warn(f"Found {out_of_range} values in {col} outside range ({min_val}, {max_val})")
                all_valid = False

    return all_valid


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate that all required columns are present

    Args:
        df (pd.DataFrame): DataFrame to validate
        required_cols (List[str]): List of required column names

    Returns:
        bool: True if all required columns are present
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        warnings.warn(f"Missing required columns: {missing_cols}")
        return False
    return True


def validate_no_duplicate_timestamps(df: pd.DataFrame, datetime_col: str = 'dt') -> bool:
    """
    Validate that there are no duplicate timestamps

    Args:
        df (pd.DataFrame): DataFrame to validate
        datetime_col (str): Name of datetime column

    Returns:
        bool: True if no duplicates found
    """
    if datetime_col not in df.columns:
        warnings.warn(f"Datetime column {datetime_col} not found")
        return False

    duplicate_count = df[datetime_col].duplicated().sum()

    if duplicate_count > 0:
        warnings.warn(f"Found {duplicate_count} duplicate timestamps")
        return False
    return True


def validate_weather_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Comprehensive validation for weather dataframe

    Args:
        df (pd.DataFrame): Weather DataFrame

    Returns:
        Dict[str, bool]: Validation results for each check
    """
    required_weather_cols = ['dt', 
                             'visibility', 
                             'wind_gust', 
                             'rain', 
                             'snow',
                             'rain_cat',
                             'snow_cat',
                             'wind_gust_cat',
                             'vis_cat']

    validation_results = {
        'required_columns': validate_required_columns(df, required_weather_cols),
        'datetime_format': validate_datetime_format(df, 'dt'),
        'weather_ranges': validate_weather_ranges(df),
        'no_duplicate_timestamps': validate_no_duplicate_timestamps(df, 'dt'),
    }

    return validation_results


def validate_accident_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Comprehensive validation for accident dataframe

    Args:
        df (pd.DataFrame): Accident DataFrame

    Returns:
        Dict[str, bool]: Validation results for each check
    """
    required_accident_cols = ['dt', 
                              'latitude',
                              'longitude',
                              'on_street_name',
                              'cross_street_name']

    validation_results = {
        'required_columns': validate_required_columns(df, required_accident_cols),
        'datetime_format': validate_datetime_format(df, 'dt'),
        'coordinate_bounds': validate_coordinate_bounds(df),
    }

    return validation_results


def validate_merged_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Comprehensive validation for merged weather + accident dataframe

    Args:
        df (pd.DataFrame): Merged DataFrame

    Returns:
        Dict[str, bool]: Validation results for each check
    """
    required_merged_cols = ['dt', 
                            'latitude', 
                            'longitude', 
                            'on_street_name',
                            'cross_street_name',
                            'visibility', 
                            'wind_gust',
                            'rain',
                            'snow',
                            'rain_cat',
                            'snow_cat',
                            'vis_cat',
                            'wind_gust_cat']

    validation_results = {
        'required_columns': validate_required_columns(df, required_merged_cols),
        'datetime_format': validate_datetime_format(df, 'dt'),
        'coordinate_bounds': validate_coordinate_bounds(df),
        'weather_ranges': validate_weather_ranges(df),
    }

    return validation_results


def validate_feature_engineered_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validation procedure for feature-engineered dataframe

    Args:
        df (pd.DataFrame): Merged + feature-engineered dataframe

    Returns:
        Dict[str, bool]: Validation results for each check
    """

    # Core required columns (should always be present)
    required_core_cols = ['dt', 'latitude', 'longitude']

    # Expected temporal features
    expected_temporal_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour']

    # Expected spatial features (if spatial processing was done)
    expected_spatial_cols = ['grid_lat', 'grid_lon', 'location_id', 'distance_from_center']

    validation_results = {
        'required_core_columns': validate_required_columns(df, required_core_cols),
        'datetime_format': validate_datetime_format(df, 'dt'),
        'coordinate_bounds': validate_coordinate_bounds(df),
    }

    # Check for temporal features
    temporal_present = sum(col in df.columns for col in expected_temporal_cols)
    validation_results['temporal_features_present'] = temporal_present >= 3  # At least some temporal features

    # Check for spatial features (if coordinates exist)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        spatial_present = sum(col in df.columns for col in expected_spatial_cols)
        validation_results['spatial_features_present'] = spatial_present >= 2  # At least some spatial features

    # Check for interaction features (if any)
    interaction_cols = [col for col in df.columns if col.startswith('interact_')]
    validation_results['interaction_features_created'] = len(interaction_cols) > 0

    # Check data completeness for critical features
    critical_cols = ['dt', 'latitude', 'longitude'] + [col for col in expected_temporal_cols if col in df.columns]
    completeness = check_data_completeness(df, critical_cols)
    validation_results['critical_columns_complete'] = all(pct >= 95 for col, pct in completeness.items() if col in critical_cols)

    return validation_results


def validate_aggregated_time_series_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validation procedure for aggregated time series dataframe

    Args:
        df (pd.DataFrame): Aggregated time series dataframe

    Returns:
        Dict[str, bool]: Validation results for each check
    """

    # Core required columns for time series data
    required_core_cols = ['accident_count', 'location_id', 'bin_id']

    # Expected aggregated columns
    expected_agg_cols = ['x_bin', 'y_bin', 'hour_index']

    # Weather columns should be present
    expected_weather_cols = ['visibility', 'wind_gust', 'rain', 'snow']

    validation_results = {
        'required_core_columns': validate_required_columns(df, required_core_cols),
        'has_datetime_index': isinstance(df.index, pd.DatetimeIndex),
        'weather_ranges': validate_weather_ranges(df),
    }

    # Check for aggregated columns
    agg_present = sum(col in df.columns for col in expected_agg_cols)
    validation_results['aggregated_columns_present'] = agg_present >= 2

    # Check for weather columns
    weather_present = sum(col in df.columns for col in expected_weather_cols)
    validation_results['weather_columns_present'] = weather_present >= 2

    # Check accident count characteristics
    if 'accident_count' in df.columns:
        validation_results['accident_count_valid'] = (
            df['accident_count'].min() >= 0 and  # No negative counts
            df['accident_count'].dtype in ['int64', 'int32', 'float64'] and  # Numeric type
            not df['accident_count'].isna().all()  # Not all null
        )
    else:
        validation_results['accident_count_valid'] = False

    # Check location consistency
    if 'location_id' in df.columns and 'bin_id' in df.columns:
        # Each location_id should map to unique bin_id
        location_bin_mapping = df.groupby('location_id')['bin_id'].nunique()
        validation_results['location_bin_consistency'] = (location_bin_mapping == 1).all()
    else:
        validation_results['location_bin_consistency'] = False

    # Check temporal coverage
    if isinstance(df.index, pd.DatetimeIndex):
        time_gaps = df.index.to_series().diff()
        median_gap = time_gaps.median()
        validation_results['consistent_time_intervals'] = (
            pd.Timedelta('59 minutes') <= median_gap <= pd.Timedelta('61 minutes')
        )
    else:
        validation_results['consistent_time_intervals'] = False

    return validation_results


def check_data_completeness(df: pd.DataFrame, critical_cols: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Check data completeness (non-null percentage) for each column

    Args:
        df (pd.DataFrame): DataFrame to check
        critical_cols (List[str], optional): Critical columns that should have high completeness

    Returns:
        Dict[str, float]: Completeness percentage for each column
    """
    completeness = {}

    for col in df.columns:
        non_null_pct = (df[col].notna().sum() / len(df)) * 100
        completeness[col] = non_null_pct

        # Warn if critical columns have low completeness
        if critical_cols and col in critical_cols and non_null_pct < 95:
            warnings.warn(f"Critical column {col} has only {non_null_pct:.1f}% completeness")

    return completeness


def print_validation_report(validation_results: Dict[str, bool], data_type: str = "data"):
    """
    Print a formatted validation report

    Args:
        validation_results (Dict[str, bool]): Results from validation functions
        data_type (str): Type of data being validated (for reporting)
    """
    print(f"\n{'='*50}")
    print(f"VALIDATION REPORT: {data_type.upper()}")
    print(f"{'='*50}")

    all_passed = True
    for check, passed in validation_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check.replace('_', ' ').title():<25} {status}")
        if not passed:
            all_passed = False

    print(f"{'='*50}")
    overall_status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"Overall Status: {overall_status}")
    print(f"{'='*50}\n")
