"""
src/features/engineering.py

Main feature engineering orchestration
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, RobustScaler
import scipy.stats.mstats as mstats




def engineer_weather_features(df: pd.DataFrame, pbar: Optional[tqdm] = None) -> pd.DataFrame:
    """
    Engineer weather-specific features (without normalization/winsorization)

    Args:
        df (pd.DataFrame): DataFrame with weather data
        pbar (Optional[tqdm]): Progress bar to update

    Returns:
        pd.DataFrame: DataFrame with engineered weather features (not normalized)
    """
    result = df.copy()

    # One-hot encode categorical weather features only
    if pbar:
        pbar.set_description("One-hot encoding weather categories")
    categorical_weather_cols = [col for col in ['rain_cat', 'snow_cat', 'wind_gust_cat', 'vis_cat']
                               if col in result.columns]
    if categorical_weather_cols:
        result = onehot_encode_features(result, categorical_weather_cols)
    if pbar:
        pbar.update(1)

    return result


def winsorize_and_normalize_all_features(df: pd.DataFrame, pbar: Optional[tqdm] = None) -> pd.DataFrame:
    """
    Winsorize and normalize all numeric features including weather and interaction features

    Args:
        df (pd.DataFrame): DataFrame with features to process
        pbar (Optional[tqdm]): Progress bar to update

    Returns:
        pd.DataFrame: DataFrame with winsorized and normalized features
    """
    result = df.copy()

    # Identify numeric columns to process
    weather_cols = ['rain', 'snow', 'wind_gust', 'snow_lag_1h', 'rain_lag_1h']
    interaction_cols = [col for col in result.columns if col.startswith('interact_')]

    # Get existing weather columns
    existing_weather_cols = [col for col in weather_cols if col in result.columns]

    # Combine all columns that need processing
    cols_to_process = existing_weather_cols + interaction_cols

    if cols_to_process:
        # Step 1: Winsorize extreme values
        if pbar:
            pbar.set_description("Winsorizing weather and interaction features")
        result = winsorize_features(result, cols_to_process)
        if pbar:
            pbar.update(1)

        # Step 2: Normalize all features
        if pbar:
            pbar.set_description("Normalizing weather and interaction features")
        result = normalize_features(result, cols_to_process, method="robust")
        print(f"Winsorized and normalized {len(cols_to_process)} features: {cols_to_process}")
        if pbar:
            pbar.update(1)
            
        # Step 3: Log transform visibility
        if pbar:
            pbar.set_description("Normalizing weather and interaction features")
        result = log_transform_visibility(result)
        print(f"Log10-transformed visiblity column")
        if pbar:
            pbar.update(1)
    else:
        print("No numeric features found to winsorize and normalize")
        if pbar:
            pbar.update(2)

    return result


def create_interaction_features(df: pd.DataFrame, pbar: Optional[tqdm] = None) -> pd.DataFrame:
    """
    Create interaction features between different variables

    Args:
        df (pd.DataFrame): Input dataframe
        pbar (Optional[tqdm]): Progress bar to update

    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    result = df.copy()

    # Weather + Time interactions
    if pbar:
        pbar.set_description("Creating weather-time interactions")

    # Rain during rush hour
    if 'rain' in result.columns and 'is_rush_hour' in result.columns:
        result['rain_during_rush'] = result['rain'] * result['is_rush_hour']

    # Snow during weekend
    if 'snow' in result.columns and 'is_weekend' in result.columns:
        result['snow_during_weekend'] = result['snow'] * result['is_weekend']

    # Poor visibility at night
    if 'visibility' in result.columns and 'hour' in result.columns:
        result['poor_vis_at_night'] = (result['visibility'] < 1000) & (
            (result['hour'] >= 20) | (result['hour'] <= 6)
        ).astype(int)

    if pbar:
        pbar.update(1)

    # Spatial + Time interactions
    if pbar:
        pbar.set_description("Creating spatial-time interactions")

    if pbar:
        pbar.update(1)

    return result



def normalize_features(df: pd.DataFrame, columns: List[str], method: str = "robust") -> pd.DataFrame:
    """
    Normalizes features to fit a standard Normal distribution, i.e., N(0, 1)

    Args:
        df (pd.DataFrame): the input dataframe to modify
        columns (List[str]): the columns to normalize
        method (str): normalization method - "standard" or "robust" (default: "robust")

    Returns:
        pd.DataFrame: the newly modified dataframe with normalized values
    """
    df = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("method must be 'standard' or 'robust'")

    df[columns] = scaler.fit_transform(df[columns])

    return df

def winsorize_features(df: pd.DataFrame, columns: List[str], lower_pct: float = 0.01, upper_pct: float = 0.995) -> pd.DataFrame:
    """
    Winsorizes (clips) features to prevent extreme outliers in columns. Done before normalization.

    Args:
        df (pd.DataFrame): the input dataframe to modify
        columns (List[str]): the columns to apply clipping
        lower_pct (float): lower percentile for clipping (default: 0.01)
        upper_pct (float): upper percentile for clipping (default: 0.995)

    Returns:
        pd.DataFrame: the newly modified dataframe with winsorized values
    """

    assert 0 <= lower_pct < upper_pct <= 1, "lower_pct must be < upper_pct and both in [0,1]"

    df = df.copy()

    for col in columns:
        if col in df.columns:
            original_max = df[col].max()
            df[col] = mstats.winsorize(df[col], limits=[lower_pct, 1 - upper_pct])
            print(f"Winsorized {col}: original max {original_max:.2f} -> {df[col].max():.2f}")

    return df


def log_transform_visibility(df: pd.DataFrame, visibility_col: str = "visibility") -> pd.DataFrame:
    """
    Apply log10 transformation to visibility data for better scaling.

    Args:
        df (pd.DataFrame): the input dataframe to modify
        visibility_col (str): the visibility column name (default: "visibility")

    Returns:
        pd.DataFrame: dataframe with log10_visibility column added
    """
    df = df.copy()
    df[visibility_col] = np.log10(df[visibility_col])
    return df

def onehot_encode_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    One-hot encode categorical columns.

    Args:
        df (pd.DataFrame): the input dataframe to modify
        columns (List[str]): the categorical columns to encode

    Returns:
        pd.DataFrame: dataframe with one-hot encoded columns
    """
    return pd.get_dummies(df, columns=columns, dtype=int)