"""
src/data/ingestion.py

Functions to load the raw data
"""

import pandas as pd

def load_weather_data(file_path) -> pd.DataFrame:
    """
    Loads and returns the weather data as a Pandas DataFrame
    
    Args:
        file_path (str): The path to the dataset
        
    Returns:
        pd.DataFrame: the file dataframe
    """
    weather_df = pd.read_csv(file_path)
    
    # basic check
    assert type(weather_df) == pd.DataFrame
    assert weather_df.shape[0] > 0
    
    return weather_df


def load_accident_data(file_path) -> pd.DataFrame:
    """
    Loads and returns the accident data as a Pandas DataFrame
    
    Args:
        file_path (str): The path to the dataset
        
    Returns:
        pd.DataFrame: the file dataframe
    """
    accident_df = pd.read_csv(file_path)
    
    # basic check
    assert type(accident_df) == pd.DataFrame
    assert accident_df.shape[0] > 0
    
    return accident_df
    