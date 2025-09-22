"""
src/data/preprocessing.py
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import mstats


def clean_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning procedure for the weather dataframe.

    - removes unnecessary features
    - formats datetime to YY-MM-DD HH:00 (hourly, to match accidents)


    Args:
        df (pd.DataFrame): the input dataframe to modify

    Returns:
        pd.DataFrame: the cleaned dataframe
    """
    df = df.copy()

    keep_columns = [
        'datetime',
        'visibility',
        'wind_gust',
        'rain_1h',
        'snow_1h',
    ]
    # filter out irrelevant columns
    df = df[keep_columns]

    # reformat dt to the YY-MM-DD HH:00 in Eastern Time (hourly to match accidents)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('US/Eastern').dt.strftime("%Y-%m-%d %H:00")
    
    # drop duplicate dt values
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    
    # drop NaN values
    df = df.dropna(axis=0, how='any')
    
    # rename weather columns (rain_1h, snow_1h) and datetime to dt
    df = df.rename(columns={"rain_1h": "rain", "snow_1h": "snow", "datetime": "dt"})
    
    # categorize the weather conditions
    def classify_snow(snow):
        if snow == 0:
            return 'no snow'
        elif snow < 1:
            return 'light snow'
        elif snow < 4:
            return 'moderate snow'
        else:
            return 'heavy snow'
        
    def classify_rain(rain):
        if rain == 0:
            return 'no rain'
        elif rain < 2.5:
            return 'light rain'
        elif rain < 10:
            return 'moderate rain'
        else:
            return 'heavy rain'
        
    def classify_visibility(visibility):
        if visibility < 1000:
            return 'low vis'
        elif visibility < 5000:
            return 'medium vis'
        else:
            return 'high vis'

    def classify_wind_gusts(wind):
        if wind < 1:
            return 'calm wind'
        elif wind < 5.5:
            return 'light wind'
        elif wind < 11:
            return 'moderate wind'
        elif wind < 17:
            return 'strong wind'
        else:
            return 'very strong wind'

    df['rain_cat'] = df['rain'].apply(classify_rain)
    df['snow_cat'] = df['snow'].apply(classify_snow)
    df['vis_cat'] = df['visibility'].apply(classify_visibility)
    df['wind_gust_cat'] = df['wind_gust'].apply(classify_wind_gusts)
    
    return df


def clean_accident_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning procedure for the accident data

    Expected input columns: 'CRASH DATE', 'CRASH TIME', 'LATITUDE', 'LONGITUDE',
                           'ON STREET NAME', 'CROSS STREET NAME', 'COLLISION_ID'
    """
    df = df.copy()
    
    keep_columns = [
        'CRASH DATE',
        'CRASH TIME',
        'LATITUDE',
        'LONGITUDE',
        'ON STREET NAME',
        'CROSS STREET NAME',
    ]
    df = df[keep_columns]

    # rename columns to lowercase
    df = df.rename(columns={col_name: col_name.lower() for col_name in df.columns})

    # rename columns to snake_case
    df = df.rename(columns={col_name: "_".join(col_name.split()) for col_name in df.columns})
    
    # convert to datetime format: %Y-%m-%d %H:%M
    df['dt'] = df['crash_date'] + " " + df['crash_time']
    df['dt'] = pd.to_datetime(df['dt']).dt.strftime("%Y-%m-%d %H:00")
    # drop the crash_date and crash_time columns since they are redundant
    df = df.drop(labels=["crash_date", "crash_time"], axis=1)

    # drop NaN locations
    """
    NOTE:
        In the notebook (notebooks/1_accidents.ipynb), 
        we used geocoding to fill in the NaN coordinates if streets were included,
        but that is very time-intensive with small gains (only 70k out of 2m+ total rows).
        
        In the interest of time and with the abundance of data, 
        we will opt to drop any NaN coordinates.
    """
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])
        
    # remove weird longitude and latitude values that don't belong in NYC
    latitudes = df['latitude']
    longitudes = df['longitude']
    df = df[(latitudes > 40) & (latitudes < 41) & (longitudes > -74.7) & (longitudes < -73.7)]

    return df


def merge_weather_accident_dfs(weather_df: pd.DataFrame, accident_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the weather and accident dataframes for further processing.
    Requires the dataframes to be cleaned and formatted prior to calling this function
        - datetimes should match YY-MM-DD HH:00 datetime format

    Args:
        weather_df (pd.DataFrame): the weather dataframe (should be cleaned and formatted)
        accident_df (pd.DataFrame): the accident dataframe (should be cleaned and formatted)

    Returns:
        pd.DataFrame: the merged dataframe
    """
    merged_df = pd.merge(left=accident_df, right=weather_df, 
                         how='left',
                         left_on=['dt'],
                         right_on=['dt'])
    
    return merged_df
    