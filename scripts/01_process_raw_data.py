"""
scripts/01_process_raw_data.py

Runs data processing pipeline on raw data using src/data
"""

from src.data.load import load_weather_data, load_accident_data
import src.data.preprocessing as preprocessing

import pandas as pd

from argparse import ArgumentParser

from tqdm import tqdm
import time


def clean_data():
    accident_df = load_accident_data("data/raw/accident_data.csv")
    weather_df = load_weather_data("data/raw/weather_data.csv")
    
    cleaned_accident_df = preprocessing.clean_accident_df(accident_df)
    cleaned_weather_df = preprocessing.clean_weather_df(weather_df)
    
    merged_df = preprocessing.merge_weather_accident_dfs(cleaned_weather_df, cleaned_accident_df)
    
    return merged_df, cleaned_weather_df

def save_df(df: pd.DataFrame, save_path: str):
    df.to_csv(save_path, index=False)
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", default="./data/processed")
    
    args = parser.parse_args()
    save_path = args.path
    return save_path

def main():
    save_path = parse_args()
    
    with tqdm(total=2, desc="Overall Progress") as pbar:
        
        # perform primary processing on raw data
        
        pbar.set_description("Loading and cleaning raw data")
        cleaned_data, cleaned_weather_data = clean_data()
        pbar.update(1)
        
        # save the results (should be a merged dataframe)
        pbar.set_description("Saving data")
        save_df(cleaned_data, save_path + "/01_merged.csv")
        save_df(cleaned_weather_data, save_path + "/00_weather.csv")
        pbar.update(1)
        
        pbar.set_description("Complete!")

    

if __name__ == "__main__":
    main()