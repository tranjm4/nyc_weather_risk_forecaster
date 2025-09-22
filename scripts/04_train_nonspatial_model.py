"""
scripts/04_train_nonspatial_models.py

This script runs the training procedure
"""

from argparse import ArgumentParser
from src.models.non_spatial_model import NonSpatialModel

import yaml

import pandas as pd

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--config-file", required=True,
                        help="The config file to read the model; reads from 'config/models/[ config_file ]'")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    config_file = args.config_file
    
    with open(f"config/models/{config_file}", "r") as file:
        temp_config = yaml.safe_load(file)
    num_samples = temp_config["num_samples"]

    model = NonSpatialModel(config_file = config_file)

    # Load data for fitting
    print("Loading training data...")
    df = pd.read_csv("data/processed/03_features.csv").dropna()
    
    if num_samples > 0:
        df = df.sample(num_samples)

    # Prepare features and target
    features = model.bayesian_model_config.metadata.features
    
    
    X = df[features]
    y = df['accident_count']

    # Check data types and values
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y dtype: {y.dtype}, y min: {y.min()}, y max: {y.max()}")
    print(f"X ranges: {X.min().to_dict()} to {X.max().to_dict()}")

    # Ensure y is integer type for Poisson
    y = y.astype(int)

    print("Building model...")
    model.build_model(X, y)
    
    
    print(f"Training model with {len(features)} features and {len(df)} samples...")
    # Fit the model
    model.fit()

    print("Model training completed!")

    # Save the model
    model_name = model.bayesian_model_config.metadata.name
    model.save(f"models/{model_name}")
    print("Model saved to models/nonspatial_model.nc")

if __name__ == "__main__":
    main()
    