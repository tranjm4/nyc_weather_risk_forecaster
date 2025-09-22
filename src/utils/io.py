"""
src/utils/io.py

I/O util functions to load/save files in the project directory
"""

import pickle
import arviz as az
import pandas as pd
import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def save_pkl(data: Any, filepath: str) -> None:
    """
    Serializes the provided data into a pkl file

    Args:
        data: The data to serialize (DataFrame, dict, or any pickle-able object)
        filepath (str): Path where to save the pickle file
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pkl(filepath: str) -> Any:
    """
    De-serializes the pkl file

    Args:
        filepath (str): Path to the pickle file to load

    Returns:
        Any: The deserialized data
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_model_results(results_dict: Dict, save_path: str) -> None:
    """
    Save model results with ArviZ for traces and pickle for metadata

    Args:
        results_dict (Dict): Dictionary containing model results
        save_path (str): Directory path to save results
    """
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    arviz_data = {}
    metadata = {}

    for location_id, result in results_dict.items():
        # Extract what we can serialize
        metadata[location_id] = {
            "summary": result["summary"],
            "model_features": result["model_features"],
        }

        # Convert approximation to InferenceData if possible
        if "approx" in result:
            try:
                # Sample from the approximation for ArviZ
                trace = result["approx"].sample(1000)
                idata = az.convert_to_inference_data(trace)
                arviz_data[location_id] = idata
            except Exception as e:
                print(f"Could not convert location {location_id} to ArviZ: {e}")

    # Save ArviZ data
    if arviz_data:
        az.to_netcdf(arviz_data, save_path / "model_traces.nc")

    # Save metadata
    with open(save_path / "model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {len(arviz_data)} location traces and metadata to {save_path}")


def load_model_results(save_path: str) -> tuple:
    """
    Load saved model results

    Args:
        save_path (str): Directory path containing saved results

    Returns:
        tuple: (metadata, arviz_data)
    """
    save_path = Path(save_path)

    # Load traces
    try:
        arviz_data = az.from_netcdf(save_path / "model_traces.nc")
    except FileNotFoundError:
        arviz_data = {}

    # Load metadata
    with open(save_path / "model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return metadata, arviz_data


def save_traces_arviz(results: Dict, base_dir: str) -> None:
    """
    Save each location's trace as a separate file

    Args:
        results (Dict): Dictionary of model results by location
        base_dir (str): Base directory to save traces
    """
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    saved_locations = {}
    summaries = {}

    for location_id, result in results.items():
        clean_id = location_id.replace('.', 'p').replace('_', '-')
        filename = f"{base_dir}/loc_{clean_id}.nc"

        try:
            # Sample from approximation to get trace
            trace = result['approx'].sample(1000)

            # Convert to InferenceData and save
            idata = az.convert_to_inference_data(trace)
            idata.to_netcdf(filename)

            print(f"Saved {location_id} to {filename}")
            saved_locations[location_id] = f"loc_{clean_id}.nc"
            summaries[location_id] = result['summary']

        except Exception as e:
            print(f"Failed to save {location_id}: {e}")

    # Save metadata
    if saved_locations:
        sample_result = next(iter(results.values()))
        metadata = {
            'model_features': sample_result['model_features'],
            'location_mapping': saved_locations,
            'total_locations': len(results),
            'saved_locations': len(saved_locations)
        }

        with open(f"{base_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        with open(f"{base_dir}/summaries.pkl", "wb") as f:
            pickle.dump(summaries, f)


def load_all_results(base_dir: str) -> tuple:
    """
    Load all saved results from individual trace files

    Args:
        base_dir (str): Base directory containing saved traces

    Returns:
        tuple: (loaded_results, model_features)
    """
    # Load metadata
    with open(f"{base_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load all location results
    location_mapping = metadata["location_mapping"]
    loaded_results = {}

    for name in location_mapping:
        mapped_filename = location_mapping[name]
        filepath = f"{base_dir}/{mapped_filename}"

        try:
            trace = az.from_netcdf(filepath)
            loaded_results[name] = {
                'trace': trace,
                'location_id': name
            }
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    return loaded_results, metadata['model_features']