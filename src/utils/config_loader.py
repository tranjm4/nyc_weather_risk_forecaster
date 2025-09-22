"""
src/config/loader.py

Configuration loader for interaction effects and experiment settings
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class InteractionConfig:
    """Configuration class for managing interaction effects"""

    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file

        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get_enabled_interaction_types(self) -> List[str]:
        """Get list of enabled interaction types"""
        enabled_types = []
        interaction_types = [
            'weather_time_interactions',
            'spatial_time_interactions',
            'weather_spatial_interactions',
            'multiway_interactions',
            'threshold_interactions',
            'polynomial_interactions'
        ]

        for interaction_type in interaction_types:
            if self.config.get(interaction_type, {}).get('enabled', False):
                enabled_types.append(interaction_type)

        return enabled_types

    def get_interactions_by_type(self, interaction_type: str) -> List[Dict[str, Any]]:
        """Get all interactions for a specific type"""
        return self.config.get(interaction_type, {}).get('interactions', [])

    def get_settings(self) -> Dict[str, Any]:
        """Get general settings"""
        return self.config.get('settings', {})

    def validate_features_exist(self, df: pd.DataFrame, required_features: List[str]) -> bool:
        """
        Validate that required features exist in the dataframe

        Args:
            df (pd.DataFrame): Input dataframe
            required_features (List[str]): List of required feature names

        Returns:
            bool: True if all features exist, False otherwise
        """
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features for interaction: {missing_features}")
            return False
        return True


def create_interaction_from_config(df: pd.DataFrame, interaction: Dict[str, Any],
                                 settings: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a single interaction feature based on configuration

    Args:
        df (pd.DataFrame): Input dataframe
        interaction (Dict[str, Any]): Interaction configuration
        settings (Dict[str, Any]): General settings

    Returns:
        pd.DataFrame: DataFrame with new interaction feature
    """
    result = df.copy()

    name = interaction['name']
    features = interaction['features']
    operation = interaction['operation']
    prefix = settings.get('interaction_prefix', 'interact_')
    feature_name = f"{prefix}{name}"

    # Validate features exist if required
    if settings.get('validate_features', True):
        if not all(f in result.columns for f in features):
            print(f"Skipping interaction {name}: missing features {features}")
            return result

    try:
        if operation == 'multiply':
            # Multiply features together
            interaction_values = result[features[0]].copy()
            for feature in features[1:]:
                interaction_values *= result[feature]
            result[feature_name] = interaction_values

        elif operation == 'threshold':
            # Create binary feature based on threshold
            feature = features[0]
            threshold = interaction['threshold']
            comparison = interaction['comparison']

            if comparison == 'greater':
                result[feature_name] = (result[feature] > threshold).astype(int)
            elif comparison == 'less':
                result[feature_name] = (result[feature] < threshold).astype(int)
            elif comparison == 'equal':
                result[feature_name] = (result[feature] == threshold).astype(int)
            else:
                raise ValueError(f"Unknown comparison type: {comparison}")

        elif operation == 'polynomial':
            # Create polynomial features
            feature = features[0]
            degree = interaction['degree']
            result[feature_name] = result[feature] ** degree

        elif operation == 'custom':
            # Execute custom logic
            custom_logic = interaction['custom_logic']
            # Replace feature names with actual column references
            for feature in features:
                custom_logic = custom_logic.replace(feature, f"result['{feature}']")
            result[feature_name] = eval(custom_logic).astype(int)

        else:
            raise ValueError(f"Unknown operation type: {operation}")

        # Handle null values if required
        if settings.get('require_non_null', True):
            # Set interaction to null if any base feature is null
            null_mask = result[features].isnull().any(axis=1)
            result.loc[null_mask, feature_name] = np.nan

        print(f"Created interaction feature: {feature_name}")

    except Exception as e:
        print(f"Error creating interaction {name}: {e}")

    return result


def create_interactions_from_config(df: pd.DataFrame, config: InteractionConfig) -> pd.DataFrame:
    """
    Create all interaction features based on configuration

    Args:
        df (pd.DataFrame): Input dataframe
        config (InteractionConfig): Configuration object

    Returns:
        pd.DataFrame: DataFrame with all interaction features
    """
    result = df.copy()
    settings = config.get_settings()

    print("Creating interaction features from configuration...")

    # Process each enabled interaction type
    for interaction_type in config.get_enabled_interaction_types():
        print(f"Processing {interaction_type}...")
        interactions = config.get_interactions_by_type(interaction_type)

        for interaction in interactions:
            result = create_interaction_from_config(result, interaction, settings)

    # Normalize interaction features if requested
    if settings.get('normalize_interactions', False):
        print("Normalizing interaction features...")
        prefix = settings.get('interaction_prefix', 'interact_')
        interaction_cols = [col for col in result.columns if col.startswith(prefix)]

        if interaction_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            result[interaction_cols] = scaler.fit_transform(result[interaction_cols])
            print(f"Normalized {len(interaction_cols)} interaction features")

    return result