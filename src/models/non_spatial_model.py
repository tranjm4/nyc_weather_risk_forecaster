"""
File: src/models/non_spatial_model.py

The non-spatial model, consisting of independent models per grid region
"""

from src.models.utils import ModelResponse

import pymc as pm
import numpy as np
import arviz as az
import yaml
import json
from pathlib import Path

from typing import List, Optional
from pydantic import BaseModel

from tqdm import tqdm

class Prior(BaseModel):
    distribution: str
    name: str
    parameters: dict
    
class Sampling(BaseModel):
    method: str
    chains: int
    iterations: int
    warmup: int
    target_accept: float
    max_treedepth: int
    
class ModelMetadata(BaseModel):
    name: str
    version: str
    description: str
    features: List[str]
    
class ModelStructure(BaseModel):
    likelihood: str
    link: str
    hierarchical: bool
        
class BayesianModelConfig(BaseModel):
    priors: List[Prior]
    structure: ModelStructure
    sampling: Sampling
    metadata: ModelMetadata
    

class NonSpatialModel:
    # Give the model a name
    _model_type = "NonSpatialModel"

    # And a version
    version = "0.1"

    def __init__(self, config_file: Optional[str] = None):
        # Only parse config if provided (not during loading)
        if config_file is not None:
            self.bayesian_model_config: BayesianModelConfig = self._parse_config_file(config_file)
        else:
            self.bayesian_model_config = None
            
        self.model = None
        self.X = None
        self.y = None

    
    def build_model(self, X, y, **kwargs) -> None:
        """
        Build the Bayesian model

        Args:
            X (pd.DataFrame): input data for the model, with the necessary features
            y (pd.Series): the target data for the model (i.e., accident counts)
        """
        # Validate and store the data
        self._validate_and_store_data(X, y)
        with pm.Model() as model:
            # Store coefficients for later use
            coefficients = {}

            # Create mutable data containers using stored data
            x_data = pm.Data("x_data", self.X)
            y_data = pm.Data("y_data", self.y)

            # Create priors using Pydantic config
            features = self.bayesian_model_config.metadata.features

            # Iterate through priors in self.model_config
            for prior in self.bayesian_model_config.priors:
                prior_name = f"beta_{prior.name}"
                distribution = prior.distribution

                if distribution == "Normal":
                    coeff = pm.Normal(
                        name=prior_name,
                        **prior.parameters
                    )
                    coefficients[prior.name] = coeff

            # Create linear predictor
            feature_coeffs = pm.math.stack([coefficients[feat] for feat in features])
            linear_pred = coefficients["intercept"] + pm.math.dot(x_data, feature_coeffs)

            # Apply link function and likelihood
            if self.bayesian_model_config.structure.likelihood == "Poisson":
                # Clip linear predictor to prevent overflow in exp()
                # This prevents lambda values > ~700 which cause NumPy overflow
                linear_pred_clipped = pm.math.clip(linear_pred, -20, 20)
                mu = pm.math.exp(linear_pred_clipped)  # log link
                y_obs = pm.Poisson("y_obs", mu=mu, observed=y_data)

            # Store the model in self.model
            self.model = model

    def _validate_and_store_data(self, X, y):
        """Validate and store input data"""
        if self.bayesian_model_config is None:
            raise ValueError("Model config not loaded. Cannot validate data.")

        # Validate features match config
        expected_features = set(self.bayesian_model_config.metadata.features)
        provided_features = set(X.columns) if hasattr(X, 'columns') else set()

        missing_features = expected_features - provided_features
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Store data for model building
        self.X = X[self.bayesian_model_config.metadata.features]  # Ensure correct order
        self.y = y

    def fit(self, **kwargs):
        """
        Runs the fitting procedure using variational inference.
        """
        if self.model is None:
            raise Exception("Model was not built yet! Call build_model(X, y) first.")

        sampling_config = self.bayesian_model_config.sampling if self.bayesian_model_config else {}

        with self.model:
            if sampling_config.method == "advi":
                advi_fit = pm.ADVI()
                tracker = pm.callbacks.Tracker(
                    mean=advi_fit.approx.mean.eval,
                    std=advi_fit.approx.std.eval,
                )
                self.approx = advi_fit.fit(
                    n = getattr(sampling_config, 'iterations', 2000),
                    score=True
                )
                losses = self.approx.hist

                # Sample from the variational approximation
                self.trace = self.approx.sample(
                    draws=getattr(sampling_config, 'iterations', 2000) // 4
                )
            elif sampling_config.method == "mcmc":
                self.trace = pm.sample(
                    draws = getattr(sampling_config, 'iterations', 2000),
                    tune = getattr(sampling_config, 'warmup', 1000),
                    chains = getattr(sampling_config, 'chains', 4),
                    target_accept = getattr(sampling_config, 'target_accept', 0.95),
                    **kwargs
                )
        
        
    def save(self, dir: str):
        """
        Serializes the model to a .nc file and the bayesian_model_config to a .json file
        
        Args:
            dir (str): The directory to which we save our trace and experiment metadata
        """
        if self.trace == None:
            raise Exception("No traces to be saved! Must run fit() or load() first.")
        
        # make the directory if it doesn't exist
        Path(dir).mkdir(exist_ok=True, parents=True)
        
        # serialize as .nc file
        az.to_netcdf(self.trace, f"{dir}/trace.nc")
        
        # if we used variational inference, we save the loss history
        if getattr(self, "approx", None) is not None:
            # serialize loss history as .npy file
            np.save(f"{dir}/loss_hist.npy", self.approx.hist)
        
        # Save metadata as JSON file (can be loaded back to BayesianModelConfig)
        metadata_json = self.bayesian_model_config.model_dump()
        with open(f"{dir}/metadata.json", "w") as metadata_file:
            json.dump(metadata_json, metadata_file, indent=4)
        
        
    def load(self, dir: str):
        """
        Deserializes the model given the directory.
        Reads from the provided directory argument and looks for trace.nc and metadata.json
        
        Loads the files onto self.trace and self.bayesian_model_config
        
        Args:
            dir (str): The directory from which to deserialize from.
                Expects trace.nc and metadata.json to be included.
        """
        self.trace = az.from_netcdf(dir + "/trace.nc")
        
        # Load metadata from JSON -> BayesianModelConfig
        with open(f"{dir}/metadata.json", "r") as metadata_file:
            json_data = json.load(metadata_file)
            self.bayesian_model_config = BayesianModelConfig.model_validate(json_data)


    def _parse_config_file(self, config_file: str) -> BayesianModelConfig:
        """Parses the config file to determine the parameters and model structure"""
        with open(f"config/models/{config_file}", 'r') as f:
            config = yaml.safe_load(f)
        
        # Get list of features (also store as metadata)
        features = config["features"]
        features_set = set(config["features"]) # use to identify default coefficients
        
        # Parse priors from YAML
        priors = []
        for coefficient in config["priors"]["coefficients"]:
            features_set.remove(coefficient)
            
            prior_config = config["priors"]["coefficients"][coefficient]
            
            distribution = prior_config["distribution"]
            params = prior_config["params"]
            
            prior = Prior(
                distribution = distribution,
                name = coefficient,
                parameters = params
            )
            
            priors.append(prior)

        # Add intercept prior
        intercept_config = config["priors"]["intercept"]
        intercept_prior = Prior(
            distribution=intercept_config["distribution"],
            name="intercept",
            parameters=intercept_config["params"]
        )
        priors.append(intercept_prior)

        # Set default priors for every other feature
        default_prior = config["priors"]["default_coefficient"]
        for feature in features_set:
            distribution = default_prior["distribution"]
            params = default_prior["params"]
            
            prior = Prior(
                distribution = distribution,
                name = feature,
                parameters = params
            )
            
            priors.append(prior)
        
        # Include metadata
        metadata = ModelMetadata(
            name = config["name"],
            description = config["description"],
            version = config["version"],
            features = features
        )
        
        # Define model specifications
        model_structure = ModelStructure(
            likelihood = config["model"]["likelihood"],
            link = config["model"]["link"],
            hierarchical = config["model"]["hierarchical"]
        )
        
        model = BayesianModelConfig(
            priors = priors,
            structure = model_structure,
            sampling = config["sampling"],
            metadata = metadata
        )
        
        return model
        

