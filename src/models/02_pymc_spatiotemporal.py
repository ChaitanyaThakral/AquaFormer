import os
if 'PYTENSOR_FLAGS' not in os.environ:
    os.environ['PYTENSOR_FLAGS'] = 'cxx='
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

def build_and_sample_model(df: pd.DataFrame, draws: int = 2000, tune: int = 1000):
    """
    Build and sample a Bayesian hierarchical spatiotemporal model using PyMC
    with the JAX/NumPyro backend for faster sampling.
    
    The model relates geography (elevation) to weather (precipitation),
    incorporating a latent variable for true atmospheric moisture.
    
    Args:
        df: pandas DataFrame with 'elevation', 'temp_celsius', 'pressure_hpa',
            and 'actual_precip_mm' columns.
        draws: number of samples to draw.
        tune: number of tuning samples.
        
    Returns:
        idata: ArviZ InferenceData object containing the sampled posteriors.
    """
    # Extract data as numpy arrays for PyMC
    elevation = df['elevation'].values
    temperature = df['temp_celsius'].values
    pressure = df['pressure_hpa'].values
    actual_precip = df['actual_precip_mm'].values
    
    # Standardize inputs to help MCMC convergence
    elev_std = (elevation - np.mean(elevation)) / (np.std(elevation) + 1e-6)
    temp_std = (temperature - np.mean(temperature)) / (np.std(temperature) + 1e-6)
    pres_std = (pressure - np.mean(pressure)) / (np.std(pressure) + 1e-6)
    
    with pm.Model() as spatiotemporal_model:
        # Hierarchical Priors
        # Geography dictates weather: mountains force air up
        elevation_effect = pm.Normal('elev', mu=0, sigma=1)
        
        # Additional effects from temperature and pressure
        temp_effect = pm.Normal('temp_effect', mu=0, sigma=1)
        pres_effect = pm.Normal('pres_effect', mu=0, sigma=1)
        
        # Intercept
        intercept = pm.Normal('intercept', mu=0, sigma=2)
        
        # Model the true atmospheric moisture as a latent (hidden) variable
        # The latent moisture is influenced by elevation, temp, and pressure
        mu_moisture = (
            intercept 
            + elevation_effect * elev_std 
            + temp_effect * temp_std 
            + pres_effect * pres_std
        )
        
        # The latent moisture itself has some variation not captured by the covariates
        latent_moisture = pm.Normal('latent_moisture', mu=mu_moisture, sigma=1, shape=len(df))
        
        # Expected precipitation depends on the latent moisture
        # We use an exponential link function to ensure expected precipitation is positive
        expected_precip = pm.Deterministic('expected_precip', pm.math.exp(latent_moisture))
        
        # Dispersion parameter for the Gamma distribution
        alpha = pm.Exponential('alpha', 1.0)
        
        # Likelihood
        # Expected value of Gamma is alpha / beta -> beta = alpha / expected_precip
        beta = alpha / expected_precip
        
        # Observed precipitation
        # Add a tiny epsilon to actual_precip to prevent 0s, which can cause log(0) issues in Gamma
        observed_precip = pm.Gamma(
            'obs_precip', 
            alpha=alpha, 
            beta=beta, 
            observed=actual_precip + 1e-4
        )
        
        # Sample using JAX/NumPyro backend for XLA compilation
        # which reduces runtime significantly
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            nuts_sampler='numpyro',
            random_seed=42,
            progressbar=True
        )
        
    # --- ADD THIS: Diagnostic Enforcement Gate ---
    # Extract the maximum Gelman-Rubin statistic across all parameters
    max_rhat = az.summary(idata)['r_hat'].max()
    print(f"Maximum R-hat: {max_rhat}")
    
    # If chains haven't converged, raise an exception to halt the pipeline
    if max_rhat > 1.01:
        raise RuntimeError(f"MCMC chains did not converge! Max R-hat ({max_rhat}) > 1.01. Pipeline halted.")
        
    return idata

def extract_moisture_posteriors(idata: az.InferenceData) -> pd.DataFrame:
    """
    Extract the posterior mean and standard deviation for the latent moisture 
    levels at every grid point.
    
    Args:
        idata: The InferenceData object returned by the MCMC sampler.
        
    Returns:
        pd.DataFrame containing 'moisture_mean' and 'moisture_sd' for each grid point.
    """
    # Extract the posterior summary for the 'latent_moisture' variable
    summary = az.summary(idata, var_names=['latent_moisture'])
    
    # Construct a clean DataFrame with the mathematically sound probabilities/metrics
    posteriors_df = pd.DataFrame({
        'moisture_mean': summary['mean'].values,
        'moisture_sd': summary['sd'].values
    })
    
    return posteriors_df
