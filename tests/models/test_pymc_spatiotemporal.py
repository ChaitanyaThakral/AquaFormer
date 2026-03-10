import os
os.environ['PYTENSOR_FLAGS'] = 'cxx='
import pytest
import pandas as pd
import numpy as np
import pymc as pm
import importlib
pymc_spat = importlib.import_module("src.models.02_pymc_spatiotemporal")
build_and_sample_model = pymc_spat.build_and_sample_model

@pytest.fixture
def dummy_weather_data():
    """Create a dummy dataset mimicking the weather data for testing."""
    np.random.seed(42)
    n_samples = 50
    
    # Generate dummy data
    data = {
        'elevation': np.random.uniform(0, 3000, n_samples),
        'temp_celsius': np.random.normal(15, 10, n_samples),
        'pressure_hpa': np.random.normal(1013, 10, n_samples),
        # Generate some positive precipitation values
        'actual_precip_mm': np.random.exponential(5, n_samples)
    }
    
    return pd.DataFrame(data)

def test_pymc_spatiotemporal_model_compiles_and_samples(dummy_weather_data):
    """
    Test that the Bayesian spatiotemporal model can be built and sampled
    using the JAX/NumPyro backend without errors.
    """
    # Use very small number of draws and tuning steps just to verify 
    # compilation and sampling pipeline works
    draws = 10
    tune = 10
    
    from unittest.mock import patch
    
    try:
        # Mock az.summary to return a DataFrame with max r_hat = 1.0
        # so the diagnostic gate passes even with low draws.
        dummy_summary = pd.DataFrame({'r_hat': [1.0]})
        with patch.object(pymc_spat.az, 'summary', return_value=dummy_summary):
            idata = build_and_sample_model(
                df=dummy_weather_data, 
                draws=draws, 
                tune=tune
            )
    except Exception as e:
        pytest.fail(f"Model building or sampling failed with error: {e}")
        
    # Check that the InferenceData object was returned and has the right structure
    assert idata is not None
    assert hasattr(idata, 'posterior')
    
    # Verify the posteriors exist for our defined priors
    posterior_vars = list(idata.posterior.data_vars.keys())
    assert 'elev' in posterior_vars
    assert 'intercept' in posterior_vars
    assert 'temp_effect' in posterior_vars
    assert 'pres_effect' in posterior_vars
    assert 'alpha' in posterior_vars
    
    # Check the shape of the samples
    # PyMC normally samples 4 chains by default
    n_chains = idata.posterior.dims['chain']
    n_draws = idata.posterior.dims['draw']
    
    assert n_chains > 0
    assert n_draws == draws
