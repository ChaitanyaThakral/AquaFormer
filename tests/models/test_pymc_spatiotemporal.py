import os
os.environ['PYTENSOR_FLAGS'] = 'cxx='
import pytest
import pandas as pd
import numpy as np
import pymc as pm
import importlib

pymc_spat = importlib.import_module("src.models.02_pymc_spatiotemporal")
build_and_sample_model = pymc_spat.build_and_sample_model
extract_moisture_posteriors = pymc_spat.extract_moisture_posteriors
evaluate_out_of_sample = pymc_spat.evaluate_out_of_sample
calculate_ece = pymc_spat.calculate_ece
plot_calibration_curve = pymc_spat.plot_calibration_curve

@pytest.fixture
def dummy_weather_data():
    """Create a dummy dataset mimicking the weather data for testing."""
    np.random.seed(42)
    n_samples = 50
    data = {
        'elevation': np.random.uniform(0, 3000, n_samples),
        'temp_celsius': np.random.normal(15, 10, n_samples),
        'pressure_hpa': np.random.normal(1013, 10, n_samples),
        'actual_precip_mm': np.random.exponential(5, n_samples)
    }
    return pd.DataFrame(data)

def test_pymc_spatiotemporal_model_compiles_and_samples(dummy_weather_data):
    draws = 10
    tune = 10
    from unittest.mock import patch
    
    try:
        dummy_summary = pd.DataFrame({'r_hat': [1.0]})
        with patch.object(pymc_spat.az, 'summary', return_value=dummy_summary):
            model, idata = build_and_sample_model(
                df=dummy_weather_data, 
                draws=draws, 
                tune=tune
            )
    except Exception as e:
        pytest.fail(f"Model building or sampling failed with error: {e}")
        
    assert idata is not None
    assert hasattr(idata, 'posterior')
    
    posterior_vars = list(idata.posterior.data_vars.keys())
    assert 'elev' in posterior_vars
    assert 'intercept' in posterior_vars
    assert 'temp_effect' in posterior_vars
    assert 'pres_effect' in posterior_vars
    assert 'alpha' in posterior_vars
    
    n_chains = idata.posterior.dims['chain']
    n_draws = idata.posterior.dims['draw']
    
    assert n_chains > 0
    assert n_draws == draws

def test_extract_moisture_posteriors(dummy_weather_data):
    draws = 10
    tune = 10
    from unittest.mock import patch
    
    dummy_summary = pd.DataFrame({'r_hat': [1.0]})
    with patch.object(pymc_spat.az, 'summary', return_value=dummy_summary):
        model, idata = build_and_sample_model(
            df=dummy_weather_data, 
            draws=draws, 
            tune=tune
        )
        
    posteriors_df = extract_moisture_posteriors(idata)
    
    assert isinstance(posteriors_df, pd.DataFrame)
    assert len(posteriors_df) == len(dummy_weather_data)
    assert 'moisture_mean' in posteriors_df.columns
    assert 'moisture_sd' in posteriors_df.columns

def test_calibration_metrics(dummy_weather_data, tmp_path):
    """Test out of sample evaluation and calibration plotting."""
    draws = 10
    tune = 10
    from unittest.mock import patch
    
    dummy_summary = pd.DataFrame({'r_hat': [1.0]})
    with patch.object(pymc_spat.az, 'summary', return_value=dummy_summary):
        model, idata = build_and_sample_model(
            df=dummy_weather_data, 
            draws=draws, 
            tune=tune
        )
    
    # Define a threshold for extreme events
    threshold = 5.0
    
    # Evaluate out of sample
    y_true_binary, y_prob = evaluate_out_of_sample(model, idata, dummy_weather_data, threshold)
    
    assert len(y_true_binary) == len(dummy_weather_data)
    assert len(y_prob) == len(dummy_weather_data)
    
    # Calculate ECE
    ece = calculate_ece(y_true_binary, y_prob, bins=10)
    assert isinstance(ece, float)
    assert 0 <= ece <= 1.0
    
    # Test plotting
    save_path = str(tmp_path / 'calibration_curve.png')
    plot_calibration_curve(y_true_binary, y_prob, ece, save_path=save_path)
    
    assert os.path.exists(save_path)
