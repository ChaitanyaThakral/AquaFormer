import os
if 'PYTENSOR_FLAGS' not in os.environ:
    os.environ['PYTENSOR_FLAGS'] = 'cxx='
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def build_and_sample_model(df: pd.DataFrame, draws: int = 2000, tune: int = 1000):
    """
    Build and sample a Bayesian hierarchical spatiotemporal model using PyMC.
    Uses pm.Data to allow out-of-sample predictions.
    
    Returns:
        (model, idata): Tuple of the PyMC model object and the InferenceData.
    """
    elevation = df['elevation'].values
    temperature = df['temp_celsius'].values
    pressure = df['pressure_hpa'].values
    actual_precip = df['actual_precip_mm'].values
    
    elev_std = (elevation - np.mean(elevation)) / (np.std(elevation) + 1e-6)
    temp_std = (temperature - np.mean(temperature)) / (np.std(temperature) + 1e-6)
    pres_std = (pressure - np.mean(pressure)) / (np.std(pressure) + 1e-6)
    
    with pm.Model() as spatiotemporal_model:
        # Wrap data in Data for later swapping
        elev_data = pm.Data('elev_std', elev_std)
        temp_data = pm.Data('temp_std', temp_std)
        pres_data = pm.Data('pres_std', pres_std)
        obs_data = pm.Data('actual_precip', actual_precip + 1e-4)
        
        # Hierarchical Priors
        elevation_effect = pm.Normal('elev', mu=0, sigma=1)
        temp_effect = pm.Normal('temp_effect', mu=0, sigma=1)
        pres_effect = pm.Normal('pres_effect', mu=0, sigma=1)
        intercept = pm.Normal('intercept', mu=0, sigma=2)
        
        mu_moisture = (
            intercept 
            + elevation_effect * elev_data 
            + temp_effect * temp_data 
            + pres_effect * pres_data
        )
        
        latent_moisture = pm.Normal('latent_moisture', mu=mu_moisture, sigma=1, shape=elev_data.shape)
        expected_precip = pm.Deterministic('expected_precip', pm.math.exp(latent_moisture))
        
        alpha = pm.Exponential('alpha', 1.0)
        beta = alpha / expected_precip
        
        observed_precip = pm.Gamma('obs_precip', alpha=alpha, beta=beta, observed=obs_data)
        
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            nuts_sampler='numpyro',
            random_seed=42,
            progressbar=True
        )
        
    max_rhat = az.summary(idata)['r_hat'].max()
    print(f"Maximum R-hat: {max_rhat}")
    
    if max_rhat > 1.01:
        raise RuntimeError(f"MCMC chains did not converge! Max R-hat ({max_rhat}) > 1.01. Pipeline halted.")
        
    return spatiotemporal_model, idata

def extract_moisture_posteriors(idata: az.InferenceData) -> pd.DataFrame:
    summary = az.summary(idata, var_names=['latent_moisture'])
    posteriors_df = pd.DataFrame({
        'moisture_mean': summary['mean'].values,
        'moisture_sd': summary['sd'].values
    })
    return posteriors_df

def evaluate_out_of_sample(model: pm.Model, idata: az.InferenceData, df_test: pd.DataFrame, threshold: float):
    """Evaluate calibration on a held-out test set."""
    elevation = df_test['elevation'].values
    temperature = df_test['temp_celsius'].values
    pressure = df_test['pressure_hpa'].values
    actual_precip = df_test['actual_precip_mm'].values
    
    elev_std = (elevation - np.mean(elevation)) / (np.std(elevation) + 1e-6)
    temp_std = (temperature - np.mean(temperature)) / (np.std(temperature) + 1e-6)
    pres_std = (pressure - np.mean(pressure)) / (np.std(pressure) + 1e-6)
    
    with model:
        # Swap in the test data
        pm.set_data({
            'elev_std': elev_std,
            'temp_std': temp_std,
            'pres_std': pres_std,
            'actual_precip': actual_precip + 1e-4
        })
        # Generate posterior predictive samples
        ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, random_seed=42, progressbar=False)
    
    sim_precip = ppc.posterior_predictive['obs_precip'].values
    n_chains, n_draws, n_obs = sim_precip.shape
    sim_precip_flat = sim_precip.reshape(n_chains * n_draws, n_obs)
    
    # Calculate probability of extreme event
    y_prob = (sim_precip_flat > threshold).mean(axis=0)
    
    # Synchronize threshold for ground truth
    y_true_binary = (actual_precip > threshold).astype(int)
    
    return y_true_binary, y_prob

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10):
    """Calculate Expected Calibration Error."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy='uniform')
    
    # Manually calculate bin counts
    hist, _ = np.histogram(y_prob, bins=bins, range=(0, 1))
    
    # Filter out empty bins like calibration_curve does
    nonzero_bins = hist > 0
    hist = hist[nonzero_bins]
    
    # Calculate ECE
    ece = np.sum(np.abs(prob_true - prob_pred) * hist) / len(y_prob)
    return ece

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, ece: float, save_path: str = 'visualizations/calibration_curve.png'):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label=f"Bayesian Model (ECE={ece:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
