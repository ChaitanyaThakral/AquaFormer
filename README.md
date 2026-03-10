# 🌊 AquaFormer: Physics-Informed Extreme Precipitation Forecaster

**AquaFormer** is a hybrid predictive system that bridges probabilistic Bayesian inference, deep learning, and atmospheric physics to forecast localized flash flood onset across the US Pacific Northwest.

Unlike standard statistical models that regress to the mean and fail on catastrophic tail-events, AquaFormer actively penalizes physically impossible predictions (e.g., predicting rain when atmospheric moisture is zero) by injecting fluid dynamic constraints directly into the neural network's loss function.

---

## 🛑 The Problem: Why Standard ML Fails on Weather
Traditional machine learning (XGBoost, LSTMs) treats weather data as purely statistical sequences. 
1. **The Averaging Trap:** When trained on decades of climate data, standard models learn to predict "average" weather extremely well but fail disastrously on catastrophic 1-in-100-year tail events (like a flash flood) because they natively regress to the mean.
2. **Physics Blindness:** Statistical models do not understand the physical world. They routinely output physically impossible predictions (e.g., predicting 5 inches of rain from a dry air mass simply because the temperature matched a historical pattern).

## 💡 The Solution: A Hybrid PIML Pipeline
AquaFormer solves this using a **Physics-Informed Machine Learning (PIML)** architecture divided into three tiers:
1. **The Probabilistic Tier (PyMC):** We use Bayesian inference to clean noisy satellite data, establish a mathematically sound "ground truth," and quantify uncertainty bounds based on topographical priors (e.g., elevation effects on wind).
2. **The Deep Learning Tier (Vision Transformer):** We feed this denoised data into a Vision Transformer (ViT), treating the 2,500+ Pacific Northwest weather grid points as a multi-channel image to capture spatial pressure gradients.
3. **The Physics Constraint:** During backpropagation, fluid dynamic equations are injected into the loss function. If the network predicts more rainfall than the total atmospheric moisture entering the grid, the loss function mathematically punishes the model.

---

## 🏗️ System Architecture & Engineering Roadmap

The project is structured across a 14-Day Engineering Roadmap.

### Phase 1: Spatial Data & Baseline Architecture `[COMPLETED]`
The goal of this phase was to build a database capable of handling multidimensional arrays over physical geography and mathematically prove why deep learning is required.

- ✅ **Spatiotemporal Schema:** Deployed a Dockerized **PostgreSQL** instance with the **PostGIS** extension. Created a `SpatialGrid` table using `GEOMETRY(Point, 4326)` for WGS 84 coordinates. Engineered a composite index on `(grid_id, reading_timestamp)` to ensure sliding-window neural network queries are lightning-fast.
- ✅ **The ERA5 Data Pipeline:** Extracted hourly climate data (Temperature, Pressure, Wind Vectors) from the Copernicus CDS API. Handled the massive NetCDF files using `xarray`. Flattened the 4D tensors and bulk-inserted **34,295,400 rows** into PostgreSQL via SQLAlchemy.
- ✅ **The Baseline & Class Imbalance Proof:** Trained an **XGBoost Classifier** using `scale_pos_weight=99` to penalize missing a flood. The model achieved 99% accuracy but only **0.28 Recall** on the extreme class, mathematically proving that non-sequential, non-spatial models fail on rare weather events.
- ✅ **Software Engineering Standards:** Refactored all ETL and modeling scripts into isolated functions. Implemented rigorous testing using `pytest`, achieving **100% test coverage** across data validation and model-logic pipelines.

### Phase 2: Bayesian Uncertainty & Denoising `[IN PROGRESS]`
The goal of this phase is to build a model that understands how geography dictates weather (e.g., mountains forcing air up, causing rain).

- ⏳ **PyMC Spatiotemporal Modeling:** Writing a Bayesian hierarchical model. Defining topographical priors (`elevation_effect = pm.Normal('elev', mu=0, sigma=1)`) and modeling the true atmospheric moisture as a latent variable. 
- 📅 **MCMC Sampling & Diagnostics:** Run the No-U-Turn Sampler (NUTS) using the **JAX/NumPyro** backend to compile MCMC sampling to XLA. Implement an automated gate calculating the Gelman-Rubin statistic ($\hat{R}$).
- 📅 **Expected Calibration Error (ECE):** Plotting a calibration curve mapping predicted probability vs empirical frequency. Target: ECE < 0.04.

### Phase 3: The Physics-Informed Transformer `[PLANNED]`
- 📅 **PyTorch 3D Tensor Construction:** Build a custom `torch.utils.data.Dataset` returning tensors of shape `(24, 2500, 6)` representing a 24-hour lookback window.
- 📅 **Vision Transformer (ViT) Architecture:** Treat the 2500 points as a $50 \times 50$ pixel "image" and implement Patch Embeddings for Multi-Head Self-Attention.
- 📅 **The Physics-Informed Loss Function:** Custom `nn.Module`: `Loss = α * MSE(Predicted, Actual) + β * ReLU(Predicted Rain - Total Column Water)`.
- 📅 **Training & Evaluation:** Train loop using AdamW and OneCycle scheduling. Evaluate using Continuous Ranked Probability Score (CRPS).

### Phase 4: Business Logic & MLOps `[PLANNED]`
- 📅 **The Cost-Matrix Optimizer:** Use `scipy.optimize` to find the probability threshold that minimizes financial cost (penalizing False Negatives 10x over False Positives).
- 📅 **Dynamic Risk Mapping:** Render interactive probability risk maps of the Pacific Northwest using `folium` and `geopandas`.
- 📅 **FastAPI & Redis Engine:** Deploy as a microservice using `FastAPI` and `Pydantic`, backed by a `Redis` cache.
- 📅 **System Hardening:** Write integration tests and explicitly unit-test the Physics Loss function with impossible water scenarios.

---

## 💻 Tech Stack
- **Data Engineering:** Python, pandas, xarray, SQLAlchemy
- **Database:** PostgreSQL, PostGIS, pgAdmin, Docker
- **Traditional ML:** XGBoost, scikit-learn
- **Bayesian Probabilistics:** PyMC, NumPyro, JAX, ArviZ
- **Deep Learning:** PyTorch
- **Testing:** pytest

---

## 🚀 How to Run (Phase 1)

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/ChaitanyaThakral/AquaFormer.git
   cd AquaFormer
   pip install -r requirements.txt
