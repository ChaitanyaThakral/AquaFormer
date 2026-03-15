import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib

pytorch_dataset = importlib.import_module("src.data.04_pytorch_dataset")
SpatiotemporalDataset = pytorch_dataset.SpatiotemporalDataset

@pytest.fixture
def valid_dataframe():
    """Create a valid dummy DataFrame spanning 48 hours."""
    np.random.seed(42)
    num_grids = 2500
    hours = 48
    
    # Create contiguous timestamps
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=hours, freq="h")
    
    # Vectorized generation for speed
    n_total = num_grids * hours
    df = pd.DataFrame({
        'timestamp': np.repeat(timestamps, num_grids),
        'grid_id': np.tile(np.arange(num_grids), hours),
        'temp_celsius': np.random.normal(15, 5, n_total),
        'pressure_hpa': np.random.normal(1013, 10, n_total),
        'wind_u': np.random.normal(0, 5, n_total),
        'wind_v': np.random.normal(0, 5, n_total),
        'moisture': np.random.normal(50, 10, n_total),
        'elevation': np.random.uniform(0, 3000, n_total),
        'actual_precip_mm': np.random.exponential(2, n_total)
    })
    return df

def test_single_sample_shape_is_24_2500_6(valid_dataframe):
    dataset = SpatiotemporalDataset(valid_dataframe, sequence_length=24, num_grids=2500)
    
    X, y = dataset[0]
    
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32
    
    assert X.shape == (24, 2500, 6)
    assert y.shape == (2500,)

def test_len_matches_expected_windows(valid_dataframe):
    # 48 contiguous hours, sequence=24 -> valid starts are 0 through 23
    # Total valid windows = 48 - 24 = 24
    dataset = SpatiotemporalDataset(valid_dataframe, sequence_length=24, num_grids=2500)
    assert len(dataset) == 24

def test_missing_hour_windows_are_dropped(valid_dataframe):
    # Drop hour 10 to create a gap
    df_missing = valid_dataframe[valid_dataframe['timestamp'] != pd.Timestamp("2026-01-01 10:00:00")]
    dataset = SpatiotemporalDataset(df_missing, sequence_length=24, num_grids=2500)
    
    # Compute expected valid indices explicitly
    times = df_missing['timestamp'].unique()
    expected_valid = 0
    for i in range(len(times) - 24):
        if pd.to_datetime(times[i + 24]) - pd.to_datetime(times[i]) == pd.Timedelta(hours=24):
            expected_valid += 1
            
    assert len(dataset) == expected_valid
    assert len(dataset) < 24 # ensure some were dropped

def test_dataloader_batches_correctly(valid_dataframe):
    dataset = SpatiotemporalDataset(valid_dataframe, sequence_length=24, num_grids=2500)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    X_batch, y_batch = next(iter(dataloader))
    assert X_batch.shape == (4, 24, 2500, 6)
    assert y_batch.shape == (4, 2500)

def test_raises_on_missing_required_columns(valid_dataframe):
    df_broken = valid_dataframe.drop(columns=['moisture'])
    with pytest.raises(ValueError, match="Missing required columns"):
        SpatiotemporalDataset(df_broken)

def test_raises_on_incomplete_grid_per_timestamp(valid_dataframe):
    # Drop exactly one row
    df_broken = valid_dataframe.drop(index=0)
    with pytest.raises(ValueError, match="exactly 2500 grid points"):
        SpatiotemporalDataset(df_broken)

def test_raises_on_duplicate_rows(valid_dataframe):
    # Duplicate the first row
    df_broken = pd.concat([valid_dataframe, valid_dataframe.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        SpatiotemporalDataset(df_broken)

def test_grid_id_layout_consistent_across_timestamps(valid_dataframe):
    df_broken = valid_dataframe.copy()
    # Replace one grid_id with an entirely new one (9999)
    # This creates 2501 unique canonical grid IDs overall.
    ts2 = df_broken['timestamp'].unique()[1]
    idx_0 = df_broken[(df_broken['timestamp'] == ts2) & (df_broken['grid_id'] == 0)].index[0]
    df_broken.at[idx_0, 'grid_id'] = 9999
    
    with pytest.raises(ValueError, match="Expected 2500 unique grid IDs"):
        SpatiotemporalDataset(df_broken, sequence_length=24, num_grids=2500)
        
def test_raises_on_empty_valid_windows(valid_dataframe):
    # Only 10 hours of data, sequence_length=24
    df_short = valid_dataframe[valid_dataframe['timestamp'] < pd.Timestamp("2026-01-01 10:00:00")]
    with pytest.raises(ValueError, match="No valid contiguous windows found"):
        SpatiotemporalDataset(df_short, sequence_length=24, num_grids=2500)
