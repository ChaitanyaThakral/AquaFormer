import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Optional

class SpatiotemporalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 24, target_col: str = 'actual_precip_mm', num_grids: int = 2500, feature_cols: Optional[List[str]] = None):
        """
        Robust PyTorch Dataset for Spatiotemporal weather data.
        
        Args:
            df: pandas DataFrame containing the weather data.
            sequence_length: Number of past hours to look back (default: 24).
            target_col: Name of the target column (default: 'actual_precip_mm').
            num_grids: Expected number of geographic grid points per timestamp (default: 2500).
            feature_cols: List of feature column names.
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.num_grids = num_grids
        
        if feature_cols is None:
            self.feature_cols = ['temp_celsius', 'pressure_hpa', 'wind_u', 'wind_v', 'moisture', 'elevation']
        else:
            self.feature_cols = feature_cols
            
        self.required_cols = ['timestamp', 'grid_id', target_col] + self.feature_cols
        
        # Explicit datetime parsing
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
            
        # 1. Schema Validation
        self._validate_schema(df)
        
        # 2. Reshaping
        self.data_array, self.targets, self.timestamps = self._reshape_and_sort(df)
        
        # 3. Valid Indices Generation
        self.valid_indices = self._build_valid_indices(self.timestamps)
        
        if len(self.valid_indices) == 0:
            raise ValueError("No valid contiguous windows found for the given sequence_length.")
        
    def _validate_schema(self, df: pd.DataFrame):
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.duplicated(subset=['timestamp', 'grid_id']).any():
            raise ValueError("Duplicate (timestamp, grid_id) pairs found.")
            
        # Check if every timestamp has exactly num_grids
        grid_counts = df.groupby('timestamp').size()
        if not (grid_counts == self.num_grids).all():
            raise ValueError(f"Not all timestamps have exactly {self.num_grids} grid points.")
            
        # Verify grid layout consistency against canonical set
        canonical_grid_ids = np.sort(df['grid_id'].unique())
        if len(canonical_grid_ids) != self.num_grids:
            raise ValueError(f"Expected {self.num_grids} unique grid IDs, found {len(canonical_grid_ids)}.")
            
        for ts, grp in df.groupby('timestamp'):
            if not np.array_equal(np.sort(grp['grid_id'].values), canonical_grid_ids):
                raise ValueError("Grid IDs are inconsistent across timestamps.")
            
    def _reshape_and_sort(self, df: pd.DataFrame):
        # Sort to ensure consistent ordering (temporally, then spatially)
        df = df.sort_values(by=['timestamp', 'grid_id']).reset_index(drop=True)
        
        # Extract unique timestamps (already sorted)
        timestamps = df['timestamp'].unique()
        num_timestamps = len(timestamps)
        
        # Reshape features to (time_steps, num_grids, features)
        data_flat = df[self.feature_cols].values
        data_array = data_flat.reshape((num_timestamps, self.num_grids, len(self.feature_cols))).astype(np.float32)
        
        # Reshape targets to (time_steps, num_grids)
        targets_flat = df[self.target_col].values
        targets_array = targets_flat.reshape((num_timestamps, self.num_grids)).astype(np.float32)
        
        return data_array, targets_array, pd.to_datetime(timestamps)
        
    def _build_valid_indices(self, times: pd.DatetimeIndex) -> List[int]:
        valid = []
        for i in range(len(times) - self.sequence_length):
            # Check if the time window is exactly 'sequence_length' hours long
            time_diff = times[i + self.sequence_length] - times[i]
            if time_diff == pd.Timedelta(hours=self.sequence_length):
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Extract the sequence window: Shape (24, 2500, 6)
        x_window = self.data_array[start_idx : start_idx + self.sequence_length]
        
        # Extract the target for the hour immediately following the window
        # Shape (2500,)
        y_target = self.targets[start_idx + self.sequence_length]
        
        x_tensor = torch.from_numpy(x_window).float()
        y_tensor = torch.from_numpy(y_target).float()
        
        return x_tensor, y_tensor
