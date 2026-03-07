import os
import glob
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine

print("Initializing Data Transformation Pipeline...")

# 1. Connect to the PostGIS Docker Database
engine = create_engine('postgresql://postgres:admin@localhost:5433/aquaformer')

# 2. Find all downloaded NetCDF files
data_dir = 'data/raw'
nc_files = glob.glob(os.path.join(data_dir, '*.nc'))

if not nc_files:
    print("No .nc files found in data/raw/. Ensure the download script finished.")
    exit()

print(f"Found {len(nc_files)} months of data to process.")

# 3. Process files one by one to prevent RAM crashing
for file_path in nc_files:
    print(f"\nProcessing {file_path}...")

    ds = xr.open_dataset(file_path, engine='netcdf4')

    print("Flattening tensor to dataframe...")
    df = ds.to_dataframe().reset_index()

    # Rename columns — valid_time is the correct coordinate name from Copernicus
    df = df.rename(columns={
        't2m':        'temp_celsius',
        'sp':         'pressure_hpa',
        'u10':        'wind_u_vector',
        'v10':        'wind_v_vector',
        'tp':         'actual_precip_mm',
        'valid_time': 'reading_timestamp'  # Copernicus uses valid_time, not time
    })

    # Convert units for machine learning
    if 'temp_celsius' in df.columns:
        df['temp_celsius'] = df['temp_celsius'] - 273.15       # Kelvin → Celsius
    if 'actual_precip_mm' in df.columns:
        df['actual_precip_mm'] = df['actual_precip_mm'] * 1000 # Meters → mm

    # Drop rows with NaN values
    df = df.dropna()

    print(f"Flattened dataset contains {len(df)} rows.")
    print("Sample of processed data:")
    print(df[['latitude', 'longitude', 'reading_timestamp', 'temp_celsius', 'actual_precip_mm']].head())

    ds.close() # Free memory

print("\nTransformation test complete! Ready for SQL injection phase.")
