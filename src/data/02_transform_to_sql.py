import os
import glob
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

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

    # Clean up unnecessary coordinate columns that Copernicus includes
    cols_to_drop = ['number', 'expver']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Drop rows with NaN values
    df = df.dropna()

    print(f"Flattened dataset contains {len(df)} rows. Injecting to PostGIS...")
    
    # --- SQL INJECTION ---
    # Push dataframe directly to the 'climate_data' table
    try:
        df.to_sql(
            name='climate_data',
            con=engine,
            if_exists='append', # Adds rows without dropping the table
            index=False,        # We don't need pandas index numbers in SQL
            chunksize=50000     # Safely upload 50k rows at a time
        )
        print(f"✅ Successfully injected {file_path} into SQL!")
    except Exception as e:
        print(f"❌ SQL Injection failed for {file_path}. Error: {e}")
        break # Stop the script so we can debug

    ds.close() # Free memory

print("\nTransformation and Injection pipeline complete!")
