import os
import glob
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ERA5 raw column names to project-standard names."""
    return df.rename(columns={
        't2m':        'temp_celsius',
        'sp':         'pressure_hpa',
        'u10':        'wind_u_vector',
        'v10':        'wind_v_vector',
        'tp':         'actual_precip_mm',
        'valid_time': 'reading_timestamp',
    })


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 native units to human-readable units.

    - Temperature: Kelvin -> Celsius
    - Precipitation: meters -> millimeters
    """
    df = df.copy()
    if 'temp_celsius' in df.columns:
        df['temp_celsius'] = df['temp_celsius'] - 273.15
    if 'actual_precip_mm' in df.columns:
        df['actual_precip_mm'] = df['actual_precip_mm'] * 1000
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary Copernicus coordinate columns and NaN rows."""
    df = df.copy()
    cols_to_drop = ['number', 'expver']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna()
    return df


def transform_dataset(ds: xr.Dataset) -> pd.DataFrame:
    """Full transform pipeline: flatten xarray Dataset to a clean DataFrame."""
    df = ds.to_dataframe().reset_index()
    df = rename_columns(df)
    df = convert_units(df)
    df = clean_dataframe(df)
    return df


if __name__ == '__main__':
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
        df = transform_dataset(ds)

        print(f"Flattened dataset contains {len(df)} rows. Injecting to PostGIS...")

        try:
            df.to_sql(
                name='climate_data',
                con=engine,
                if_exists='append',
                index=False,
                chunksize=50000,
            )
            print(f"✅ Successfully injected {file_path} into SQL!")
        except Exception as e:
            print(f"❌ SQL Injection failed for {file_path}. Error: {e}")
            break

        ds.close()

    print("\nTransformation and Injection pipeline complete!")
