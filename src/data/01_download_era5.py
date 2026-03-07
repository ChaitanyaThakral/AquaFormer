import cdsapi
import os
import time
import zipfile
import shutil
import xarray as xr

os.makedirs('data/raw', exist_ok=True)
c = cdsapi.Client()

# N , W , S , E
pnw_bounding_box = [49.0, -125.0, 42.0, -114.0]

years = ['2021', '2022', '2023']
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

print("Starting ERA5 data extraction via Python API...")

for year in years:
    for month in months:
        file_path = f'data/raw/pnw_climate_{year}_{month}.nc'

        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping...")
            continue

        print(f"Requesting data for {year}-{month}...")
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        '2m_temperature',
                        'surface_pressure',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        'total_precipitation'
                    ],
                    'year': [year],
                    'month': [month],
                    'day': [str(i).zfill(2) for i in range(1, 32)],
                    'time': [f"{str(i).zfill(2)}:00" for i in range(24)],
                    'area': pnw_bounding_box,
                    'data_format': 'netcdf',
                    'download_format': 'unarchived'
                },
                file_path
            )


            if zipfile.is_zipfile(file_path):
                print(f"  -> API sent a ZIP. Extracting and merging all variables...")
                extract_dir = f'data/raw/tmp_{year}_{month}'
                os.makedirs(extract_dir, exist_ok=True)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                extracted = [
                    os.path.join(extract_dir, f)
                    for f in os.listdir(extract_dir)
                    if f.endswith('.nc')
                ]

                if extracted:
                    print(f"  -> Found {len(extracted)} variable files. Merging...")
                    
                    datasets = [xr.open_dataset(f, engine='netcdf4') for f in extracted]
                    ds_merged = xr.merge(datasets)
                    ds_merged.load() 
                    
                    for ds in datasets:
                        ds.close() 
                    os.remove(file_path)           
                    ds_merged.to_netcdf(file_path) 
                    ds_merged.close()

                shutil.rmtree(extract_dir)         
                print(f"  -> Merge complete!")

            print(f"Successfully downloaded and extracted {year}-{month}.")
            time.sleep(2)

        except Exception as e:
            print(f"Failed to download {year}-{month}. Error: {e}")
            continue

print("Extraction pipeline complete!")