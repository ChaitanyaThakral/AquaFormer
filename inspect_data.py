import xarray as xr
import numpy as np

ds = xr.open_dataset('data/raw/pnw_climate_2021_01.nc')
nlat = ds.dims['latitude']
nlon = ds.dims['longitude']
print(f"Grid: {nlat}x{nlon} = {nlat*nlon} points")
print(f"Timesteps: {ds.dims['valid_time']}")
print(f"Variables: {list(ds.data_vars)}")
print(f"Lat range: {float(ds.latitude.min()):.2f} to {float(ds.latitude.max()):.2f}")
print(f"Lon range: {float(ds.longitude.min()):.2f} to {float(ds.longitude.max()):.2f}")

tp = ds['tp'].values
print(f"Precip (m): min={np.nanmin(tp):.6f}, max={np.nanmax(tp):.6f}, mean={np.nanmean(tp):.6f}")
print(f"Precip (mm): min={np.nanmin(tp)*1000:.4f}, max={np.nanmax(tp)*1000:.4f}, mean={np.nanmean(tp)*1000:.4f}")

# Check if there's humidity/moisture
print(f"Has specific_humidity: {'q' in ds.data_vars or 'specific_humidity' in ds.data_vars}")
ds.close()
