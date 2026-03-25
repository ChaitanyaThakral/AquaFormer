import os
import torch
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import importlib

vit_module = importlib.import_module("src.models.05_vision_transformer")
SpatiotemporalViT = vit_module.SpatiotemporalViT

tr_mod = importlib.import_module("src.models.09_train_real_data")
load_data = tr_mod.load_data

def main():
    print("Building Dynamic Risk Map for Dec 4-5, 2023 Atmospheric River...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    feats, tgts, timestamps, raw_moist = load_data('data/raw')
    
    # Normalize features (using 2021-2022 stats)
    tr_end = 15312
    f_mean = feats[:tr_end].mean(axis=(0, 1), keepdims=True)
    f_std = feats[:tr_end].std(axis=(0, 1), keepdims=True) + 1e-6
    feats_norm = (feats - f_mean) / f_std
    
    # Find timestamp for Dec 4, 2023
    # timestamps are numpy datetime64
    target_date = np.datetime64('2023-12-04T12:00:00')
    idx = np.abs(timestamps - target_date).argmin()
    
    actual_time = timestamps[idx]
    print(f"Target timestamp found: {actual_time}")
    
    # Extract 24-hour sequence BEFORE this timestamp
    # idx is the target hour, so sequence is [idx-24 : idx]
    seq_x = feats_norm[idx-24:idx]
    seq_m = raw_moist[idx]
    true_y = tgts[idx]
    
    # Run model
    model = SpatiotemporalViT(
        in_features=7, seq_length=24, grid_h=50, grid_w=50,
        patch_size=5, embed_dim=128, depth=4, num_heads=8,
        dim_feedforward=1280
    ).to(device)
    
    model.load_state_dict(torch.load('runs/best_model_v7.pth', map_location=device, weights_only=True))
    model.eval()
    
    x_tensor = torch.from_numpy(seq_x).float().unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(x_tensor)
        p_real = torch.expm1(p).cpu().squeeze().numpy()
        
    moisture_bound = seq_m * 10000.0
    p_clamped = np.minimum(p_real, moisture_bound)
    p_clamped = np.maximum(p_clamped, 0.0)
    p_clamped = p_clamped.reshape(50, 50)
    
    # Generate Folium Map
    # Bounding box: N=49.0, W=-125.0, S=42.0, E=-114.0
    # Center of PNW
    m = folium.Map(location=[45.5, -119.5], zoom_start=6, tiles='OpenStreetMap')
    
    lats = np.linspace(49.0, 42.0, 50)
    lons = np.linspace(-125.0, -114.0, 50)
    
    # Build HeatMap data: [lat, lon, weight]
    heat_data = []
    for i in range(50):
        for j in range(50):
            # Only add significant rainfall to heatmap
            val = float(p_clamped[i, j])
            if val > 1.0:
                heat_data.append([float(lats[i]), float(lons[j]), val])
                
    # Add HeatMap
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1, gradient={
        0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'
    }).add_to(m)
    
    # Add a title/marker for the event
    folium.Marker(
        location=[47.6, -122.3], # Seattle
        popup=f"Seattle\\nPredicted: {p_clamped[0, 12]:.2f}mm",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    map_file = 'aquaformer_risk_map.html'
    m.save(map_file)
    print(f"Successfully saved dynamic risk map to {map_file}!")

if __name__ == '__main__':
    main()
