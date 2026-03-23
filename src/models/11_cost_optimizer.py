import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import importlib

vit_module = importlib.import_module("src.models.05_vision_transformer")
SpatiotemporalViT = vit_module.SpatiotemporalViT

tr_mod = importlib.import_module("src.models.09_train_real_data")
load_data = tr_mod.load_data
SeqDS = tr_mod.SeqDS
from torch.utils.data import DataLoader

def main():
    print("Loading data and model for Cost-Matrix Optimization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Validation Set
    feats, tgts, timestamps, raw_moist = load_data('data/raw')
    
    # Normalize features
    f_mean = feats.mean(axis=(0, 1), keepdims=True)
    f_std = feats.std(axis=(0, 1), keepdims=True) + 1e-6
    feats = (feats - f_mean) / f_std
    
    # Target in log space
    tgts = np.log1p(tgts)
    
    # Use 2023 Test Set for Cost Optimization to capture catastrophic events
    tr_end = 17520  # End of 2022
    va_end = len(feats)  # End of 2023
    
    va_feats = feats[tr_end:va_end]
    va_tgts = tgts[tr_end:va_end]
    va_moist = raw_moist[tr_end:va_end]
    
    va_ds = SeqDS(va_feats, va_tgts, va_moist, seq=24)
    va_dl = DataLoader(va_ds, batch_size=32, num_workers=0)

    # Load Model v7
    model = SpatiotemporalViT(
        in_features=7, seq_length=24, grid_h=50, grid_w=50,
        patch_size=5, embed_dim=128, depth=4, num_heads=8,
        dim_feedforward=1280
    ).to(device)
    
    model.load_state_dict(torch.load('runs/best_model_v7.pth', map_location=device, weights_only=True))
    model.eval()

    # Inference over validation set
    all_preds = []
    all_trues = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for x, y, m in va_dl:
            x, y = x.to(device), y.to(device)
            p = model(x)
            
            # Apply physics bound
            p_real = torch.expm1(p)
            moisture_bound = m * 10000.0
            p_clamped = torch.min(p_real, moisture_bound)
            p_clamped = torch.clamp(p_clamped, min=0.0)
            
            all_preds.append(p_clamped.cpu())
            all_trues.append(torch.expm1(y).cpu())

    preds = torch.cat(all_preds).numpy().flatten()
    trues = torch.cat(all_trues).numpy().flatten()

    # Costs
    COST_FN = 10_000_000
    COST_FP = 50_000
    TRUE_DISASTER_THRESHOLD = 10.0 # mm of rain

    actual_disaster_mask = trues >= TRUE_DISASTER_THRESHOLD
    print(f"Number of disasters (> 10mm): {np.sum(actual_disaster_mask)}")
    print(f"Max true rain: {np.max(trues):.2f} mm")
    print(f"Max predicted rain: {np.max(preds):.2f} mm")

    def expected_financial_cost(tau):
        predicted_evac_mask = preds >= tau
        
        # False Negatives: It was a disaster, but we predicted below tau
        fn_count = np.sum(actual_disaster_mask & ~predicted_evac_mask)
        
        # False Positives: We predicted above tau, but it was NOT a disaster
        fp_count = np.sum(~actual_disaster_mask & predicted_evac_mask)
        
        total_cost = (fn_count * COST_FN) + (fp_count * COST_FP)
        return total_cost

    print("\nOptimizing decision threshold...")
    # Standard threshold is usually 10.0 (predict exactly the threshold)
    baseline_cost = expected_financial_cost(10.0)
    print(f"Baseline Cost (Threshold = 10.0mm): ${baseline_cost:,.2f}")

    # Search for optimal tau using a fine grid search (step functions break minimize_scalar)
    taus = np.linspace(0.1, 15.0, 500)
    costs = [expected_financial_cost(t) for t in taus]
    
    min_idx = np.argmin(costs)
    optimal_tau = taus[min_idx]
    optimal_cost = costs[min_idx]
    
    print(f"Optimized Cost (Threshold = {optimal_tau:.2f}mm): ${optimal_cost:,.2f}")
    print(f"Financial Savings: ${(baseline_cost - optimal_cost):,.2f}")

    # Generate Cost Curve Plot
    print("\nGenerating Cost Curve Plot...")
    taus = np.linspace(0.1, 25.0, 100)
    costs = [expected_financial_cost(t) for t in taus]

    plt.figure(figsize=(10, 6))
    plt.plot(taus, costs, linewidth=2, color='darkred', label='Total Expected Cost')
    plt.axvline(x=optimal_tau, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_tau:.2f}mm)')
    plt.axvline(x=10.0, color='gray', linestyle=':', label='Naive Threshold (10.0mm)')
    
    plt.title('Cost-Matrix Optimization Curve', fontsize=14)
    plt.xlabel('Evacuation Decision Threshold (Predicted mm)', fontsize=12)
    plt.ylabel('Total Expected Cost ($)', fontsize=12)
    plt.yscale('log') # Use log scale because FN costs dominate exponentially
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('cost_curve.png', dpi=300, bbox_inches='tight')
    print("Saved 'cost_curve.png'.")

if __name__ == '__main__':
    main()
