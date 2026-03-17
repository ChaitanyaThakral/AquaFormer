import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, moisture_scale=1.0, aggregation_mode='last', moisture_idx=4):
        """
        Physics-Informed Surrogate Loss.
        Constrains predictions against physical bounds (water proxy).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.moisture_scale = moisture_scale
        if aggregation_mode not in ['last', 'mean', 'max']:
            raise ValueError("aggregation_mode must be 'last', 'mean', or 'max'")
        self.aggregation_mode = aggregation_mode
        self.moisture_idx = moisture_idx
        
    def forward(self, y_pred, y_true, x_input):
        # y_pred: (B, 2500)
        # y_true: (B, 2500)
        # x_input: (B, 24, 2500, 6)
        
        # 1. Base Accuracy Term
        mse_loss = F.mse_loss(y_pred, y_true)
        
        # 2. Physics Proxy Calculation
        # Extract moisture: (B, 24, 2500)
        moisture = x_input[:, :, :, self.moisture_idx]
        
        if self.aggregation_mode == 'last':
            water_proxy = moisture[:, -1, :] # (B, 2500)
        elif self.aggregation_mode == 'mean':
            water_proxy = moisture.mean(dim=1)
        elif self.aggregation_mode == 'max':
            water_proxy = moisture.max(dim=1).values
            
        # Scale proxy
        water_proxy = water_proxy * self.moisture_scale
        
        # 3. Asymmetric Physics Penalty
        # Penalty: ReLU(y_pred - water_proxy)
        penalty = F.relu(y_pred - water_proxy).mean()
        
        # 4. Total Loss Assembly
        total_loss = self.alpha * mse_loss + self.beta * penalty
        
        # Return components for logging visibility
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'physics_penalty': penalty
        }
