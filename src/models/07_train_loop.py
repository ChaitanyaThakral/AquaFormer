import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.05_vision_transformer import SpatiotemporalViT
from src.models.06_physics_loss import PhysicsInformedLoss
from src.data.04_pytorch_dataset import AquaDataset # Assuming this is the dataset from Day 4

def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_penalty = 0.0
    
    for batch_idx, (x_input, y_true) in enumerate(dataloader):
        x_input = x_input.to(device)
        y_true = y_true.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x_input)
        
        # Physics-informed loss calculation
        losses = criterion(y_pred, y_true, x_input)
        loss = losses['loss']
        
        # Backward pass
        loss.backward()
        
        # --- Gradient Clipping (Day 9 Fix) ---
        # Protect the Softplus() gradients from collapsing under heavy physics penalties
        # and prevent exploding gradients in the transformer backbone
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        # -------------------------------------
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        epoch_loss += loss.item()
        epoch_mse += losses['mse_loss'].item()
        epoch_penalty += losses['physics_penalty'].item()
        
    num_batches = len(dataloader)
    return {
        'loss': epoch_loss / num_batches,
        'mse': epoch_mse / num_batches,
        'penalty': epoch_penalty / num_batches
    }

if __name__ == "__main__":
    # Example setup for production training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Model
    model = SpatiotemporalViT(
        in_features=6, 
        seq_length=24, 
        grid_h=50, 
        grid_w=50, 
        patch_size=5, 
        embed_dim=256, 
        depth=4, 
        num_heads=8
    ).to(device)
    
    # 2. Initialize Physics-Informed Loss
    criterion = PhysicsInformedLoss(
        alpha=1.0, 
        beta=10.0, 
        moisture_scale=1.0, 
        aggregation_mode='last', 
        moisture_idx=4
    ).to(device)
    
    # 3. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Optional: Dummy dataloader for illustration
    # dataset = AquaDataset(...)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Training loop initialized with gradient clipping enabled.")
    # for epoch in range(num_epochs):
    #     metrics = train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0)
    #     print(f"Epoch {epoch+1} | Loss: {metrics['loss']:.4f} | MSE: {metrics['mse']:.4f} | Penalty: {metrics['penalty']:.4f}")
