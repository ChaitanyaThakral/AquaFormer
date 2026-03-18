"""
Day 10: Production Training & Cost-Aware Evaluation Pipeline.

Implements a fully instrumented training loop with:
  - AdamW optimizer with weight decay
  - OneCycleLR scheduler (stepped per batch)
  - Gradient clipping to protect Softplus activations
  - TensorBoard logging of scalars, learning rate, and spatial heatmaps
  - Validation loop computing rare-event R², cost-aware error, and violation rate
"""

import os
import importlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

# --- Internal imports (handle numbered module names) ---
vit_module = importlib.import_module("src.models.05_vision_transformer")
loss_module = importlib.import_module("src.models.06_physics_loss")
metrics_module = importlib.import_module("src.models.08_evaluation_metrics")

SpatiotemporalViT = vit_module.SpatiotemporalViT
PhysicsInformedLoss = loss_module.PhysicsInformedLoss
calculate_rare_event_r2 = metrics_module.calculate_rare_event_r2
calculate_cost_aware_error = metrics_module.calculate_cost_aware_error
calculate_physical_violation_rate = metrics_module.calculate_physical_violation_rate


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Heatmap Utility
# ─────────────────────────────────────────────────────────────────────────────

def _make_spatial_heatmap(y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int):
    """
    Create a Matplotlib figure with side-by-side 50×50 heatmaps of
    predicted vs actual rainfall for the first sample in the batch.

    Returns a matplotlib Figure suitable for ``writer.add_figure()``.
    """
    pred_grid = y_pred[0].detach().cpu().reshape(50, 50).numpy()
    true_grid = y_true[0].detach().cpu().reshape(50, 50).numpy()

    vmin = min(pred_grid.min(), true_grid.min())
    vmax = max(pred_grid.max(), true_grid.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(true_grid, cmap='Blues', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'Ground Truth (Epoch {epoch})')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_grid, cmap='Blues', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f'Prediction (Epoch {epoch})')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training Epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                max_grad_norm=1.0):
    """
    Run one training epoch.

    The OneCycleLR scheduler is stepped **per batch** (not per epoch).
    Gradient clipping is applied before every optimizer step.
    """
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_penalty = 0.0

    for x_input, y_true in dataloader:
        x_input = x_input.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()

        # Forward
        y_pred = model(x_input)
        losses = criterion(y_pred, y_true, x_input)
        loss = losses['loss']

        # Backward
        loss.backward()

        # Gradient clipping — protects Softplus gradients under heavy
        # physics penalties and prevents transformer gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # OneCycleLR must be stepped per batch
        scheduler.step()

        running_loss += loss.item()
        running_mse += losses['mse_loss'].item()
        running_penalty += losses['physics_penalty'].item()

    n = len(dataloader)
    return {
        'loss': running_loss / n,
        'mse': running_mse / n,
        'penalty': running_penalty / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation Epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, moisture_idx=4,
                   moisture_scale=1.0):
    """
    Run one validation pass and compute the custom evaluation metrics:
      - Rare-event R²  (99th percentile)
      - Cost-aware error (10× FN penalty)
      - Physical violation rate
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_proxies = []

    for x_input, y_true in dataloader:
        x_input = x_input.to(device)
        y_true = y_true.to(device)

        y_pred = model(x_input)
        losses = criterion(y_pred, y_true, x_input)

        running_loss += losses['loss'].item()

        # Collect for epoch-level metric calculation
        all_preds.append(y_pred)
        all_targets.append(y_true)

        # Extract the moisture proxy the same way PhysicsInformedLoss does
        moisture = x_input[:, :, :, moisture_idx]
        water_proxy = moisture[:, -1, :] * moisture_scale
        all_proxies.append(water_proxy)

    # Concatenate across all batches
    y_pred_all = torch.cat(all_preds, dim=0)
    y_true_all = torch.cat(all_targets, dim=0)
    proxy_all = torch.cat(all_proxies, dim=0)

    # --- Custom Metrics ---
    rare_r2 = calculate_rare_event_r2(y_pred_all, y_true_all, percentile_val=99.0)
    cost_result = calculate_cost_aware_error(y_pred_all, y_true_all, threshold=10.0, fn_weight=10.0)
    violation_rate = calculate_physical_violation_rate(y_pred_all, proxy_all)

    n = len(dataloader)
    return {
        'loss': running_loss / n,
        'rare_event_r2': rare_r2.item(),
        'cost_score': cost_result['cost_score'].item(),
        'fn_count': cost_result['fn_count'].item(),
        'fa_count': cost_result['fa_count'].item(),
        'violation_rate': violation_rate.item(),
        # Keep the last batch for spatial heatmaps
        '_last_pred': all_preds[-1],
        '_last_true': all_targets[-1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Run
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    log_dir='runs/aquaformer',
    device=None,
):
    """
    Full production training run with TensorBoard instrumentation.

    Args:
        model: The SpatiotemporalViT instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of epochs.
        lr: Peak learning rate for OneCycleLR.
        weight_decay: AdamW weight decay.
        max_grad_norm: Maximum gradient norm for clipping.
        log_dir: TensorBoard log directory.
        device: torch.device to use.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # --- Loss ---
    criterion = PhysicsInformedLoss(
        alpha=1.0, beta=10.0,
        moisture_scale=1.0, aggregation_mode='last', moisture_idx=4,
    ).to(device)

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- OneCycleLR Scheduler (total_steps = epochs × batches) ---
    total_steps = num_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
    )

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Training on {device} | {num_epochs} epochs | "
          f"{len(train_loader)} train batches | {len(val_loader)} val batches")
    print(f"TensorBoard logs -> {log_dir}")
    print("-" * 70)

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            max_grad_norm=max_grad_norm,
        )

        # ---- Validate ----
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # ---- TensorBoard Scalars ----
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Train_MSE', train_metrics['mse'], epoch)
        writer.add_scalar('Loss/Physics_Penalty', train_metrics['penalty'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        writer.add_scalar('Metric/Rare_Event_R2', val_metrics['rare_event_r2'], epoch)
        writer.add_scalar('Metric/Cost_Score', val_metrics['cost_score'], epoch)
        writer.add_scalar('Metric/Violation_Rate', val_metrics['violation_rate'], epoch)
        writer.add_scalar('Metric/FN_Count', val_metrics['fn_count'], epoch)
        writer.add_scalar('Metric/FA_Count', val_metrics['fa_count'], epoch)

        # Current learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LR/OneCycleLR', current_lr, epoch)

        # ---- Spatial Heatmaps (every 5 epochs) ----
        if epoch % 5 == 0 or epoch == 1:
            fig = _make_spatial_heatmap(val_metrics['_last_pred'],
                                        val_metrics['_last_true'], epoch)
            writer.add_figure('Images/Spatial_Heatmap', fig, epoch)
            plt.close(fig)

        # ---- Console ----
        print(f"Epoch {epoch:>3d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Rare R2: {val_metrics['rare_event_r2']:.4f} | "
              f"Violation: {val_metrics['violation_rate']:.1f}% | "
              f"LR: {current_lr:.2e}")

    writer.close()
    print("-" * 70)
    print("Training complete. Run  tensorboard --logdir runs  to visualize.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpatiotemporalViT(
        in_features=6, seq_length=24,
        grid_h=50, grid_w=50, patch_size=5,
        embed_dim=256, depth=4, num_heads=8,
    )

    # ---- Smoke test with synthetic data ----
    print("Running smoke test with synthetic data ...")
    from torch.utils.data import TensorDataset

    num_samples = 64
    x_dummy = torch.randn(num_samples, 24, 2500, 6)
    # Set moisture channel to something sensible
    x_dummy[:, :, :, 4] = torch.rand(num_samples, 24, 2500) * 20.0
    y_dummy = torch.rand(num_samples, 2500) * 15.0

    dataset = TensorDataset(x_dummy, y_dummy)
    train_ds, val_ds = random_split(dataset, [48, 16])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = run_training(
        model, train_loader, val_loader,
        num_epochs=10, lr=1e-3,
        log_dir='runs/aquaformer_smoke',
        device=device,
    )
    print("Smoke test passed.")
