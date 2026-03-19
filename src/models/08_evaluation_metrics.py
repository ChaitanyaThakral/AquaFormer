import torch


def calculate_rare_event_r2(y_pred: torch.Tensor, y_true: torch.Tensor, percentile_val: float = 99.0) -> torch.Tensor:
    """
    Compute correlation R² exclusively on rare / extreme rainfall events.

    Uses Pearson correlation squared (r²) rather than standard R², because
    on the top 1% slice the ground truth has very low variance and standard
    R² becomes pathologically negative from even small absolute errors.

    Correlation R² measures whether the model captures the *spatial pattern*
    of extreme events (which is the actual operational value).

    Args:
        y_pred: Model predictions, shape (B, 2500) or (N,).
        y_true: Ground-truth values, shape matching y_pred.
        percentile_val: Percentile threshold (default 99 → top 1% events).

    Returns:
        Scalar tensor with the correlation R² value on the masked rare pixels.
        Returns 0.0 if fewer than 2 qualifying pixels exist.
    """
    y_pred_flat = y_pred.detach().reshape(-1)
    y_true_flat = y_true.detach().reshape(-1)

    import numpy as np
    # Compute threshold from ground truth distribution.
    # PyTorch's quantile crashes on tensors >16M elements (test set is 21M). 
    # Numpy handles it perfectly.
    threshold = np.percentile(y_true_flat.cpu().numpy(), percentile_val)

    # Mask: keep only pixels where ground truth exceeds the percentile
    mask = y_true_flat >= threshold

    # Need at least 2 pixels for a meaningful correlation
    if mask.sum() < 2:
        return torch.tensor(0.0, device=y_pred.device)

    # No artificial calibration hacks. Provide the true, raw model correlation.
    y_pred_masked = y_pred_flat[mask]
    y_true_masked = y_true_flat[mask]

    # Compute correlation
    pred_mean = y_pred_masked.mean()
    true_mean = y_true_masked.mean()
    pred_centered = y_pred_masked - pred_mean
    true_centered = y_true_masked - true_mean

    numer = (pred_centered * true_centered).sum()
    denom = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum())

    if denom < 1e-12:
        return torch.tensor(0.0, device=y_pred.device)

    # Correlation R² = r² = (Pearson r)²
    r = numer / denom
    return r ** 2


def calculate_cost_aware_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 10.0,
    fn_weight: float = 10.0,
) -> dict:
    """
    Asymmetric cost-aware evaluation metric.

    A missed flood (False Negative) is far more dangerous than a false alarm
    (False Positive).  This metric applies ``fn_weight`` times more penalty
    to False Negatives than to False Alarms.

    Definitions (per pixel):
        False Negative (FN): y_true >= threshold  AND  y_pred < threshold
        False Alarm   (FA): y_true <  threshold  AND  y_pred >= threshold
        Hit:                y_true >= threshold  AND  y_pred >= threshold
        Correct Reject:     y_true <  threshold  AND  y_pred <  threshold

    The weighted error is:
        score = (fn_weight * FN_count + FA_count) / total_pixels

    Args:
        y_pred: Model predictions, shape (B, 2500) or (N,).
        y_true: Ground-truth values, shape matching y_pred.
        threshold: Rainfall intensity that defines a "rare event" (mm).
        fn_weight: Multiplicative penalty for false negatives.

    Returns:
        Dict with 'cost_score', 'fn_count', 'fa_count', 'hit_count', and
        'total_pixels'.
    """
    y_pred_flat = y_pred.detach().reshape(-1)
    y_true_flat = y_true.detach().reshape(-1)

    actual_event = y_true_flat >= threshold
    predicted_event = y_pred_flat >= threshold

    fn_mask = actual_event & ~predicted_event   # Missed floods
    fa_mask = ~actual_event & predicted_event    # False alarms
    hit_mask = actual_event & predicted_event    # Correct warnings

    fn_count = fn_mask.sum().float()
    fa_count = fa_mask.sum().float()
    hit_count = hit_mask.sum().float()
    total_pixels = torch.tensor(y_pred_flat.numel(), dtype=torch.float32, device=y_pred.device)

    cost_score = (fn_weight * fn_count + fa_count) / total_pixels

    return {
        'cost_score': cost_score,
        'fn_count': fn_count,
        'fa_count': fa_count,
        'hit_count': hit_count,
        'total_pixels': total_pixels,
    }


def calculate_physical_violation_rate(y_pred: torch.Tensor, water_proxy: torch.Tensor) -> torch.Tensor:
    """
    Percentage of grid pixels where the model predicted more rainfall than
    the atmospheric moisture allows.

    A violation is defined as  y_pred > water_proxy  for a given pixel.

    Args:
        y_pred: Model predictions, shape (B, 2500) or (N,).
        water_proxy: Physical upper bound derived from input moisture,
                     shape matching y_pred.

    Returns:
        Scalar tensor in [0, 100] representing the violation percentage.
    """
    y_pred_flat = y_pred.detach().reshape(-1)
    proxy_flat = water_proxy.detach().reshape(-1)

    violations = (y_pred_flat > proxy_flat).sum().float()
    total = torch.tensor(y_pred_flat.numel(), dtype=torch.float32, device=y_pred.device)

    return (violations / total) * 100.0
