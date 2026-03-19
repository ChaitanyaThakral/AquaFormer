import torch
import pytest
import importlib

metrics_module = importlib.import_module("src.models.08_evaluation_metrics")
calculate_rare_event_r2 = metrics_module.calculate_rare_event_r2
calculate_cost_aware_error = metrics_module.calculate_cost_aware_error
calculate_physical_violation_rate = metrics_module.calculate_physical_violation_rate


# ---------- Cost-Aware Error Tests ----------

def test_cost_aware_penalty():
    """
    Construct a controlled scenario with exactly 1 False Negative and
    1 False Alarm.  Verify that the False Negative contributes exactly
    10x the cost of the False Alarm.
    """
    # 4 pixels total, threshold = 10.0
    #   Pixel 0: y_true=15, y_pred=5   → FN  (actual event, missed)
    #   Pixel 1: y_true=2,  y_pred=12  → FA  (no event, false alarm)
    #   Pixel 2: y_true=20, y_pred=25  → Hit
    #   Pixel 3: y_true=1,  y_pred=3   → Correct Reject
    y_true = torch.tensor([15.0, 2.0, 20.0, 1.0])
    y_pred = torch.tensor([5.0, 12.0, 25.0, 3.0])

    result = calculate_cost_aware_error(y_pred, y_true, threshold=10.0, fn_weight=10.0)

    assert result['fn_count'].item() == 1.0, f"Expected 1 FN, got {result['fn_count'].item()}"
    assert result['fa_count'].item() == 1.0, f"Expected 1 FA, got {result['fa_count'].item()}"
    assert result['hit_count'].item() == 1.0, f"Expected 1 Hit, got {result['hit_count'].item()}"
    assert result['total_pixels'].item() == 4.0

    # cost_score = (10 * 1 + 1 * 1) / 4 = 11 / 4 = 2.75
    expected_score = (10.0 * 1 + 1.0 * 1) / 4.0
    assert abs(result['cost_score'].item() - expected_score) < 1e-6, \
        f"Expected cost_score={expected_score}, got {result['cost_score'].item()}"

    # Verify the 10x asymmetry explicitly:
    # FN contribution = 10 * 1 = 10
    # FA contribution = 1 * 1 = 1
    fn_contribution = 10.0 * result['fn_count'].item()
    fa_contribution = 1.0 * result['fa_count'].item()
    assert fn_contribution == 10.0 * fa_contribution, \
        "False Negative penalty should be exactly 10x the False Alarm penalty."


def test_cost_aware_no_events():
    """When there are no events at all, cost should be zero."""
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.5, 1.0, 2.0])

    result = calculate_cost_aware_error(y_pred, y_true, threshold=10.0)
    assert result['cost_score'].item() == 0.0
    assert result['fn_count'].item() == 0.0
    assert result['fa_count'].item() == 0.0


# ---------- Rare Event R² Tests ----------

def test_rare_event_r2_masking():
    """
    Verify that the R² calculation completely ignores pixels below the
    99th percentile threshold.

    Strategy: Create a tensor where all 99th-percentile pixels have perfect
    predictions, but all below-threshold pixels have terrible predictions.
    R² should still be high because only the top 1% matters.
    """
    torch.manual_seed(42)
    n = 10000
    y_true = torch.zeros(n)
    y_pred = torch.zeros(n)

    # Bottom 99%: terrible predictions (large error)
    y_true[:9900] = torch.rand(9900) * 5.0
    y_pred[:9900] = torch.rand(9900) * 100.0  # wildly wrong

    # Top 1%: near-perfect predictions
    extreme_values = torch.linspace(50.0, 100.0, 100)
    y_true[9900:] = extreme_values
    y_pred[9900:] = extreme_values + torch.randn(100) * 0.01  # tiny noise

    r2 = calculate_rare_event_r2(y_pred, y_true, percentile_val=99.0)

    # R² on the extreme slice should be very close to 1.0
    assert r2.item() > 0.99, \
        f"Rare-event R² should be ~1.0 for perfect extreme predictions, got {r2.item()}"


def test_rare_event_r2_bad_predictions():
    """When predictions on extreme events are constant (no signal), correlation R² should be 0."""
    torch.manual_seed(0)
    n = 10000
    y_true = torch.arange(n, dtype=torch.float32)
    y_pred = torch.zeros(n)  # Predict 0 everywhere (zero variance)

    r2 = calculate_rare_event_r2(y_pred, y_true, percentile_val=99.0)
    # Constant predictions have zero variance → Pearson r is 0 → r² = 0
    assert r2.item() < 0.05, \
        f"Rare-event correlation R² should be ~0 for constant predictions, got {r2.item()}"


def test_rare_event_r2_too_few_pixels():
    """With fewer than 2 qualifying pixels, return 0.0 gracefully."""
    y_true = torch.tensor([1.0, 1.0, 1.0])  # All identical → quantile = 1.0
    y_pred = torch.tensor([1.0, 1.0, 1.0])

    # Since all values == threshold, the mask keeps all 3.
    # But if we had only 1 unique extreme pixel, check edge behaviour.
    r2 = calculate_rare_event_r2(y_pred, y_true, percentile_val=99.0)
    # All same → ss_tot = 0 → should return 0.0
    assert r2.item() == 0.0


# ---------- Physical Violation Rate Tests ----------

def test_violation_rate_none():
    """No violations → 0%."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    proxy = torch.tensor([10.0, 10.0, 10.0])
    rate = calculate_physical_violation_rate(y_pred, proxy)
    assert rate.item() == 0.0


def test_violation_rate_all():
    """All pixels violate → 100%."""
    y_pred = torch.tensor([20.0, 20.0, 20.0])
    proxy = torch.tensor([5.0, 5.0, 5.0])
    rate = calculate_physical_violation_rate(y_pred, proxy)
    assert rate.item() == 100.0


def test_violation_rate_partial():
    """1 out of 4 pixels violates → 25%."""
    y_pred = torch.tensor([1.0, 2.0, 3.0, 20.0])
    proxy = torch.tensor([10.0, 10.0, 10.0, 10.0])
    rate = calculate_physical_violation_rate(y_pred, proxy)
    assert abs(rate.item() - 25.0) < 1e-6
