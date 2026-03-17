import torch
import pytest
import importlib
loss_module = importlib.import_module("src.models.06_physics_loss")
PhysicsInformedLoss = loss_module.PhysicsInformedLoss

@pytest.fixture
def dummy_data():
    y_true = torch.ones(2, 2500)
    # Set moisture (index 4) to 10.0 everywhere
    x_input = torch.zeros(2, 24, 2500, 6)
    x_input[:, :, :, 4] = 10.0 
    return y_true, x_input

def test_physics_penalty_activates(dummy_data):
    y_true, x_input = dummy_data
    # Predict 15.0 > moisture (10.0) -> Penalty activates
    y_pred = torch.ones(2, 2500) * 15.0
    
    criterion = PhysicsInformedLoss(alpha=1.0, beta=1.0)
    losses = criterion(y_pred, y_true, x_input)
    
    assert losses['physics_penalty'] > 0.0, "Penalty should be positive when prediction exceeds proxy."
    assert abs(losses['physics_penalty'].item() - 5.0) < 1e-6 # (15 - 10) = 5

def test_physics_penalty_vanishes(dummy_data):
    y_true, x_input = dummy_data
    # Predict 5.0 < moisture (10.0) -> Penalty vanishes
    y_pred = torch.ones(2, 2500) * 5.0
    
    criterion = PhysicsInformedLoss(alpha=1.0, beta=1.0)
    losses = criterion(y_pred, y_true, x_input)
    
    assert losses['physics_penalty'].item() == 0.0, "Penalty should be strictly 0 when prediction is within physical bounds."

def test_loss_gradient_flow(dummy_data):
    y_true, x_input = dummy_data
    y_pred = torch.full((2, 2500), 15.0, requires_grad=True)
    
    criterion = PhysicsInformedLoss()
    losses = criterion(y_pred, y_true, x_input)
    losses['loss'].backward()
    
    assert y_pred.grad is not None, "Gradients did not flow back to y_pred."
    assert torch.sum(torch.abs(y_pred.grad)) > 0, "Gradients are zero."

def test_moisture_aggregation_modes():
    # Construct input where moisture changes over time
    x_input = torch.zeros(2, 24, 2500, 6)
    x_input[:, -1, :, 4] = 5.0  # last hour is 5.0
    x_input[:, 0, :, 4] = 15.0  # first hour is 15.0
    # Mean will be (15 + 5) / 24 = 20 / 24 = 0.833
    
    y_pred = torch.ones(2, 2500) * 10.0
    y_true = torch.ones(2, 2500)
    
    criterion_last = PhysicsInformedLoss(aggregation_mode='last')
    loss_last = criterion_last(y_pred, y_true, x_input)['physics_penalty'].item()
    # 10.0 - 5.0 = 5.0
    assert abs(loss_last - 5.0) < 1e-6
    
    criterion_mean = PhysicsInformedLoss(aggregation_mode='mean')
    loss_mean = criterion_mean(y_pred, y_true, x_input)['physics_penalty'].item()
    # 10.0 - 0.833 = 9.166
    assert loss_mean > loss_last, "Mean aggregation should yield a different penalty than Last aggregation in this scenario."
