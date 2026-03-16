import torch
import pytest
import importlib
vit_module = importlib.import_module("src.models.05_vision_transformer")
SpatiotemporalViT = vit_module.SpatiotemporalViT
PatchEmbed = vit_module.PatchEmbed

@pytest.fixture
def dummy_input():
    # Shape: (Batch, Sequence, GridSize, Features)
    # B=2, Seq=24, Grid=2500, Features=6
    return torch.randn(2, 24, 2500, 6)

def test_vit_forward_shape(dummy_input):
    model = SpatiotemporalViT()
    out = model(dummy_input)
    assert out.shape == (2, 2500), f"Expected shape (2, 2500), got {out.shape}"

def test_vit_patch_tokenization(dummy_input):
    B = dummy_input.shape[0]
    # Simulate the initial reshape happening inside the model
    # (B, seq, H, W, features) -> (B, seq, features, H, W) -> (B, seq*features, H, W)
    x = dummy_input.view(B, 24, 50, 50, 6)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    x = x.view(B, 144, 50, 50)
    
    embedder = PatchEmbed(in_channels=144, embed_dim=256, patch_size=5)
    tokens = embedder(x)
    # 50x50 grid with 5x5 patches -> exactly 100 patches
    assert tokens.shape == (B, 100, 256), f"Expected 100 tokens, got {tokens.shape}"

def test_vit_non_negative_output(dummy_input):
    model = SpatiotemporalViT()
    out = model(dummy_input)
    # Enforced by Softplus
    assert torch.all(out >= 0.0), "Output contains negative values, softplus failed."

def test_vit_gradient_flow(dummy_input):
    model = SpatiotemporalViT()
    out = model(dummy_input)
    loss = out.sum()
    loss.backward()
    
    # Check patch embed grads
    assert model.patch_embed.proj.weight.grad is not None, "Patch embed has no gradients."
    
    # Check transformer grads
    # Need to access layers[0].linear1.weight to ensure the transformer block got gradients
    assert model.transformer.layers[0].linear1.weight.grad is not None, "Transformer has no gradients."
