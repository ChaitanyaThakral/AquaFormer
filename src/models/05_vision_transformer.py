import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=144, embed_dim=256, patch_size=5):
        super().__init__()
        # Convolutional layer cleanly tokenizes the 50x50 spatial grid
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, in_channels, 50, 50) -> (B, embed_dim, 10, 10)
        x = self.proj(x)
        # Flatten spatial dims: (B, embed_dim, 100)
        x = x.flatten(2)
        # Transpose: (B, 100, embed_dim)
        x = x.transpose(1, 2)
        return x

class RainfallHead(nn.Module):
    def __init__(self, embed_dim=256, patch_size=5, num_patches_h=10, num_patches_w=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.grid_h = num_patches_h * patch_size
        self.grid_w = num_patches_w * patch_size
        
        # MLP for explicit spatial mixing before projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Project directly to the full 50x50 spatial grid
        self.proj = nn.Linear(embed_dim, self.grid_h * self.grid_w)
        
    def forward(self, x):
        # x: (B, 100, embed_dim)
        
        # Global Average Pooling across the 100 tokens
        x = x.mean(dim=1) # (B, embed_dim)
        
        # Apply spatial mixing
        x = self.mlp(x) # (B, embed_dim)
        
        # Project directly to 2500 vector
        x = self.proj(x) # (B, 2500)
        
        return x

class SpatiotemporalViT(nn.Module):
    def __init__(self, in_features=6, seq_length=24, grid_h=50, grid_w=50, patch_size=5, embed_dim=256, depth=4, num_heads=8):
        """
        Baseline Grid Encoder acting as a Spatiotemporal Vision Transformer.
        """
        super().__init__()
        self.in_channels = in_features * seq_length
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.patch_size = patch_size
        
        num_patches_h = grid_h // patch_size
        num_patches_w = grid_w // patch_size
        num_patches = num_patches_h * num_patches_w
        
        self.patch_embed = PatchEmbed(in_channels=self.in_channels, embed_dim=embed_dim, patch_size=patch_size)
        
        # Learnable Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder Head
        self.head = RainfallHead(embed_dim, patch_size, num_patches_h, num_patches_w)
        
        # Enforce mathematical non-negativity
        self.activation = nn.Softplus()
        
    def forward(self, x):
        # x: (B, seq_length, 2500, features)
        B = x.shape[0]
        
        # Baseline Data Transformation: Reshape to (B, seq, grid_h, grid_w, features)
        x = x.view(B, x.size(1), self.grid_h, self.grid_w, x.size(-1))
        # Bring seq and features to channel dim: (B, seq, features, grid_h, grid_w)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        # Flatten time and features into the channel dim: (B, 144, 50, 50)
        x = x.view(B, -1, self.grid_h, self.grid_w)
        
        # Patch Tokenization
        tokens = self.patch_embed(x) # (B, 100, embed_dim)
        
        # Add Positional Encoding
        tokens = tokens + self.pos_embed
        
        # Multi-Head Self-Attention
        tokens = self.transformer(tokens) # (B, 100, embed_dim)
        
        # Linear Projection back to Grid
        out = self.head(tokens) # (B, 2500)
        
        # Non-negative constraint
        out = self.activation(out)
        return out
