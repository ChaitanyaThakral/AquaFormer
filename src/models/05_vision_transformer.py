import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=144, embed_dim=128, patch_size=5):
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
    def __init__(self, embed_dim=128, patch_size=5, num_patches_h=10, num_patches_w=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Per-token MLP for spatial mixing (operates on each of the 100 tokens)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Each token reconstructs its own 5x5 = 25 pixel patch
        self.proj = nn.Linear(embed_dim, patch_size * patch_size)
        
    def forward(self, x):
        # x: (B, 100, embed_dim) — one token per spatial patch
        B = x.shape[0]
        
        # Per-token MLP with residual connection and layer norm
        x = x + self.mlp(x)
        x = self.norm(x)
        
        # Project each token to its patch pixels: (B, 100, 25)
        x = self.proj(x)
        
        # Reassemble spatial grid from patches
        # (B, 100, 25) -> (B, 10, 10, 5, 5)
        x = x.view(B, self.num_patches_h, self.num_patches_w,
                    self.patch_size, self.patch_size)
        # (B, 10, 5, 10, 5) -> (B, 50, 50)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.num_patches_h * self.patch_size,
                    self.num_patches_w * self.patch_size)
        # Flatten to (B, 2500)
        x = x.view(B, -1)
        return x

class SpatiotemporalViT(nn.Module):
    def __init__(self, in_features=6, seq_length=24, grid_h=50, grid_w=50,
                 patch_size=5, embed_dim=128, depth=4, num_heads=8,
                 dim_feedforward=1280):
        """
        Physics-Informed Spatiotemporal Vision Transformer (~2.1M parameters).
        
        Treats the Pacific Northwest weather grid as a multi-channel image,
        tokenizes it into spatial patches, and uses multi-head self-attention
        to capture long-range spatial dependencies before projecting back
        to a dense precipitation grid.
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1,
        )
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
