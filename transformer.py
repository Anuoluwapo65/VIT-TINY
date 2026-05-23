import torch     
import torch.nn as nn    

class MultiheadAttentionMechanism(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout = 0.1):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.transformer = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_dropout, _ = self.transformer(x_norm, x_norm, x_norm)
        x = x + attn_dropout
        x_norm = self.layer_norm2(x)
        mlp_dropout = self.mlp(x_norm)
        x = x + mlp_dropout
        return x