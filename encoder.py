import torch     
import torch.nn as nn     
import torchvision  
from transformer import MultiheadAttentionMechanism
from dataloader import img_patches


class VIT_transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_classes, patch_size, num_channels,  num_patches, num_layers, dropout = 0.0):
        super().__init__()

        self.patch_size = patch_size


        self.input_layer = nn.Linear(num_channels *(patch_size ** 2), embed_dim)

        self.transformer = nn.Sequential(*[MultiheadAttentionMechanism(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)
        
        self.cls_token  = nn.Parameter(torch.randn(1,1,embed_dim))
        self.positional_encoding = nn.Parameter(torch.randn(1, 1+ num_patches, embed_dim))


    def forward(self, x):
        x = img_patches(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_encoding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
                                      
                                      
                                      
                                        