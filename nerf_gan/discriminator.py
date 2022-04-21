import torch
import torch.nn as nn
from vit_pytorch.vit import Transformer

from einops import repeat


class Discriminator(nn.Module):

    def __init__(self,
                 num_classes=1,
                 dim=1024,
                 depth=6,
                 heads=16,
                 patch_dim=3075,
                 mlp_dim=2048,
                 dropout=0.1,
                 emb_dropout=0.1,
                 dim_head=64):

        super().__init__()

        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes))

    def forward(self, patches, poses):
        B = patches.shape[0]
        trans = poses[:, :3, 3]
        x = torch.cat([patches.reshape(1, B, -1),
                       trans.reshape(1, B, -1)],
                      dim=2)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)
