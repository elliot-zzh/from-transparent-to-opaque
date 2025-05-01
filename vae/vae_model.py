import torch
import torch.nn as nn
from torch.optim import Adam

from vae.parameters import device


class VAE(nn.Module):
    def __init__(self, embed_dim, compress_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.compress_dim = compress_dim

        # compressing
        self.norm1 = nn.RMSNorm(embed_dim)
        self.wc1 = nn.Linear(embed_dim, ff_dim)
        self.wcv = nn.Linear(embed_dim, ff_dim)
        self.silu = nn.SiLU()
        self.wc2 = nn.Linear(ff_dim, compress_dim, bias=True)
        self.res_proj1 = nn.Linear(embed_dim, compress_dim, bias=True)

        # uncompressing
        self.norm2 = nn.RMSNorm(compress_dim)
        self.wuc = nn.Linear(compress_dim, ff_dim)
        self.wuv = nn.Linear(compress_dim, ff_dim)
        # self.silu = nn.SiLU()
        self.w_back = nn.Linear(ff_dim, embed_dim)
        self.res_proj2 = nn.Linear(compress_dim, embed_dim)

    def uncompress(self, x):
        x = self.norm2(x)
        return self.w_back(self.silu(self.wuc(x)) * self.wuv(x)) + self.res_proj2(x)

    def forward(self, x, compressing=False):
        x = self.norm1(x)
        x = self.wc2(self.silu(self.wc1(x)) * self.wcv(x)) + self.res_proj1(x)
        if compressing: return x
        return self.uncompress(x)

vae = VAE(2048, 256, 2048 * 4)

optimizer = Adam(vae.parameters(), lr=1e-3)

vae = vae.to(device)
vae.train()
scaler = torch.amp.GradScaler(device=device)
lossf = nn.MSELoss()