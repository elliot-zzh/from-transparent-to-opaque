import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CurrentStepMixerGater(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.w = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        # self.act = nn.Tanh()
        # self.act = nn.SiLU()
        self.act = nn.Sigmoid()
        # torch.nn.init.constant_(self.w.weight, 1 / embed_dim)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, hidden, embed):
        x = torch.cat([hidden, embed], dim=-1)
        return self.act(self.w(x))


class Gate(nn.Module):
    def __init__(self, embed_dim, inject_scale, zero_init=True, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.inject_scale = inject_scale

        self.norm = nn.RMSNorm(embed_dim)
        self.time_mixing_gate = CurrentStepMixerGater(embed_dim)

    @torch.compile
    def forward(self, hidden, embed):
        hidden = self.norm(hidden)
        gate = self.time_mixing_gate(hidden, embed) * 2
        return embed * gate + (embed**2).mean() ** 0.5 * (1 - gate) * hidden

    @torch.compile
    def forward_hidden(self, hidden, embed):  # forward hidden only
        gate = self.time_mixing_gate(hidden, embed) * 2
        return (embed**2).sum() ** 0.5 * (1 - gate) * hidden, gate
