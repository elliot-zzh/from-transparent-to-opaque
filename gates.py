import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from parameters import (
    double_gate,
    enbale_injection_scale,
)


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
        return self.act(self.w(x)) * 2


class Gate(nn.Module):
    def __init__(self, embed_dim, inject_scale, zero_init=True, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.inject_scale = inject_scale
        self.double_gate = double_gate
        self.enbale_injection_scale = enbale_injection_scale

        self.norm = nn.RMSNorm(embed_dim)
        self.injection_gate = CurrentStepMixerGater(embed_dim)
        if double_gate:
            self.origin_gate = CurrentStepMixerGater(embed_dim)

    def forward(self, hidden, embed):
        hidden = self.norm(hidden)
        injection_gate = 1 - self.injection_gate(hidden, embed)
        injection_scale = (embed**2).mean() ** 0.5 if self.enbale_injection_scale else 1
        if self.double_gate:
            origin_gate = self.origin_gate(hidden, embed)
            return embed * origin_gate + injection_scale * injection_gate * hidden
        else:
            return (
                embed * (1 - injection_gate) + injection_scale * injection_gate * hidden
            )

    def forward_hidden(self, hidden, embed):  # forward hidden only
        injection_gate = 1 - self.injection_gate(hidden, embed)
        injection_scale = (embed**2).mean() ** 0.5 if self.enbale_injection_scale else 1
        return injection_scale * hidden, injection_gate
