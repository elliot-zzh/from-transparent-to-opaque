import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CurrentStepMixerGater(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.w = nn.Linear(embed_dim * 2, embed_dim)
        # self.act = nn.Tanh()
        # self.act = nn.SiLU()
        self.act = nn.Sigmoid()
        # torch.nn.init.constant_(self.w.weight, 1 / embed_dim)
        nn.init.zeros_(self.w.weight)

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
        if zero_init:
            self.gate = nn.Parameter(
                torch.zeros(embed_dim)
            )  # all from model embeddings first for stability
        else:
            self.gate = nn.Parameter(torch.ones(embed_dim) * 0.5)
        self.time_mixing_gate = CurrentStepMixerGater(embed_dim)

    @torch.compile
    def forward(self, hidden, embed):
        hidden = self.norm(hidden)
        return embed * (
            1 - self.gate
        ) + self.inject_scale * self.gate * hidden * self.time_mixing_gate(
            hidden, embed
        )

    @torch.compile
    def forward_hidden(self, hidden, embed):  # forward hidden only
        return self.gate * hidden * self.time_mixing_gate(hidden, embed)

    def print_gates(self):
        print("gate value:", self.gate[:20])

    def print_heatmap(self):
        plt.imshow(
            self.gate.detach().cpu().numpy()[:20], cmap="hot", interpolation="nearest"
        )
        plt.colorbar()
        plt.show()
