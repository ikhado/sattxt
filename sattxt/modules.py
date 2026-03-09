import torch
from torch import nn


class LinearProj(nn.Module):
    """Stable projector for frozen towers: LN -> Linear (no bias)."""

    def __init__(self, in_dim: int, out_dim: int, with_ln: bool = True, orthogonal_init: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim) if with_ln else nn.Identity()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.init_weights(orthogonal_init=orthogonal_init)

    def init_weights(self, orthogonal_init: bool = True):
        if isinstance(self.proj, nn.Linear):
            if orthogonal_init:
                nn.init.orthogonal_(self.proj.weight)
            else:
                nn.init.normal_(
                    self.proj.weight,
                    std=self.proj.in_features ** -0.5,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))
