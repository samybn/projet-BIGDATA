import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class MambaConfig:
    def __init__(self, d_model: int, n_layer: int, d_state: int = 16, expand: int = 2):
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.expand = expand

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_in = config.d_model
        d_expand = config.d_model * config.expand
        self.in_proj = nn.Linear(d_in, d_expand * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_expand,
            out_channels=d_expand,
            kernel_size=3,
            groups=d_expand,
            padding=1,
        )
        self.dt_rank = math.ceil(config.d_model / 16)
        self.x_proj = nn.Linear(d_expand, config.d_state * 2 + self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_expand, bias=True)
        # Initialisation spéciale pour dt_proj
        dt_init_std = self.dt_rank ** -0.5 * 0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # A matrix is now per-channel for better expressivity (standard Mamba)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(d_expand, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_expand))
        self.out_proj = nn.Linear(d_expand, d_in, bias=False)

    def selective_scan(self, u, delta, A, B_ssm, C_ssm, D):
        # u: (B, L, D), delta: (B, L, D)
        # A: (D, N), B_ssm: (B, L, N), C_ssm: (B, L, N)
        batch_size, seq_len, dim = u.shape
        n_state = A.shape[1]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_u = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, D, N)
        x = torch.zeros(batch_size, dim, n_state, device=u.device)
        ys = []
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x * C_ssm[:, i].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y + u * D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        xz = self.in_proj(x)  # (B, L, 2*E)
        x, z = xz.chunk(2, dim=-1)  # (B, L, E) chacun
        x = x.transpose(1, 2)  # (B, E, L)
        x = F.silu(self.conv1d(x)[..., :L])
        x = x.transpose(1, 2)  # (B, L, E)
        y = self.x_proj(x)  # (B, L, N*2+rank)
        B_ssm, C_ssm, dt = y.split(
            [self.config.d_state, self.config.d_state, self.dt_rank], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, L, E)
        A = -torch.exp(self.A_log.float())  # (1, N)
        y = self.selective_scan(
            x, delta,
            A,
            B_ssm,
            C_ssm,
            self.D
        )
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

class MambaModel(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return self.norm_f(x)
class MambaClassifier(nn.Module):
    def __init__(self, d_model, n_layer, d_state=16, expand=2, num_classes=2):
        super().__init__()
        config = MambaConfig(d_model=d_model, n_layer=n_layer, d_state=d_state, expand=expand)
        self.mamba = MambaModel(config)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape : (batch, seq_len, d_model)
        x = self.mamba(x)
        x = x.mean(dim=1)   # pooling sur la dimension de sequence
        return self.classifier(x)
