import torch
import torch.nn as nn
import torch.nn.functional as F


class _NHiTSBlock(nn.Module):
    """Single N-HiTS block: MaxPool → MLP → basis expansion for backcast + forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        pool_size: int,
        n_theta: int,
        mlp_units: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pool_size = pool_size
        self.n_theta = n_theta

        pooled_len = max(1, seq_len // pool_size)
        self.pooled_len = pooled_len

        # MLP: input is the pooled look-back
        layers = []
        in_dim = pooled_len
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, mlp_units), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = mlp_units
        self.mlp = nn.Sequential(*layers)
        self.theta_b = nn.Linear(mlp_units, n_theta)  # backcast basis coefficients
        self.theta_f = nn.Linear(mlp_units, n_theta)  # forecast basis coefficients

        # Basis functions: simple linear interpolation via a learned matrix
        self.basis_b = nn.Linear(n_theta, seq_len, bias=False)
        self.basis_f = nn.Linear(n_theta, pred_len, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T) — channel-independent; each channel processed independently
        B, C, T = x.shape
        # Pool
        pooled = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size,
                              padding=0)  # (B, C, T_pool)
        # Flatten channel → MLP
        h = self.mlp(pooled.reshape(B * C, self.pooled_len))  # (B*C, mlp_units)
        theta_b = self.theta_b(h)  # (B*C, n_theta)
        theta_f = self.theta_f(h)  # (B*C, n_theta)
        backcast = self.basis_b(theta_b).reshape(B, C, T)   # (B, C, T)
        forecast = self.basis_f(theta_f).reshape(B, C, self.pred_len)  # (B, C, P)
        return backcast, forecast


class _NHiTSStack(nn.Module):
    """Stack of N-HiTS blocks at a single pooling scale."""

    def __init__(self, seq_len, pred_len, pool_size, n_blocks, n_theta, mlp_units,
                 n_layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            _NHiTSBlock(seq_len, pred_len, pool_size, n_theta, mlp_units, n_layers,
                        dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor):
        total_forecast = torch.zeros(
            x.shape[0], x.shape[1], self.blocks[0].pred_len, device=x.device
        )
        for block in self.blocks:
            backcast, forecast = block(x)
            x = x - backcast         # residual learning
            total_forecast = total_forecast + forecast
        return x, total_forecast


class NHiTS(nn.Module):
    """N-HiTS — Neural Hierarchical Interpolation for Time Series (Challu et al., AAAI 2023).

    Hierarchical multi-scale forecasting using stacked doubly-residual blocks.
    Each stack operates at a different temporal resolution via MaxPool. The
    final forecast is the sum of all stack forecasts.

    Paper: *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*
    https://ojs.aaai.org/index.php/AAAI/article/view/26253

    Args:
        seq_len (int): Look-back window length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input channels.
        n_stacks (int): Number of hierarchical stacks. Defaults to 3.
        n_blocks (int): Number of blocks per stack. Defaults to 1.
        n_theta (int): Basis function width. Defaults to 512.
        mlp_units (int): MLP hidden size. Defaults to 512.
        n_layers (int): Number of MLP layers per block. Defaults to 2.
        dropout (float): Dropout probability. Defaults to 0.1.
        output_prob (int): If > 0, output class logits. Defaults to 0.

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_stacks: int = 3,
        n_blocks: int = 1,
        n_theta: int = 512,
        mlp_units: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_prob: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.output_prob = output_prob

        # Pool sizes increase geometrically: 1, 2, 4, ...
        pool_sizes = [2 ** i for i in range(n_stacks)]

        self.stacks = nn.ModuleList([
            _NHiTSStack(seq_len, pred_len, ps, n_blocks, n_theta, mlp_units,
                        n_layers, dropout)
            for ps in pool_sizes
        ])

        if output_prob > 0:
            self.cls_head = nn.Linear(enc_in * pred_len, output_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) → process channel-independently as (B, C, L)
        x = x.transpose(1, 2)  # (B, C, L)
        forecast = torch.zeros(x.shape[0], x.shape[1], self.pred_len, device=x.device)
        residual = x
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            forecast = forecast + stack_forecast
        out = forecast.transpose(1, 2)  # (B, pred_len, C)
        if self.output_prob > 0:
            out = self.cls_head(out.reshape(out.shape[0], -1))
        return out
