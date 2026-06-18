import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.decomp import SeriesDecomp


class _PDM(nn.Module):
    """Past Decomposable Mixing block (one scale pair).

    Mixes the seasonal components bottom-up (fine → coarse) and the trend
    components top-down (coarse → fine).
    """

    def __init__(self, down_len: int, up_len: int, d_model: int, dropout: float):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=25)
        # Seasonal: project from fine scale → coarse scale.
        self.seasonal_mixing = nn.Sequential(
            nn.Linear(down_len, up_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Trend: project from coarse scale → fine scale.
        self.trend_mixing = nn.Sequential(
            nn.Linear(up_len, down_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_fine: torch.Tensor, x_coarse: torch.Tensor):
        # x_fine: (B, T_fine, C),  x_coarse: (B, T_coarse, C)
        seasonal_fine, trend_fine = self.decomp(x_fine)
        seasonal_coarse, trend_coarse = self.decomp(x_coarse)

        # Bottom-up seasonal mixing: project fine → coarse and add residual
        seasonal_coarse_out = seasonal_coarse + self.seasonal_mixing(
            seasonal_fine.transpose(1, 2)        # (B, C, T_fine)
        ).transpose(1, 2)                         # (B, T_coarse, C)

        # Top-down trend mixing: project coarse → fine and update fine trend
        trend_fine_out = trend_fine + self.trend_mixing(
            trend_coarse.transpose(1, 2)          # (B, C, T_coarse)
        ).transpose(1, 2)                         # (B, T_fine, C)

        # Updated coarse-scale representation (returned to update scales[i+1])
        coarse_out = seasonal_coarse_out + trend_coarse
        # Updated fine-scale representation (used to update scales[i])
        fine_out = seasonal_fine + trend_fine_out
        return coarse_out, fine_out


class _Predictor(nn.Module):
    """Single-scale predictor head."""

    def __init__(self, seq_len: int, pred_len: int, n_features: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(seq_len, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, pred_len, C)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class TimeMixer(nn.Module):
    """TimeMixer — Multi-scale Mixing for Time Series Forecasting (Wang et al., ICLR 2024).

    Decomposes the input into multiple temporal scales, mixes seasonal
    components bottom-up (fine → coarse) and trend components top-down
    (coarse → fine), then combines multi-scale predictions via a Future
    Multipredictor Mixing (FMM) layer.

    Paper: *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*
    https://openreview.net/forum?id=7oLshfEIC2

    Args:
        seq_len (int): Look-back window length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input channels.
        n_heads (int): Number of heads in the FMM mixing MLP. Defaults to 4.
        d_model (int): Hidden dimension for FMM. Defaults to 32.
        e_layers (int): Number of PDM mixing layers. Defaults to 3.
        dropout (float): Dropout probability. Defaults to 0.1.
        down_sampling_window (int): Downsampling factor between scales. Defaults to 2.
        down_sampling_layers (int): Number of downsampling scales. Defaults to 3.
        output_prob (int): If > 0, output class logits for classification.
            Defaults to 0.

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_heads: int = 4,
        d_model: int = 32,
        e_layers: int = 3,
        dropout: float = 0.1,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 3,
        output_prob: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.output_prob = output_prob

        # Compute scale lengths
        self.scale_lens = [seq_len]
        for _ in range(down_sampling_layers):
            self.scale_lens.append(max(1, self.scale_lens[-1] // down_sampling_window))

        # Normalisation per scale
        self.normalize_layers = nn.ModuleList([
            nn.LayerNorm(self.scale_lens[i]) for i in range(len(self.scale_lens))
        ])

        # PDM mixing blocks: one block per adjacent scale pair per layer
        self.mixing_layers = nn.ModuleList()
        for _ in range(e_layers):
            layer_blocks = nn.ModuleList()
            for i in range(len(self.scale_lens) - 1):
                layer_blocks.append(
                    _PDM(
                        down_len=self.scale_lens[i],
                        up_len=self.scale_lens[i + 1],
                        d_model=d_model,
                        dropout=dropout,
                    )
                )
            self.mixing_layers.append(layer_blocks)

        # Scale predictors
        self.predictors = nn.ModuleList([
            _Predictor(l, pred_len, enc_in, dropout) for l in self.scale_lens
        ])

        # FMM: mix predictions from all scales
        n_scales = len(self.scale_lens)
        self.fmm = nn.Sequential(
            nn.Linear(n_scales * pred_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        if output_prob > 0:
            self.cls_head = nn.Linear(enc_in * pred_len, output_prob)

    def _downsample(self, x: torch.Tensor) -> list:
        """Produce multi-scale representations via average pooling."""
        scales = [x]
        for _ in range(self.down_sampling_layers):
            scales.append(
                F.avg_pool1d(
                    scales[-1].transpose(1, 2),
                    kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                    padding=0,
                ).transpose(1, 2)
            )
        return scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        scales = self._downsample(x)
        # Apply per-scale normalization (over the time dimension)
        scales = [
            norm(s.transpose(1, 2)).transpose(1, 2)
            for norm, s in zip(self.normalize_layers, scales)
        ]

        # Apply e_layers of PDM mixing (sequential within a layer so each
        # block feeds the next one's input, matching the paper's cascaded design).
        for layer_blocks in self.mixing_layers:
            for i, block in enumerate(layer_blocks):
                coarse_out, fine_out = block(scales[i], scales[i + 1])
                scales[i] = fine_out
                scales[i + 1] = coarse_out

        # Predict from each scale → (B, pred_len, C)
        preds = [pred(s) for pred, s in zip(self.predictors, scales)]

        # FMM: stack pred_lens along feature dim per channel
        # stack along last dim: (B, pred_len, C, n_scales) → (B, pred_len*n_scales, C)
        stacked = torch.cat(
            [p.transpose(1, 2) for p in preds], dim=-1
        )  # (B, C, n_scales * pred_len)
        out = self.fmm(stacked).transpose(1, 2)  # (B, pred_len, C)

        if self.output_prob > 0:
            out = self.cls_head(out.reshape(out.shape[0], -1))
        return out
