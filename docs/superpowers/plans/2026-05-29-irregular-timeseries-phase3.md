# Irregular Time Series — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Prerequisite:** Phases 1 and 2 must be complete (all DataModules, GRU-D, IrregularClassificationExp, IrregularInterpolationExp, IrregularForecastExp, GRUD combo classes, registry wiring).

**Goal:** Add four more irregular time-series models — mTAN, LatentODE, NeuralCDE, Raindrop — as `nn.Module` classes with lazy-import optional dependencies, plus all three task combo classes for each model (classification, interpolation, forecast), wired into the experiment registry.

**Architecture:** Each model lives in `torch_timeseries/model/irregular/`. Models with optional dependencies use lazy imports in their module body with `ImportError` messages that include the `pip install torch-timeseries[irregular]` hint. Combo experiment classes follow the same `Model + TaskExp` dataclass pattern established in Phase 1/2. mTAN and GRU-D are the only no-external-dep models; LatentODE requires `torchdiffeq`, NeuralCDE requires `torchcde`, Raindrop requires `torch_geometric`. Per the spec, NeuralCDE and Raindrop support classification only (their architectures don't generalize naturally to interpolation/forecast).

**Tech Stack:** PyTorch ≥2.0, torchmetrics. Optional: `torchdiffeq>=0.2.3`, `torchcde>=0.2.5`, `torch_geometric>=2.0.0` (install via `pip install torch-timeseries[irregular]`).

---

## File Map

**Create:**
```
torch_timeseries/model/irregular/mtan.py          # mTAN (no external deps)
torch_timeseries/model/irregular/latent_ode.py    # LatentODE (requires torchdiffeq)
torch_timeseries/model/irregular/neural_cde.py    # NeuralCDE (requires torchcde)
torch_timeseries/model/irregular/raindrop.py      # Raindrop (requires torch_geometric)
torch_timeseries/experiments/mTAN.py              # mTAN combo classes
torch_timeseries/experiments/LatentODE.py         # LatentODE combo class
torch_timeseries/experiments/NeuralCDE.py         # NeuralCDE combo class
torch_timeseries/experiments/Raindrop.py          # Raindrop combo class
tests/model/test_mtan.py
tests/model/test_lazy_imports.py
tests/experiments/test_phase3_combos.py
```

**Modify:**
```
torch_timeseries/model/irregular/__init__.py      # add mTAN, LatentODE, NeuralCDE, Raindrop exports
torch_timeseries/experiments/__init__.py          # add Phase 3 imports
```

---

### Task 1: mTAN (Multi-Time Attention Network)

**Files:**
- Create: `torch_timeseries/model/irregular/mtan.py`
- Test: `tests/model/test_mtan.py`

mTAN learns a set of `num_ref_points` reference time points and uses learned time embeddings to compute soft-attention between query times and reference times. The encoder maps irregular observations to a fixed-size representation; the decoder queries at arbitrary times.

mTAN time embedding: `phi(t) = [sin(w_k * t + b_k), cos(w_k * t + b_k)]` for `k=1..d/2`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/model/test_mtan.py
import torch
import pytest


def _batch(B=4, T=10, F=3):
    x = torch.randn(B, T, F)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    mask = (torch.rand(B, T, F) > 0.4).float()
    return x, t, mask


def test_mtan_classification_forward():
    """mTAN returns (B, num_classes) for classification (no t_query)."""
    from torch_timeseries.model.irregular.mtan import mTAN
    B, T, F, C = 4, 10, 3, 2
    model = mTAN(input_size=F, hidden_size=32, output_size=C,
                  num_ref_points=8, num_heads=2)
    x, t, mask = _batch(B, T, F)
    out = model(x, t, mask)
    assert out.shape == (B, C)


def test_mtan_seq2seq_forward():
    """mTAN with t_query returns (B, Tq, F)."""
    from torch_timeseries.model.irregular.mtan import mTAN
    B, T, F, Tq = 4, 10, 3, 5
    model = mTAN(input_size=F, hidden_size=32, output_size=F,
                  num_ref_points=8, num_heads=2)
    x, t, mask = _batch(B, T, F)
    t_query = torch.linspace(0.5, 1.0, Tq).unsqueeze(0).expand(B, -1)
    out = model(x, t, mask, t_query=t_query)
    assert out.shape == (B, Tq, F)


def test_mtan_no_nan():
    from torch_timeseries.model.irregular.mtan import mTAN
    B, T, F = 4, 10, 3
    model = mTAN(input_size=F, hidden_size=32, output_size=2, num_ref_points=8)
    x, t, mask = _batch(B, T, F)
    mask = torch.zeros_like(mask)   # all missing
    out = model(x, t, mask)
    assert not torch.isnan(out).any()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_mtan.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `mtan.py`**

```python
# torch_timeseries/model/irregular/mtan.py
"""Multi-Time Attention Network (mTAN).

Reference: Shukla & Marlin, 2021 — "Multi-Time Attention Networks for
Irregularly Sampled Time Series".
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class TimeEmbedding(nn.Module):
    """Sine/cosine time embedding with learned frequencies."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.w = nn.Parameter(torch.randn(embed_dim // 2))
        self.b = nn.Parameter(torch.randn(embed_dim // 2))

    def forward(self, t: Tensor) -> Tensor:
        # t: (..., ) → (..., embed_dim)
        t = t.unsqueeze(-1)                                # (..., 1)
        arg = t * self.w + self.b                          # (..., D/2)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class mTANEncoder(nn.Module):
    """Encodes irregular observations to reference time points via attention."""

    def __init__(self, input_size: int, hidden_size: int,
                 num_ref_points: int, num_heads: int,
                 time_embed_dim: int = 32) -> None:
        super().__init__()
        self.num_ref_points = num_ref_points
        self.hidden_size = hidden_size
        self.time_embed = TimeEmbedding(time_embed_dim)

        # Reference time points (learned)
        self.ref_times = nn.Parameter(
            torch.linspace(0.0, 1.0, num_ref_points))

        # Input projection: (F + time_embed_dim) → hidden_size
        self.in_proj = nn.Linear(input_size + time_embed_dim, hidden_size)

        # Cross-attention: query=ref, key/value=input
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        # x: (B, T, F), t: (B, T), mask: (B, T, F)
        B, T, F = x.shape

        # Time embeddings for observation times
        t_emb = self.time_embed(t)                        # (B, T, D)
        # Mask-weight the input: multiply by mean observed mask across features
        obs_weight = mask.mean(dim=-1, keepdim=True)      # (B, T, 1)
        x_in = torch.cat([x * obs_weight, t_emb], dim=-1) # (B, T, F+D)
        v = self.in_proj(x_in)                             # (B, T, H)
        k = v                                              # key = value

        # Reference time embeddings as queries
        ref_t_emb = self.time_embed(
            self.ref_times.unsqueeze(0).expand(B, -1))    # (B, R, D)
        ref_in = torch.cat([
            torch.zeros(B, self.num_ref_points, F, device=x.device),
            ref_t_emb
        ], dim=-1)                                         # (B, R, F+D)
        q = self.in_proj(ref_in)                           # (B, R, H)

        # Mask padding positions (mask=0 in ALL features → padding)
        # attn key_padding_mask: True = ignore
        key_pad = (mask.sum(dim=-1) == 0)                  # (B, T)

        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_pad)
        return attn_out                                    # (B, R, H)


class mTAN(nn.Module):
    """Multi-Time Attention Network for irregular time series.

    - ``forward(x, t, mask)`` → (B, output_size) for classification.
    - ``forward(x, t, mask, t_query=...)`` → (B, Tq, input_size) for interp/forecast.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_ref_points: int = 16,
        num_heads: int = 2,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = mTANEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_ref_points=num_ref_points,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
        )
        self.drop = nn.Dropout(dropout)
        # Classification head: flatten ref outputs → output_size
        self.fc_cls = nn.Linear(num_ref_points * hidden_size, output_size)
        # Seq2Seq head: project per-query to input_size
        self.time_embed_dec = TimeEmbedding(time_embed_dim)
        self.fc_dec = nn.Linear(hidden_size + time_embed_dim, input_size)
        self.dec_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        x: Tensor,              # (B, T, F)
        t: Tensor,              # (B, T)
        mask: Tensor,           # (B, T, F)
        x_time: Tensor = None,  # ignored
        t_query: Tensor = None, # (B, Tq) — enables seq2seq mode
    ) -> Tensor:
        ref = self.encoder(x, t, mask)           # (B, R, H)
        B, R, H = ref.shape

        if t_query is None:
            # Classification: flatten and project
            flat = ref.reshape(B, R * H)
            return self.fc_cls(self.drop(flat))  # (B, output_size)

        # Seq2Seq: attend from query times to reference representations
        Tq = t_query.shape[1]
        t_q_emb = self.time_embed_dec(t_query)    # (B, Tq, D)
        # Pad to H dim for attention query
        q = nn.functional.pad(
            t_q_emb, (0, H - t_q_emb.shape[-1])
        ) if t_q_emb.shape[-1] < H else t_q_emb[:, :, :H]

        attn_out, _ = self.dec_attn(q, ref, ref)  # (B, Tq, H)
        combined = torch.cat([
            attn_out,
            t_q_emb[:, :, :H] if t_q_emb.shape[-1] >= H
            else nn.functional.pad(t_q_emb, (0, H - t_q_emb.shape[-1]))
        ], dim=-1)[:, :, :H + t_q_emb.shape[-1]]
        # Use concat of attn_out and t_q_emb
        combined2 = torch.cat([attn_out, t_q_emb], dim=-1)  # (B, Tq, H+D)
        return self.fc_dec(self.drop(combined2))             # (B, Tq, F)
```

Note: the decoder attention requires `q` and `ref` to have matching embedding dim. The implementation above concatenates the decoded hidden with the query time embedding and projects to F. Fix the cleaner version:

```python
# Cleaner seq2seq decoder in mTAN.forward:
        # Tq path — rewrite fc_dec to take H+D → F
        # Make sure fc_dec was defined with H+time_embed_dim:
        # self.fc_dec = nn.Linear(hidden_size + time_embed_dim, input_size)
        q_proj = nn.functional.pad(t_q_emb, (0, max(0, H - t_q_emb.shape[-1])))[:, :, :H]
        attn_out, _ = self.dec_attn(q_proj, ref, ref)        # (B, Tq, H)
        combined = torch.cat([attn_out, t_q_emb], dim=-1)    # (B, Tq, H+D)
        return self.fc_dec(self.drop(combined))               # (B, Tq, F)
```

The final implementation to use (replacing the `if t_query is None` block onward):

```python
        if t_query is None:
            flat = ref.reshape(B, R * H)
            return self.fc_cls(self.drop(flat))

        Tq = t_query.shape[1]
        t_q_emb = self.time_embed_dec(t_query)                 # (B, Tq, D)
        # Pad/slice to H for cross-attention query
        q_proj = torch.zeros(B, Tq, H, device=x.device)
        min_d = min(H, t_q_emb.shape[-1])
        q_proj[:, :, :min_d] = t_q_emb[:, :, :min_d]
        attn_out, _ = self.dec_attn(q_proj, ref, ref)          # (B, Tq, H)
        combined = torch.cat([attn_out, t_q_emb], dim=-1)      # (B, Tq, H+D)
        return self.fc_dec(self.drop(combined))                 # (B, Tq, F)
```

- [ ] **Step 4: Run mTAN tests**

```bash
pytest tests/model/test_mtan.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add torch_timeseries/model/irregular/mtan.py tests/model/test_mtan.py
git commit -m "feat: add mTAN model for irregular time series"
```

---

### Task 2: LatentODE (lazy import from `torchdiffeq`)

**Files:**
- Create: `torch_timeseries/model/irregular/latent_ode.py`
- Test: `tests/model/test_lazy_imports.py`

LatentODE: RNN encoder over reversed time sequence → latent `z0` → ODE solver forward in time → decoder at query times. The `torchdiffeq` package provides the ODE solver.

- [ ] **Step 1: Write the failing lazy-import test**

```python
# tests/model/test_lazy_imports.py
import pytest


def test_latent_ode_import_error_without_torchdiffeq(monkeypatch):
    """Importing LatentODE without torchdiffeq raises ImportError with install hint."""
    import sys
    # Remove torchdiffeq from sys.modules if present, then mock builtins.__import__
    if "torchdiffeq" in sys.modules:
        pytest.skip("torchdiffeq is installed; this test needs it absent")

    from torch_timeseries.model.irregular import latent_ode as _mod
    import importlib
    # Force reimport to trigger lazy import check
    # Since torchdiffeq may not be installed, just instantiating should raise
    with pytest.raises(ImportError, match="torchdiffeq"):
        _mod.LatentODE(input_size=3, latent_size=8, hidden_size=16, output_size=2)


def test_neural_cde_import_error_without_torchcde(monkeypatch):
    """Importing NeuralCDE without torchcde raises ImportError with install hint."""
    import sys
    if "torchcde" in sys.modules:
        pytest.skip("torchcde is installed")

    from torch_timeseries.model.irregular import neural_cde as _mod
    with pytest.raises(ImportError, match="torchcde"):
        _mod.NeuralCDE(input_size=3, hidden_size=16, output_size=2)


def test_raindrop_import_error_without_torch_geometric(monkeypatch):
    """Importing Raindrop without torch_geometric raises ImportError with install hint."""
    import sys
    if "torch_geometric" in sys.modules:
        pytest.skip("torch_geometric is installed")

    from torch_timeseries.model.irregular import raindrop as _mod
    with pytest.raises(ImportError, match="torch_geometric"):
        _mod.Raindrop(input_size=3, hidden_size=16, output_size=2,
                      num_nodes=3, num_heads=1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_lazy_imports.py -v
```
Expected: `ImportError` or tests skip depending on installed packages.

- [ ] **Step 3: Write `latent_ode.py`**

```python
# torch_timeseries/model/irregular/latent_ode.py
"""Latent ODE for irregular time series.

Requires: pip install torch-timeseries[irregular]  (installs torchdiffeq)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class _ODEFunc(nn.Module):
    """Simple MLP ODE function dz/dt = f(z)."""
    def __init__(self, latent_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, t, z):
        return self.net(z)


class LatentODE(nn.Module):
    """Variational Latent ODE for irregular time series.

    Requires ``torchdiffeq``:
        pip install torch-timeseries[irregular]

    Architecture:
        1. RNN encoder over reversed observations → (mu, logvar) for z0.
        2. Sample z0 ~ N(mu, exp(0.5*logvar)).
        3. ODE solver from t=0 to t=1 via z0, queried at t_query.
        4. Linear decoder: z(t) → output.
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        output_size: int,
        ode_method: str = "dopri5",
    ) -> None:
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                "LatentODE requires torchdiffeq. "
                "Install it with: pip install torch-timeseries[irregular]"
            )
        super().__init__()
        self.latent_size = latent_size
        self.ode_method = ode_method

        # Encoder: GRU over reversed sequence, outputs z0 distribution
        self.encoder_rnn = nn.GRU(
            input_size * 2,   # [x; mask] at each step
            hidden_size,
            batch_first=True,
        )
        self.z0_proj = nn.Linear(hidden_size, latent_size * 2)  # mu, logvar

        # ODE dynamics
        self.ode_func = _ODEFunc(latent_size, hidden_size)

        # Decoder: latent → output
        self.decoder = nn.Linear(latent_size, output_size)

        # Classification head (if no t_query)
        self.fc_cls = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def _encode(self, x: Tensor, t: Tensor, mask: Tensor):
        # Reverse observations in time
        x_in = torch.cat([x, mask], dim=-1)    # (B, T, F*2)
        # Reverse along time axis
        x_rev = torch.flip(x_in, dims=[1])
        _, h = self.encoder_rnn(x_rev)          # h: (1, B, H)
        h = h.squeeze(0)                        # (B, H)
        z0_params = self.z0_proj(h)             # (B, latent*2)
        mu, logvar = z0_params.chunk(2, dim=-1)
        return mu, logvar

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        x: Tensor,              # (B, T, F)
        t: Tensor,              # (B, T)
        mask: Tensor,           # (B, T, F)
        x_time: Tensor = None,  # ignored
        t_query: Tensor = None, # (B, Tq)
    ) -> Tensor:
        from torchdiffeq import odeint

        mu, logvar = self._encode(x, t, mask)
        z0 = self._reparameterize(mu, logvar)      # (B, latent)

        if t_query is None:
            # Classification: use z0 directly
            return self.fc_cls(z0)                  # (B, output_size)

        # Solve ODE from 0 to 1, evaluated at sorted unique query times
        B, Tq = t_query.shape
        # Use the batch's mean query times as the integration time grid
        # (simplified: use linspace from 0 to 1)
        t_grid = torch.linspace(0.0, 1.0, Tq + 1, device=x.device)
        z_traj = odeint(self.ode_func, z0, t_grid, method=self.ode_method)
        # z_traj: (Tq+1, B, latent) — evaluate at query times (indices 1..)
        z_at_query = z_traj[1:].permute(1, 0, 2)  # (B, Tq, latent)
        return self.decoder(z_at_query)             # (B, Tq, output_size)
```

- [ ] **Step 4: Write `neural_cde.py`**

```python
# torch_timeseries/model/irregular/neural_cde.py
"""Neural Controlled Differential Equation for irregular time series.

Requires: pip install torch-timeseries[irregular]  (installs torchcde)
Only supports classification (terminal hidden state) per Phase 3 scope.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class _CDEFunc(nn.Module):
    """CDE vector field: dz/dt = f(z) * dX/dt."""
    def __init__(self, input_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * input_channels),
        )
        self.hidden_size = hidden_size
        self.input_channels = input_channels

    def forward(self, t, z):
        out = self.net(z)                                      # (B, H*C)
        return out.view(z.shape[0], self.hidden_size, self.input_channels)


class NeuralCDE(nn.Module):
    """Neural CDE for irregular time-series classification.

    Requires ``torchcde``:
        pip install torch-timeseries[irregular]

    Fits a natural cubic spline to the irregular observations and drives
    a CDE with that spline. Returns ``(B, output_size)`` logits.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        interpolation: str = "cubic",
    ) -> None:
        try:
            import torchcde
        except ImportError:
            raise ImportError(
                "NeuralCDE requires torchcde. "
                "Install it with: pip install torch-timeseries[irregular]"
            )
        super().__init__()
        self.interpolation = interpolation
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input_channels = F + 1 (features + time channel)
        self.cde_func = _CDEFunc(input_size + 1, hidden_size)
        self.initial_proj = nn.Linear(input_size + 1, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor,              # (B, T, F)
        t: Tensor,              # (B, T)
        mask: Tensor,           # (B, T, F)
        x_time: Tensor = None,
        t_query: Tensor = None,
    ) -> Tensor:                # (B, output_size)
        import torchcde

        B, T, F = x.shape
        # Add time as first channel: (B, T, F+1)
        t_expand = t.unsqueeze(-1)                   # (B, T, 1)
        X = torch.cat([t_expand, x], dim=-1)         # (B, T, F+1)

        # Fit natural cubic spline
        coeffs = torchcde.natural_cubic_coeffs(X)
        X_spline = torchcde.NaturalCubicSpline(coeffs)

        z0 = self.initial_proj(X[:, 0, :])           # (B, H)

        z_T = torchcde.cdeint(
            X=X_spline,
            func=self.cde_func,
            z0=z0,
            t=t[0],                                   # use first batch's time points
            method="rk4",
        )
        z_T = z_T[:, -1]                              # terminal state (B, H)
        return self.fc(z_T)                           # (B, output_size)
```

- [ ] **Step 5: Write `raindrop.py`**

```python
# torch_timeseries/model/irregular/raindrop.py
"""Raindrop: Graph-guided network for irregular multivariate time series.

Requires: pip install torch-timeseries[irregular]  (installs torch_geometric)
Only supports classification per Phase 3 scope.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Raindrop(nn.Module):
    """Raindrop model (Zhang et al., 2022) for irregular time-series classification.

    Requires ``torch_geometric``:
        pip install torch-timeseries[irregular]

    Models each feature as a graph node; attention between sensor nodes
    depends on temporal proximity and feature correlations.
    Returns ``(B, output_size)`` logits.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_nodes: int = None,   # defaults to input_size
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        try:
            from torch_geometric.nn import GATConv
        except ImportError:
            raise ImportError(
                "Raindrop requires torch_geometric. "
                "Install it with: pip install torch-timeseries[irregular]"
            )
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        num_nodes = num_nodes or input_size

        # Temporal encoder: per-feature GRU
        self.feature_gru = nn.GRU(
            1, hidden_size, batch_first=True)    # each feature independently

        # Graph attention across feature nodes
        self.gat = GATConv(
            in_channels=hidden_size,
            out_channels=hidden_size // num_heads,
            heads=num_heads,
            dropout=dropout,
        )

        # Global pooling → classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        x: Tensor,              # (B, T, F)
        t: Tensor,              # (B, T)
        mask: Tensor,           # (B, T, F)
        x_time: Tensor = None,
        t_query: Tensor = None,
    ) -> Tensor:                # (B, output_size)
        from torch_geometric.nn import GATConv
        import torch_geometric.utils as pyg_utils

        B, T, F = x.shape

        # Per-feature temporal encoding: reshape to (B*F, T, 1), run GRU
        x_pf = x.permute(0, 2, 1).unsqueeze(-1)     # (B, F, T, 1)
        x_pf = x_pf.reshape(B * F, T, 1)
        mask_pf = mask.permute(0, 2, 1).reshape(B * F, T, 1)
        x_pf = x_pf * mask_pf                        # zero-out missing
        _, h = self.feature_gru(x_pf)                # h: (1, B*F, H)
        h = h.squeeze(0).reshape(B, F, self.hidden_size)   # (B, F, H)

        # Graph attention across feature nodes (fully connected per sample)
        # Build fully-connected edge index for F nodes
        src = torch.arange(F, device=x.device).repeat_interleave(F)
        dst = torch.arange(F, device=x.device).repeat(F)
        edge_index = torch.stack([src, dst], dim=0)    # (2, F*F)

        node_feats = []
        for b in range(B):
            nf = self.gat(h[b], edge_index)            # (F, H)
            node_feats.append(nf.mean(dim=0))          # mean pool over nodes → (H,)
        graph_repr = torch.stack(node_feats, dim=0)    # (B, H)

        return self.fc(graph_repr)                     # (B, output_size)
```

- [ ] **Step 6: Run lazy-import tests**

```bash
pytest tests/model/test_lazy_imports.py -v
```
Expected: tests pass (or skip if the library is already installed).

- [ ] **Step 7: Commit all four models**

```bash
git add torch_timeseries/model/irregular/ tests/model/test_mtan.py tests/model/test_lazy_imports.py
git commit -m "feat: add mTAN, LatentODE, NeuralCDE, Raindrop models (Phase 3)"
```

---

### Task 3: Model `__init__.py` update

**Files:**
- Modify: `torch_timeseries/model/irregular/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
# torch_timeseries/model/irregular/__init__.py
from .grud import GRUD
# Lazy-import models — only available with torch-timeseries[irregular]
from . import mtan, latent_ode, neural_cde, raindrop

# Expose at module level for convenience (import errors only on instantiation)
mTAN = mtan.mTAN
LatentODE = latent_ode.LatentODE
NeuralCDE = neural_cde.NeuralCDE
Raindrop = raindrop.Raindrop

__all__ = ["GRUD", "mTAN", "LatentODE", "NeuralCDE", "Raindrop"]
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from torch_timeseries.model.irregular import GRUD, mTAN; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add torch_timeseries/model/irregular/__init__.py
git commit -m "feat: update model/irregular/__init__.py with all Phase 3 model exports"
```

---

### Task 4: mTAN combo experiment classes

**Files:**
- Create: `torch_timeseries/experiments/mTAN.py`
- Test: `tests/experiments/test_phase3_combos.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_phase3_combos.py
import pytest
import numpy as np


class _ToyIrregular:
    num_features = 3
    num_classes = 2
    labels = None

    def __init__(self, n=30, has_labels=True):
        rng = np.random.default_rng(0)
        self.samples, self.times, self.masks = [], [], []
        lbs = []
        for i in range(n):
            T = rng.integers(6, 15)
            self.samples.append(rng.normal(size=(T, 3)).astype("float32"))
            self.times.append(np.sort(rng.uniform(0, 48, T)).astype("float32"))
            self.masks.append((rng.random((T, 3)) > 0.2).astype("float32"))
            lbs.append(i % 2)
        if has_labels:
            self.labels = np.array(lbs, dtype=np.int64)
        self.num_classes = 2

    def __len__(self):
        return len(self.samples)


def test_mtan_classification_exp_runs(tmp_path):
    from torch_timeseries.experiments.mTAN import mTANIrregularClassification
    exp = mTANIrregularClassification(
        dataset_type="__toy__", epochs=2, patience=5, batch_size=8,
        hidden_size=16, num_ref_points=4, num_heads=1,
        device="cpu", save_dir=str(tmp_path),
    )
    exp._toy_dataset = _ToyIrregular(n=30)
    result = exp.run(seed=1)
    assert "accuracy" in result


def test_mtan_registered(tmp_path):
    from torch_timeseries.experiments import get_experiment_class
    cls = get_experiment_class("mTAN", "IrregularClassification")
    assert cls is not None
    cls2 = get_experiment_class("mTAN", "IrregularInterpolation")
    assert cls2 is not None
    cls3 = get_experiment_class("mTAN", "IrregularForecast")
    assert cls3 is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/experiments/test_phase3_combos.py::test_mtan_classification_exp_runs -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `mTAN.py`**

```python
# torch_timeseries/experiments/mTAN.py
from dataclasses import dataclass
from ..model.irregular.mtan import mTAN
from .irregular_classification import IrregularClassificationExp
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp


@dataclass
class mTANParameters:
    hidden_size: int = 64
    num_ref_points: int = 16
    num_heads: int = 2
    mtan_dropout: float = 0.1


@dataclass
class mTANIrregularClassification(IrregularClassificationExp, mTANParameters):
    model_type: str = "mTAN"

    def _init_model(self) -> None:
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_classes,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
            dropout=self.mtan_dropout,
        ).to(self.device)


@dataclass
class mTANIrregularInterpolation(IrregularInterpolationExp, mTANParameters):
    model_type: str = "mTAN"

    def _init_model(self) -> None:
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
            dropout=self.mtan_dropout,
        ).to(self.device)


@dataclass
class mTANIrregularForecast(IrregularForecastExp, mTANParameters):
    model_type: str = "mTAN"

    def _init_model(self) -> None:
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
            dropout=self.mtan_dropout,
        ).to(self.device)
```

- [ ] **Step 4: Add mTAN imports to `experiments/__init__.py`**

```python
from .mTAN import mTANIrregularClassification, mTANIrregularInterpolation, mTANIrregularForecast
```

- [ ] **Step 5: Run mTAN tests**

```bash
pytest tests/experiments/test_phase3_combos.py::test_mtan_classification_exp_runs -v
pytest tests/experiments/test_phase3_combos.py::test_mtan_registered -v
```
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add torch_timeseries/experiments/mTAN.py torch_timeseries/experiments/__init__.py \
        tests/experiments/test_phase3_combos.py
git commit -m "feat: add mTAN experiment combos for all 3 irregular tasks"
```

---

### Task 5: LatentODE, NeuralCDE, Raindrop combo classes + full registry wiring

**Files:**
- Create: `torch_timeseries/experiments/LatentODE.py`
- Create: `torch_timeseries/experiments/NeuralCDE.py`
- Create: `torch_timeseries/experiments/Raindrop.py`
- Modify: `torch_timeseries/experiments/__init__.py`

Per the spec, NeuralCDE and Raindrop support classification only. LatentODE supports all 3 tasks.

- [ ] **Step 1: Write `LatentODE.py`**

```python
# torch_timeseries/experiments/LatentODE.py
from dataclasses import dataclass
from .irregular_classification import IrregularClassificationExp
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp


@dataclass
class LatentODEParameters:
    latent_size: int = 16
    hidden_size_ode: int = 32
    ode_method: str = "dopri5"


@dataclass
class LatentODEIrregularClassification(IrregularClassificationExp, LatentODEParameters):
    model_type: str = "LatentODE"

    def _init_model(self) -> None:
        from ..model.irregular.latent_ode import LatentODE
        self.model = LatentODE(
            input_size=self.dm.num_features,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size_ode,
            output_size=self.dm.num_classes,
            ode_method=self.ode_method,
        ).to(self.device)


@dataclass
class LatentODEIrregularInterpolation(IrregularInterpolationExp, LatentODEParameters):
    model_type: str = "LatentODE"

    def _init_model(self) -> None:
        from ..model.irregular.latent_ode import LatentODE
        self.model = LatentODE(
            input_size=self.dm.num_features,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size_ode,
            output_size=self.dm.num_features,
            ode_method=self.ode_method,
        ).to(self.device)


@dataclass
class LatentODEIrregularForecast(IrregularForecastExp, LatentODEParameters):
    model_type: str = "LatentODE"

    def _init_model(self) -> None:
        from ..model.irregular.latent_ode import LatentODE
        self.model = LatentODE(
            input_size=self.dm.num_features,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size_ode,
            output_size=self.dm.num_features,
            ode_method=self.ode_method,
        ).to(self.device)
```

- [ ] **Step 2: Write `NeuralCDE.py`**

```python
# torch_timeseries/experiments/NeuralCDE.py
from dataclasses import dataclass
from .irregular_classification import IrregularClassificationExp


@dataclass
class NeuralCDEParameters:
    ncde_hidden_size: int = 32
    interpolation: str = "cubic"


@dataclass
class NeuralCDEIrregularClassification(IrregularClassificationExp, NeuralCDEParameters):
    model_type: str = "NeuralCDE"

    def _init_model(self) -> None:
        from ..model.irregular.neural_cde import NeuralCDE
        self.model = NeuralCDE(
            input_size=self.dm.num_features,
            hidden_size=self.ncde_hidden_size,
            output_size=self.dm.num_classes,
            interpolation=self.interpolation,
        ).to(self.device)
```

- [ ] **Step 3: Write `Raindrop.py`**

```python
# torch_timeseries/experiments/Raindrop.py
from dataclasses import dataclass
from .irregular_classification import IrregularClassificationExp


@dataclass
class RaindropParameters:
    raindrop_hidden_size: int = 32
    raindrop_num_heads: int = 2
    raindrop_dropout: float = 0.1


@dataclass
class RaindropIrregularClassification(IrregularClassificationExp, RaindropParameters):
    model_type: str = "Raindrop"

    def _init_model(self) -> None:
        from ..model.irregular.raindrop import Raindrop
        self.model = Raindrop(
            input_size=self.dm.num_features,
            hidden_size=self.raindrop_hidden_size,
            output_size=self.dm.num_classes,
            num_nodes=self.dm.num_features,
            num_heads=self.raindrop_num_heads,
            dropout=self.raindrop_dropout,
        ).to(self.device)
```

- [ ] **Step 4: Add all imports to `experiments/__init__.py`**

```python
from .LatentODE import (
    LatentODEIrregularClassification,
    LatentODEIrregularInterpolation,
    LatentODEIrregularForecast,
)
from .NeuralCDE import NeuralCDEIrregularClassification
from .Raindrop import RaindropIrregularClassification
```

- [ ] **Step 5: Add registry tests for all combos**

```python
# append to tests/experiments/test_phase3_combos.py

def test_all_phase3_registered():
    from torch_timeseries.experiments import get_experiment_class
    expected = [
        ("LatentODE", "IrregularClassification"),
        ("LatentODE", "IrregularInterpolation"),
        ("LatentODE", "IrregularForecast"),
        ("NeuralCDE", "IrregularClassification"),
        ("Raindrop", "IrregularClassification"),
    ]
    for model, task in expected:
        cls = get_experiment_class(model, task)
        assert cls is not None, f"{model}/{task} not in registry"
```

- [ ] **Step 6: Run full Phase 3 tests**

```bash
pytest tests/experiments/test_phase3_combos.py -v
pytest tests/model/test_mtan.py tests/model/test_lazy_imports.py -v
```
Expected: all pass.

- [ ] **Step 7: Run full regression check**

```bash
pytest tests/ -v --ignore=tests/experiments/test_autoformer.py -x
```
Expected: all pass.

- [ ] **Step 8: Final commit**

```bash
git add torch_timeseries/experiments/LatentODE.py \
        torch_timeseries/experiments/NeuralCDE.py \
        torch_timeseries/experiments/Raindrop.py \
        torch_timeseries/experiments/__init__.py \
        tests/experiments/test_phase3_combos.py
git commit -m "feat: add LatentODE, NeuralCDE, Raindrop experiment combos (Phase 3 complete)"
```

---

## Self-Review

**Spec coverage check (Phase 3):**
- mTAN model (no external deps): Task 1 ✅
- LatentODE model (lazy import torchdiffeq): Task 2 ✅
- NeuralCDE model (lazy import torchcde): Task 2 ✅
- Raindrop model (lazy import torch_geometric): Task 2 ✅
- test_latent_ode_lazy_import_error: Task 2 ✅
- mTAN combo: classification + interpolation + forecast: Task 4 ✅
- LatentODE combo: classification + interpolation + forecast: Task 5 ✅
- NeuralCDE combo: classification only (spec non-goal for interp/forecast): Task 5 ✅
- Raindrop combo: classification only (spec non-goal for interp/forecast): Task 5 ✅
- Registry wiring for all 5 models × eligible tasks: Tasks 4–5 ✅
- `[irregular]` optional deps already added in Phase 1 pyproject.toml: ✅

**Non-goals confirmed:**
- NeuralCDE interpolation/forecast: not implemented — spec Section 7 ✅
- Raindrop interpolation/forecast: not implemented — spec Section 7 ✅
- MIMIC auto-download: not implemented — spec Section 7 ✅
- Leaderboard UI / remote upload: not implemented — spec Section 7 ✅
