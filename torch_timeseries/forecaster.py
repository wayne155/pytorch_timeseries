"""High-level scikit-learn-style API for time-series forecasting.

Also provides :func:`compare` — a one-liner for benchmarking multiple models
on the same dataset::

    from torch_timeseries import compare

    results = compare(
        ["DLinear", "PatchTST", "iTransformer"],
        X_train=X[:800],
        X_test=X[800:],
        seq_len=96,
        pred_len=24,
        epochs=5,
    )
    # returns a dict: {"DLinear": {"mse": ..., "mae": ..., "rmse": ...}, ...}


Users bring their own data as a NumPy array or pandas DataFrame — no dataset
class, no YAML config, no understanding of the experiment infrastructure needed.

Quick start::

    import numpy as np
    from torch_timeseries import Forecaster

    # 1000 timesteps, 3 channels
    X = np.random.randn(1000, 3)

    fc = Forecaster("DLinear", seq_len=96, pred_len=24)
    fc.fit(X)

    # Predict next 24 steps given the last 96
    y_hat = fc.predict(X[-96:])          # shape: (24, 3)

    # Evaluate MSE / MAE on a held-out slice
    print(fc.score(X[-200:]))            # {"mse": 0.97, "mae": 0.78}

Any model from ``torch_timeseries.model.forecasting_models`` is supported by
name.  Model-specific hyper-parameters can be passed as keyword arguments::

    fc = Forecaster(
        "PatchTST",
        seq_len=336,
        pred_len=96,
        d_model=128,
        n_heads=8,
        e_layers=3,
    )
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .scaler.standard import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


class _WindowDataset(Dataset):
    """Sliding-window dataset over a normalised 2-D time series."""

    def __init__(self, X: np.ndarray, seq_len: int, pred_len: int) -> None:
        self.X = torch.from_numpy(X).float()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = max(0, len(X) - seq_len - pred_len + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = self.X[i : i + self.seq_len]
        y = self.X[i + self.seq_len : i + self.seq_len + self.pred_len]
        return x, y


class _EarlyStopping:
    """In-memory early stopping with best-weight capture."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights: Optional[dict] = None
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - 1e-8:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def _to_numpy(X) -> np.ndarray:
    """Accept NumPy arrays, pandas DataFrames/Series, or torch Tensors."""
    try:
        import pandas as pd
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy(dtype=np.float32)
    except ImportError:
        pass
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


class Forecaster:
    """scikit-learn-style wrapper for any model in ``forecasting_models``.

    Parameters
    ----------
    model:
        Either a model name (string) from ``torch_timeseries.model.forecasting_models``
        or a pre-constructed ``nn.Module``.  When a name is given, the model is
        built with ``seq_len``, ``pred_len``, ``enc_in`` (inferred from the
        training data), and any extra ``**model_kwargs``.
    seq_len:
        Number of input look-back timesteps.
    pred_len:
        Number of steps to forecast.
    device:
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``.
    epochs:
        Maximum training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    patience:
        Early-stopping patience (epochs without improvement on val loss).
    normalize:
        If ``True`` (default) fit a :class:`StandardScaler` on the training
        slice and apply it before every forward pass.  Predictions are returned
        in the *original* scale.
    verbose:
        Print epoch summaries and a progress bar.
    **model_kwargs:
        Extra keyword arguments forwarded to the model constructor.

    Examples
    --------
    >>> import numpy as np
    >>> from torch_timeseries import Forecaster
    >>> X = np.random.randn(500, 3)
    >>> fc = Forecaster("DLinear", seq_len=48, pred_len=12, epochs=3)
    >>> fc.fit(X)
    >>> fc.predict(X[-48:]).shape
    (12, 3)
    """

    def __init__(
        self,
        model: Union[str, nn.Module],
        seq_len: int = 96,
        pred_len: int = 24,
        *,
        device: str = "cpu",
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 5,
        normalize: bool = True,
        verbose: bool = True,
        **model_kwargs,
    ) -> None:
        self.model_spec = model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.normalize = normalize
        self.verbose = verbose
        self.model_kwargs = model_kwargs

        self._model: Optional[nn.Module] = None
        self._scaler: Optional[StandardScaler] = None
        self._enc_in: Optional[int] = None

    # ── construction ──────────────────────────────────────────────────────────

    def _build_model(self, enc_in: int) -> nn.Module:
        if isinstance(self.model_spec, nn.Module):
            return self.model_spec.to(self.device)

        name = self.model_spec
        import torch_timeseries.model as _m

        if not hasattr(_m, name):
            available = sorted(getattr(_m, "forecasting_models", []))
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available forecasting models: {available}"
            )
        cls = getattr(_m, name)
        return cls(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=enc_in,
            **self.model_kwargs,
        ).to(self.device)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X,
        *,
        val_split: float = 0.1,
    ) -> "Forecaster":
        """Train on a univariate or multivariate time series.

        Parameters
        ----------
        X:
            Time series data.  Accepted shapes:

            * ``(N, C)`` — N timesteps, C channels (most common).
            * ``(N,)``   — single-channel; treated as ``(N, 1)``.

            Also accepts ``pd.DataFrame``, ``pd.Series``, or ``torch.Tensor``.
        val_split:
            Fraction of timesteps (from the *end* of the training series) to
            hold out for validation and early stopping.  The scaler is fit only
            on the training portion.

        Returns
        -------
        self
        """
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]
        N, C = X.shape
        self._enc_in = C

        min_len = self.seq_len + self.pred_len
        if N < min_len:
            raise ValueError(
                f"X has {N} timesteps but seq_len + pred_len = {min_len}. "
                "Provide more data or reduce seq_len / pred_len."
            )

        # Train / val time split
        n_train = max(min_len, int(N * (1.0 - val_split)))
        X_train_raw = X[:n_train]
        # Validation context: keep last seq_len of train for context
        X_val_raw = X[max(0, n_train - self.seq_len):]

        # Normalise
        if self.normalize:
            self._scaler = StandardScaler()
            self._scaler.fit(X_train_raw)
            X_train = self._scaler.transform(X_train_raw)
            X_val = self._scaler.transform(X_val_raw)
        else:
            self._scaler = None
            X_train = X_train_raw.astype(np.float32)
            X_val = X_val_raw.astype(np.float32)

        train_ds = _WindowDataset(X_train, self.seq_len, self.pred_len)
        val_ds = _WindowDataset(X_val, self.seq_len, self.pred_len)

        if len(train_ds) == 0:
            raise ValueError(
                f"Training slice is too short ({n_train} timesteps) to form "
                f"even one window of seq_len={self.seq_len} + pred_len={self.pred_len}."
            )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        self._model = self._build_model(C)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(self.patience)

        for epoch in range(1, self.epochs + 1):
            # ── train ────────────────────────────────────────────────────────
            self._model.train()
            train_losses: List[float] = []
            for x_b, y_b in train_loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                optimiser.zero_grad()
                pred = self._model(x_b)
                loss = loss_fn(pred, y_b)
                loss.backward()
                optimiser.step()
                train_losses.append(loss.item())

            # ── validate ─────────────────────────────────────────────────────
            val_loss = self._eval_loss(val_loader, loss_fn) if len(val_ds) > 0 else float("inf")
            train_loss_avg = float(np.mean(train_losses))

            if self.verbose:
                print(
                    f"Epoch {epoch:3d}/{self.epochs}  "
                    f"train_loss={train_loss_avg:.6f}  "
                    f"val_loss={val_loss:.6f}"
                )

            stopper(val_loss, self._model)
            if stopper.stop:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        stopper.restore_best(self._model)
        self._model.eval()
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X) -> np.ndarray:
        """Forecast the next ``pred_len`` timesteps.

        Parameters
        ----------
        X:
            Context window(s).  Accepted shapes:

            * ``(seq_len, C)``    — single context; returns ``(pred_len, C)``.
            * ``(N, C)`` where N > seq_len — uses the last ``seq_len`` rows.
            * ``(B, seq_len, C)`` — batch of B contexts; returns
              ``(B, pred_len, C)``.

        Returns
        -------
        np.ndarray
            Forecast in the *original* (un-normalised) scale.
        """
        self._check_fitted()
        X = _to_numpy(X)
        batched = X.ndim == 3
        if not batched:
            if X.ndim == 1:
                X = X[:, None]
            if len(X) > self.seq_len:
                X = X[-self.seq_len:]
            X = X[None]  # (1, seq_len, C)

        if self.normalize and self._scaler is not None:
            # Normalise each window independently using the fitted scaler
            X = np.stack([self._scaler.transform(w) for w in X])

        x_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(x_t).cpu().numpy()  # (B, pred_len, C)

        if self.normalize and self._scaler is not None:
            pred = np.stack([self._scaler.inverse_transform(p) for p in pred])

        return pred[0] if not batched else pred  # squeeze batch dim if single

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X) -> Dict[str, float]:
        """Evaluate MSE and MAE over sliding windows of X.

        The model predicts each window's next ``pred_len`` steps and compares
        them to the ground truth in the *original* scale.

        Parameters
        ----------
        X:
            Time series data, shape ``(N, C)``.  Needs at least
            ``seq_len + pred_len`` timesteps.

        Returns
        -------
        dict
            ``{"mse": float, "mae": float, "rmse": float}``
        """
        self._check_fitted()
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]

        if self.normalize and self._scaler is not None:
            X_norm = self._scaler.transform(X)
        else:
            X_norm = X.astype(np.float32)

        ds = _WindowDataset(X_norm, self.seq_len, self.pred_len)
        if len(ds) == 0:
            raise ValueError(
                f"X has only {len(X)} timesteps; need at least "
                f"seq_len + pred_len = {self.seq_len + self.pred_len}."
            )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        loss_fn = nn.MSELoss(reduction="sum")
        mae_fn = nn.L1Loss(reduction="sum")
        total_mse = total_mae = 0.0
        n_elements = 0

        self._model.eval()
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                pred = self._model(x_b).cpu()    # normalised predictions
                # Invert normalisation for a fair comparison
                if self.normalize and self._scaler is not None:
                    pred_np = pred.numpy()
                    y_np = y_b.numpy()
                    pred_np = np.stack([self._scaler.inverse_transform(p) for p in pred_np])
                    y_np = np.stack([self._scaler.inverse_transform(y) for y in y_np])
                    pred = torch.from_numpy(pred_np)
                    y_b = torch.from_numpy(y_np)

                total_mse += loss_fn(pred, y_b).item()
                total_mae += mae_fn(pred, y_b).item()
                n_elements += pred.numel()

        mse = total_mse / n_elements
        mae = total_mae / n_elements
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}

    # ── utilities ─────────────────────────────────────────────────────────────

    def _eval_loss(self, loader: DataLoader, loss_fn: nn.Module) -> float:
        self._model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                pred = self._model(x_b)
                total += loss_fn(pred, y_b).item() * x_b.size(0)
                n += x_b.size(0)
        return total / n if n > 0 else float("inf")

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Forecaster is not fitted yet.  Call fit() first.")

    @property
    def model(self) -> nn.Module:
        """The underlying PyTorch model (after fitting)."""
        self._check_fitted()
        return self._model

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        self._check_fitted()
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        name = self.model_spec if isinstance(self.model_spec, str) else type(self.model_spec).__name__
        status = "fitted" if self._model is not None else "not fitted"
        return (
            f"Forecaster(model={name!r}, seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}, [{status}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# compare()
# ──────────────────────────────────────────────────────────────────────────────


def compare(
    models: List[Union[str, nn.Module]],
    X_train,
    X_test,
    *,
    seq_len: int = 96,
    pred_len: int = 24,
    device: str = "cpu",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 5,
    normalize: bool = True,
    verbose: bool = True,
    val_split: float = 0.1,
    **shared_model_kwargs,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate multiple models on the same dataset.

    Each model is trained independently on ``X_train`` and scored on ``X_test``.
    All models share the same ``seq_len``, ``pred_len``, and training hyper-
    parameters unless you pass a :class:`Forecaster` instance directly (in
    which case its own settings take precedence).

    Parameters
    ----------
    models:
        List of model names (strings) or pre-configured :class:`Forecaster`
        instances.  Names are looked up in
        ``torch_timeseries.model.forecasting_models``.
    X_train:
        Training time series, shape ``(N_train, C)``.
    X_test:
        Test time series, shape ``(N_test, C)``.  Needs at least
        ``seq_len + pred_len`` timesteps.
    seq_len:
        Shared look-back length (ignored for :class:`Forecaster` instances).
    pred_len:
        Shared forecast horizon (ignored for :class:`Forecaster` instances).
    verbose:
        Print per-model progress.
    **shared_model_kwargs:
        Extra kwargs forwarded to every model's constructor.

    Returns
    -------
    dict
        ``{model_name: {"mse": float, "mae": float, "rmse": float}, ...}``
        Sorted by ascending MSE.

    Examples
    --------
    >>> results = compare(
    ...     ["DLinear", "NLinear", "PatchTST"],
    ...     X_train=X[:800], X_test=X[800:],
    ...     seq_len=96, pred_len=24, epochs=5,
    ... )
    >>> for name, m in results.items():
    ...     print(f"{name:30s}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")
    """
    results: Dict[str, Dict[str, float]] = {}

    for i, spec in enumerate(models):
        if isinstance(spec, Forecaster):
            fc = spec
            name = (
                spec.model_spec
                if isinstance(spec.model_spec, str)
                else type(spec.model_spec).__name__
            )
        else:
            name = spec if isinstance(spec, str) else type(spec).__name__
            fc = Forecaster(
                spec,
                seq_len=seq_len,
                pred_len=pred_len,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                normalize=normalize,
                verbose=verbose,
                **shared_model_kwargs,
            )

        if verbose:
            print(f"\n[{i + 1}/{len(models)}] {name}")

        try:
            fc.fit(X_train, val_split=val_split)
            metrics = fc.score(X_test)
        except Exception as exc:
            if verbose:
                print(f"  ERROR: {exc}")
            metrics = {"mse": float("inf"), "mae": float("inf"), "rmse": float("inf"), "error": str(exc)}

        results[name] = metrics

    # Sort by ascending MSE
    results = dict(sorted(results.items(), key=lambda kv: kv[1].get("mse", float("inf"))))
    return results
