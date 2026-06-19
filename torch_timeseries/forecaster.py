"""High-level scikit-learn-style API for time-series forecasting.

Also provides :func:`compare` — a one-liner for benchmarking multiple models
on the same dataset, and :func:`list_models` for discovery::

    from torch_timeseries import compare, list_models

    print(list_models())          # sorted list of all available model names

    results = compare(
        ["DLinear", "PatchTST", "iTransformer"],
        X_train=X[:800],
        X_test=X[800:],
        seq_len=96,
        pred_len=24,
        epochs=5,
    )
    # prints a ranked table and returns a dict:
    # {"DLinear": {"mse": ..., "mae": ..., "rmse": ..., "smape": ...}, ...}


Users bring their own data as a NumPy array or pandas DataFrame — no dataset
class, no YAML config, no understanding of the experiment infrastructure needed.

Quick start::

    import numpy as np
    from torch_timeseries import Forecaster

    # 1000 timesteps, 3 channels
    X = np.random.randn(1000, 3)

    fc = Forecaster("DLinear", seq_len=96, pred_len=24, scheduler="cosine")
    fc.fit(X)

    # Inspect per-epoch loss history
    print(fc.history_[:3])
    # [{"epoch": 1, "train_loss": 1.02, "val_loss": 0.99}, ...]

    # Predict next 24 steps given the last 96
    y_hat = fc.predict(X[-96:])          # shape: (24, 3)

    # Evaluate with extended metrics
    print(fc.score(X[-200:]))
    # {"mse": 0.97, "mae": 0.78, "rmse": 0.99, "smape": 12.3}

    # Persist and reload
    fc.save("my_model.pt")
    fc2 = Forecaster.load("my_model.pt")

    # Walk-forward cross-validation
    cv = fc.cross_validate(X, n_splits=5)
    print(cv["mean_mse"], cv["std_mse"])

    # Hyperparameter grid search
    best = fc.tune(X, param_grid={"lr": [1e-3, 1e-4], "batch_size": [16, 32]})
    best.fit(X)

    # Clone an unfitted copy for independent experiments
    clone = fc.clone()

Any model from ``torch_timeseries.model.forecasting_models`` is supported by
name.  Model-specific hyper-parameters can be passed as keyword arguments::

    fc = Forecaster(
        "PatchTST",
        seq_len=336,
        pred_len=96,
        d_model=128,
        n_heads=8,
        e_layers=3,
        loss="mae",
        scheduler="cosine",
        grad_clip=1.0,
    )
"""
from __future__ import annotations

import copy
import time
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
# Scheduler factory
# ──────────────────────────────────────────────────────────────────────────────

_SCHEDULER_CHOICES = ("cosine", "plateau", "step")


def _make_scheduler(optimiser, scheduler_name: Optional[str], epochs: int, patience: int):
    """Create an LR scheduler from a string name, or None for no scheduling."""
    if scheduler_name is None:
        return None
    name = scheduler_name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=max(1, patience // 2), factor=0.5
        )
    if name == "step":
        step_size = max(1, epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=0.5)
    raise ValueError(
        f"Unknown scheduler {scheduler_name!r}. "
        f"Choose from {_SCHEDULER_CHOICES} or None."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Loss factory
# ──────────────────────────────────────────────────────────────────────────────

_LOSS_MAP = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def _resolve_loss(loss) -> nn.Module:
    """Return an ``nn.Module`` loss from a string name or module instance."""
    if loss is None:
        return nn.MSELoss()
    if isinstance(loss, nn.Module):
        return loss
    key = loss.lower().replace(" ", "_")
    if key not in _LOSS_MAP:
        raise ValueError(
            f"Unknown loss {loss!r}. Choose from {sorted(_LOSS_MAP)} or pass an nn.Module."
        )
    return _LOSS_MAP[key]()


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
        scheduler: Optional[str] = None,
        loss: Optional[Union[str, nn.Module]] = None,
        warm_start: bool = False,
        grad_clip: Optional[float] = None,
        weight_decay: float = 0.0,
        callbacks: Optional[List] = None,
        progress_bar: bool = False,
        augmentation=None,
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
        self.scheduler = scheduler
        self.loss = loss
        self.warm_start = warm_start
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.callbacks = callbacks or []
        self.progress_bar = progress_bar
        self.augmentation = augmentation
        self.model_kwargs = model_kwargs

        self._model: Optional[nn.Module] = None
        self._scaler: Optional[StandardScaler] = None
        self._enc_in: Optional[int] = None
        self.history_: List[dict] = []

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

        if self.warm_start and self._model is not None and self._enc_in == C:
            # Reuse existing weights; only reset history
            pass
        else:
            self._model = self._build_model(C)
        self.history_ = []
        optimiser = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = _make_scheduler(optimiser, self.scheduler, self.epochs, self.patience)
        loss_fn = _resolve_loss(self.loss)
        stopper = _EarlyStopping(self.patience)

        try:
            from tqdm import tqdm as _tqdm
            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

        epoch_iter = range(1, self.epochs + 1)
        if self.progress_bar and _has_tqdm:
            epoch_iter = _tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # ── train ────────────────────────────────────────────────────────
            self._model.train()
            train_losses: List[float] = []
            for x_b, y_b in train_loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                if self.augmentation is not None:
                    x_b = self.augmentation(x_b)
                optimiser.zero_grad()
                pred = self._model(x_b)
                loss = loss_fn(pred, y_b)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
                optimiser.step()
                train_losses.append(loss.item())

            # ── validate ─────────────────────────────────────────────────────
            val_loss = self._eval_loss(val_loader, loss_fn) if len(val_ds) > 0 else float("inf")
            train_loss_avg = float(np.mean(train_losses))

            entry = {
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss,
                "lr": optimiser.param_groups[0]["lr"],
            }
            self.history_.append(entry)

            for cb in self.callbacks:
                cb(self, entry)

            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_loss)
                else:
                    sched.step()

            if self.verbose:
                lr_now = optimiser.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:3d}/{self.epochs}  "
                    f"train_loss={train_loss_avg:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"lr={lr_now:.2e}"
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
            ``{"mse": float, "mae": float, "rmse": float, "smape": float}``
            SMAPE is the symmetric mean absolute percentage error (×100, in %).
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
        total_mse = total_mae = total_smape = 0.0
        n_elements = 0

        self._model.eval()
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                pred = self._model(x_b).cpu()
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
                # SMAPE: 2|y-ŷ| / (|y| + |ŷ| + ε) * 100
                denom = pred.abs() + y_b.abs() + 1e-8
                total_smape += (2.0 * (pred - y_b).abs() / denom * 100.0).sum().item()
                n_elements += pred.numel()

        mse = total_mse / n_elements
        mae = total_mae / n_elements
        smape = total_smape / n_elements
        result = {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)), "smape": smape}
        return result

    # ── utilities ─────────────────────────────────────────────────────────────

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the fitted forecaster to a file.

        Saves model weights, scaler state, hyperparameters, and training history
        so the forecaster can be fully restored with :meth:`load`.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"model.pt"``).
        """
        self._check_fitted()
        payload = {
            "model_spec": self.model_spec if isinstance(self.model_spec, str) else None,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "enc_in": self._enc_in,
            "model_kwargs": self.model_kwargs,
            "model_state": self._model.state_dict(),
            "scaler_state": self._scaler.__dict__.copy() if self._scaler is not None else None,
            "history": self.history_,
            "normalize": self.normalize,
            "scheduler": self.scheduler,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "Forecaster":
        """Restore a :class:`Forecaster` saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to a file written by :meth:`save`.
        device:
            Target device for the loaded model.

        Returns
        -------
        Forecaster
            A ready-to-use (fitted) :class:`Forecaster`.
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        model_spec = payload.get("model_spec")
        if model_spec is None:
            raise ValueError(
                "Cannot reload a Forecaster that was built from a raw nn.Module "
                "(no model name was recorded at save time)."
            )
        fc = cls(
            model_spec,
            seq_len=payload["seq_len"],
            pred_len=payload["pred_len"],
            device=device,
            normalize=payload.get("normalize", True),
            scheduler=payload.get("scheduler"),
            verbose=False,
            **payload.get("model_kwargs", {}),
        )
        fc._enc_in = payload["enc_in"]
        fc._model = fc._build_model(fc._enc_in)
        fc._model.load_state_dict(payload["model_state"])
        fc._model.eval()
        if payload.get("scaler_state") is not None:
            fc._scaler = StandardScaler()
            fc._scaler.__dict__.update(payload["scaler_state"])
        fc.history_ = payload.get("history", [])
        return fc

    # ── convenience ───────────────────────────────────────────────────────────

    def fit_predict(self, X_train, X_context, *, val_split: float = 0.1) -> np.ndarray:
        """Fit on *X_train*, then predict from *X_context* in one call.

        Returns
        -------
        np.ndarray
            Forecast array with shape ``(pred_len, C)`` or
            ``(B, pred_len, C)`` for a batch context.
        """
        return self.fit(X_train, val_split=val_split).predict(X_context)

    def cross_validate(
        self,
        X,
        *,
        n_splits: int = 5,
        val_split: float = 0.1,
    ) -> Dict[str, float]:
        """Walk-forward (expanding-window) cross-validation.

        The series is partitioned into ``n_splits + 1`` roughly equal chunks.
        For fold *k* (0-indexed), the model trains on chunks ``0..k+1`` and is
        evaluated on chunk ``k+2`` (using the same scoring as :meth:`score`).
        This mirrors real deployment where training always precedes the test
        window.

        Parameters
        ----------
        X:
            Full time series, shape ``(N, C)``.
        n_splits:
            Number of folds.  More splits mean smaller test chunks.
        val_split:
            Fraction held out for early-stopping within each training window.

        Returns
        -------
        dict
            ``{"mean_mse": float, "std_mse": float, "mean_mae": float,
            "std_mae": float, "mean_rmse": float, "std_rmse": float,
            "mean_smape": float, "std_smape": float, "n_splits_used": int}``
        """
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]
        N = len(X)
        min_len = self.seq_len + self.pred_len
        fold_size = N // (n_splits + 2)

        if fold_size < min_len:
            raise ValueError(
                f"Fold size ({fold_size}) is smaller than seq_len + pred_len "
                f"({min_len}). Provide more data or reduce n_splits / seq_len."
            )

        fold_results: List[Dict[str, float]] = []
        for k in range(n_splits):
            train_end = fold_size * (k + 2)
            test_end = min(train_end + fold_size, N)

            X_train = X[:train_end]
            X_test = X[train_end:test_end]

            if len(X_train) < min_len or len(X_test) < min_len:
                continue

            fc = Forecaster(
                self.model_spec,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                device=str(self.device),
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                patience=self.patience,
                normalize=self.normalize,
                verbose=False,
                scheduler=self.scheduler,
                **self.model_kwargs,
            )
            fc.fit(X_train, val_split=val_split)
            fold_results.append(fc.score(X_test))

        if not fold_results:
            raise ValueError(
                "No valid folds completed.  Provide more data or reduce "
                "n_splits / seq_len / pred_len."
            )

        metric_keys = [k for k in fold_results[0] if k not in ("error",)]
        out: Dict[str, float] = {}
        for key in metric_keys:
            vals = [r[key] for r in fold_results if key in r]
            out[f"mean_{key}"] = float(np.mean(vals))
            out[f"std_{key}"] = float(np.std(vals))
        out["n_splits_used"] = len(fold_results)
        return out

    def plot_history(self, *, ax=None):
        """Plot training and validation loss curves from the last :meth:`fit`.

        Requires ``matplotlib``.

        Parameters
        ----------
        ax:
            An existing ``matplotlib.axes.Axes`` to draw on.  If *None*
            (default) a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._check_fitted()
        if not self.history_:
            raise RuntimeError("Training history is empty.  Call fit() first.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_history(). "
                "Install it with: pip install matplotlib"
            ) from exc

        epochs = [h["epoch"] for h in self.history_]
        train_losses = [h["train_loss"] for h in self.history_]
        val_losses = [h["val_loss"] for h in self.history_]

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(epochs, train_losses, label="train_loss")
        ax.plot(epochs, val_losses, label="val_loss", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        name = self.model_spec if isinstance(self.model_spec, str) else type(self.model_spec).__name__
        ax.set_title(f"{name} — Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if created_fig:
            plt.tight_layout()
        return ax

    def partial_fit(
        self,
        X,
        *,
        val_split: float = 0.1,
    ) -> "Forecaster":
        """Continue training on new data without discarding prior weights.

        Unlike :meth:`fit`, ``partial_fit`` always uses ``warm_start=True``
        semantics regardless of the instance's ``warm_start`` setting.  The
        model is trained for ``self.epochs`` additional epochs.  The scaler is
        re-fitted on the new data.

        Use this for online / streaming scenarios where data arrives in chunks.

        Parameters
        ----------
        X:
            New time series data for the next training round.
        val_split:
            Val fraction passed to the training loop.

        Returns
        -------
        self
        """
        original_warm_start = self.warm_start
        self.warm_start = True
        try:
            self.fit(X, val_split=val_split)
        finally:
            self.warm_start = original_warm_start
        return self

    def compare_horizons(
        self,
        X,
        *,
        horizons: List[int],
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, float]]:
        """Train and score the same model class across multiple forecast horizons.

        For each horizon in *horizons*, a fresh clone of the current
        :class:`Forecaster` is trained and scored on *X*.  This reveals how
        model performance degrades (or remains stable) as the prediction window
        grows.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)``.
        horizons:
            List of ``pred_len`` values to evaluate.
        val_split:
            Val fraction passed to each clone's :meth:`fit`.
        verbose:
            Print per-horizon progress.

        Returns
        -------
        dict
            ``{pred_len: {"mse": float, "mae": float, "rmse": float,
                          "smape": float, "elapsed_s": float}, ...}``
            Sorted by ascending ``pred_len``.

        Examples
        --------
        >>> results = fc.compare_horizons(X, horizons=[6, 12, 24, 48])
        >>> for h, m in results.items():
        ...     print(f"pred_len={h:3d}  MSE={m['mse']:.4f}")
        """
        results: Dict[int, Dict[str, float]] = {}
        for h in sorted(horizons):
            fc = self.clone()
            fc.pred_len = h
            if verbose:
                print(f"  horizon={h}")
            try:
                t0 = time.perf_counter()
                fc.fit(X, val_split=val_split)
                elapsed = time.perf_counter() - t0
                metrics = fc.score(X)
                metrics["elapsed_s"] = elapsed
            except Exception as exc:
                if verbose:
                    print(f"  ERROR at horizon {h}: {exc}")
                metrics = {
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "rmse": float("inf"),
                    "smape": float("inf"),
                    "elapsed_s": float("nan"),
                    "error": str(exc),
                }
            results[h] = metrics
        return results

    def benchmark(
        self,
        *,
        n_runs: int = 100,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Measure inference latency of the fitted model.

        Runs ``n_runs`` forward passes with a random input and returns timing
        statistics.  Useful for comparing deployment efficiency across models.

        Parameters
        ----------
        n_runs:
            Number of timed forward passes.
        batch_size:
            Batch size for each pass.
        seq_len:
            Input sequence length.  Defaults to ``self.seq_len``.
        warmup:
            Number of un-timed warm-up passes before recording (eliminates
            first-pass JIT / memory allocation cost).

        Returns
        -------
        dict
            ``{"mean_ms": float, "std_ms": float,
               "min_ms": float, "max_ms": float,
               "throughput_samples_per_sec": float}``
        """
        self._check_fitted()
        seq = seq_len or self.seq_len
        enc_in = self._enc_in or 1
        x_dummy = torch.randn(batch_size, seq, enc_in, device=self.device)

        self._model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                self._model(x_dummy)

        times_ms: List[float] = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                self._model(x_dummy)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(times_ms))
        return {
            "mean_ms": mean_ms,
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "throughput_samples_per_sec": float(batch_size / (mean_ms / 1000.0 + 1e-9)),
        }

    def to_onnx(self, path: str, *, opset_version: int = 17) -> None:
        """Export the fitted model to ONNX format.

        The exported model accepts an input of shape
        ``(batch_size, seq_len, enc_in)`` and produces
        ``(batch_size, pred_len, enc_in)``.

        Parameters
        ----------
        path:
            Destination ``.onnx`` file path.
        opset_version:
            ONNX opset version (default 17).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ImportError
            If ``onnx`` is not installed.

        Examples
        --------
        >>> fc.to_onnx("dlinear.onnx")
        >>> # Load back with onnxruntime:
        >>> import onnxruntime; sess = onnxruntime.InferenceSession("dlinear.onnx")
        """
        self._check_fitted()
        try:
            import torch.onnx
        except ImportError as exc:
            raise ImportError("torch.onnx is required but not available.") from exc

        enc_in = self._enc_in or 1
        x_dummy = torch.randn(1, self.seq_len, enc_in, device=self.device)
        self._model.eval()
        torch.onnx.export(
            self._model,
            x_dummy,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=opset_version,
        )

    def evaluate(self, X, *, seasonal_period: int = 1) -> Dict[str, float]:
        """Extended evaluation including SMAPE and MASE.

        Computes MSE, MAE, RMSE, SMAPE, and MASE over all sliding windows of
        *X*.  Results are in the *original* (un-normalised) scale.

        Parameters
        ----------
        X:
            Time series data, shape ``(N, C)``.  Needs at least
            ``seq_len + pred_len`` timesteps.
        seasonal_period:
            Period used for the naive seasonal baseline in MASE (default 1,
            i.e. the naïve one-step-ahead forecast).

        Returns
        -------
        dict
            ``{"mse", "mae", "rmse", "smape", "mase"}``
        """
        self._check_fitted()
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]

        base = self.score(X)

        # Compute MASE: MAE / naive-mae
        # Naive: y_t = y_{t - seasonal_period}; error on same data
        m = seasonal_period
        if len(X) > m:
            naive_errors = np.abs(X[m:] - X[:-m])
            naive_mae = float(naive_errors.mean()) + 1e-8
        else:
            naive_mae = 1.0
        mase = base["mae"] / naive_mae

        return {**base, "mase": mase}

    def feature_importance(
        self,
        X,
        *,
        metric: str = "mse",
        n_repeats: int = 5,
        random_state: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Estimate channel importance via permutation.

        For each channel *c*, the channel's values are randomly shuffled across
        timesteps (repeated ``n_repeats`` times) and the resulting score is
        compared to the baseline.  A large increase in the chosen metric after
        shuffling indicates that the model relies on that channel.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)``.  Must have at least
            ``seq_len + pred_len`` timesteps.
        metric:
            Metric to evaluate — ``"mse"``, ``"mae"``, or ``"smape"``.
        n_repeats:
            Number of shuffle repetitions per channel.
        random_state:
            NumPy random seed for reproducible shuffling.

        Returns
        -------
        dict
            ``{"importances_mean": np.ndarray shape (C,),
               "importances_std": np.ndarray shape (C,),
               "baseline_score": float}``
            Higher ``importances_mean[c]`` means channel *c* matters more.
        """
        self._check_fitted()
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]
        baseline = self.score(X)[metric]
        rng = np.random.default_rng(random_state)
        C = X.shape[1]
        importances = np.zeros((C, n_repeats))
        for c in range(C):
            for r in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, c] = rng.permutation(X_perm[:, c])
                importances[c, r] = self.score(X_perm)[metric] - baseline
        return {
            "importances_mean": importances.mean(axis=1),
            "importances_std": importances.std(axis=1),
            "baseline_score": baseline,
        }

    def predict_uncertainty(
        self,
        X,
        *,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Estimate predictive uncertainty via MC-Dropout.

        Activates Dropout layers at inference time (``model.train()`` mode) and
        runs ``n_samples`` stochastic forward passes to build a sample
        distribution.  Works best with models that contain Dropout layers.

        Parameters
        ----------
        X:
            Context window(s).  Same shapes as :meth:`predict`.
        n_samples:
            Number of MC samples.

        Returns
        -------
        dict
            ``{"mean": np.ndarray, "std": np.ndarray,
               "lower": np.ndarray, "upper": np.ndarray}``
            All arrays have the same shape as a single :meth:`predict` output.
            ``lower``/``upper`` are the 5th and 95th percentiles.
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        batched = X_np.ndim == 3
        if not batched:
            if X_np.ndim == 1:
                X_np = X_np[:, None]
            if len(X_np) > self.seq_len:
                X_np = X_np[-self.seq_len:]
            X_np = X_np[None]

        if self.normalize and self._scaler is not None:
            X_np = np.stack([self._scaler.transform(w) for w in X_np])

        x_t = torch.from_numpy(X_np.astype(np.float32)).to(self.device)

        # Activate dropout at inference (MC-Dropout)
        self._model.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self._model(x_t).cpu().numpy()
                if self.normalize and self._scaler is not None:
                    pred = np.stack([self._scaler.inverse_transform(p) for p in pred])
                samples.append(pred[0] if not batched else pred)
        self._model.eval()

        samples_arr = np.stack(samples)  # (n_samples, ..., pred_len, C)
        return {
            "mean": samples_arr.mean(axis=0),
            "std": samples_arr.std(axis=0),
            "lower": np.percentile(samples_arr, 5, axis=0),
            "upper": np.percentile(samples_arr, 95, axis=0),
        }

    def residuals(self, X) -> np.ndarray:
        """Compute prediction residuals over sliding windows of X.

        For every valid window ``x`` of length ``seq_len``, compute
        ``y_true - y_pred`` where ``y_true`` is the following ``pred_len``
        timesteps and ``y_pred`` is :meth:`predict`.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_windows, pred_len, C)`` — signed residuals in the
            original scale.
        """
        self._check_fitted()
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]
        min_len = self.seq_len + self.pred_len
        if len(X) < min_len:
            raise ValueError(
                f"X has only {len(X)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X) - self.seq_len - self.pred_len + 1
        windows = np.stack([X[i : i + self.seq_len] for i in range(n_windows)])
        truths = np.stack(
            [X[i + self.seq_len : i + self.seq_len + self.pred_len] for i in range(n_windows)]
        )
        preds = self.predict(windows)  # (n_windows, pred_len, C)
        return truths - preds

    def predict_rolling(
        self,
        X,
        *,
        n_steps: Optional[int] = None,
        stride: int = 1,
    ) -> np.ndarray:
        """Generate rolling forecasts by sliding the context window across X.

        Unlike :meth:`predict`, which forecasts a single fixed horizon from one
        context window, ``predict_rolling`` slides the ``seq_len`` window along
        the series and emits one forecast per position.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)`` or ``(N,)``.  Must have at least
            ``seq_len + pred_len`` timesteps.
        n_steps:
            Maximum number of forecast positions.  If *None* (default) all
            valid positions are used.
        stride:
            Step size between consecutive context windows (default 1).

        Returns
        -------
        np.ndarray
            Shape ``(n_positions, pred_len, C)`` — one forecast per window.
        """
        self._check_fitted()
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[:, None]
        N = len(X)
        min_len = self.seq_len + self.pred_len
        if N < min_len:
            raise ValueError(
                f"X has only {N} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        positions = list(range(0, N - self.seq_len - self.pred_len + 1, stride))
        if n_steps is not None:
            positions = positions[:n_steps]

        windows = np.stack([X[i : i + self.seq_len] for i in positions])  # (P, S, C)
        return self.predict(windows)  # (P, pred_len, C)

    # ── sklearn-compatible params ─────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Return hyperparameters as a dict (scikit-learn compatible).

        Returns
        -------
        dict
            All constructor parameters plus any extra ``model_kwargs``.
        """
        params = {
            "model": self.model_spec,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "device": str(self.device),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience,
            "normalize": self.normalize,
            "verbose": self.verbose,
            "scheduler": self.scheduler,
            "loss": self.loss,
            "warm_start": self.warm_start,
            "grad_clip": self.grad_clip,
            "weight_decay": self.weight_decay,
            "progress_bar": self.progress_bar,
            "augmentation": self.augmentation,
        }
        params.update(self.model_kwargs)
        return params

    def set_params(self, **params) -> "Forecaster":
        """Set hyperparameters (scikit-learn compatible).

        Parameters are applied directly as attributes; unknown keys are added
        to ``model_kwargs``.  Returns ``self`` so calls can be chained.
        """
        _direct = {"seq_len", "pred_len", "epochs", "batch_size", "lr",
                   "patience", "normalize", "verbose", "scheduler", "loss",
                   "warm_start", "grad_clip", "weight_decay", "callbacks",
                   "progress_bar", "augmentation"}
        for k, v in params.items():
            if k == "model":
                self.model_spec = v
            elif k == "device":
                self.device = torch.device(v)
            elif k in _direct:
                setattr(self, k, v)
            else:
                self.model_kwargs[k] = v
        return self

    # ── internal helpers ──────────────────────────────────────────────────────

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

    def clone(self) -> "Forecaster":
        """Return an unfitted copy with the same hyperparameters.

        Useful for hyperparameter search: keep the original forecaster and work
        with independent copies for each candidate configuration.

        Returns
        -------
        Forecaster
            A fresh (not fitted) :class:`Forecaster` with identical settings.
        """
        return Forecaster(
            self.model_spec,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            device=str(self.device),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            normalize=self.normalize,
            verbose=self.verbose,
            scheduler=self.scheduler,
            loss=self.loss,
            warm_start=self.warm_start,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            progress_bar=self.progress_bar,
            augmentation=self.augmentation,
            **self.model_kwargs,
        )

    @classmethod
    def from_dataset(
        cls,
        model: Union[str, nn.Module],
        dataset_name: str,
        *,
        root: str = "~/.torchtimeseries/data",
        train_fraction: float = 0.7,
        val_split: float = 0.1,
        seq_len: int = 96,
        pred_len: int = 24,
        **kwargs,
    ) -> "Forecaster":
        """Construct and fit a :class:`Forecaster` directly from a built-in dataset.

        Downloads the dataset on first use (cached in *root*).

        Parameters
        ----------
        model:
            Model name or ``nn.Module``.
        dataset_name:
            Name of a built-in dataset, e.g. ``"ETTh1"``.  Use
            ``torch_timeseries.list_datasets()`` to see all options.
        root:
            Dataset cache directory.
        train_fraction:
            Fraction of the dataset used for training.  The remainder is
            held out as test data.
        val_split:
            Validation fraction passed to :meth:`fit`.
        seq_len:
            Look-back window length.
        pred_len:
            Forecast horizon.
        **kwargs:
            Additional keyword arguments forwarded to the :class:`Forecaster`
            constructor.

        Returns
        -------
        Forecaster
            A fitted forecaster ready for :meth:`predict` and :meth:`score`.

        Examples
        --------
        >>> fc = Forecaster.from_dataset("DLinear", "ETTh1", epochs=5)
        >>> y = fc.predict(...)
        """
        from .dataset import load_dataset as _load
        X = _load(dataset_name, root=root)
        n_train = int(len(X) * train_fraction)
        X_train = X[:n_train]
        fc = cls(model, seq_len=seq_len, pred_len=pred_len, **kwargs)
        fc.fit(X_train, val_split=val_split)
        fc._dataset_name = dataset_name
        fc._X_test = X[n_train:]
        return fc

    @classmethod
    def from_config(cls, config: dict) -> "Forecaster":
        """Construct a :class:`Forecaster` from a flat config dict.

        This is the inverse of :meth:`get_params`: any key that maps to a
        constructor parameter is applied; the rest are passed as
        ``model_kwargs``.

        Parameters
        ----------
        config:
            A dict such as the one returned by :meth:`get_params`.

        Returns
        -------
        Forecaster
        """
        _known = {"model", "seq_len", "pred_len", "device", "epochs", "batch_size",
                  "lr", "patience", "normalize", "verbose", "scheduler",
                  "loss", "warm_start", "grad_clip"}
        cfg = dict(config)
        model = cfg.pop("model", "DLinear")
        seq_len = cfg.pop("seq_len", 96)
        pred_len = cfg.pop("pred_len", 24)
        kwargs = {k: cfg.pop(k) for k in list(cfg) if k in _known}
        return cls(model, seq_len=seq_len, pred_len=pred_len,
                   **kwargs, **cfg)

    def tune(
        self,
        X,
        *,
        param_grid: dict,
        n_splits: int = 3,
        val_split: float = 0.1,
        metric: str = "mean_mse",
        verbose: bool = True,
        n_iter: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> "Forecaster":
        """Grid-search hyperparameters using walk-forward cross-validation.

        Iterates over all combinations in *param_grid*, evaluates each with
        :meth:`cross_validate`, and returns the best-scoring unfitted
        :class:`Forecaster` ready to be retrained on the full dataset.

        Parameters
        ----------
        X:
            Full time series for cross-validation, shape ``(N, C)``.
        param_grid:
            Dict mapping parameter names to lists of candidate values.
            Example::

                {"lr": [1e-3, 1e-4], "batch_size": [16, 32]}

        n_splits:
            Number of CV folds passed to :meth:`cross_validate`.
        val_split:
            Val fraction within each training window.
        metric:
            CV result key to minimise (default ``"mean_mse"``).
        verbose:
            Print per-combination progress.
        n_iter:
            If given, randomly sample ``n_iter`` combinations from the full
            grid (random search).  Useful when the grid is large.
        random_state:
            Seed for the random sampler used when ``n_iter`` is set.

        Returns
        -------
        Forecaster
            An unfitted :class:`Forecaster` configured with the best params
            found.  Call ``.fit(X)`` on the returned forecaster to train it.

        Examples
        --------
        >>> best = fc.tune(X, param_grid={"lr": [1e-3, 1e-4], "batch_size": [16, 64]})
        >>> best.fit(X)
        """
        import itertools
        import random as _random

        keys = list(param_grid)
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        if n_iter is not None and n_iter < len(combos):
            rng = _random.Random(random_state)
            combos = rng.sample(combos, n_iter)

        best_score = float("inf")
        best_params: Optional[dict] = None

        for i, vals in enumerate(combos):
            candidate = dict(zip(keys, vals))
            fc = self.clone()
            fc.set_params(**candidate)
            try:
                cv_result = fc.cross_validate(X, n_splits=n_splits, val_split=val_split)
                score = cv_result.get(metric, float("inf"))
            except Exception as exc:
                score = float("inf")
                cv_result = {metric: float("inf"), "_error": str(exc)}

            if verbose:
                print(
                    f"  [{i + 1}/{len(combos)}] {candidate}  "
                    f"{metric}={score:.6f}"
                )

            if score < best_score:
                best_score = score
                best_params = candidate

        if best_params is None:
            raise RuntimeError("All hyperparameter combinations failed.")

        if verbose:
            print(f"\nBest params ({metric}={best_score:.6f}): {best_params}")

        best_fc = self.clone()
        best_fc.set_params(**best_params)
        return best_fc

    def summary(self) -> str:
        """Return a human-readable summary string of the forecaster.

        Includes model name, hyperparameters, parameter count (if fitted), and
        the last epoch's train/val loss.

        Returns
        -------
        str
        """
        name = self.model_spec if isinstance(self.model_spec, str) else type(self.model_spec).__name__
        lines = [
            f"Forecaster — {name}",
            f"  seq_len     : {self.seq_len}",
            f"  pred_len    : {self.pred_len}",
            f"  epochs      : {self.epochs}",
            f"  lr          : {self.lr}",
            f"  batch_size  : {self.batch_size}",
            f"  normalize   : {self.normalize}",
            f"  scheduler   : {self.scheduler}",
            f"  loss        : {self.loss!r}",
            f"  warm_start  : {self.warm_start}",
            f"  grad_clip   : {self.grad_clip}",
            f"  weight_decay: {self.weight_decay}",
            f"  callbacks   : {len(self.callbacks)}",
            f"  device      : {self.device}",
        ]
        if self._model is not None:
            lines.append(f"  parameters  : {self.n_parameters:,}")
        if self.history_:
            last = self.history_[-1]
            lines.append(
                f"  last epoch  : {last['epoch']}  "
                f"train={last['train_loss']:.6f}  "
                f"val={last['val_loss']:.6f}"
            )
        return "\n".join(lines)

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


def time_series_split(
    X,
    *,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    gap: int = 0,
) -> List[tuple]:
    """Generate time-series aware (walk-forward) train/test index splits.

    Unlike random K-fold, each fold has all data up to a cutoff as training
    and the next ``test_size`` timesteps as test — simulating real deployment.

    Parameters
    ----------
    X:
        Time series array or any object with ``len()``.
    n_splits:
        Number of folds.
    test_size:
        Number of test timesteps per fold.  If *None*, uses
        ``len(X) // (n_splits + 1)``.
    gap:
        Number of timesteps between the end of the training slice and the
        start of the test slice (to avoid leakage from overlapping windows).

    Returns
    -------
    list of (train_indices, test_indices) tuples
        Each element is a pair of ``np.ndarray`` index arrays.

    Examples
    --------
    >>> splits = time_series_split(X, n_splits=5)
    >>> for train_idx, test_idx in splits:
    ...     X_train, X_test = X[train_idx], X[test_idx]
    ...     fc.fit(X_train).score(X_test)
    """
    N = len(_to_numpy(X)) if not isinstance(X, int) else X
    if test_size is None:
        test_size = N // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        test_start = N - (n_splits - i) * test_size
        test_end = test_start + test_size
        train_end = test_start - gap
        if train_end <= 0 or test_start >= N:
            continue
        splits.append((np.arange(train_end), np.arange(test_start, test_end)))
    return splits


def list_models() -> List[str]:
    """Return a sorted list of all available forecasting model names.

    Examples
    --------
    >>> from torch_timeseries import list_models
    >>> names = list_models()
    >>> print(names[:5])
    ['CARD', 'CATS', 'CycleNet', ...]
    """
    import torch_timeseries.model as _m
    return sorted(getattr(_m, "forecasting_models", []))


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
    print_table: bool = True,
    n_jobs: int = 1,
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
        Print per-model epoch progress.
    print_table:
        Print a ranked summary table after all models finish (default True).
    n_jobs:
        Number of parallel workers.  ``1`` (default) runs sequentially.
        ``-1`` uses all available CPU cores.  Parallel execution uses
        ``concurrent.futures.ProcessPoolExecutor`` and requires models to be
        specified by name (not ``nn.Module`` instances).
    **shared_model_kwargs:
        Extra kwargs forwarded to every model's constructor.

    Returns
    -------
    dict
        ``{model_name: {"mse": float, "mae": float, "rmse": float, "smape": float}, ...}``
        Sorted by ascending MSE.

    Examples
    --------
    >>> results = compare(
    ...     ["DLinear", "NLinear", "PatchTST"],
    ...     X_train=X[:800], X_test=X[800:],
    ...     seq_len=96, pred_len=24, epochs=5,
    ... )
    # ── Results ────────────────────────────────────────────────
    # Rank  Model              MSE      MAE      RMSE     SMAPE%
    #    1  DLinear           0.9234   0.7112   0.9609   12.34
    #    2  NLinear           0.9901   0.7451   0.9950   13.02
    #    3  PatchTST          1.0123   0.7678   1.0061   13.55
    """
    X_train_np = _to_numpy(X_train)
    X_test_np = _to_numpy(X_test)

    def _run_one(spec, idx):
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
        if verbose and n_jobs == 1:
            print(f"\n[{idx + 1}/{len(models)}] {name}")
        try:
            t0 = time.perf_counter()
            fc.fit(X_train_np, val_split=val_split)
            elapsed = time.perf_counter() - t0
            metrics = fc.score(X_test_np)
            metrics["elapsed_s"] = elapsed
        except Exception as exc:
            if verbose and n_jobs == 1:
                print(f"  ERROR: {exc}")
            metrics = {
                "mse": float("inf"),
                "mae": float("inf"),
                "rmse": float("inf"),
                "smape": float("inf"),
                "elapsed_s": float("nan"),
                "error": str(exc),
            }
        return name, metrics

    results: Dict[str, Dict[str, float]] = {}

    if n_jobs == 1:
        for i, spec in enumerate(models):
            name, metrics = _run_one(spec, i)
            results[name] = metrics
    else:
        import concurrent.futures
        import os
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, spec, i): i for i, spec in enumerate(models)}
            for fut in concurrent.futures.as_completed(futures):
                name, metrics = fut.result()
                results[name] = metrics

    # Sort by ascending MSE
    results = dict(sorted(results.items(), key=lambda kv: kv[1].get("mse", float("inf"))))

    if print_table:
        _print_compare_table(results)

    return results


def _print_compare_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a ranked summary table for :func:`compare` results."""
    if not results:
        return
    cols = ["MSE", "MAE", "RMSE", "SMAPE%", "Time(s)"]
    keys = ["mse", "mae", "rmse", "smape", "elapsed_s"]
    name_w = max(len(n) for n in results) + 2
    col_w = 10
    sep = "─" * (6 + name_w + col_w * len(cols))
    header = f"{'Rank':>5}  {'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    print("\n" + sep)
    print(header)
    print(sep)
    for rank, (name, m) in enumerate(results.items(), 1):
        row = f"{rank:>5}  {name:<{name_w}}"
        for k in keys:
            v = m.get(k, float("nan"))
            if v == float("inf"):
                row += f"{'ERR':>{col_w}}"
            elif v != v:  # nan
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# StackedForecaster
# ──────────────────────────────────────────────────────────────────────────────


class StackedForecaster:
    """Two-stage residual stacking of :class:`Forecaster` instances.

    The first stage trains on the raw series.  The second stage trains on
    the *residuals* (true − predicted) from the first stage, learning what the
    first stage missed.  At inference, the final prediction is the sum of both
    stages' outputs.

    This implements a simple two-level boosting scheme that can reduce
    systematic bias of the base model.

    Parameters
    ----------
    base:
        A :class:`Forecaster` (or any object with ``fit``, ``predict``,
        and ``residuals`` methods).  Used as the first stage.
    meta:
        A :class:`Forecaster` used as the second (residual) stage.  If
        *None*, a clone of *base* is used.

    Examples
    --------
    >>> base = Forecaster("DLinear", seq_len=96, pred_len=24, epochs=10)
    >>> meta = Forecaster("NLinear", seq_len=96, pred_len=24, epochs=10)
    >>> sf = StackedForecaster(base, meta)
    >>> sf.fit(X_train)
    >>> y_hat = sf.predict(X_train[-96:])
    """

    def __init__(self, base: Forecaster, meta: Optional[Forecaster] = None) -> None:
        self.base = base
        self.meta = meta if meta is not None else base.clone()

    def fit(self, X, *, val_split: float = 0.1) -> "StackedForecaster":
        """Fit both stages sequentially.

        Parameters
        ----------
        X:
            Training time series.
        val_split:
            Val fraction passed to each stage's :meth:`Forecaster.fit`.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]

        self.base.fit(X_np, val_split=val_split)
        residuals = self.base.residuals(X_np)  # (n_windows, pred_len, C)

        # Build a pseudo-series from residuals by concatenating them:
        # use the *last* residual per position, giving a (N', pred_len, C)
        # array.  To train the meta model with the same window format, we
        # treat each residual vector as the target and use the corresponding
        # original input window as context.  The simplest approach: train
        # the meta Forecaster on the residual time series directly (concat
        # the first timestep of each residual block to form a 1-D-per-channel
        # residual series).
        n_windows = len(residuals)
        seq_len = self.base.seq_len
        pred_len = self.base.pred_len
        min_len = seq_len + pred_len

        # Residual series: take the mean across pred_len to get per-position
        # residual signals, then form a series of length n_windows.
        residual_series = residuals.mean(axis=1)  # (n_windows, C)
        if len(residual_series) < min_len:
            # Not enough data for meta; skip meta training
            return self

        self.meta.fit(residual_series, val_split=val_split)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict = base prediction + meta residual correction.

        Parameters
        ----------
        X:
            Context window(s).  Same shapes accepted as :meth:`Forecaster.predict`.

        Returns
        -------
        np.ndarray
            Shape ``(pred_len, C)`` or ``(B, pred_len, C)`` for batch input.
        """
        base_pred = self.base.predict(X)
        meta_pred = self.meta.predict(X)
        return base_pred + meta_pred

    def score(self, X) -> Dict[str, float]:
        """Score the stacked forecaster using :meth:`Forecaster.score` logic."""
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        min_len = self.base.seq_len + self.base.pred_len
        if len(X_np) < min_len:
            raise ValueError(
                f"X has only {len(X_np)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X_np) - self.base.seq_len - self.base.pred_len + 1
        windows = np.stack([X_np[i : i + self.base.seq_len] for i in range(n_windows)])
        truths = np.stack(
            [X_np[i + self.base.seq_len : i + self.base.seq_len + self.base.pred_len]
             for i in range(n_windows)]
        )
        preds = self.predict(windows)  # (n_windows, pred_len, C)
        diff = preds - truths
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        denom = np.abs(preds) + np.abs(truths) + 1e-8
        smape = float((2.0 * np.abs(diff) / denom * 100.0).mean())
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)), "smape": smape}

    def __repr__(self) -> str:
        return f"StackedForecaster(base={self.base!r}, meta={self.meta!r})"


# ──────────────────────────────────────────────────────────────────────────────
# BaggingForecaster
# ──────────────────────────────────────────────────────────────────────────────


class BaggingForecaster:
    """Bootstrap-aggregation (bagging) ensemble of :class:`Forecaster` clones.

    Trains ``n_estimators`` independent clones, each on a bootstrapped (random
    subsample) of the training series.  At inference, predictions are averaged
    across all members.  Bagging reduces variance and provides uncertainty
    estimates via member disagreement.

    Parameters
    ----------
    base:
        A (not-yet-fitted) :class:`Forecaster` used as the template.
    n_estimators:
        Number of bootstrap members (default 10).
    subsample:
        Fraction of training timesteps sampled per member (default 0.8).
        Sampling is done *with* replacement, preserving temporal order within
        each contiguous draw.
    random_state:
        NumPy seed for bootstrap sampling.

    Examples
    --------
    >>> base = Forecaster("DLinear", seq_len=96, pred_len=24, epochs=10)
    >>> bag = BaggingForecaster(base, n_estimators=5)
    >>> bag.fit(X_train)
    >>> result = bag.predict(X_train[-96:])
    >>> print(result["mean"].shape)   # (24, C)
    >>> print(result["std"].shape)    # (24, C) — member disagreement
    """

    def __init__(
        self,
        base: Forecaster,
        n_estimators: int = 10,
        subsample: float = 0.8,
        random_state: Optional[int] = None,
    ) -> None:
        if not 0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1].")
        self.base = base
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.random_state = random_state
        self.estimators_: List[Forecaster] = []

    def fit(self, X, *, val_split: float = 0.1) -> "BaggingForecaster":
        """Train all bootstrap members.

        Parameters
        ----------
        X:
            Training time series.
        val_split:
            Val fraction passed to each member's :meth:`Forecaster.fit`.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N = len(X_np)
        n_sample = max(self.base.seq_len + self.base.pred_len,
                       int(N * self.subsample))
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            start = rng.integers(0, N - n_sample + 1)
            X_boot = X_np[start : start + n_sample]
            fc = self.base.clone()
            fc.fit(X_boot, val_split=val_split)
            self.estimators_.append(fc)
        return self

    def predict(self, X) -> Dict[str, np.ndarray]:
        """Predict with ensemble; returns mean and member disagreement.

        Parameters
        ----------
        X:
            Context window(s).  Same shapes as :meth:`Forecaster.predict`.

        Returns
        -------
        dict
            ``{"mean": np.ndarray, "std": np.ndarray,
               "lower": np.ndarray, "upper": np.ndarray}``
            All arrays have the same shape as a single ``Forecaster.predict``
            output.  ``std`` is the cross-member standard deviation.
        """
        if not self.estimators_:
            raise RuntimeError("BaggingForecaster is not fitted yet.  Call fit() first.")
        preds = np.stack([e.predict(X) for e in self.estimators_])  # (M, ..., C)
        return {
            "mean": preds.mean(axis=0),
            "std": preds.std(axis=0),
            "lower": np.percentile(preds, 5, axis=0),
            "upper": np.percentile(preds, 95, axis=0),
        }

    def score(self, X) -> Dict[str, float]:
        """Score the ensemble (mean prediction) with MSE/MAE/RMSE/SMAPE."""
        if not self.estimators_:
            raise RuntimeError("BaggingForecaster is not fitted yet.  Call fit() first.")
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        min_len = self.base.seq_len + self.base.pred_len
        if len(X_np) < min_len:
            raise ValueError(
                f"X has only {len(X_np)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X_np) - self.base.seq_len - self.base.pred_len + 1
        windows = np.stack([X_np[i : i + self.base.seq_len] for i in range(n_windows)])
        truths = np.stack(
            [X_np[i + self.base.seq_len : i + self.base.seq_len + self.base.pred_len]
             for i in range(n_windows)]
        )
        result = self.predict(windows)
        preds = result["mean"]
        diff = preds - truths
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        denom = np.abs(preds) + np.abs(truths) + 1e-8
        smape = float((2.0 * np.abs(diff) / denom * 100.0).mean())
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)), "smape": smape}

    def __repr__(self) -> str:
        fitted = f"{len(self.estimators_)} estimators" if self.estimators_ else "not fitted"
        return (
            f"BaggingForecaster(base={self.base!r}, "
            f"n_estimators={self.n_estimators}, [{fitted}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# compare_to_dataframe helper
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class Pipeline:
    """Chain a preprocessing step with a :class:`Forecaster`.

    The preprocessor is applied to raw data before training and inference.
    Typical use: apply a custom callable (e.g. differencing, log-transform,
    or a rolling-mean detrend) upstream of the :class:`Forecaster`'s built-in
    normalisation.

    Parameters
    ----------
    preprocessor:
        A callable ``(X: np.ndarray) -> np.ndarray`` applied to data before
        passing it to the forecaster.  Must accept ``(N, C)`` arrays and
        return the same shape.  Examples:

        * ``lambda X: np.diff(X, axis=0)`` — first differences
        * ``lambda X: np.log1p(np.abs(X)) * np.sign(X)`` — signed log
        * ``lambda X: X / X.std(axis=0)`` — channel-wise std scaling
    forecaster:
        A (not-yet-fitted) :class:`Forecaster`.

    Examples
    --------
    >>> import numpy as np
    >>> from torch_timeseries import Forecaster, Pipeline
    >>>
    >>> def log_transform(X):
    ...     return np.sign(X) * np.log1p(np.abs(X))
    >>>
    >>> pipe = Pipeline(log_transform, Forecaster("DLinear", seq_len=96, pred_len=24))
    >>> pipe.fit(X_train)
    >>> y_hat = pipe.predict(X_train[-96:])  # in original space
    """

    def __init__(self, preprocessor, forecaster: Forecaster) -> None:
        if not callable(preprocessor):
            raise ValueError("preprocessor must be callable.")
        self.preprocessor = preprocessor
        self.forecaster = forecaster
        self._inv_preprocessor = None

    def set_inverse(self, inv_preprocessor) -> "Pipeline":
        """Register the inverse of the preprocessor for output inversion.

        Parameters
        ----------
        inv_preprocessor:
            Callable ``(y: np.ndarray) -> np.ndarray`` that undoes
            ``preprocessor``.  If set, :meth:`predict` will apply it to
            the forecast before returning.

        Returns
        -------
        self
        """
        self._inv_preprocessor = inv_preprocessor
        return self

    def fit(self, X, *, val_split: float = 0.1) -> "Pipeline":
        """Apply the preprocessor then fit the forecaster.

        Parameters
        ----------
        X:
            Raw training time series.
        val_split:
            Val fraction for the forecaster.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        X_proc = self.preprocessor(X_np)
        self.forecaster.fit(X_proc, val_split=val_split)
        return self

    def predict(self, X) -> np.ndarray:
        """Preprocess *X*, forecast, and optionally apply inverse transform.

        Parameters
        ----------
        X:
            Context window(s).  Same shapes as :meth:`Forecaster.predict`.

        Returns
        -------
        np.ndarray
            Forecast in the *preprocessed* space if no inverse was registered,
            otherwise in the *original* space.
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 2:
            X_proc = self.preprocessor(X_np)
        else:
            X_proc = np.stack([self.preprocessor(w) for w in X_np])
        pred = self.forecaster.predict(X_proc)
        if self._inv_preprocessor is not None:
            if pred.ndim == 2:
                pred = self._inv_preprocessor(pred)
            else:
                pred = np.stack([self._inv_preprocessor(p) for p in pred])
        return pred

    def score(self, X) -> Dict[str, float]:
        """Evaluate MSE/MAE/RMSE/SMAPE on preprocessed data."""
        X_np = _to_numpy(X)
        X_proc = self.preprocessor(X_np)
        return self.forecaster.score(X_proc)

    @property
    def history_(self):
        """Training history from the underlying forecaster."""
        return self.forecaster.history_

    def __repr__(self) -> str:
        return f"Pipeline(preprocessor={self.preprocessor!r}, forecaster={self.forecaster!r})"


def compare_to_dataframe(results: Dict[str, Dict[str, float]]):
    """Convert :func:`compare` results to a ``pandas.DataFrame``.

    Parameters
    ----------
    results:
        Dict returned by :func:`compare`.

    Returns
    -------
    pandas.DataFrame
        One row per model, columns for each metric.  Sorted by MSE ascending.
        Returns ``None`` if pandas is not installed.

    Examples
    --------
    >>> df = compare_to_dataframe(results)
    >>> df.sort_values("mse").head()
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    if not results:
        return pd.DataFrame()
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df
