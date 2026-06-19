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
from typing import Dict, List, Optional, Tuple, Union

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

    def plot_forecast(
        self,
        X,
        *,
        channel: int = 0,
        n_context: Optional[int] = None,
        ax=None,
        title: Optional[str] = None,
    ):
        """Plot context, ground truth, and model forecast for one channel.

        Requires ``matplotlib``.

        Parameters
        ----------
        X:
            Full time series, shape ``(N, C)`` with
            ``N >= seq_len + pred_len``.  The last ``seq_len`` rows are used
            as the context window; the next ``pred_len`` rows (if available)
            are plotted as ground truth.
        channel:
            Channel index to plot (default ``0``).
        n_context:
            How many context timesteps to show to the left of the prediction.
            Defaults to ``seq_len`` (i.e., the full context window).
        ax:
            Existing ``matplotlib.axes.Axes``.  If ``None``, a new figure is
            created.
        title:
            Optional axes title.  Defaults to
            ``"<model> — channel <channel> forecast"``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._check_fitted()
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_forecast(). "
                "Install it with: pip install matplotlib"
            ) from exc

        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N, C = X_np.shape
        seq = self.seq_len
        pred = self.pred_len

        if N < seq:
            raise ValueError(
                f"X has only {N} timesteps; need at least seq_len={seq}."
            )

        ctx_window = X_np[-seq - pred : -pred] if N >= seq + pred else X_np[-seq:]
        if len(ctx_window) < seq:
            ctx_window = X_np[:seq]

        y_pred = self.predict(ctx_window)  # (pred_len, C)

        n_ctx = min(n_context or seq, seq)
        ctx_show = ctx_window[-n_ctx:, channel]
        ctx_t = np.arange(-n_ctx, 0)
        pred_t = np.arange(0, pred)

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(ctx_t, ctx_show, color="steelblue", label="context")

        # Ground truth future (if available)
        if N >= seq + pred:
            truth_start = max(0, N - pred)
            gt = X_np[truth_start : truth_start + pred, channel]
            if len(gt) == pred:
                ax.plot(pred_t, gt, color="green", linestyle="--",
                        label="ground truth")

        ax.plot(pred_t, y_pred[:, channel], color="tomato", label="forecast")
        ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)

        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — channel {channel} forecast")
        ax.set_xlabel("Timestep (0 = forecast start)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

    def plot_intervals(
        self,
        intervals: Dict[str, np.ndarray],
        *,
        channel: int = 0,
        X_context=None,
        X_truth=None,
        ax=None,
        title: Optional[str] = None,
        alpha: float = 0.25,
    ):
        """Visualise a prediction interval dict on a matplotlib Axes.

        Accepts the output of :meth:`predict_interval` or
        :meth:`predict_uncertainty` directly.

        Requires ``matplotlib``.

        Parameters
        ----------
        intervals:
            Dict with keys ``"mean"``, ``"lower"``, ``"upper"`` — all shape
            ``(pred_len, C)``.  Typically the return value of
            :meth:`predict_interval` or :meth:`predict_uncertainty`.
        channel:
            Channel index to plot (default ``0``).
        X_context:
            Optional context window, shape ``(seq_len, C)`` — plotted to the
            left of the forecast region.
        X_truth:
            Optional ground truth, shape ``(pred_len, C)`` — plotted as a
            dashed line over the forecast region.
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.
        alpha:
            Transparency of the confidence band (default 0.25).

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_intervals(). "
                "Install it with: pip install matplotlib"
            ) from exc

        mean = np.asarray(intervals["mean"])
        lower = np.asarray(intervals["lower"])
        upper = np.asarray(intervals["upper"])
        pred = mean.shape[0]
        pred_t = np.arange(0, pred)

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(10, 4))

        # Context
        if X_context is not None:
            ctx = np.asarray(X_context)
            if ctx.ndim == 1:
                ctx = ctx[:, None]
            n_ctx = len(ctx)
            ax.plot(np.arange(-n_ctx, 0), ctx[:, channel],
                    color="steelblue", label="context")

        # Confidence band
        ax.fill_between(pred_t, lower[:, channel], upper[:, channel],
                        alpha=alpha, color="tomato", label="interval")
        ax.plot(pred_t, mean[:, channel], color="tomato", label="forecast mean")

        # Ground truth
        if X_truth is not None:
            gt = np.asarray(X_truth)
            if gt.ndim == 1:
                gt = gt[:, None]
            ax.plot(pred_t, gt[:pred, channel], color="green",
                    linestyle="--", label="ground truth")

        ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — channel {channel} intervals")
        ax.set_xlabel("Timestep (0 = forecast start)")
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

    def simulate(
        self,
        X_seed,
        steps: int,
        *,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Auto-regressively generate *steps* future timesteps one at a time.

        At each step, the model forecasts the next ``pred_len`` values, appends
        the *first* predicted timestep to the rolling buffer, and repeats.
        Optionally adds Gaussian noise to each generated step (for stochastic
        simulation / scenario generation).

        Parameters
        ----------
        X_seed:
            Seed context window, shape ``(seq_len, C)`` or longer.
        steps:
            Number of future timesteps to generate.
        noise_scale:
            Standard deviation of additive Gaussian noise applied to each
            generated step.  ``0.0`` (default) gives deterministic output.
        random_state:
            NumPy seed for reproducible stochastic simulation.

        Returns
        -------
        np.ndarray
            Shape ``(steps, C)`` — the generated future sequence.
        """
        self._check_fitted()
        X_np = _to_numpy(X_seed)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        if len(X_np) > self.seq_len:
            X_np = X_np[-self.seq_len :]
        rng = np.random.default_rng(random_state)
        buffer = X_np.copy()
        generated = []
        for _ in range(steps):
            pred = self.predict(buffer)    # (pred_len, C)
            next_step = pred[0].copy()     # take first step
            if noise_scale > 0.0:
                next_step = next_step + rng.normal(0.0, noise_scale, size=next_step.shape)
            generated.append(next_step)
            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = next_step
        return np.stack(generated)  # (steps, C)

    def plot_scenarios(
        self,
        mc_result: Dict[str, np.ndarray],
        *,
        channel: int = 0,
        X_context=None,
        n_scenarios_to_plot: int = 20,
        ax=None,
        title: Optional[str] = None,
        alpha_scenarios: float = 0.08,
    ):
        """Fan chart of Monte Carlo scenario trajectories.

        Visualises the output of :meth:`montecarlo_forecast`: individual
        scenario lines, the mean trajectory, and the inter-quantile band.

        Requires ``matplotlib``.

        Parameters
        ----------
        mc_result:
            Dict returned by :meth:`montecarlo_forecast`.
        channel:
            Channel index to plot (default ``0``).
        X_context:
            Optional context window, shape ``(seq_len, C)``, plotted to the
            left of the forecast region.
        n_scenarios_to_plot:
            How many raw scenario lines to draw (default 20).
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.
        alpha_scenarios:
            Transparency of individual scenario lines (default 0.08).

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_scenarios(). "
                "Install it with: pip install matplotlib"
            ) from exc

        mean = mc_result["mean"][:, channel]
        scenarios = mc_result["scenarios"][:, :, channel]
        steps = len(mean)
        pred_t = np.arange(0, steps)

        q_keys = sorted(mc_result["quantiles"].keys())
        n_q = len(q_keys)

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(10, 4))

        # Context
        if X_context is not None:
            ctx = np.asarray(X_context)
            if ctx.ndim == 1:
                ctx = ctx[:, None]
            n_ctx = len(ctx)
            ax.plot(np.arange(-n_ctx, 0), ctx[:, channel],
                    color="steelblue", label="context")

        # Individual scenario lines
        n_plot = min(n_scenarios_to_plot, len(scenarios))
        for s in scenarios[:n_plot]:
            ax.plot(pred_t, s, color="grey", alpha=alpha_scenarios, linewidth=0.6)

        # Inter-quantile bands
        if n_q >= 2:
            lo = mc_result["quantiles"][q_keys[0]][:, channel]
            hi = mc_result["quantiles"][q_keys[-1]][:, channel]
            ax.fill_between(pred_t, lo, hi, alpha=0.15, color="tomato",
                            label=f"[{q_keys[0]:.0%}, {q_keys[-1]:.0%}]")

        # Mean trajectory
        ax.plot(pred_t, mean, color="tomato", linewidth=1.5, label="mean")
        ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)

        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — scenario fan chart (ch {channel})")
        ax.set_xlabel("Timestep")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

    def montecarlo_forecast(
        self,
        X_seed,
        steps: int,
        *,
        n_scenarios: int = 100,
        noise_scale: float = 0.05,
        random_state: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run N stochastic simulations and summarize the ensemble.

        Calls :meth:`simulate` with ``noise_scale`` applied, ``n_scenarios``
        times, then aggregates the resulting trajectories into mean, std, and
        optional quantile bands.

        Parameters
        ----------
        X_seed:
            Seed context window, shape ``(seq_len, C)`` or longer.
        steps:
            Number of future timesteps to generate per scenario.
        n_scenarios:
            Number of independent simulation runs (default 100).
        noise_scale:
            Noise std for each scenario (default 0.05).
        random_state:
            Base seed; each scenario gets ``random_state + i``.
        quantiles:
            Quantile levels to compute (default ``[0.05, 0.25, 0.75, 0.95]``).

        Returns
        -------
        dict
            ``{"mean": (steps, C), "std": (steps, C),
               "quantiles": {q: (steps, C) for q in quantiles},
               "scenarios": (n_scenarios, steps, C)}``
        """
        self._check_fitted()
        quantiles = quantiles or [0.05, 0.25, 0.75, 0.95]
        base_seed = random_state if random_state is not None else 0
        scenarios = np.stack([
            self.simulate(X_seed, steps, noise_scale=noise_scale,
                          random_state=base_seed + i)
            for i in range(n_scenarios)
        ])  # (n_scenarios, steps, C)
        return {
            "mean": scenarios.mean(axis=0),
            "std": scenarios.std(axis=0),
            "quantiles": {q: np.quantile(scenarios, q, axis=0) for q in quantiles},
            "scenarios": scenarios,
        }

    def stream_predict(self, X):
        """Yield rolling forecasts as a generator over incoming timesteps.

        The generator maintains a ``seq_len``-length buffer.  For each
        new timestep appended from *X* (beyond the initial buffer), it
        yields the model's prediction for the next ``pred_len`` steps.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)`` with ``N > seq_len``.  The
            first ``seq_len`` rows seed the buffer; each subsequent row
            advances the window by one and triggers a prediction.

        Yields
        ------
        np.ndarray
            Shape ``(pred_len, C)`` — the forecast at each step.

        Examples
        --------
        >>> for pred in fc.stream_predict(X_test):
        ...     process(pred)
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N = len(X_np)
        if N <= self.seq_len:
            return
        buffer = X_np[:self.seq_len].copy()
        for i in range(self.seq_len, N):
            yield self.predict(buffer)
            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = X_np[i]

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

    @staticmethod
    def compute_metrics(
        y_true,
        y_pred,
        *,
        seasonal_period: int = 1,
    ) -> Dict[str, float]:
        """Compute forecast metrics from raw arrays without a fitted model.

        A static utility for comparing predictions from any source.

        Parameters
        ----------
        y_true:
            Ground truth, shape ``(N, pred_len, C)`` or ``(pred_len, C)``
            or ``(N,)``.
        y_pred:
            Predictions, same shape as *y_true*.
        seasonal_period:
            Period for MASE (default 1 = non-seasonal MAE baseline).

        Returns
        -------
        dict
            ``{"mse", "mae", "rmse", "smape", "mase"}`` — all floats.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        if y_t.shape != y_p.shape:
            raise ValueError(
                f"y_true shape {y_t.shape} != y_pred shape {y_p.shape}."
            )
        diff = y_p - y_t
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        rmse = float(np.sqrt(mse))
        denom = np.abs(y_p) + np.abs(y_t) + 1e-8
        smape = float((2.0 * np.abs(diff) / denom * 100.0).mean())
        # MASE: scale by seasonal naive MAE
        y_flat = y_t.ravel()
        if len(y_flat) > seasonal_period:
            naive_errors = np.abs(
                y_flat[seasonal_period:] - y_flat[:-seasonal_period]
            )
            scale = naive_errors.mean() if naive_errors.size > 0 else 1.0
        else:
            scale = 1.0
        mase = mae / (scale + 1e-8)
        return {"mse": mse, "mae": mae, "rmse": rmse, "smape": smape, "mase": mase}

    @staticmethod
    def smooth(X, window: int = 5, method: str = "mean") -> np.ndarray:
        """Apply a rolling smoothing filter along the time axis.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)`` or ``(N,)``.
        window:
            Rolling window size (default 5).
        method:
            ``"mean"`` (default) or ``"median"``.

        Returns
        -------
        np.ndarray
            Smoothed series, same shape as *X*.  The first ``window - 1``
            positions are filled with the expanding window (no padding).
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N, C = X_np.shape
        out = np.empty_like(X_np)
        for t in range(N):
            start = max(0, t - window + 1)
            block = X_np[start : t + 1]
            if method == "mean":
                out[t] = block.mean(axis=0)
            elif method == "median":
                out[t] = np.median(block, axis=0)
            else:
                raise ValueError(f"Unknown method {method!r}. Choose 'mean' or 'median'.")
        return out.squeeze() if X.ndim == 1 else out  # type: ignore[union-attr]

    @staticmethod
    def create_windows(
        X,
        seq_len: int,
        pred_len: int,
        *,
        stride: int = 1,
        gap: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised (context, target) window arrays from a raw series.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)`` or ``(N,)``.
        seq_len:
            Context window length.
        pred_len:
            Forecast horizon length.
        stride:
            Step size between consecutive windows (default 1).
        gap:
            Number of timesteps between the end of the context and the start
            of the target (default 0 = immediate).

        Returns
        -------
        X_windows: np.ndarray
            Shape ``(n_windows, seq_len, C)``.
        y_windows: np.ndarray
            Shape ``(n_windows, pred_len, C)``.
        """
        X_np = np.asarray(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N, C = X_np.shape
        total = seq_len + gap + pred_len
        if N < total:
            raise ValueError(
                f"X has only {N} timesteps; need at least "
                f"seq_len + gap + pred_len = {total}."
            )
        starts = range(0, N - total + 1, stride)
        X_wins = np.stack([X_np[i : i + seq_len] for i in starts])
        y_wins = np.stack(
            [X_np[i + seq_len + gap : i + seq_len + gap + pred_len] for i in starts]
        )
        return X_wins, y_wins

    def plot_channel_scores(
        self,
        X,
        *,
        metric: str = "mse",
        ax=None,
        title: Optional[str] = None,
        channel_names: Optional[List[str]] = None,
    ):
        """Bar chart of per-channel forecast error.

        Requires ``matplotlib``.

        Parameters
        ----------
        X:
            Time series used for evaluation, shape ``(N, C)``.
        metric:
            Which metric to plot: ``"mse"`` (default), ``"mae"``, or
            ``"rmse"``.
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.  Defaults to ``"<model> — per-channel <metric>"``.
        channel_names:
            Optional list of length C for x-axis tick labels.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._check_fitted()
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_channel_scores(). "
                "Install it with: pip install matplotlib"
            ) from exc

        scores = self.score_per_channel(X)
        if metric not in scores:
            raise ValueError(
                f"metric must be one of {list(scores.keys())}; got {metric!r}."
            )
        values = scores[metric]
        C = len(values)
        labels = channel_names if channel_names else [f"ch{i}" for i in range(C)]

        created_fig = ax is None
        if created_fig:
            fig_w = max(4, C * 0.6 + 1.5)
            _, ax = plt.subplots(figsize=(fig_w, 4))

        colors = plt.cm.coolwarm(np.linspace(0, 1, C))
        ax.bar(labels, values, color=colors)
        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — per-channel {metric.upper()}")
        ax.set_xlabel("Channel")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

    def fit_score(
        self,
        X,
        *,
        test_size: Optional[float] = 0.2,
        val_split: float = 0.1,
        seasonal_period: int = 1,
    ) -> Dict[str, float]:
        """Fit on the first portion of *X* and score on the rest.

        Performs a single chronological train/test split of *X* by
        ``test_size`` fraction.  This is the simplest way to get a
        quick performance estimate without a separate test set.

        Parameters
        ----------
        X:
            Full time series, shape ``(N, C)``.
        test_size:
            Fraction of timesteps held out for testing (default 0.2).
        val_split:
            Validation fraction within the training portion.
        seasonal_period:
            Passed to :meth:`evaluate` for MASE computation.

        Returns
        -------
        dict
            Extended metrics from :meth:`evaluate`:
            ``{"mse", "mae", "rmse", "smape", "mase"}``.
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        n_test = max(self.seq_len + self.pred_len, int(len(X_np) * test_size))
        n_train = len(X_np) - n_test
        if n_train < self.seq_len + self.pred_len:
            raise ValueError(
                f"Training slice too short after test_size={test_size} split. "
                "Reduce test_size or provide more data."
            )
        self.fit(X_np[:n_train], val_split=val_split)
        return self.evaluate(X_np[n_train:], seasonal_period=seasonal_period)

    def plot_residuals(
        self,
        X,
        *,
        channel: int = 0,
        bins: int = 40,
        ax=None,
        title: Optional[str] = None,
    ):
        """Histogram of forecast residuals for one channel.

        Uses :meth:`residuals` to compute ``true - predicted`` over all valid
        windows of *X*, then plots the distribution.  Requires ``matplotlib``.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)``.
        channel:
            Channel index to plot (default ``0``).
        bins:
            Number of histogram bins (default 40).
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._check_fitted()
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_residuals(). "
                "Install it with: pip install matplotlib"
            ) from exc

        res = self.residuals(X)          # (W, pred_len, C)
        flat = res[:, :, channel].ravel()

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.hist(flat, bins=bins, edgecolor="white", linewidth=0.4,
                color="steelblue", alpha=0.85)
        ax.axvline(0, color="tomato", linewidth=1.2, linestyle="--",
                   label="zero")
        ax.axvline(flat.mean(), color="green", linewidth=1.2, linestyle=":",
                   label=f"mean={flat.mean():.3f}")

        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — residuals (channel {channel})")
        ax.set_xlabel("Residual (true − predicted)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

    def score_per_channel(self, X) -> Dict[str, np.ndarray]:
        """Per-channel MSE, MAE, and RMSE over sliding windows of *X*.

        Useful for diagnosing which channels are hard to forecast.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)``.

        Returns
        -------
        dict
            ``{"mse": (C,), "mae": (C,), "rmse": (C,)}`` — one value per
            channel.
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        seq, pred = self.seq_len, self.pred_len
        min_len = seq + pred
        if len(X_np) < min_len:
            raise ValueError(
                f"X has only {len(X_np)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X_np) - seq - pred + 1
        windows = np.stack([X_np[i : i + seq] for i in range(n_windows)])
        truths = np.stack(
            [X_np[i + seq : i + seq + pred] for i in range(n_windows)]
        )
        preds = self.predict(windows)  # (W, pred, C)
        diff = preds - truths          # (W, pred, C)
        mse_c = (diff ** 2).mean(axis=(0, 1))      # (C,)
        mae_c = np.abs(diff).mean(axis=(0, 1))     # (C,)
        return {
            "mse": mse_c,
            "mae": mae_c,
            "rmse": np.sqrt(mse_c),
        }

    def detect_anomalies(
        self,
        X,
        *,
        threshold: Optional[float] = None,
        n_sigma: float = 3.0,
    ) -> np.ndarray:
        """Detect timestep-level anomalies using prediction residuals.

        For each timestep, the model predicts the next ``pred_len`` steps.
        A timestep is flagged as anomalous if any of its predicted values
        deviate from the true values by more than ``threshold`` (or by more
        than ``n_sigma`` standard deviations of the residuals if no threshold
        is given).

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)``.  Needs at least
            ``seq_len + pred_len`` timesteps.
        threshold:
            Absolute residual threshold.  If *None* (default), uses
            ``n_sigma * std(residuals)`` as an adaptive threshold.
        n_sigma:
            Number of standard deviations used when ``threshold=None``.

        Returns
        -------
        np.ndarray, dtype=bool, shape (N,)
            True at positions flagged as anomalous.  The first ``seq_len``
            positions are always False (no context available).
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N = len(X_np)
        min_len = self.seq_len + self.pred_len
        if N < min_len:
            raise ValueError(
                f"X has only {N} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        residuals = self.residuals(X_np)  # (n_windows, pred_len, C)
        # Use L2 norm per window as anomaly score
        scores = np.sqrt((residuals ** 2).mean(axis=(1, 2)))  # (n_windows,)

        if threshold is None:
            threshold = n_sigma * scores.std() + scores.mean()

        # Map window scores back to timesteps (each window -> start position)
        anomaly_mask = np.zeros(N, dtype=bool)
        for i, score in enumerate(scores):
            if score > threshold:
                # Flag the input window start position
                anomaly_mask[i + self.seq_len] = True

        return anomaly_mask

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

    def timestep_importance(
        self,
        X,
        *,
        metric: str = "mse",
        n_repeats: int = 5,
        random_state: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Estimate which *timestep positions* within the context window matter most.

        For each position *t* in ``[0, seq_len)``, the values at that position
        (across all channels) are permuted within the window, and the resulting
        score change vs the baseline is recorded.  High increase → that timestep
        strongly influences the forecast.

        Parameters
        ----------
        X:
            Time series of shape ``(N, C)``.  Must have at least
            ``seq_len + pred_len`` timesteps.
        metric:
            ``"mse"``, ``"mae"``, or ``"smape"``.
        n_repeats:
            Shuffle repetitions per timestep position.
        random_state:
            NumPy seed for reproducibility.

        Returns
        -------
        dict
            ``{"importances_mean": (seq_len,), "importances_std": (seq_len,),
               "baseline_score": float}``
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N, C = X_np.shape
        seq, pred = self.seq_len, self.pred_len
        min_len = seq + pred
        if N < min_len:
            raise ValueError(
                f"X has only {N} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = N - seq - pred + 1
        rng = np.random.default_rng(random_state)

        # Baseline: stack all windows and score
        windows = np.stack([X_np[i : i + seq] for i in range(n_windows)])
        truths = np.stack([X_np[i + seq : i + seq + pred] for i in range(n_windows)])
        preds_base = self.predict(windows)
        diff_base = preds_base - truths

        def _score(diff: np.ndarray) -> float:
            if metric == "mse":
                return float((diff ** 2).mean())
            elif metric == "mae":
                return float(np.abs(diff).mean())
            elif metric == "smape":
                p, t = preds_base, truths
                return float((2 * np.abs(diff) / (np.abs(p) + np.abs(t) + 1e-8) * 100).mean())
            raise ValueError(f"Unknown metric {metric!r}.")

        baseline = _score(diff_base)
        importances = np.zeros((seq, n_repeats))

        for t_pos in range(seq):
            for r in range(n_repeats):
                perm_windows = windows.copy()
                # Shuffle position t_pos across all windows independently
                for w in range(n_windows):
                    perm_windows[w, t_pos, :] = rng.permutation(
                        perm_windows[w, t_pos, :]
                    )
                preds_perm = self.predict(perm_windows)
                diff_perm = preds_perm - truths
                importances[t_pos, r] = _score(diff_perm) - baseline

        return {
            "importances_mean": importances.mean(axis=1),
            "importances_std": importances.std(axis=1),
            "baseline_score": baseline,
        }

    def plot_timestep_importance(
        self,
        X,
        *,
        metric: str = "mse",
        n_repeats: int = 5,
        random_state: Optional[int] = None,
        ax=None,
        title: Optional[str] = None,
    ):
        """Plot :meth:`timestep_importance` as a line chart.

        Requires ``matplotlib``.

        Parameters
        ----------
        X, metric, n_repeats, random_state:
            Forwarded to :meth:`timestep_importance`.
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_timestep_importance(). "
                "Install it with: pip install matplotlib"
            ) from exc

        result = self.timestep_importance(
            X, metric=metric, n_repeats=n_repeats, random_state=random_state
        )
        mean = result["importances_mean"]
        std = result["importances_std"]
        xs = np.arange(len(mean))

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(9, 4))

        ax.plot(xs, mean, color="steelblue", label="importance (mean)")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color="steelblue")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — timestep importance ({metric})")
        ax.set_xlabel("Timestep position in context window")
        ax.set_ylabel(f"Δ {metric.upper()} after shuffle")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

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
        mean = samples_arr.mean(axis=0)
        std = samples_arr.std(axis=0)
        scale = getattr(self, "_interval_scale_", 1.0)
        half = scale * std * 1.6449
        return {
            "mean": mean,
            "std": std,
            "lower": mean - half,
            "upper": mean + half,
        }

    def calibrate(
        self,
        X_cal,
        *,
        target_coverage: float = 0.90,
        n_samples: int = 50,
    ) -> "Forecaster":
        """Post-hoc calibrate MC-Dropout intervals on a held-out set.

        Fits a scalar ``_interval_scale_`` so that the ``predict_uncertainty``
        lower/upper bounds achieve approximately *target_coverage* empirical
        coverage on the calibration data.  After calling this method,
        :meth:`predict_uncertainty` automatically applies the scale factor.

        The calibration uses a simple binary search over interval scale
        factors in ``[0.01, 20]``.

        Parameters
        ----------
        X_cal:
            Calibration time series, shape ``(N, C)`` with
            ``N >= seq_len + pred_len``.
        target_coverage:
            Desired fraction of ground-truth values inside the predicted
            interval (default 0.90).
        n_samples:
            MC-Dropout samples per window.

        Returns
        -------
        self
        """
        self._check_fitted()
        if not (0.0 < target_coverage < 1.0):
            raise ValueError(
                f"target_coverage must be in (0, 1); got {target_coverage}."
            )

        X_np = _to_numpy(X_cal)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        seq, pred = self.seq_len, self.pred_len
        min_len = seq + pred
        if len(X_np) < min_len:
            raise ValueError(
                f"X_cal has only {len(X_np)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X_np) - seq - pred + 1

        # Collect MC samples for every calibration window.
        all_samples: List[np.ndarray] = []
        all_truths: List[np.ndarray] = []
        for i in range(n_windows):
            window = X_np[i : i + seq]
            truth = X_np[i + seq : i + seq + pred]
            unc = self.predict_uncertainty(window, n_samples=n_samples)
            # Keep raw mean and std; reconstruct intervals with varying scale
            all_samples.append(unc["mean"])
            all_truths.append((unc["std"], truth))

        means = np.stack(all_samples)          # (W, pred, C)
        stds = np.stack([t[0] for t in all_truths])    # (W, pred, C)
        truths = np.stack([t[1] for t in all_truths])  # (W, pred, C)

        def coverage_at(scale: float) -> float:
            half = scale * stds * 1.6449  # ≈ 90% interval for N(0,1)
            inside = ((truths >= means - half) & (truths <= means + half))
            return float(inside.mean())

        # Binary search for scale that achieves target coverage
        lo, hi = 0.01, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if coverage_at(mid) < target_coverage:
                lo = mid
            else:
                hi = mid
        self._interval_scale_: float = (lo + hi) / 2.0
        return self

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

    def predict_interval(
        self,
        X,
        X_cal,
        *,
        coverage: float = 0.90,
    ) -> Dict[str, np.ndarray]:
        """Conformal prediction interval using empirical residual quantiles.

        Computes the empirical ``(1-coverage)/2`` and ``(1+coverage)/2``
        quantiles of the absolute residuals on *X_cal* and applies them as
        lower/upper offsets around the point forecast on *X*.  No Dropout is
        required — works for any model architecture.

        Parameters
        ----------
        X:
            Context window for the forecast, shape ``(seq_len, C)`` or longer.
        X_cal:
            Calibration set, shape ``(N, C)`` with ``N >= seq_len + pred_len``.
        coverage:
            Target empirical coverage (default ``0.90``).

        Returns
        -------
        dict
            ``{"mean": np.ndarray, "lower": np.ndarray, "upper": np.ndarray}``
            All arrays have shape ``(pred_len, C)``.
        """
        self._check_fitted()
        if not (0.0 < coverage < 1.0):
            raise ValueError(f"coverage must be in (0, 1); got {coverage}.")

        # Compute residuals on calibration set
        res = self.residuals(X_cal)  # (W, pred_len, C) — signed
        abs_res = np.abs(res)        # absolute error per step

        alpha = 1.0 - coverage
        # Per-step, per-channel quantile
        q_high = np.quantile(abs_res, 1.0 - alpha / 2, axis=0)  # (pred_len, C)
        q_low = np.quantile(abs_res, alpha / 2, axis=0)           # (pred_len, C)

        mean = self.predict(X)  # (pred_len, C)
        return {
            "mean": mean,
            "lower": mean - q_high,
            "upper": mean + q_high,
            "half_width": q_high,
        }

    def forecast(self, X, steps: int) -> np.ndarray:
        """Auto-regressive forecast for *steps* steps, any horizon length.

        When ``steps <= pred_len``, this is equivalent to :meth:`predict`
        trimmed to the first *steps* values.  When ``steps > pred_len``, the
        method chains multiple calls: each predicted window is appended to the
        context and used as input for the next call.

        Parameters
        ----------
        X:
            Context window(s) of shape ``(seq_len, C)`` or ``(N, C)`` where
            N > seq_len (last seq_len rows are used).
        steps:
            Total number of future timesteps to produce.

        Returns
        -------
        np.ndarray
            Shape ``(steps, C)`` — concatenated forecast.

        Notes
        -----
        Chaining introduces **compounding errors** for long horizons: the
        predicted values (potentially noisy) are fed back as inputs.  For
        horizons much larger than ``pred_len`` this degrades accuracy.
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        if len(X_np) > self.seq_len:
            X_np = X_np[-self.seq_len:]

        accumulated: List[np.ndarray] = []
        context = X_np.copy()
        remaining = steps

        while remaining > 0:
            pred = self.predict(context)  # (pred_len, C)
            take = min(remaining, self.pred_len)
            accumulated.append(pred[:take])
            remaining -= take
            if remaining > 0:
                # Extend context with the newly predicted block
                context = np.concatenate([context, pred], axis=0)[-self.seq_len:]

        return np.concatenate(accumulated, axis=0)

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

    def reset(self) -> "Forecaster":
        """Un-fit the forecaster in-place.

        Clears the fitted model, scaler, encoder input size, and training
        history.  The hyperparameters are unchanged, so the forecaster can be
        immediately re-fitted with :meth:`fit`.

        Returns
        -------
        self

        Examples
        --------
        >>> fc.fit(X)
        >>> fc.reset()  # back to unfitted state
        >>> "not fitted" in repr(fc)
        True
        """
        self._model = None
        self._scaler = None
        self._enc_in = None
        self.history_ = []
        return self

    def add_callback(self, fn) -> "Forecaster":
        """Register a callback to be called after each training epoch.

        The callback receives ``(forecaster, epoch_dict)`` where *epoch_dict*
        contains at least ``epoch``, ``train_loss``, and ``val_loss``.

        Parameters
        ----------
        fn:
            Callable with signature ``fn(forecaster, epoch_dict) -> None``.

        Returns
        -------
        self
        """
        if not callable(fn):
            raise TypeError(f"callback must be callable; got {type(fn).__name__}.")
        self.callbacks.append(fn)
        return self

    def remove_callback(self, fn) -> "Forecaster":
        """Unregister a previously added callback.

        Parameters
        ----------
        fn:
            The exact callable passed to :meth:`add_callback`.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If *fn* is not registered.
        """
        try:
            self.callbacks.remove(fn)
        except ValueError:
            raise ValueError(
                "Callback not found in registered callbacks.  "
                "Pass the same object that was given to add_callback()."
            )
        return self

    def inspect_layers(self) -> str:
        """Return a formatted string describing the model's layer structure.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Multi-line string with layer names, types, and parameter counts.
            Each line is: ``<layer_name>  <type>  <n_params> params``

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_fitted()
        lines = [f"Model: {type(self._model).__name__}  ({self.n_parameters:,} total params)"]
        lines.append("─" * 60)
        for name, module in self._model.named_modules():
            if name == "":
                continue
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            lines.append(f"  {name:<40}  {type(module).__name__:<20}  {n_params:>8,}")
        return "\n".join(lines)

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

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of hyperparameters.

        Only includes parameters that are JSON-serializable (strings, numbers,
        booleans, and None).  Non-serializable params (``callbacks``,
        ``augmentation``, ``loss`` as nn.Module) are omitted.

        Returns
        -------
        dict
            Flat dict suitable for ``json.dumps()``.

        Examples
        --------
        >>> import json
        >>> cfg = fc.to_dict()
        >>> json.dumps(cfg)  # round-trip friendly
        >>> fc2 = Forecaster.from_dict(cfg)
        """
        _json_safe_types = (str, int, float, bool, type(None))
        result: dict = {}
        for k, v in self.get_params().items():
            if isinstance(v, _json_safe_types):
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "Forecaster":
        """Construct a :class:`Forecaster` from a dict (inverse of :meth:`to_dict`).

        Equivalent to :meth:`from_config` but named to pair with :meth:`to_dict`.

        Parameters
        ----------
        d:
            A dict as returned by :meth:`to_dict` or any flat config dict.

        Returns
        -------
        Forecaster
        """
        return cls.from_config(d)

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

    def explain(
        self,
        X,
        *,
        n_repeats: int = 3,
        seasonal_period: int = 1,
        channel_names: Optional[List[str]] = None,
    ) -> str:
        """Return a comprehensive diagnostic report for this forecaster.

        Combines the output of :meth:`summary`, :meth:`evaluate`,
        :meth:`score_per_channel`, :meth:`feature_importance`, and
        :meth:`timestep_importance` into a single human-readable string.

        Parameters
        ----------
        X:
            Evaluation time series, shape ``(N, C)``.
        n_repeats:
            Permutation repeats for importance methods (default 3 for speed).
        seasonal_period:
            Passed to :meth:`evaluate` for MASE computation.
        channel_names:
            Optional channel labels for display.

        Returns
        -------
        str
            Multi-section diagnostic report.
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        _, C = X_np.shape
        ch_labels = channel_names or [f"ch{i}" for i in range(C)]

        lines = ["=" * 60, "Forecaster Diagnostic Report", "=" * 60]

        # Model summary
        lines.append("")
        lines.append("─── Model ───")
        for ln in self.summary().splitlines():
            lines.append("  " + ln)

        # Overall metrics
        lines.append("")
        lines.append("─── Evaluation metrics ───")
        try:
            metrics = self.evaluate(X_np, seasonal_period=seasonal_period)
            for k, v in metrics.items():
                lines.append(f"  {k:10s}: {v:.6f}")
        except Exception as e:
            lines.append(f"  (evaluate failed: {e})")

        # Per-channel metrics
        lines.append("")
        lines.append("─── Per-channel MSE ───")
        try:
            ch_scores = self.score_per_channel(X_np)
            for i, label in enumerate(ch_labels):
                lines.append(f"  {label:12s}: {ch_scores['mse'][i]:.6f}")
        except Exception as e:
            lines.append(f"  (score_per_channel failed: {e})")

        # Channel importance
        lines.append("")
        lines.append("─── Channel importance (MSE increase on shuffle) ───")
        try:
            fi = self.feature_importance(X_np, n_repeats=n_repeats, random_state=0)
            ranked = np.argsort(fi["importances_mean"])[::-1]
            for idx in ranked:
                label = ch_labels[idx]
                mean = fi["importances_mean"][idx]
                std = fi["importances_std"][idx]
                lines.append(f"  {label:12s}: {mean:+.6f} ± {std:.6f}")
        except Exception as e:
            lines.append(f"  (feature_importance failed: {e})")

        # Timestep importance summary (top 3 and bottom 3)
        lines.append("")
        lines.append("─── Timestep importance (top/bottom 3 positions) ───")
        try:
            ti = self.timestep_importance(X_np, n_repeats=n_repeats, random_state=0)
            imp = ti["importances_mean"]
            top3 = np.argsort(imp)[::-1][:3]
            bot3 = np.argsort(imp)[:3]
            lines.append("  Most important:")
            for t in top3:
                lines.append(f"    t={t:3d}: {imp[t]:+.6f}")
            lines.append("  Least important:")
            for t in bot3:
                lines.append(f"    t={t:3d}: {imp[t]:+.6f}")
        except Exception as e:
            lines.append(f"  (timestep_importance failed: {e})")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(
        self,
        X,
        directory: str,
        *,
        n_repeats: int = 3,
        seasonal_period: int = 1,
        channel_names: Optional[List[str]] = None,
        save_plots: bool = True,
    ) -> str:
        """Save a diagnostic report and optional plots to *directory*.

        Writes:
        - ``report.txt``: output of :meth:`explain`
        - ``history.png`` (if fitted and matplotlib available): loss curves
        - ``residuals.png`` (if matplotlib available): residual histogram

        Parameters
        ----------
        X:
            Evaluation data, shape ``(N, C)``.
        directory:
            Target directory path (created if it doesn't exist).
        n_repeats:
            Permutation repeats passed to :meth:`explain`.
        seasonal_period:
            Passed to :meth:`explain`.
        channel_names:
            Optional channel labels.
        save_plots:
            Whether to save matplotlib plots (default ``True``).

        Returns
        -------
        str
            Path to the ``report.txt`` file.
        """
        import os
        os.makedirs(directory, exist_ok=True)

        report_text = self.explain(
            X, n_repeats=n_repeats, seasonal_period=seasonal_period,
            channel_names=channel_names,
        )
        report_path = os.path.join(directory, "report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        if save_plots:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                if self.history_:
                    ax = self.plot_history()
                    ax.get_figure().savefig(
                        os.path.join(directory, "history.png"), dpi=100, bbox_inches="tight"
                    )
                    plt.close("all")

                ax = self.plot_residuals(X)
                ax.get_figure().savefig(
                    os.path.join(directory, "residuals.png"), dpi=100, bbox_inches="tight"
                )
                plt.close("all")
            except ImportError:
                pass  # matplotlib not installed; skip plots

        return report_path

    def hyperparameter_sensitivity(
        self,
        X,
        param_ranges: Dict[str, List],
        *,
        val_split: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, List[Dict]]:
        """One-at-a-time sensitivity analysis over scalar hyperparameters.

        For each parameter in *param_ranges*, trains clones at each candidate
        value (holding all other hyperparameters fixed at their current
        settings), and records the final validation loss.

        Parameters
        ----------
        X:
            Training data, shape ``(N, C)``.
        param_ranges:
            Dict mapping hyperparameter name → list of candidate values.
            Example: ``{"lr": [1e-4, 1e-3, 1e-2], "batch_size": [16, 32, 64]}``.
        val_split:
            Val fraction for each clone's fit.
        verbose:
            If ``True``, print progress.

        Returns
        -------
        dict
            Maps each param name to a list of ``{"value": v, "val_loss": f}``
            dicts sorted by value.
        """
        results: Dict[str, List[Dict]] = {}
        for param, values in param_ranges.items():
            param_results = []
            for v in values:
                fc = self.clone()
                fc.set_params(**{param: v})
                if not verbose:
                    fc.verbose = False
                fc.fit(X, val_split=val_split)
                val_loss = fc.history_[-1]["val_loss"] if fc.history_ else float("nan")
                param_results.append({"value": v, "val_loss": val_loss})
                if verbose:
                    print(f"  {param}={v!r}  val_loss={val_loss:.6f}")
            results[param] = sorted(param_results, key=lambda d: d["value"]
                                    if isinstance(d["value"], (int, float)) else 0)
        return results

    def plot_sensitivity(
        self,
        sensitivity_results: Dict[str, List[Dict]],
        *,
        ax=None,
        title: Optional[str] = None,
    ):
        """Plot sensitivity curves from :meth:`hyperparameter_sensitivity`.

        Requires ``matplotlib``.

        Parameters
        ----------
        sensitivity_results:
            The dict returned by :meth:`hyperparameter_sensitivity`.
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_sensitivity(). "
                "Install it with: pip install matplotlib"
            ) from exc

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(8, 4))

        for param, records in sensitivity_results.items():
            xs = [r["value"] for r in records]
            ys = [r["val_loss"] for r in records]
            ax.plot(range(len(xs)), ys, marker="o", label=param)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs], rotation=30)

        ax.set_xlabel("Hyperparameter value")
        ax.set_ylabel("Val loss")
        ax.set_title(title or "Hyperparameter sensitivity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

    def learning_curve(
        self,
        X,
        *,
        train_fractions: Optional[List[float]] = None,
        val_split: float = 0.1,
        metric: str = "mse",
        verbose: bool = False,
    ) -> List[Dict]:
        """Train on increasing fractions of *X* and record performance.

        Reveals whether the model is data-limited (error still dropping at
        large fractions) or capacity-limited (error plateauing early).

        Parameters
        ----------
        X:
            Full training series, shape ``(N, C)``.
        train_fractions:
            List of fractions in ``(0, 1]`` to try.  Defaults to
            ``[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]``.
        val_split:
            Validation fraction held out inside each fraction's training run.
        metric:
            Metric key returned by :meth:`cross_validate` (default ``"mean_mse"``).
            Also ``"mean_mae"``, ``"mean_rmse"``, ``"mean_smape"``.
        verbose:
            Print progress if ``True``.

        Returns
        -------
        list of dict
            Each dict has ``{"fraction": float, "n_samples": int,
            "val_loss": float}`` sorted by fraction.
        """
        fracs = train_fractions or [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N = len(X_np)
        min_samples = self.seq_len + self.pred_len + 1
        results = []
        for frac in fracs:
            n = max(min_samples, int(N * frac))
            n = min(n, N)
            subset = X_np[:n]
            fc = self.clone()
            if not verbose:
                fc.verbose = False
            fc.fit(subset, val_split=val_split)
            val_loss = fc.history_[-1]["val_loss"] if fc.history_ else float("nan")
            if verbose:
                print(f"  fraction={frac:.2f}  n={n}  val_loss={val_loss:.6f}")
            results.append({"fraction": frac, "n_samples": n, "val_loss": val_loss})
        return results

    def plot_learning_curve(
        self,
        learning_curve_results: List[Dict],
        *,
        ax=None,
        title: Optional[str] = None,
    ):
        """Plot output of :meth:`learning_curve`.

        Requires ``matplotlib``.

        Parameters
        ----------
        learning_curve_results:
            List of dicts returned by :meth:`learning_curve`.
        ax:
            Existing ``matplotlib.axes.Axes``.  New figure if ``None``.
        title:
            Axes title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_learning_curve(). "
                "Install it with: pip install matplotlib"
            ) from exc

        ns = [r["n_samples"] for r in learning_curve_results]
        losses = [r["val_loss"] for r in learning_curve_results]

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.plot(ns, losses, marker="o", color="steelblue")
        ax.fill_between(ns, losses, alpha=0.12, color="steelblue")
        model_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        ax.set_title(title or f"{model_name} — learning curve")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Val loss")
        ax.grid(True, alpha=0.3)

        if created_fig:
            plt.tight_layout()
        return ax

    def freeze_layers(self, names: Optional[List[str]] = None) -> "Forecaster":
        """Freeze model parameters (set ``requires_grad=False``).

        Useful for transfer learning: freeze a pre-trained backbone and
        fine-tune only the head layers via :meth:`partial_fit`.

        Parameters
        ----------
        names:
            List of sub-module name prefixes to freeze (e.g.
            ``["encoder"]``).  If ``None``, freezes the entire model.

        Returns
        -------
        self
        """
        self._check_fitted()
        for name, param in self._model.named_parameters():
            if names is None or any(name.startswith(n) for n in names):
                param.requires_grad_(False)
        return self

    def unfreeze_layers(self, names: Optional[List[str]] = None) -> "Forecaster":
        """Unfreeze model parameters (set ``requires_grad=True``).

        Parameters
        ----------
        names:
            List of sub-module name prefixes to unfreeze.  If ``None``,
            unfreezes the entire model.

        Returns
        -------
        self
        """
        self._check_fitted()
        for name, param in self._model.named_parameters():
            if names is None or any(name.startswith(n) for n in names):
                param.requires_grad_(True)
        return self

    def frozen_parameter_count(self) -> int:
        """Return the number of frozen (non-trainable) parameters."""
        self._check_fitted()
        return sum(p.numel() for p in self._model.parameters()
                   if not p.requires_grad)

    def export_predictions(
        self,
        X,
        path: str,
        *,
        channel_names: Optional[List[str]] = None,
        stride: int = 1,
    ) -> str:
        """Export rolling forecast predictions to a CSV file.

        Runs :meth:`predict_rolling` over *X* and writes one row per
        predicted timestep.  The index column is the timestep offset from
        the start of the first prediction window.

        Requires ``pandas``.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)``.
        path:
            Destination file path (``*.csv``).
        channel_names:
            Optional column labels.  Defaults to ``["ch0", "ch1", ...]``.
        stride:
            Stride between context windows (default 1).

        Returns
        -------
        str
            The path the file was written to.
        """
        self._check_fitted()
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for export_predictions(). "
                "Install it with: pip install pandas"
            ) from exc

        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        preds = self.predict_rolling(X_np, stride=stride)  # (W, pred, C)
        W, pred, C = preds.shape
        cols = channel_names if channel_names else [f"ch{i}" for i in range(C)]

        # Flatten: one row per (window, step)
        rows = []
        for w in range(W):
            for t in range(pred):
                row = {"window": w, "step": t}
                for c_i, col in enumerate(cols):
                    row[col] = float(preds[w, t, c_i])
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return path

    def compare_models(
        self,
        models: List[str],
        X_train,
        X_test,
        *,
        include_self: bool = True,
        n_jobs: int = 1,
        print_table: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark *models* using this forecaster's hyperparameters as defaults.

        A convenience wrapper around the module-level :func:`compare` that
        inherits ``seq_len``, ``pred_len``, ``epochs``, ``lr``,
        ``batch_size``, and ``normalize`` from the current instance.

        Parameters
        ----------
        models:
            List of model name strings.  The current model is added
            automatically when *include_self* is ``True``.
        X_train:
            Training data, shape ``(N, C)``.
        X_test:
            Test data, shape ``(M, C)``.
        include_self:
            If ``True`` (default), prepend the current model to *models*.
        n_jobs:
            Parallel workers (default 1).
        print_table:
            Print a comparison table (default ``True``).
        verbose:
            Verbose training output.

        Returns
        -------
        dict
            Same format as :func:`compare`.
        """
        names = list(models)
        self_name = (
            self.model_spec
            if isinstance(self.model_spec, str)
            else type(self.model_spec).__name__
        )
        if include_self and self_name not in names:
            names = [self_name] + names

        return compare(
            names,
            X_train=X_train,
            X_test=X_test,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            normalize=self.normalize,
            n_jobs=n_jobs,
            print_table=print_table,
            verbose=verbose,
        )

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


def make_forecaster(name: str, **kwargs) -> "Forecaster":
    """Convenience factory: create a :class:`Forecaster` by model name.

    Equivalent to ``Forecaster(name, **kwargs)`` but slightly more readable
    in scripts and notebooks.

    Parameters
    ----------
    name:
        A model name from ``torch_timeseries.model.forecasting_models``.
    **kwargs:
        Forwarded to :class:`Forecaster`.

    Returns
    -------
    Forecaster

    Examples
    --------
    >>> from torch_timeseries import make_forecaster
    >>> fc = make_forecaster("DLinear", seq_len=96, pred_len=24, epochs=10)
    >>> fc.fit(X)
    """
    return Forecaster(name, **kwargs)


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


# ──────────────────────────────────────────────────────────────────────────────
# MultiChannelForecaster
# ──────────────────────────────────────────────────────────────────────────────


class MultiChannelForecaster:
    """Train one independent :class:`Forecaster` per channel.

    The entire time series is split into individual univariate channels and
    each channel is modelled independently (channel-independent / CI mode
    at the API level).  This gives each channel its own model weights,
    potentially improving accuracy when channels have very different dynamics.

    Parameters
    ----------
    base:
        A (not-yet-fitted) :class:`Forecaster` used as the per-channel
        template.  Its ``enc_in`` will be forced to 1 for each clone.

    Examples
    --------
    >>> base = Forecaster("DLinear", seq_len=96, pred_len=24, epochs=5)
    >>> mcf = MultiChannelForecaster(base)
    >>> mcf.fit(X_train)         # X_train: (N, C)
    >>> y = mcf.predict(X_train[-96:])   # y: (24, C)
    """

    def __init__(self, base: Forecaster) -> None:
        self.base = base
        self.channel_forecasters_: List[Forecaster] = []
        self._n_channels: Optional[int] = None

    def fit(self, X, *, val_split: float = 0.1) -> "MultiChannelForecaster":
        """Fit one cloned Forecaster per channel.

        Parameters
        ----------
        X:
            Training time series, shape ``(N, C)``.
        val_split:
            Val fraction passed to each channel's :meth:`Forecaster.fit`.

        Returns
        -------
        self
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        N, C = X_np.shape
        self._n_channels = C
        self.channel_forecasters_ = []
        for c in range(C):
            fc = self.base.clone()
            fc.model_kwargs = {**fc.model_kwargs}  # shallow copy
            fc.fit(X_np[:, c : c + 1], val_split=val_split)
            self.channel_forecasters_.append(fc)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict all channels from a context window.

        Parameters
        ----------
        X:
            Context, shape ``(seq_len, C)`` or ``(N, C)`` with N > seq_len.

        Returns
        -------
        np.ndarray
            Shape ``(pred_len, C)``.
        """
        if not self.channel_forecasters_:
            raise RuntimeError("MultiChannelForecaster not fitted.  Call fit() first.")
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        if len(X_np) > self.base.seq_len:
            X_np = X_np[-self.base.seq_len:]
        preds = []
        for c, fc in enumerate(self.channel_forecasters_):
            y_c = fc.predict(X_np[:, c : c + 1])  # (pred_len, 1)
            preds.append(y_c)
        return np.concatenate(preds, axis=1)  # (pred_len, C)

    def score(self, X) -> Dict[str, float]:
        """Evaluate the per-channel ensemble on *X*."""
        if not self.channel_forecasters_:
            raise RuntimeError("MultiChannelForecaster not fitted.  Call fit() first.")
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        seq = self.base.seq_len
        pred = self.base.pred_len
        min_len = seq + pred
        if len(X_np) < min_len:
            raise ValueError(
                f"X has only {len(X_np)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X_np) - seq - pred + 1
        windows = np.stack([X_np[i : i + seq] for i in range(n_windows)])
        truths = np.stack([X_np[i + seq : i + seq + pred] for i in range(n_windows)])
        preds = np.stack([self.predict(w) for w in windows])
        diff = preds - truths
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        denom = np.abs(preds) + np.abs(truths) + 1e-8
        smape = float((2.0 * np.abs(diff) / denom * 100.0).mean())
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)), "smape": smape}

    def __repr__(self) -> str:
        n = len(self.channel_forecasters_)
        status = f"{n} channels fitted" if n > 0 else "not fitted"
        return f"MultiChannelForecaster(base={self.base!r}, [{status}])"


# ──────────────────────────────────────────────────────────────────────────────
# EnsembleForecaster
# ──────────────────────────────────────────────────────────────────────────────


class EnsembleForecaster:
    """Average predictions from a list of heterogeneous :class:`Forecaster` objects.

    Unlike :class:`BaggingForecaster` (which clones *one* model on bootstrap
    sub-samples), ``EnsembleForecaster`` accepts *different* already-configured
    Forecasters and combines their predictions via a weighted average.  All
    member forecasters must share the same ``seq_len`` and ``pred_len``.

    Parameters
    ----------
    forecasters:
        List of (name, :class:`Forecaster`) tuples, or plain list of
        :class:`Forecaster` objects.  Mixed model architectures are fine.
    weights:
        Optional per-forecaster weights for the weighted average.  Must
        sum to a positive value.  If ``None``, uniform weights are used.

    Examples
    --------
    >>> f1 = Forecaster("DLinear",  seq_len=96, pred_len=24, epochs=5)
    >>> f2 = Forecaster("NLinear",  seq_len=96, pred_len=24, epochs=5)
    >>> f3 = Forecaster("PatchTST", seq_len=96, pred_len=24, epochs=5)
    >>> ens = EnsembleForecaster([f1, f2, f3])
    >>> ens.fit(X_train)
    >>> y = ens.predict(X_train[-96:])   # shape (24, C)
    """

    def __init__(
        self,
        forecasters,
        weights: Optional[List[float]] = None,
    ) -> None:
        # Normalise: accept both [(name, fc), ...] and [fc, ...]
        if forecasters and isinstance(forecasters[0], tuple):
            self.named_forecasters: List[Tuple[str, Forecaster]] = list(forecasters)
        else:
            self.named_forecasters = [
                (f"fc_{i}", fc) for i, fc in enumerate(forecasters)
            ]
        self.weights = weights
        self._fitted = False

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def forecasters_(self) -> List[Forecaster]:
        return [fc for _, fc in self.named_forecasters]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "EnsembleForecaster not fitted.  Call fit() first."
            )

    def _norm_weights(self) -> np.ndarray:
        n = len(self.named_forecasters)
        if self.weights is None:
            return np.ones(n) / n
        w = np.asarray(self.weights, dtype=float)
        if len(w) != n:
            raise ValueError(
                f"weights length {len(w)} != number of forecasters {n}."
            )
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        return w / total

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X, *, val_split: float = 0.1) -> "EnsembleForecaster":
        """Fit all member forecasters on *X*.

        Parameters
        ----------
        X:
            Training series, shape ``(N, C)``.
        val_split:
            Validation fraction forwarded to each member's
            :meth:`Forecaster.fit`.

        Returns
        -------
        self
        """
        if not self.named_forecasters:
            raise ValueError("EnsembleForecaster needs at least one forecaster.")
        for _, fc in self.named_forecasters:
            fc.fit(X, val_split=val_split)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Weighted-average prediction from all members.

        Parameters
        ----------
        X:
            Context window, shape ``(seq_len, C)`` or longer.

        Returns
        -------
        np.ndarray
            Shape ``(pred_len, C)``.
        """
        self._check_fitted()
        w = self._norm_weights()
        preds = np.stack(
            [fc.predict(X) for fc in self.forecasters_], axis=0
        )  # (n_members, pred_len, C)
        return (preds * w[:, None, None]).sum(axis=0)

    def predict_std(self, X) -> Dict[str, np.ndarray]:
        """Return mean, std, lower (5th pct), upper (95th pct) across members.

        Parameters
        ----------
        X:
            Context window, shape ``(seq_len, C)`` or longer.

        Returns
        -------
        dict with keys ``mean``, ``std``, ``lower``, ``upper``.
        """
        self._check_fitted()
        preds = np.stack(
            [fc.predict(X) for fc in self.forecasters_], axis=0
        )  # (n, pred_len, C)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return {
            "mean": mean,
            "std": std,
            "lower": np.percentile(preds, 5, axis=0),
            "upper": np.percentile(preds, 95, axis=0),
        }

    def score(self, X) -> Dict[str, float]:
        """Evaluate the ensemble on *X*."""
        self._check_fitted()
        # Use the first member's seq_len/pred_len as reference
        ref = self.forecasters_[0]
        seq, pred = ref.seq_len, ref.pred_len
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        min_len = seq + pred
        if len(X_np) < min_len:
            raise ValueError(
                f"X has only {len(X_np)} timesteps; need at least "
                f"seq_len + pred_len = {min_len}."
            )
        n_windows = len(X_np) - seq - pred + 1
        truths = np.stack([X_np[i + seq : i + seq + pred] for i in range(n_windows)])
        preds = np.stack([self.predict(X_np[i : i + seq]) for i in range(n_windows)])
        diff = preds - truths
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        denom = np.abs(preds) + np.abs(truths) + 1e-8
        smape = float((2.0 * np.abs(diff) / denom * 100.0).mean())
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)), "smape": smape}

    def __repr__(self) -> str:
        names = [n for n, _ in self.named_forecasters]
        status = "fitted" if self._fitted else "not fitted"
        return f"EnsembleForecaster([{', '.join(names)}], [{status}])"


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


def compare_plot(
    results: Dict[str, Dict[str, float]],
    metric: str = "mse",
    *,
    top_n: Optional[int] = None,
    ax=None,
    title: Optional[str] = None,
    color: str = "steelblue",
    highlight_best: bool = True,
):
    """Bar chart of :func:`compare` results sorted by *metric*.

    Requires ``matplotlib``.

    Parameters
    ----------
    results:
        Dict returned by :func:`compare`.
    metric:
        Metric column to plot (default ``"mse"``).
    top_n:
        Only show the best *top_n* models.  ``None`` shows all.
    ax:
        Existing ``matplotlib.axes.Axes``.  If ``None``, a new figure is
        created.
    title:
        Axes title.  Defaults to ``"Model comparison — <metric>"``.
    color:
        Bar colour.
    highlight_best:
        If ``True``, colour the best bar in orange.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for compare_plot(). "
            "Install it with: pip install matplotlib"
        ) from exc

    if not results:
        raise ValueError("results dict is empty.")

    # Build sorted list
    items = [
        (name, metrics.get(metric, float("nan")))
        for name, metrics in results.items()
        if not isinstance(metrics, Exception)
    ]
    items.sort(key=lambda x: x[1])
    if top_n is not None:
        items = items[:top_n]

    names = [i[0] for i in items]
    values = [i[1] for i in items]

    colors = [color] * len(names)
    if highlight_best and names:
        colors[0] = "tomato"

    created_fig = ax is None
    if created_fig:
        fig_w = max(6, len(names) * 0.6 + 2)
        _, ax = plt.subplots(figsize=(fig_w, 4))

    ax.bar(names, values, color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Model comparison — {metric.upper()}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    if created_fig:
        plt.tight_layout()
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# SklearnForecaster — scikit-learn compatible wrapper
# ──────────────────────────────────────────────────────────────────────────────


class SklearnForecaster:
    """Scikit-learn compatible wrapper for :class:`Forecaster`.

    Adapts the Forecaster API to the ``BaseEstimator`` / ``RegressorMixin``
    interface so it can be used with ``sklearn.model_selection.GridSearchCV``,
    ``cross_val_score``, and ``Pipeline``.

    sklearn's ``fit(X, y)`` convention maps to forecasting as follows:

    - ``X``: shape ``(N, seq_len * C)`` — flattened context windows.
    - ``y``: shape ``(N, pred_len * C)`` — flattened target windows.

    Alternatively, pass a 2D time series ``X_ts`` of shape ``(T, C)`` to
    :meth:`fit_ts` for the standard sliding-window workflow.

    Parameters
    ----------
    model:
        Model name string.
    seq_len, pred_len, epochs, lr, batch_size, normalize, verbose:
        Forwarded to :class:`Forecaster`.
    **model_kwargs:
        Extra model constructor arguments.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> sk = SklearnForecaster("DLinear", seq_len=96, pred_len=24, epochs=5)
    >>> # Prepare X/y via time_series_split + manual windowing, then:
    >>> sk.fit(X_windows, y_windows)
    >>> sk.predict(X_new)
    """

    def __init__(
        self,
        model: str = "DLinear",
        *,
        seq_len: int = 96,
        pred_len: int = 24,
        epochs: int = 20,
        lr: float = 1e-4,
        batch_size: int = 32,
        normalize: bool = True,
        verbose: bool = False,
        **model_kwargs,
    ) -> None:
        self.model = model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.normalize = normalize
        self.verbose = verbose
        self.model_kwargs = model_kwargs
        self._fc: Optional[Forecaster] = None

    # ── sklearn interface ─────────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Return parameters for sklearn's clone / GridSearchCV."""
        params = {
            "model": self.model,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "verbose": self.verbose,
        }
        params.update(self.model_kwargs)
        return params

    def set_params(self, **params) -> "SklearnForecaster":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.model_kwargs[k] = v
        self._fc = None  # invalidate fitted state
        return self

    def fit(self, X, y=None) -> "SklearnForecaster":
        """Fit from pre-windowed arrays.

        Parameters
        ----------
        X:
            ``(N, seq_len * C)`` or ``(N, seq_len, C)`` — context windows.
        y:
            Ignored (the model learns to forecast from X to X's continuation).
            The convention here is that the model is fitted on the raw context
            windows; for proper supervised fit, use :meth:`fit_ts`.

        Returns
        -------
        self
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            # Assume (N, seq_len) univariate or (N, seq_len * C) flattened
            N = X_arr.shape[0]
            X_arr = X_arr.reshape(N, self.seq_len, -1)
        # Flatten windows into a pseudo-series for the Forecaster
        N, T, C = X_arr.shape
        pseudo = X_arr.reshape(N * T, C)
        self._fc = Forecaster(
            self.model,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            normalize=self.normalize,
            verbose=self.verbose,
            **self.model_kwargs,
        )
        self._fc.fit(pseudo)
        return self

    def fit_ts(self, X_ts, *, val_split: float = 0.1) -> "SklearnForecaster":
        """Fit directly on a raw time series (the natural use case).

        Parameters
        ----------
        X_ts:
            Raw time series, shape ``(T, C)`` or ``(T,)``.

        Returns
        -------
        self
        """
        self._fc = Forecaster(
            self.model,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            normalize=self.normalize,
            verbose=self.verbose,
            **self.model_kwargs,
        )
        self._fc.fit(X_ts, val_split=val_split)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict from context windows.

        Parameters
        ----------
        X:
            ``(N, seq_len * C)`` or ``(N, seq_len, C)`` context windows,
            or a single ``(seq_len, C)`` window.

        Returns
        -------
        np.ndarray
            Shape ``(N, pred_len * C)`` — flattened predictions.
        """
        if self._fc is None:
            raise RuntimeError("SklearnForecaster not fitted.  Call fit() first.")
        X_arr = np.asarray(X)
        single = X_arr.ndim <= 2 and X_arr.shape[0] == self.seq_len
        if single:
            X_arr = X_arr[None]
        if X_arr.ndim == 2:
            N = X_arr.shape[0]
            X_arr = X_arr.reshape(N, self.seq_len, -1)
        N, T, C = X_arr.shape
        preds = self._fc.predict(X_arr)  # (N, pred_len, C)
        return preds.reshape(N, -1)

    def score(self, X, y=None) -> float:
        """Return negative MSE (for sklearn's maximise-score convention)."""
        if self._fc is None:
            raise RuntimeError("SklearnForecaster not fitted.  Call fit() first.")
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            N = X_arr.shape[0]
            X_arr = X_arr.reshape(N, self.seq_len, -1)
        preds = self._fc.predict(X_arr)
        if y is not None:
            y_arr = np.asarray(y).reshape(preds.shape[0], self.pred_len, -1)
            mse = float(((preds - y_arr) ** 2).mean())
        else:
            mse = float(((preds - preds) ** 2).mean())  # trivial fallback
        return -mse

    @property
    def forecaster_(self) -> Optional[Forecaster]:
        """The underlying fitted :class:`Forecaster`."""
        return self._fc

    def __repr__(self) -> str:
        status = "fitted" if self._fc is not None else "not fitted"
        return (
            f"SklearnForecaster(model={self.model!r}, "
            f"seq_len={self.seq_len}, pred_len={self.pred_len}, [{status}])"
        )
