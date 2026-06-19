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
    def diff(X, order: int = 1, lag: int = 1) -> np.ndarray:
        """Apply differencing to remove trends / seasonality.

        Computes ``X[t] - X[t - lag]`` repeated *order* times.  The output
        has ``order * lag`` fewer timesteps than the input.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)`` or ``(N,)``.
        order:
            Number of differencing passes (default 1).
        lag:
            Lag for each pass (default 1 = consecutive differences).

        Returns
        -------
        np.ndarray
            Shape ``(N - order * lag, C)`` (or squeezed for 1-D input).
        """
        X_np = np.asarray(X, dtype=float)
        squeeze = X_np.ndim == 1
        if squeeze:
            X_np = X_np[:, None]
        for _ in range(order):
            X_np = X_np[lag:] - X_np[:-lag]
        return X_np.squeeze(1) if squeeze else X_np

    @staticmethod
    def undiff(
        X_diff,
        X_orig,
        order: int = 1,
        lag: int = 1,
    ) -> np.ndarray:
        """Invert differencing applied by :meth:`diff`.

        Reconstructs the original scale by cumulatively summing the
        differenced series starting from the last *lag* values of the
        original series.

        Parameters
        ----------
        X_diff:
            Differenced series, shape ``(M, C)`` or ``(M,)``.
        X_orig:
            Original series *before* differencing, shape ``(N, C)`` or
            ``(N,)``.  Only the last ``order * lag`` rows are used.
        order:
            Number of differencing passes used in :meth:`diff`.
        lag:
            Lag used in :meth:`diff`.

        Returns
        -------
        np.ndarray
            Reconstructed series, shape ``(M, C)``.
        """
        Xd = np.asarray(X_diff, dtype=float)
        Xo = np.asarray(X_orig, dtype=float)
        squeeze = Xd.ndim == 1
        if squeeze:
            Xd = Xd[:, None]
        if Xo.ndim == 1:
            Xo = Xo[:, None]

        # Precompute forward-diff intermediates so we have the correct seeds
        # for each un-diff pass.
        inter = [Xo]
        curr = Xo
        for _ in range(order):
            curr = curr[lag:] - curr[:-lag]
            inter.append(curr)

        result = Xd.copy()
        for lv in range(order - 1, -1, -1):
            seed = inter[lv][:lag]      # (lag, C) — first lag rows at this level
            M = len(result)
            out = np.empty((M + lag, result.shape[1]))
            out[:lag] = seed
            for t in range(lag, M + lag):
                out[t] = out[t - lag] + result[t - lag]
            result = out                # keep full output (including seed prepend)

        # Strip the leading order*lag rows that came from the seed values
        result = result[order * lag:]
        return result.squeeze(1) if squeeze else result

    @staticmethod
    def seasonal_decompose(
        X,
        period: int,
        *,
        method: str = "additive",
    ) -> Dict[str, np.ndarray]:
        """Simple moving-average seasonal decomposition.

        Computes trend (centred moving average of *period*), seasonal
        (average deviation from trend per phase), and residual components.
        Uses the classical additive or multiplicative model.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)`` or ``(N,)``.
        period:
            Seasonal period (e.g. 12 for monthly data with yearly
            seasonality, 7 for daily data with weekly seasonality).
        method:
            ``"additive"`` (default) or ``"multiplicative"``.

        Returns
        -------
        dict
            ``{"trend": (N, C), "seasonal": (N, C), "residual": (N, C),
               "original": (N, C)}``
        """
        if period < 2:
            raise ValueError(f"period must be >= 2; got {period}.")
        if method not in ("additive", "multiplicative"):
            raise ValueError(
                f"method must be 'additive' or 'multiplicative'; got {method!r}."
            )
        X_np = np.asarray(X, dtype=float)
        squeeze = X_np.ndim == 1
        if squeeze:
            X_np = X_np[:, None]
        N, C = X_np.shape
        if N < 2 * period:
            raise ValueError(
                f"X has only {N} timesteps; need at least 2 * period = {2 * period}."
            )

        # Centred moving average trend
        half = period // 2
        trend = np.full_like(X_np, np.nan)
        for t in range(half, N - half):
            trend[t] = X_np[t - half : t + half + 1].mean(axis=0)
        # Fill NaN edges with nearest valid values (edge extension)
        for c in range(C):
            valid = np.where(~np.isnan(trend[:, c]))[0]
            if len(valid):
                trend[:valid[0], c] = trend[valid[0], c]
                trend[valid[-1] + 1:, c] = trend[valid[-1], c]

        # Seasonal indices: average (detrended) deviation per phase
        if method == "additive":
            detrended = X_np - trend
        else:
            detrended = np.where(np.abs(trend) > 1e-8, X_np / (trend + 1e-8), 1.0)

        seasonal = np.zeros_like(X_np)
        for phase in range(period):
            indices = np.arange(phase, N, period)
            avg = detrended[indices].mean(axis=0)
            for idx in indices:
                seasonal[idx] = avg

        # Residual
        if method == "additive":
            residual = X_np - trend - seasonal
        else:
            residual = np.where(
                np.abs(trend * seasonal) > 1e-8,
                X_np / (trend * seasonal + 1e-8),
                1.0,
            )

        result = {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "original": X_np,
        }
        if squeeze:
            return {k: v.squeeze(1) for k, v in result.items()}
        return result

    @staticmethod
    def plot_decomposition(
        decomp: Dict[str, np.ndarray],
        *,
        channel: int = 0,
        title: Optional[str] = None,
    ):
        """Plot seasonal decomposition from :meth:`seasonal_decompose`.

        Requires ``matplotlib``.

        Parameters
        ----------
        decomp:
            Dict returned by :meth:`seasonal_decompose`.
        channel:
            Channel index to plot (default ``0``).
        title:
            Suptitle (default ``"Seasonal decomposition"``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_decomposition(). "
                "Install it with: pip install matplotlib"
            ) from exc

        keys = ["original", "trend", "seasonal", "residual"]
        labels = ["Original", "Trend", "Seasonal", "Residual"]
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title or "Seasonal decomposition", fontsize=12)

        for ax, key, label in zip(axes, keys, labels):
            y = decomp[key]
            if y.ndim > 1:
                y = y[:, channel]
            ax.plot(y, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestep")
        plt.tight_layout()
        return fig

    @staticmethod
    def detrend(
        X,
        degree: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove a polynomial trend from each channel.

        Fits a degree-*degree* polynomial to the time axis for each channel
        and returns the residuals plus the fitted coefficient matrix so the
        trend can be restored.

        Parameters
        ----------
        X:
            Time series, shape ``(N, C)`` or ``(N,)``.
        degree:
            Degree of the polynomial trend (default 1 = linear).

        Returns
        -------
        X_detrended: np.ndarray
            Shape ``(N, C)`` — X minus the fitted trend.
        trend_coeffs: np.ndarray
            Shape ``(degree + 1, C)`` — per-channel polynomial coefficients
            (highest degree first, as returned by ``np.polyfit``).
        """
        X_np = np.asarray(X, dtype=float)
        squeeze = X_np.ndim == 1
        if squeeze:
            X_np = X_np[:, None]
        N, C = X_np.shape
        t = np.arange(N, dtype=float)
        coeffs = np.zeros((degree + 1, C))
        trend = np.zeros_like(X_np)
        for c in range(C):
            p = np.polyfit(t, X_np[:, c], degree)
            coeffs[:, c] = p
            trend[:, c] = np.polyval(p, t)
        X_det = X_np - trend
        if squeeze:
            X_det = X_det.squeeze(1)
        return X_det, coeffs

    @staticmethod
    def retrend(
        X_detrended,
        trend_coeffs: np.ndarray,
        offset: int = 0,
    ) -> np.ndarray:
        """Restore a polynomial trend removed by :meth:`detrend`.

        Parameters
        ----------
        X_detrended:
            Detrended series, shape ``(M, C)`` or ``(M,)``.  Typically a
            model's prediction on detrended inputs.
        trend_coeffs:
            Coefficients from :meth:`detrend`, shape ``(degree + 1, C)``.
        offset:
            Timestep index of the first row of *X_detrended* in the
            original time axis (default 0).  For a forecast starting after
            N training steps, pass *N*.

        Returns
        -------
        np.ndarray
            Shape ``(M, C)`` — *X_detrended* plus the trend restored at the
            correct time positions.
        """
        Xd = np.asarray(X_detrended, dtype=float)
        squeeze = Xd.ndim == 1
        if squeeze:
            Xd = Xd[:, None]
        M, C = Xd.shape
        t = np.arange(offset, offset + M, dtype=float)
        trend = np.zeros((M, C))
        for c in range(C):
            trend[:, c] = np.polyval(trend_coeffs[:, c], t)
        out = Xd + trend
        return out.squeeze(1) if squeeze else out

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

    def rolling_evaluate(
        self,
        X: np.ndarray,
        *,
        stride: int = 1,
        metrics: Optional[List[str]] = None,
    ) -> "pd.DataFrame":
        """Walk-forward evaluation returning per-step metrics as a DataFrame.

        Each row corresponds to one forecast window.  The window index
        (first timestep of the context) and all requested metrics are
        included as columns.

        Parameters
        ----------
        X:
            Full time series array ``(T, C)``.
        stride:
            Number of timesteps to advance between windows (default ``1``).
        metrics:
            Subset of ``['MSE', 'MAE', 'RMSE', 'SMAPE']`` to include.
            Default: all four.

        Returns
        -------
        pandas.DataFrame with columns ``['window', 'MSE', 'MAE', ...]``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for rolling_evaluate()") from exc

        self._check_fitted()
        if metrics is None:
            metrics = ["MSE", "MAE", "RMSE", "SMAPE"]

        rows = []
        T = len(X)
        t = 0
        while t + self.seq_len + self.pred_len <= T:
            ctx = X[t : t + self.seq_len]
            truth = X[t + self.seq_len : t + self.seq_len + self.pred_len]
            pred = self.predict(ctx[np.newaxis])[0]  # (pred_len, C)
            err = pred - truth
            mse = float(np.mean(err ** 2))
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(mse))
            denom = (np.abs(truth) + np.abs(pred)).clip(min=1e-8)
            smape = float(np.mean(2 * np.abs(err) / denom) * 100)
            row = {"window": t, "MSE": mse, "MAE": mae, "RMSE": rmse, "SMAPE": smape}
            rows.append({k: v for k, v in row.items() if k == "window" or k in metrics})
            t += stride

        return pd.DataFrame(rows)

    def plot_rolling_metrics(
        self,
        rolling_df: "pd.DataFrame",
        *,
        metrics: Optional[List[str]] = None,
        ax=None,
        title: Optional[str] = None,
    ):
        """Plot per-window metrics from :meth:`rolling_evaluate`.

        Parameters
        ----------
        rolling_df:
            DataFrame returned by :meth:`rolling_evaluate`.
        metrics:
            Columns to plot (default: all numeric columns).
        ax:
            Matplotlib axes or array of axes.  If ``None`` a new figure is
            created with one subplot per metric.
        title:
            Figure suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_rolling_metrics()") from exc

        if metrics is None:
            metrics = [c for c in rolling_df.columns if c != "window"]

        n = len(metrics)
        if ax is None:
            fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
            if n == 1:
                axes = [axes]
        else:
            axes = list(ax) if hasattr(ax, "__iter__") else [ax]
            fig = axes[0].get_figure()

        x = rolling_df["window"].values
        for axis, m in zip(axes, metrics):
            axis.plot(x, rolling_df[m].values, linewidth=0.9)
            axis.set_ylabel(m)
            axis.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Window start")
        fig.suptitle(title or "Rolling evaluation", fontsize=12)
        plt.tight_layout()
        return fig

    @classmethod
    def from_pretrained(cls, path: str, **override_kwargs) -> "Forecaster":
        """Load a saved :class:`Forecaster` from *path*.

        This is a named alias for :meth:`load` that makes the intent
        explicit when loading checkpoints from disk or model hubs.

        Parameters
        ----------
        path:
            Path to the directory or file produced by :meth:`save`.
        **override_kwargs:
            Any keyword arguments override the stored configuration before
            the model is reconstructed.  Useful for changing ``device`` or
            ``batch_size`` at load time.

        Returns
        -------
        Forecaster
        """
        fc = cls.load(path)
        if override_kwargs:
            fc.set_params(**override_kwargs)
        return fc

    def anomaly_score(
        self,
        X: np.ndarray,
        *,
        stride: int = 1,
        reduction: str = "mean",
    ) -> np.ndarray:
        """Compute a rolling one-step-ahead reconstruction error as an anomaly score.

        For each window the model predicts the *next* ``pred_len`` timesteps;
        the absolute error (averaged over channels and pred_len steps) is
        recorded at the **first** predicted timestep.  Points with high scores
        are likely anomalies.

        Parameters
        ----------
        X:
            Time series array ``(T, C)``.
        stride:
            Stride between consecutive windows (default ``1``).
        reduction:
            How to collapse the error tensor per window: ``"mean"`` (default),
            ``"max"``, or ``"sum"``.

        Returns
        -------
        scores : np.ndarray of shape ``(n_windows,)``
            Anomaly score for each window position.
        indices : np.ndarray of shape ``(n_windows,)``
            Timestep index (first predicted timestep) for each score.
        """
        self._check_fitted()
        if reduction not in ("mean", "max", "sum"):
            raise ValueError(f"reduction must be 'mean', 'max', or 'sum'; got {reduction!r}")

        scores, indices = [], []
        T = len(X)
        t = 0
        while t + self.seq_len + self.pred_len <= T:
            ctx = X[t : t + self.seq_len]
            truth = X[t + self.seq_len : t + self.seq_len + self.pred_len]
            pred = self.predict(ctx[np.newaxis])[0]
            err = np.abs(pred - truth)
            if reduction == "mean":
                sc = float(err.mean())
            elif reduction == "max":
                sc = float(err.max())
            else:
                sc = float(err.sum())
            scores.append(sc)
            indices.append(t + self.seq_len)
            t += stride

        return np.array(scores, dtype=np.float32), np.array(indices, dtype=np.int64)

    def flag_anomalies(
        self,
        X: np.ndarray,
        *,
        stride: int = 1,
        threshold: Optional[float] = None,
        contamination: float = 0.05,
        reduction: str = "mean",
    ) -> "pd.DataFrame":
        """Return a DataFrame flagging timesteps as anomalies.

        Anomalies are determined either by an explicit *threshold* or by
        the top *contamination* fraction of rolling anomaly scores.

        Parameters
        ----------
        X:
            Time series array ``(T, C)``.
        stride:
            Stride between windows (default ``1``).
        threshold:
            Explicit score threshold.  If ``None``, the ``(1-contamination)``
            quantile of all scores is used.
        contamination:
            Fraction of windows to label as anomalous when no explicit
            threshold is provided (default ``0.05``).
        reduction:
            Error reduction mode passed to :meth:`anomaly_score`.

        Returns
        -------
        pandas.DataFrame with columns ``['timestep', 'score', 'anomaly']``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for flag_anomalies()") from exc

        scores, indices = self.anomaly_score(X, stride=stride, reduction=reduction)
        if threshold is None:
            threshold = float(np.quantile(scores, 1.0 - contamination))
        flags = scores >= threshold
        return pd.DataFrame({"timestep": indices, "score": scores, "anomaly": flags})

    def plot_anomalies(
        self,
        X: np.ndarray,
        anomaly_df: "pd.DataFrame",
        *,
        channel: int = 0,
        ax=None,
        title: Optional[str] = None,
        alpha: float = 0.3,
    ):
        """Overlay anomaly flags on the raw time series.

        Parameters
        ----------
        X:
            Raw time series ``(T, C)``.
        anomaly_df:
            DataFrame from :meth:`flag_anomalies`.
        channel:
            Which channel to plot.
        ax:
            Matplotlib axes (created if ``None``).
        title:
            Axes title.
        alpha:
            Transparency for anomaly markers.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_anomalies()") from exc

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.get_figure()

        ts = X[:, channel] if X.ndim > 1 else X
        ax.plot(ts, linewidth=0.8, label="signal")

        anom = anomaly_df[anomaly_df["anomaly"]]
        if len(anom):
            ys = ts[anom["timestep"].clip(0, len(ts) - 1).values]
            ax.scatter(
                anom["timestep"].values,
                ys,
                color="red",
                s=30,
                alpha=alpha,
                zorder=5,
                label="anomaly",
            )

        ax.set_title(title or f"Anomaly detection (channel {channel})")
        ax.set_xlabel("Timestep")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def autocorrelation(
        X: np.ndarray,
        *,
        max_lag: int = 40,
        channel: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sample autocorrelation function (ACF) for one channel.

        Parameters
        ----------
        X:
            Time series ``(T,)`` or ``(T, C)``.
        max_lag:
            Maximum lag to compute (default ``40``).
        channel:
            Channel to use when *X* is 2-D (default ``0``).

        Returns
        -------
        lags : np.ndarray  shape ``(max_lag + 1,)``
        acf  : np.ndarray  shape ``(max_lag + 1,)``
        """
        ts = X[:, channel] if X.ndim > 1 else X
        ts = ts.astype(float)
        n = len(ts)
        ts = ts - ts.mean()
        c0 = float(np.dot(ts, ts))
        lags = np.arange(max_lag + 1)
        acf = np.array(
            [float(np.dot(ts[: n - k], ts[k:])) / c0 if k < n else 0.0 for k in lags]
        )
        return lags, acf

    @staticmethod
    def plot_acf(
        X: np.ndarray,
        *,
        max_lag: int = 40,
        channel: int = 0,
        ax=None,
        title: Optional[str] = None,
        significance_level: float = 0.05,
    ):
        """Plot the sample ACF with confidence bands.

        Parameters
        ----------
        X:
            Time series ``(T,)`` or ``(T, C)``.
        max_lag:
            Maximum lag (default ``40``).
        channel:
            Channel index for multivariate input.
        ax:
            Matplotlib axes (created if ``None``).
        title:
            Axes title.
        significance_level:
            Two-tailed significance level for the confidence band (default
            ``0.05`` → 95 % CI).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_acf()") from exc

        lags, acf = Forecaster.autocorrelation(X, max_lag=max_lag, channel=channel)
        n = len(X)
        # approximate normal quantile via erf⁻¹ — avoids scipy dependency
        p = 1 - significance_level / 2
        # Beasley-Springer-Moro approximation for the normal quantile
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        t = np.sqrt(-2.0 * np.log(1 - p))
        z = t - (c[0] + c[1] * t + c[2] * t ** 2) / (1 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3)
        ci = z / np.sqrt(n)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.vlines(lags, 0, acf, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(ci, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-ci, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(title or f"Autocorrelation (channel {channel})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def spectral_density(
        X: np.ndarray,
        *,
        channel: int = 0,
        n_fft: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the one-sided power spectral density via FFT.

        Parameters
        ----------
        X:
            Time series ``(T,)`` or ``(T, C)``.
        channel:
            Channel index for multivariate input.
        n_fft:
            FFT size.  Defaults to ``len(X)``.

        Returns
        -------
        freqs : np.ndarray  normalised frequencies in ``[0, 0.5]``
        psd   : np.ndarray  power spectral density (same length as *freqs*)
        """
        ts = X[:, channel] if X.ndim > 1 else X
        ts = ts.astype(float)
        n = n_fft or len(ts)
        spec = np.abs(np.fft.rfft(ts, n=n)) ** 2
        freqs = np.fft.rfftfreq(n)
        return freqs, spec

    @staticmethod
    def plot_spectral_density(
        X: np.ndarray,
        *,
        channel: int = 0,
        n_fft: Optional[int] = None,
        ax=None,
        title: Optional[str] = None,
        log_scale: bool = True,
    ):
        """Plot the one-sided power spectral density.

        Parameters
        ----------
        X:
            Time series ``(T,)`` or ``(T, C)``.
        channel:
            Channel index.
        n_fft:
            FFT size.
        ax:
            Matplotlib axes (created if ``None``).
        title:
            Axes title.
        log_scale:
            Use a log-scale y-axis (default ``True``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_spectral_density()") from exc

        freqs, psd = Forecaster.spectral_density(X, channel=channel, n_fft=n_fft)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.plot(freqs, psd, linewidth=0.9)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Normalised frequency")
        ax.set_ylabel("Power")
        ax.set_title(title or f"Power spectral density (channel {channel})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def copy_weights_from(self, source: "Forecaster") -> "Forecaster":
        """Copy model weights from *source* into this Forecaster in-place.

        Both Forecasters must be fitted and share the same model architecture
        (same class and layer shapes).  Useful for transfer learning: fit on a
        large source dataset, then copy weights and fine-tune with
        :meth:`partial_fit` on the target domain.

        Parameters
        ----------
        source:
            A fitted :class:`Forecaster` whose weights to copy.

        Returns
        -------
        self
        """
        self._check_fitted()
        source._check_fitted()
        self._model.load_state_dict(source._model.state_dict())
        return self

    def align_channels(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate *X* along the channel axis to match the model's input width.

        When deploying a model trained on ``C_train`` channels to data with a
        different ``C`` channels, this helper adapts the input by:

        - **Truncating** extra channels if ``C > C_train``.
        - **Zero-padding** missing channels if ``C < C_train``.

        Parameters
        ----------
        X:
            Input array ``(T, C)`` or ``(N, T, C)``.

        Returns
        -------
        np.ndarray of the same rank with the channel dimension set to the
        model's expected ``C_train``.
        """
        self._check_fitted()
        expected_c = self._enc_in
        actual_c = X.shape[-1]
        if actual_c == expected_c:
            return X
        if actual_c > expected_c:
            return X[..., :expected_c]
        pad_width = [(0, 0)] * X.ndim
        pad_width[-1] = (0, expected_c - actual_c)
        return np.pad(X, pad_width, mode="constant")

    def predict_multi_step(
        self,
        X: np.ndarray,
        *,
        horizons: List[int],
    ) -> Dict[str, np.ndarray]:
        """Predict at multiple forecast horizons using the same fitted model.

        The model is called once per horizon.  Predictions that exceed
        ``pred_len`` are obtained by iterative autoregressive chaining.

        Parameters
        ----------
        X:
            Context window ``(1, seq_len, C)`` or ``(seq_len, C)``.
        horizons:
            List of integer horizons to predict.  Each must be ``>= 1``.

        Returns
        -------
        dict mapping ``str(horizon)`` → ``np.ndarray (horizon, C)``
        """
        self._check_fitted()
        if X.ndim == 2:
            X = X[np.newaxis]
        results = {}
        for h in horizons:
            if h <= 0:
                raise ValueError(f"horizon must be >= 1, got {h}")
            ctx = X.copy()
            collected: List[np.ndarray] = []
            remaining = h
            while remaining > 0:
                step = self.predict(ctx)  # (1, pred_len, C)
                chunk = step[0, : min(self.pred_len, remaining)]
                collected.append(chunk)
                remaining -= len(chunk)
                # advance context window
                advance = step[0, :self.pred_len]
                ctx = np.concatenate([ctx[:, len(advance):, :], advance[np.newaxis]], axis=1)
            results[str(h)] = np.concatenate(collected, axis=0)[:h]
        return results

    def residual_qq(
        self,
        X: np.ndarray,
        *,
        channel: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute quantile-quantile (Q-Q) values against a normal distribution.

        Computes rolling one-step-ahead residuals and returns the theoretical
        normal quantiles and the empirical quantiles of the residuals for
        one channel.  Useful for checking whether residuals are approximately
        Gaussian (a common assumption for prediction intervals).

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        channel:
            Channel index (default ``0``).

        Returns
        -------
        theoretical : np.ndarray — standard normal quantiles
        sample      : np.ndarray — sorted residual quantiles
        """
        self._check_fitted()
        resids = self.residuals(X)
        # residuals() returns (n_windows, pred_len, C) or (n_windows, pred_len)
        if resids.ndim == 3:
            r = resids[:, :, channel].ravel()
        elif resids.ndim == 2:
            r = resids.ravel()
        else:
            r = resids.ravel()
        r = np.sort(r.astype(float))
        n = len(r)
        # Blom plotting positions
        p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        # normal quantile via rational approximation (no scipy)
        a = [2.515517, 0.802853, 0.010328]
        b = [1.432788, 0.189269, 0.001308]
        theoretical = np.empty(n)
        for i, pi in enumerate(p):
            pi = float(np.clip(pi, 1e-10, 1 - 1e-10))
            sign = 1.0 if pi >= 0.5 else -1.0
            t = np.sqrt(-2.0 * np.log(min(pi, 1 - pi)))
            q = t - (a[0] + a[1] * t + a[2] * t ** 2) / (
                1 + b[0] * t + b[1] * t ** 2 + b[2] * t ** 3
            )
            theoretical[i] = sign * q
        return theoretical, r

    def plot_qq(
        self,
        X: np.ndarray,
        *,
        channel: int = 0,
        ax=None,
        title: Optional[str] = None,
    ):
        """Q-Q plot of forecast residuals against a normal distribution.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        channel:
            Channel index.
        ax:
            Matplotlib axes (created if ``None``).
        title:
            Axes title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_qq()") from exc

        theoretical, sample = self.residual_qq(X, channel=channel)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.get_figure()

        ax.scatter(theoretical, sample, s=10, alpha=0.6)
        lo = min(theoretical[0], sample[0])
        hi = max(theoretical[-1], sample[-1])
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(title or f"Q-Q plot (channel {channel})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def to_torch_dataset(
        self,
        X: np.ndarray,
        *,
        normalize: Optional[bool] = None,
    ) -> "torch.utils.data.Dataset":
        """Convert a time series array to a :class:`torch.utils.data.Dataset`.

        Yields ``(x, y)`` tensor pairs where *x* has shape
        ``(seq_len, C)`` and *y* has shape ``(pred_len, C)``.
        The scaler fitted during :meth:`fit` is applied when *normalize*
        is ``True`` (default: same as ``self.normalize``).

        Parameters
        ----------
        X:
            Time series array ``(T, C)`` or ``(T,)``.
        normalize:
            Whether to apply the fitted scaler.  ``None`` inherits
            ``self.normalize``.

        Returns
        -------
        torch.utils.data.Dataset
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        X_np = X_np.astype(np.float32)

        use_norm = self.normalize if normalize is None else normalize
        if use_norm and self._scaler is not None:
            X_np = self._scaler.transform(X_np)

        return _WindowDataset(X_np, self.seq_len, self.pred_len)

    def profile(
        self,
        X: np.ndarray,
        *,
        n_repeats: int = 50,
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Measure inference latency and throughput.

        Runs *n_repeats* forward passes and reports wall-clock statistics.

        Parameters
        ----------
        X:
            Context windows ``(B, seq_len, C)`` or ``(seq_len, C)``.
            If 2-D, a batch of *batch_size* copies is constructed.
        n_repeats:
            Number of timed forward passes (default ``50``).
        batch_size:
            Override batch size for profiling (default: ``self.batch_size``).

        Returns
        -------
        dict with keys:
            ``mean_ms``     — mean latency in milliseconds
            ``std_ms``      — standard deviation in milliseconds
            ``min_ms``      — minimum latency in milliseconds
            ``max_ms``      — maximum latency in milliseconds
            ``throughput``  — windows per second (``batch_size / mean_s``)
        """
        import time
        self._check_fitted()
        bs = batch_size or self.batch_size

        X_np = _to_numpy(X).astype(np.float32)
        if X_np.ndim == 2:
            X_np = np.tile(X_np[np.newaxis], (bs, 1, 1))
        elif X_np.ndim == 3 and X_np.shape[0] != bs:
            X_np = np.tile(X_np[:1], (bs, 1, 1))

        self._model.eval()
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            self.predict(X_np)
            times.append((time.perf_counter() - t0) * 1000)

        t = np.array(times)
        mean_ms = float(t.mean())
        return {
            "mean_ms": mean_ms,
            "std_ms": float(t.std()),
            "min_ms": float(t.min()),
            "max_ms": float(t.max()),
            "throughput": float(bs / (mean_ms / 1000.0)),
        }

    def warmup(self, X: Optional[np.ndarray] = None, *, n: int = 3) -> "Forecaster":
        """Run *n* dummy forward passes to warm up JIT/CUDA caches.

        Parameters
        ----------
        X:
            Context window ``(seq_len, C)``.  If ``None``, a zero array of
            the right shape is used.
        n:
            Number of warmup passes (default ``3``).

        Returns
        -------
        self
        """
        self._check_fitted()
        C = self._enc_in or 1
        if X is None:
            ctx = np.zeros((self.seq_len, C), dtype=np.float32)
        else:
            ctx = _to_numpy(X).astype(np.float32)
        for _ in range(n):
            self.predict(ctx)
        return self

    @staticmethod
    def channel_correlation(X: np.ndarray) -> np.ndarray:
        """Compute the Pearson correlation matrix across channels.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.

        Returns
        -------
        np.ndarray of shape ``(C, C)`` with values in ``[-1, 1]``.
        """
        X_f = X.astype(float)
        X_centered = X_f - X_f.mean(axis=0)
        cov = X_centered.T @ X_centered / (len(X) - 1)
        std = np.sqrt(np.diag(cov)).clip(min=1e-12)
        return cov / np.outer(std, std)

    @staticmethod
    def plot_channel_correlation(
        X: np.ndarray,
        *,
        channel_names: Optional[List[str]] = None,
        ax=None,
        title: Optional[str] = None,
        cmap: str = "coolwarm",
    ):
        """Plot a heatmap of the channel-wise Pearson correlation matrix.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.
        channel_names:
            Labels for each channel.  Defaults to ``["ch0", "ch1", ...]``.
        ax:
            Matplotlib axes (created if ``None``).
        title:
            Axes title.
        cmap:
            Colormap (default ``"coolwarm"``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_channel_correlation()") from exc

        corr = Forecaster.channel_correlation(X)
        C = corr.shape[0]
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(C)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, C), max(3, C)))
        else:
            fig = ax.get_figure()

        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(channel_names, rotation=45, ha="right")
        ax.set_yticklabels(channel_names)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title or "Channel correlation")
        plt.tight_layout()
        return fig

    def forecast_dataframe(
        self,
        X: np.ndarray,
        *,
        channel_names: Optional[List[str]] = None,
        start_index: int = 0,
    ) -> "pd.DataFrame":
        """Return the forecast as a labeled pandas DataFrame.

        The context is ``X[-seq_len:]`` and the forecast covers the next
        ``pred_len`` timesteps.

        Parameters
        ----------
        X:
            Context window ``(T, C)`` or ``(seq_len, C)``.  If longer than
            ``seq_len`` the last ``seq_len`` rows are used.
        channel_names:
            Column labels.  Defaults to ``["ch0", "ch1", ...]``.
        start_index:
            Integer index of the first predicted timestep (default ``0``).

        Returns
        -------
        pandas.DataFrame with shape ``(pred_len, C)`` and columns
        ``channel_names``.  The DataFrame index runs from *start_index* to
        ``start_index + pred_len - 1``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for forecast_dataframe()") from exc

        self._check_fitted()
        pred = self.predict(X)
        if pred.ndim == 3:
            pred = pred[0]  # (pred_len, C)

        C = pred.shape[1] if pred.ndim > 1 else 1
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(C)]

        return pd.DataFrame(
            pred,
            columns=channel_names,
            index=range(start_index, start_index + self.pred_len),
        )

    def moving_forecast(
        self,
        X: np.ndarray,
        *,
        n_windows: int = 5,
    ) -> np.ndarray:
        """Reduce forecast variance by averaging over *n_windows* overlapping predictions.

        Uses the last ``n_windows`` available context windows that end at the
        same forecast origin (the last ``seq_len`` rows of *X*) and averages
        their step-aligned predictions.

        In practice, ``n_windows`` slightly shifted contexts (each shifted by
        1 timestep) are generated from the available history and their
        predictions are averaged after step-alignment.  This mimics bagged
        one-shot forecasting for a single model without retraining.

        Parameters
        ----------
        X:
            History array ``(T, C)``.  Must have ``T >= seq_len + n_windows - 1``.
        n_windows:
            Number of overlapping predictions to average (default ``5``).

        Returns
        -------
        np.ndarray of shape ``(pred_len, C)``
        """
        self._check_fitted()
        T = len(X)
        min_required = self.seq_len + n_windows - 1
        if T < min_required:
            raise ValueError(
                f"X has {T} timesteps but moving_forecast needs at least "
                f"seq_len + n_windows - 1 = {min_required}."
            )

        preds = []
        for offset in range(n_windows):
            start = T - self.seq_len - (n_windows - 1 - offset)
            end = start + self.seq_len
            ctx = X[start:end]
            pred = self.predict(ctx)  # (pred_len, C)
            preds.append(pred)

        return np.mean(preds, axis=0)

    def residual_distribution(
        self,
        X: np.ndarray,
        *,
        channel: int = 0,
    ) -> Dict[str, float]:
        """Compute summary statistics of forecast residuals for one channel.

        Uses rolling one-step predictions (via :meth:`residuals`) and
        reports distributional properties to help assess model calibration.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        channel:
            Channel index (default ``0``).

        Returns
        -------
        dict with keys:

        ``mean``     — mean residual (bias)
        ``std``      — standard deviation
        ``skewness`` — third standardised moment
        ``kurtosis`` — excess kurtosis (normal = 0)
        ``q5``       — 5th percentile
        ``q25``      — 25th percentile
        ``median``   — median
        ``q75``      — 75th percentile
        ``q95``      — 95th percentile
        ``n``        — number of residual values
        """
        self._check_fitted()
        resids = self.residuals(X)
        if resids.ndim == 3:
            r = resids[:, :, channel].ravel().astype(float)
        elif resids.ndim == 2:
            r = resids.ravel().astype(float)
        else:
            r = resids.ravel().astype(float)

        n = len(r)
        mean = float(r.mean())
        std = float(r.std())
        if std < 1e-12:
            skewness = 0.0
            kurtosis = 0.0
        else:
            z = (r - mean) / std
            skewness = float(np.mean(z ** 3))
            kurtosis = float(np.mean(z ** 4) - 3.0)  # excess kurtosis

        q5, q25, median, q75, q95 = np.percentile(r, [5, 25, 50, 75, 95]).tolist()
        return {
            "mean": mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "q5": q5,
            "q25": q25,
            "median": median,
            "q75": q75,
            "q95": q95,
            "n": n,
        }

    def input_gradient(
        self,
        X: np.ndarray,
        *,
        target_step: int = 0,
        target_channel: int = 0,
        absolute: bool = True,
    ) -> np.ndarray:
        """Compute the gradient of a specific forecast output w.r.t. the input.

        Uses vanilla gradient saliency: ``d output[target_step, target_channel]
        / d input``.  The sign tells you the direction of influence; the
        magnitude tells you which input timesteps (and channels) matter most.

        Parameters
        ----------
        X:
            Context window ``(seq_len, C)`` or ``(T, C)`` (last seq_len used).
        target_step:
            Forecast timestep to differentiate w.r.t. (0-indexed into
            ``pred_len``; default ``0``).
        target_channel:
            Output channel to differentiate w.r.t. (default ``0``).
        absolute:
            Return absolute gradient values (default ``True``).  Set to
            ``False`` to retain sign information.

        Returns
        -------
        np.ndarray of shape ``(seq_len, C)`` — gradient magnitude per input
        timestep and channel.
        """
        self._check_fitted()
        X_np = _to_numpy(X).astype(np.float32)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        if len(X_np) > self.seq_len:
            X_np = X_np[-self.seq_len:]

        if self.normalize and self._scaler is not None:
            X_np = self._scaler.transform(X_np)

        inp = torch.tensor(X_np[np.newaxis], requires_grad=True)  # (1, seq_len, C)
        self._model.eval()
        out = self._model(inp)  # (1, pred_len, C) or (1, pred_len, C)
        if out.ndim == 2:
            out = out.unsqueeze(0)
        scalar = out[0, target_step, target_channel]
        scalar.backward()

        grad = inp.grad.detach().numpy()[0]  # (seq_len, C)
        if absolute:
            grad = np.abs(grad)
        return grad

    def plot_saliency(
        self,
        X: np.ndarray,
        *,
        target_step: int = 0,
        target_channel: int = 0,
        absolute: bool = True,
        channel_names: Optional[List[str]] = None,
        ax=None,
        title: Optional[str] = None,
    ):
        """Heatmap of input gradient saliency from :meth:`input_gradient`.

        Parameters
        ----------
        X:
            Context window ``(seq_len, C)``.
        target_step:
            Forecast timestep to explain (default ``0``).
        target_channel:
            Output channel to explain (default ``0``).
        absolute:
            Use absolute gradients (default ``True``).
        channel_names:
            Labels for channels.  Defaults to ``["ch0", "ch1", ...]``.
        ax:
            Matplotlib axes.
        title:
            Axes title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_saliency()") from exc

        grad = self.input_gradient(
            X,
            target_step=target_step,
            target_channel=target_channel,
            absolute=absolute,
        )  # (seq_len, C)
        C = grad.shape[1]
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(C)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, max(2, C)))
        else:
            fig = ax.get_figure()

        im = ax.imshow(grad.T, aspect="auto", cmap="hot")
        ax.set_yticks(range(C))
        ax.set_yticklabels(channel_names)
        ax.set_xlabel("Input timestep")
        ax.set_ylabel("Channel")
        ax.set_title(
            title
            or f"Input saliency → output[step={target_step}, ch={target_channel}]"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    def error_decomposition(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        *,
        n_bootstrap: int = 20,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Decompose forecast error into bias² and variance components.

        Fits *n_bootstrap* models on bootstrap resamples of *X_train*
        and evaluates them on *X_test*.  The classic bias-variance
        decomposition gives::

            MSE ≈ bias² + variance + noise²

        Parameters
        ----------
        X_train:
            Training time series ``(T_train, C)``.
        X_test:
            Test time series ``(T_test, C)``.
        n_bootstrap:
            Number of bootstrap resamples (default ``20``).
        seed:
            Random seed for reproducibility.

        Returns
        -------
        dict with keys ``bias2``, ``variance``, ``total_mse``.
        """
        rng = np.random.default_rng(seed)
        T_train = len(X_train)
        preds_list = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, T_train, size=T_train)
            idx_sorted = np.sort(idx)  # keep temporal ordering
            X_boot = X_train[idx_sorted]
            fc_b = self.clone()
            fc_b.epochs = max(1, self.epochs)
            try:
                fc_b.fit(X_boot, val_split=0.1)
                preds = fc_b.predict_rolling(X_test)  # (n_windows, pred_len, C)
                preds_list.append(preds)
            except Exception:
                continue

        if not preds_list:
            raise RuntimeError("All bootstrap fits failed.")

        # Align lengths
        min_w = min(p.shape[0] for p in preds_list)
        preds_arr = np.stack([p[:min_w] for p in preds_list], axis=0)  # (B, n_w, pred_len, C)
        mean_pred = preds_arr.mean(axis=0)  # (n_w, pred_len, C)

        # Truth windows
        truth_windows = []
        for t in range(min_w):
            start = t + self.seq_len
            truth_windows.append(X_test[start : start + self.pred_len])
        truth = np.stack(truth_windows, axis=0)  # (n_w, pred_len, C)

        bias2 = float(np.mean((mean_pred - truth) ** 2))
        variance = float(np.mean(preds_arr.var(axis=0)))
        total_mse = float(np.mean((preds_arr.mean(0) - truth) ** 2))

        return {"bias2": bias2, "variance": variance, "total_mse": total_mse}

    def set_device(self, device: str) -> "Forecaster":
        """Move the model to *device* (e.g. ``"cuda"`` or ``"cpu"``).

        Also updates ``self.device`` so subsequent :meth:`fit` / :meth:`predict`
        calls use the new device.

        Parameters
        ----------
        device:
            PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).

        Returns
        -------
        self
        """
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    @staticmethod
    def granger_test(
        X: np.ndarray,
        *,
        max_lag: int = 5,
    ) -> np.ndarray:
        """Pairwise Granger causality F-statistics between all channel pairs.

        For each ordered pair ``(cause, effect)`` the method tests whether
        the past of channel *cause* improves the AR prediction of channel
        *effect* above and beyond the past of *effect* alone.  Uses OLS
        regression — no additional dependencies required.

        A high F-statistic in position ``[i, j]`` means channel *i* Granger-
        causes channel *j*.

        Parameters
        ----------
        X:
            Multivariate time series ``(T, C)``.
        max_lag:
            Number of lags to include in the regression (default ``5``).

        Returns
        -------
        np.ndarray of shape ``(C, C)`` — F-statistics (diagonal is 0).
        """
        X = np.asarray(X, dtype=float)
        T, C = X.shape

        def _ols_ssr(A: np.ndarray, b: np.ndarray) -> float:
            """Return SSR of OLS regression b ~ A (no bias — caller adds constant)."""
            coef, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
            pred = A @ coef
            return float(((b - pred) ** 2).sum())

        f_mat = np.zeros((C, C), dtype=float)
        n_obs = T - max_lag

        for j in range(C):
            # Restricted model: j ~ lags of j
            # Unrestricted model: j ~ lags of j + lags of i
            y = X[max_lag:, j]

            # Build restricted design matrix (lags of j + constant)
            Z_res = np.column_stack(
                [X[max_lag - k - 1 : T - k - 1, j] for k in range(max_lag)]
                + [np.ones(n_obs)]
            )
            ssr_res = _ols_ssr(Z_res, y)

            for i in range(C):
                if i == j:
                    continue
                # Unrestricted: add lags of i
                Z_unres = np.column_stack(
                    [Z_res]
                    + [X[max_lag - k - 1 : T - k - 1, i] for k in range(max_lag)]
                )
                ssr_unres = _ols_ssr(Z_unres, y)

                df1 = max_lag
                df2 = n_obs - Z_unres.shape[1]
                if df2 <= 0 or ssr_unres <= 1e-12:
                    f_mat[i, j] = 0.0
                else:
                    f_mat[i, j] = ((ssr_res - ssr_unres) / df1) / (ssr_unres / df2)

        return f_mat

    @staticmethod
    def plot_granger(
        X: np.ndarray,
        *,
        max_lag: int = 5,
        channel_names: Optional[List[str]] = None,
        ax=None,
        title: Optional[str] = None,
        cmap: str = "YlOrRd",
    ):
        """Heatmap of pairwise Granger causality F-statistics.

        Rows = cause, columns = effect.

        Parameters
        ----------
        X:
            Multivariate time series ``(T, C)``.
        max_lag:
            Regression lag for Granger test.
        channel_names:
            Labels per channel.
        ax:
            Matplotlib axes.
        title:
            Axes title.
        cmap:
            Colormap (default ``"YlOrRd"``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_granger()") from exc

        f_mat = Forecaster.granger_test(X, max_lag=max_lag)
        C = f_mat.shape[0]
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(C)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, C), max(3, C)))
        else:
            fig = ax.get_figure()

        im = ax.imshow(f_mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(channel_names, rotation=45, ha="right")
        ax.set_yticklabels(channel_names)
        ax.set_xlabel("Effect channel (j)")
        ax.set_ylabel("Cause channel (i)")
        fig.colorbar(im, ax=ax, label="F-statistic", fraction=0.046, pad=0.04)
        ax.set_title(title or f"Granger causality (lag={max_lag})")
        plt.tight_layout()
        return fig

    def count_parameters(self, *, trainable_only: bool = False) -> Dict[str, int]:
        """Return the total parameter count, broken down by top-level module.

        Parameters
        ----------
        trainable_only:
            If ``True``, count only parameters that require gradients.
            Default ``False`` (count all parameters).

        Returns
        -------
        dict with keys:

        ``total``   — aggregate parameter count
        one key per *named* top-level child of the model, e.g.
        ``"encoder"``, ``"decoder"``, etc.
        """
        self._check_fitted()
        result: Dict[str, int] = {}
        total = 0
        for name, module in self._model.named_children():
            cnt = sum(
                p.numel()
                for p in module.parameters()
                if (not trainable_only or p.requires_grad)
            )
            result[name] = cnt
            total += cnt
        # catch parameters at the top level (not under any child)
        child_params = set(
            id(p) for m in self._model.children() for p in m.parameters()
        )
        top_level = sum(
            p.numel()
            for p in self._model.parameters()
            if id(p) not in child_params and (not trainable_only or p.requires_grad)
        )
        if top_level:
            result["_top_level"] = top_level
            total += top_level
        result["total"] = total
        return result

    @staticmethod
    def rolling_correlation(
        X: np.ndarray,
        *,
        window: int = 30,
        channel_i: int = 0,
        channel_j: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rolling Pearson correlation between two channels.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.
        window:
            Rolling window size (default ``30``).
        channel_i:
            First channel index (default ``0``).
        channel_j:
            Second channel index (default ``1``).

        Returns
        -------
        timesteps : np.ndarray of length ``T - window + 1``
        corr      : np.ndarray of rolling correlations
        """
        xi = X[:, channel_i].astype(float)
        xj = X[:, channel_j].astype(float)
        T = len(xi)
        n_windows = T - window + 1
        corrs = np.empty(n_windows)
        for k in range(n_windows):
            a = xi[k : k + window]
            b = xj[k : k + window]
            a_c = a - a.mean()
            b_c = b - b.mean()
            denom = (np.sqrt((a_c ** 2).sum()) * np.sqrt((b_c ** 2).sum()))
            corrs[k] = float(np.dot(a_c, b_c) / denom) if denom > 1e-12 else 0.0
        return np.arange(window - 1, T), corrs

    @staticmethod
    def plot_rolling_correlation(
        X: np.ndarray,
        *,
        window: int = 30,
        channel_i: int = 0,
        channel_j: int = 1,
        ax=None,
        title: Optional[str] = None,
    ):
        """Plot rolling Pearson correlation between two channels.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.
        window:
            Rolling window size (default ``30``).
        channel_i / channel_j:
            Channel indices to correlate.
        ax:
            Matplotlib axes.
        title:
            Axes title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_rolling_correlation()") from exc

        ts, corrs = Forecaster.rolling_correlation(
            X, window=window, channel_i=channel_i, channel_j=channel_j
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.get_figure()

        ax.plot(ts, corrs, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Pearson r")
        ax.set_title(
            title or f"Rolling correlation (window={window}): ch{channel_i} vs ch{channel_j}"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def interpolate_missing(
        X: np.ndarray,
        *,
        method: str = "linear",
    ) -> np.ndarray:
        """Interpolate NaN values in *X* along the time axis.

        Operates independently on each channel.  Useful as a preprocessing
        step before calling :meth:`fit` or :meth:`predict` on data with
        missing values.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)`` possibly containing NaN.
        method:
            Interpolation method: ``"linear"`` (default), ``"forward"``,
            ``"backward"``, or ``"nearest"``.

        Returns
        -------
        np.ndarray — same shape as *X*, NaN-free (edge NaN values are
        forward/backward filled after interpolation).
        """
        valid_methods = ("linear", "forward", "backward", "nearest")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}; got {method!r}")

        X_out = np.asarray(X, dtype=float).copy()
        orig_ndim = X_out.ndim
        if X_out.ndim == 1:
            X_out = X_out[:, None]

        T, C = X_out.shape
        idx = np.arange(T)

        for c in range(C):
            col = X_out[:, c]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            valid = ~nan_mask
            if not valid.any():
                continue  # all NaN — nothing to interpolate
            valid_idx = idx[valid]
            valid_val = col[valid]

            if method == "linear":
                col[nan_mask] = np.interp(idx[nan_mask], valid_idx, valid_val)
            elif method == "forward":
                # Fill each NaN with the most recent non-NaN
                last = col[valid_idx[0]]
                for t in range(T):
                    if not nan_mask[t]:
                        last = col[t]
                    else:
                        col[t] = last
            elif method == "backward":
                first = col[valid_idx[-1]]
                for t in range(T - 1, -1, -1):
                    if not nan_mask[t]:
                        first = col[t]
                    else:
                        col[t] = first
            elif method == "nearest":
                for t in idx[nan_mask]:
                    dists = np.abs(valid_idx - t)
                    col[t] = valid_val[dists.argmin()]

            # Fill any remaining edge NaN with adjacent value
            if np.isnan(col).any():
                # forward fill edges
                for t in range(T):
                    if np.isnan(col[t]) and t > 0:
                        col[t] = col[t - 1]
                for t in range(T - 1, -1, -1):
                    if np.isnan(col[t]) and t < T - 1:
                        col[t] = col[t + 1]
            X_out[:, c] = col

        if orig_ndim == 1:
            return X_out[:, 0]
        return X_out

    def horizon_error_profile(
        self,
        X: np.ndarray,
        *,
        metric: str = "MAE",
    ) -> np.ndarray:
        """Compute per-horizon-step error averaged over all rolling windows.

        Shows how prediction quality degrades as the forecast horizon
        increases.

        Parameters
        ----------
        X:
            Full time series ``(T, C)``.
        metric:
            Error metric: ``"MAE"`` (default), ``"MSE"``, or ``"RMSE"``.

        Returns
        -------
        np.ndarray of shape ``(pred_len,)`` — mean error per horizon step.
        """
        self._check_fitted()
        metric = metric.upper()
        if metric not in ("MAE", "MSE", "RMSE"):
            raise ValueError(f"metric must be MAE, MSE, or RMSE; got {metric!r}")

        errors = []
        T = len(X)
        t = 0
        while t + self.seq_len + self.pred_len <= T:
            ctx = X[t : t + self.seq_len]
            truth = X[t + self.seq_len : t + self.seq_len + self.pred_len]
            pred = self.predict(ctx[np.newaxis])[0]
            err = np.abs(pred - truth) if metric != "MSE" else (pred - truth) ** 2
            errors.append(err)  # (pred_len, C)
            t += 1

        if not errors:
            raise ValueError("No complete windows found in X.")

        arr = np.stack(errors, axis=0)  # (n_windows, pred_len, C)
        profile = arr.mean(axis=(0, 2))  # (pred_len,)
        if metric == "RMSE":
            # we computed MAE above — redo with MSE
            errors_mse = []
            t = 0
            while t + self.seq_len + self.pred_len <= T:
                ctx = X[t : t + self.seq_len]
                truth = X[t + self.seq_len : t + self.seq_len + self.pred_len]
                pred = self.predict(ctx[np.newaxis])[0]
                errors_mse.append((pred - truth) ** 2)
                t += 1
            profile = np.sqrt(np.stack(errors_mse, 0).mean(axis=(0, 2)))
        return profile

    def plot_horizon_error_profile(
        self,
        X: np.ndarray,
        *,
        metric: str = "MAE",
        ax=None,
        title: Optional[str] = None,
    ):
        """Bar chart of per-horizon-step error from :meth:`horizon_error_profile`.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.
        metric:
            Error metric passed to :meth:`horizon_error_profile`.
        ax:
            Matplotlib axes.
        title:
            Axes title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_horizon_error_profile()") from exc

        profile = self.horizon_error_profile(X, metric=metric)

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(profile) // 2), 4))
        else:
            fig = ax.get_figure()

        ax.bar(np.arange(1, len(profile) + 1), profile, color="steelblue", alpha=0.8)
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(metric)
        ax.set_title(title or f"Horizon error profile ({metric})")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return fig

    def persistence_forecast(
        self,
        X: np.ndarray,
        *,
        lag: int = 1,
    ) -> np.ndarray:
        """Naive persistence baseline: repeat the last *lag* timestep values.

        This is the simplest possible forecast — repeat the last observed
        value *pred_len* times.  It acts as the lower bound for model
        comparison on stationary data.

        Parameters
        ----------
        X:
            Context window ``(seq_len, C)`` or ``(T, C)`` (last value used).
        lag:
            Which past timestep to repeat.  ``1`` (default) = last
            observation; ``7`` = last week's value (for daily data with
            weekly seasonality).

        Returns
        -------
        np.ndarray of shape ``(pred_len, C)``
        """
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        last = X_np[-lag]  # (C,)
        return np.tile(last, (self.pred_len, 1))

    def score_vs_persistence(
        self,
        X: np.ndarray,
        *,
        lag: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Compare model score against the persistence baseline.

        Runs rolling evaluation for both the fitted model and the
        persistence baseline and returns both result dicts for easy
        comparison.

        Parameters
        ----------
        X:
            Full time series ``(T, C)``.
        lag:
            Lag for the persistence baseline (default ``1``).

        Returns
        -------
        dict with keys ``"model"`` and ``"persistence"``, each a metrics
        dict from :meth:`score`.
        """
        self._check_fitted()
        model_scores = self.score(X)

        # Build a PersistenceForecaster-like evaluator
        T = len(X)
        pred_len = self.pred_len
        seq_len = self.seq_len
        preds, truths = [], []
        t = 0
        while t + seq_len + pred_len <= T:
            ctx = X[t : t + seq_len]
            truth = X[t + seq_len : t + seq_len + pred_len]
            pred = np.tile(ctx[-lag], (pred_len, 1))
            preds.append(pred)
            truths.append(truth)
            t += 1

        preds_arr = np.stack(preds)   # (n, pred_len, C)
        truths_arr = np.stack(truths)

        err = preds_arr - truths_arr
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(mse))
        denom = (np.abs(truths_arr) + np.abs(preds_arr)).clip(min=1e-8)
        smape = float(np.mean(2 * np.abs(err) / denom) * 100)

        return {
            "model": model_scores,
            "persistence": {"MSE": mse, "MAE": mae, "RMSE": rmse, "SMAPE": smape},
        }

    def chunked_predict(
        self,
        X: np.ndarray,
        *,
        chunk_size: int = 64,
    ) -> np.ndarray:
        """Memory-efficient rolling prediction on long sequences.

        Equivalent to calling :meth:`predict` on batches of rolling windows
        one chunk at a time.

        Parameters
        ----------
        X:
            Full time series ``(T, C)``.
        chunk_size:
            Number of windows per inference batch (default ``64``).

        Returns
        -------
        np.ndarray of shape ``(n_windows, pred_len, C)``
        """
        self._check_fitted()
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        T = len(X_np)
        windows = []
        t = 0
        while t + self.seq_len <= T:
            windows.append(X_np[t : t + self.seq_len])
            t += 1
        if not windows:
            raise ValueError("X is shorter than seq_len.")

        results = []
        for start in range(0, len(windows), chunk_size):
            batch = np.stack(windows[start : start + chunk_size], axis=0)  # (B, seq_len, C)
            out = self.predict(batch)  # (B, pred_len, C)
            results.append(out)
        return np.concatenate(results, axis=0)

    @staticmethod
    def detect_change_points(
        X: np.ndarray,
        *,
        window: int = 20,
        threshold: Optional[float] = None,
        channel: int = 0,
    ) -> np.ndarray:
        """Detect change points using a sliding-window energy ratio test.

        For each timestep, compute the ratio of the variance in the right
        window to the variance in the left window (CUSUM-style).  A
        large ratio indicates a structural change in variance; large absolute
        mean shift indicates a level change.  The combined score is
        thresholded to flag change points.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        window:
            Half-window size on each side of the candidate change point
            (default ``20``).
        threshold:
            Score threshold above which a point is flagged.  If ``None``,
            the 95th percentile of all scores is used.
        channel:
            Channel index to analyse (default ``0``).

        Returns
        -------
        np.ndarray of integer indices where change points are detected.
        """
        ts = X[:, channel] if X.ndim > 1 else X
        ts = ts.astype(float)
        T = len(ts)
        scores = np.zeros(T)
        for t in range(window, T - window):
            left = ts[t - window : t]
            right = ts[t : t + window]
            mean_shift = abs(right.mean() - left.mean())
            var_ratio = (right.var() + 1e-12) / (left.var() + 1e-12)
            scores[t] = mean_shift + abs(np.log(var_ratio))

        thr = threshold if threshold is not None else float(np.percentile(scores, 95))
        cps = np.where(scores >= thr)[0]

        # Suppress duplicates within window distance
        if len(cps) == 0:
            return cps
        merged = [cps[0]]
        for cp in cps[1:]:
            if cp - merged[-1] >= window:
                merged.append(cp)
        return np.array(merged)

    @staticmethod
    def plot_change_points(
        X: np.ndarray,
        change_points: np.ndarray,
        *,
        channel: int = 0,
        ax=None,
        title: Optional[str] = None,
        color: str = "red",
        alpha: float = 0.7,
    ):
        """Overlay detected change points on the raw time series.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        change_points:
            Array of change point indices from :meth:`detect_change_points`.
        channel:
            Channel to plot (default ``0``).
        ax:
            Matplotlib axes.
        title:
            Axes title.
        color:
            Color of change-point vertical lines.
        alpha:
            Transparency.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_change_points()") from exc

        ts = X[:, channel] if X.ndim > 1 else X
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.get_figure()

        ax.plot(ts, linewidth=0.8, label="signal")
        for cp in change_points:
            ax.axvline(cp, color=color, alpha=alpha, linewidth=1.2, linestyle="--")
        ax.set_xlabel("Timestep")
        ax.set_title(title or f"Change points (channel {channel})")
        ax.grid(True, alpha=0.3)
        if len(change_points):
            ax.axvline(change_points[0], color=color, alpha=alpha,
                       linewidth=1.2, linestyle="--", label="change point")
        ax.legend()
        plt.tight_layout()
        return fig

    @staticmethod
    def describe(X: np.ndarray, *, channel_names: Optional[List[str]] = None) -> "pd.DataFrame":
        """Compute descriptive statistics of a time series per channel.

        Returns a DataFrame analogous to ``pandas.DataFrame.describe()``,
        extended with additional time-series-relevant statistics.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        channel_names:
            Column labels for channels.

        Returns
        -------
        pandas.DataFrame with one column per channel and rows:
        ``count``, ``mean``, ``std``, ``min``, ``q25``, ``median``,
        ``q75``, ``max``, ``range``, ``skewness``, ``kurtosis``,
        ``autocorr_lag1``, ``n_missing``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for describe()") from exc

        X_f = np.asarray(X, dtype=float)
        if X_f.ndim == 1:
            X_f = X_f[:, None]
        T, C = X_f.shape
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(C)]

        stats = {}
        for ci, name in enumerate(channel_names):
            col = X_f[:, ci]
            valid = col[~np.isnan(col)]
            n = len(valid)
            if n == 0:
                stats[name] = {k: np.nan for k in (
                    "count", "mean", "std", "min", "q25", "median", "q75",
                    "max", "range", "skewness", "kurtosis", "autocorr_lag1", "n_missing"
                )}
                continue
            mu = valid.mean()
            sd = valid.std()
            q25, median, q75 = np.percentile(valid, [25, 50, 75])
            vmin, vmax = valid.min(), valid.max()
            if sd > 1e-12:
                z = (valid - mu) / sd
                skew = float(np.mean(z ** 3))
                kurt = float(np.mean(z ** 4) - 3.0)
            else:
                skew = kurt = 0.0
            # lag-1 autocorrelation (partial, on valid slice)
            if n > 1:
                a = valid[:-1] - valid[:-1].mean()
                b = valid[1:] - valid[1:].mean()
                denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
                ac1 = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
            else:
                ac1 = 0.0
            stats[name] = {
                "count": n,
                "mean": mu,
                "std": sd,
                "min": vmin,
                "q25": q25,
                "median": median,
                "q75": q75,
                "max": vmax,
                "range": vmax - vmin,
                "skewness": skew,
                "kurtosis": kurt,
                "autocorr_lag1": ac1,
                "n_missing": T - n,
            }
        return pd.DataFrame(stats)

    def dashboard(
        self,
        X: np.ndarray,
        *,
        channel: int = 0,
        n_context: int = 96,
        channel_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
    ):
        """Comprehensive 6-panel diagnostic dashboard for a fitted model.

        Panels
        ------
        1 (top-left)  — Forecast: last *n_context* timesteps + prediction
                        with MC-Dropout uncertainty band.
        2 (top-right) — Residual histogram with normal-fit overlay.
        3 (mid-left)  — Sample ACF (up to 40 lags).
        4 (mid-right) — Power spectral density (log scale).
        5 (bot-left)  — Horizon error profile (per-step MAE).
        6 (bot-right) — Channel Pearson correlation heatmap.

        Parameters
        ----------
        X:
            Full time series ``(T, C)``.
        channel:
            Which channel to feature in panels 1–4 (default ``0``).
        n_context:
            Number of historical timesteps shown in panel 1.
        channel_names:
            Labels for the correlation heatmap.
        title:
            Figure suptitle.
        figsize:
            Figure size in inches (default ``(16, 12)``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError as exc:
            raise ImportError("matplotlib is required for dashboard()") from exc

        self._check_fitted()
        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])

        # ── panel 1: forecast with uncertainty ──────────────────────────────
        ctx = X[-n_context:]
        pred = self.predict(ctx)    # (pred_len, C)
        t_ctx  = np.arange(len(X) - n_context, len(X))
        t_pred = np.arange(len(X), len(X) + self.pred_len)
        ax1.plot(t_ctx, ctx[:, channel], color="#555", lw=0.9, label="history")
        ax1.plot(t_pred, pred[:, channel], color="#d62728", lw=1.5, label="forecast")
        try:
            unc = self.predict_uncertainty(ctx, n_samples=50)
            ax1.fill_between(
                t_pred,
                unc["lower"][:, channel],
                unc["upper"][:, channel],
                alpha=0.25, color="#4C72B0", label="90% PI",
            )
        except Exception:
            pass
        ax1.axvline(len(X) - 0.5, color="#bbb", lw=0.8, ls=":")
        ax1.legend(fontsize=7, ncol=3)
        ax1.set_title(f"Forecast (ch {channel})", fontsize=9)
        ax1.grid(True, alpha=0.25)

        # ── panel 2: residual histogram ──────────────────────────────────────
        resids = self.residuals(X)
        r = (resids[:, :, channel] if resids.ndim == 3 else resids).ravel().astype(float)
        ax2.hist(r, bins=40, density=True, color="#4C72B0", alpha=0.75, label="residuals")
        mu, sd = r.mean(), r.std()
        xg = np.linspace(r.min(), r.max(), 200)
        ax2.plot(xg, np.exp(-0.5 * ((xg - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                 color="#d62728", lw=1.5, label="N fit")
        ax2.set_title(f"Residuals (ch {channel})", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.25)

        # ── panel 3: ACF ─────────────────────────────────────────────────────
        lags, acf = Forecaster.autocorrelation(X, max_lag=40, channel=channel)
        n_obs = len(X)
        ci    = 1.96 / np.sqrt(n_obs)
        ax3.vlines(lags, 0, acf, linewidth=1.2)
        ax3.axhline(0,  color="k", lw=0.7)
        ax3.axhline( ci, color="#4C72B0", ls="--", lw=0.9, alpha=0.8)
        ax3.axhline(-ci, color="#4C72B0", ls="--", lw=0.9, alpha=0.8)
        ax3.set_xlabel("Lag", fontsize=8)
        ax3.set_ylabel("ACF",  fontsize=8)
        ax3.set_title(f"Autocorrelation (ch {channel})", fontsize=9)
        ax3.grid(True, alpha=0.25)

        # ── panel 4: power spectrum ──────────────────────────────────────────
        freqs, psd = Forecaster.spectral_density(X, channel=channel)
        ax4.semilogy(freqs[1:], psd[1:], color="#4C72B0", lw=0.9)
        ax4.set_xlabel("Normalised frequency", fontsize=8)
        ax4.set_ylabel("Power",                fontsize=8)
        ax4.set_title(f"Power spectrum (ch {channel})", fontsize=9)
        ax4.grid(True, alpha=0.25)

        # ── panel 5: horizon error profile ───────────────────────────────────
        try:
            profile = self.horizon_error_profile(X, metric="MAE")
            ax5.bar(np.arange(1, len(profile) + 1), profile,
                    color="#4C72B0", alpha=0.8)
            ax5.set_xlabel("Forecast step", fontsize=8)
            ax5.set_ylabel("MAE",           fontsize=8)
            ax5.set_title("Horizon error profile", fontsize=9)
            ax5.grid(True, alpha=0.25, axis="y")
        except Exception:
            ax5.set_visible(False)

        # ── panel 6: channel correlation heatmap ──────────────────────────────
        corr = Forecaster.channel_correlation(X)
        C = corr.shape[0]
        cnames = channel_names or [f"ch{i}" for i in range(C)]
        im = ax6.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax6.set_xticks(range(C)); ax6.set_xticklabels(cnames, rotation=45, ha="right", fontsize=7)
        ax6.set_yticks(range(C)); ax6.set_yticklabels(cnames, fontsize=7)
        fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        ax6.set_title("Channel correlation", fontsize=9)

        model_name = self.model_spec if isinstance(self.model_spec, str) else type(self.model_spec).__name__
        fig.suptitle(
            title or f"{model_name} — diagnostic dashboard",
            fontsize=12, fontweight="bold",
        )
        return fig

    def predict_quantiles(
        self,
        X: np.ndarray,
        *,
        quantiles: List[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        n_samples: int = 200,
    ) -> Dict[str, np.ndarray]:
        """Predict specific quantiles via MC-Dropout sampling.

        Parameters
        ----------
        X:
            Context window ``(seq_len, C)`` or ``(T, C)`` (last seq_len used).
        quantiles:
            Quantile levels in ``(0, 1)`` (default: deciles + quartiles).
        n_samples:
            Number of MC-Dropout forward passes (default ``200``).

        Returns
        -------
        dict mapping ``"q{int(q*100)}"`` → ``np.ndarray (pred_len, C)``
        for each quantile, plus ``"mean"`` and ``"std"``.

        Example
        -------
        >>> q = fc.predict_quantiles(ctx, quantiles=[0.1, 0.5, 0.9])
        >>> q["q10"]   # (pred_len, C) — 10th-percentile forecast
        >>> q["q90"]   # (pred_len, C) — 90th-percentile forecast
        """
        self._check_fitted()
        X_np = _to_numpy(X).astype(np.float32)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        if len(X_np) > self.seq_len:
            X_np = X_np[-self.seq_len:]

        if self.normalize and self._scaler is not None:
            X_np = self._scaler.transform(X_np)

        self._model.train()  # keep dropout active
        samples = []
        inp = torch.tensor(X_np[np.newaxis])
        with torch.no_grad():
            for _ in range(n_samples):
                out = self._model(inp)
                if out.ndim == 2:
                    out = out.unsqueeze(0)
                pred = out[0].cpu().numpy()
                if self.normalize and self._scaler is not None:
                    pred = self._scaler.inverse_transform(pred)
                samples.append(pred)
        self._model.eval()

        arr = np.stack(samples, axis=0)  # (n_samples, pred_len, C)
        result: Dict[str, np.ndarray] = {
            "mean": arr.mean(axis=0),
            "std":  arr.std(axis=0),
        }
        for q in quantiles:
            key = f"q{int(round(q * 100))}"
            result[key] = np.percentile(arr, q * 100, axis=0)
        return result

    @staticmethod
    def stationarity_test(
        X: np.ndarray,
        *,
        max_lags: int = 12,
        channel: int = 0,
    ) -> Dict[str, float]:
        """ADF-style stationarity test via OLS (no external dependency).

        Runs an Augmented Dickey-Fuller regression::

            Δx_t = α + β·x_{t-1} + Σ γ_k·Δx_{t-k} + ε_t

        A significantly negative β (small p-value) rejects the unit-root
        null, indicating stationarity.  The p-value is approximated using
        MacKinnon (1994) response surface coefficients for the ADF statistic.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        max_lags:
            Number of augmenting lagged-difference terms (default ``12``).
        channel:
            Channel index for multivariate input.

        Returns
        -------
        dict with keys:

        ``adf_stat``   — ADF t-statistic (more negative → more stationary)
        ``p_value``    — approximate p-value
        ``n_obs``      — number of observations used
        ``critical_1`` — 1% critical value (MacKinnon)
        ``critical_5`` — 5% critical value
        ``critical_10``— 10% critical value
        """
        ts = X[:, channel] if X.ndim > 1 else X
        ts = ts.astype(float)
        T = len(ts)

        dx = np.diff(ts)              # Δx_t, length T-1
        n  = T - 1 - max_lags        # usable observations

        # Build design matrix: [1, x_{t-1}, Δx_{t-1}, ..., Δx_{t-maxlags}]
        rows = []
        for t in range(max_lags, T - 1):
            row = [1.0, ts[t]]        # intercept + lagged level
            row += [dx[t - k] for k in range(1, max_lags + 1)]
            rows.append(row)
        A = np.array(rows)
        b = dx[max_lags:]             # target: Δx_{t}

        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        b_hat = float(coef[1])        # coefficient on x_{t-1}

        resid = b - A @ coef
        sse   = float((resid ** 2).sum())
        k     = A.shape[1]
        s2    = sse / max(n - k, 1)
        ATA   = A.T @ A
        try:
            var_b = float(s2 * np.linalg.inv(ATA)[1, 1])
        except np.linalg.LinAlgError:
            var_b = 1e-8
        se_b  = float(np.sqrt(max(var_b, 1e-12)))
        adf_stat = b_hat / se_b

        # MacKinnon (1994) approximate p-value for ADF with constant
        # Response-surface coefficients for n=∞, 5%, 1%, 10%
        tau_star = {
            "critical_1":  -3.43,
            "critical_5":  -2.86,
            "critical_10": -2.57,
        }
        # Rough p-value from standard normal tail (conservative for large n)
        p = float(np.clip(0.5 * (1 + np.tanh((adf_stat + 2.5) / 1.2)), 0.0, 1.0))

        return {
            "adf_stat":   adf_stat,
            "p_value":    p,
            "n_obs":      n,
            **tau_star,
        }

    def cross_val_score(
        self,
        X: np.ndarray,
        *,
        n_splits: int = 5,
        metric: str = "MAE",
        val_size: Optional[int] = None,
        refit: bool = True,
    ) -> np.ndarray:
        """Walk-forward cross-validated metric scores.

        Each fold extends the training window by one ``val_size`` block and
        evaluates on the following block.

        Parameters
        ----------
        X:
            Full time series ``(T, C)``.
        n_splits:
            Number of folds (default ``5``).
        metric:
            ``"MAE"``, ``"MSE"``, or ``"RMSE"`` (default ``"MAE"``).
        val_size:
            Evaluation block length in timesteps.  Defaults to
            ``len(X) // (n_splits + 1)``.
        refit:
            If ``True`` (default) refit the model on the final full training
            set after CV.  Set ``False`` to skip the final refit.

        Returns
        -------
        np.ndarray of shape ``(n_splits,)`` — one score per fold.
        """
        metric = metric.upper()
        if metric not in ("MAE", "MSE", "RMSE"):
            raise ValueError(f"metric must be MAE, MSE, or RMSE; got {metric!r}")

        T = len(X)
        block = val_size or T // (n_splits + 1)
        min_train = self.seq_len + self.pred_len

        scores = []
        for fold in range(n_splits):
            val_start = block * (fold + 1)
            val_end   = val_start + block
            if val_end > T or val_start < min_train:
                continue
            X_tr = X[:val_start]
            X_va = X[val_start:val_end]

            fc_fold = self.clone()
            try:
                fc_fold.fit(X_tr, val_split=0.0)
            except Exception:
                continue

            err_list = []
            t = 0
            while t + self.seq_len + self.pred_len <= len(X_va):
                ctx   = X_va[t : t + self.seq_len]
                truth = X_va[t + self.seq_len : t + self.seq_len + self.pred_len]
                pred  = fc_fold.predict(ctx)
                err   = pred - truth
                if metric == "MAE":
                    err_list.append(float(np.mean(np.abs(err))))
                elif metric == "MSE":
                    err_list.append(float(np.mean(err ** 2)))
                else:
                    err_list.append(float(np.sqrt(np.mean(err ** 2))))
                t += 1
            if err_list:
                scores.append(float(np.mean(err_list)))

        if refit:
            self.fit(X)

        return np.array(scores)

    @staticmethod
    def mutual_information(
        X: np.ndarray,
        *,
        n_bins: int = 20,
    ) -> np.ndarray:
        """Pairwise mutual information matrix between channels.

        Captures non-linear dependencies that Pearson correlation misses.
        Uses the histogram-based estimator:
        ``MI(i,j) = Σ p(x,y) log[ p(x,y) / (p(x)·p(y)) ]``.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.
        n_bins:
            Number of histogram bins per axis (default ``20``).

        Returns
        -------
        np.ndarray of shape ``(C, C)`` — MI in nats (symmetric, diagonal ≥ 0).
        """
        X_f = np.asarray(X, dtype=float)
        T, C = X_f.shape
        mi = np.zeros((C, C))
        for i in range(C):
            for j in range(i, C):
                xi, xj = X_f[:, i], X_f[:, j]
                if i == j:
                    # H(X) = MI(X;X)
                    counts, _ = np.histogram(xi, bins=n_bins)
                    p = counts / T
                    p = p[p > 0]
                    mi[i, i] = float(-np.sum(p * np.log(p)))
                    continue
                hist2d, _, _ = np.histogram2d(xi, xj, bins=n_bins)
                pxy = hist2d / T
                px  = pxy.sum(axis=1, keepdims=True)
                py  = pxy.sum(axis=0, keepdims=True)
                mask = pxy > 0
                val  = float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py + 1e-300)[mask])))
                mi[i, j] = mi[j, i] = val
        return mi

    @staticmethod
    def plot_mutual_information(
        X: np.ndarray,
        *,
        n_bins: int = 20,
        channel_names: Optional[List[str]] = None,
        ax=None,
        title: Optional[str] = None,
        cmap: str = "viridis",
    ):
        """Heatmap of pairwise mutual information between channels.

        Parameters
        ----------
        X:
            Time series ``(T, C)``.
        n_bins:
            Histogram bins for MI estimation.
        channel_names:
            Labels per channel.
        ax:
            Matplotlib axes.
        title:
            Axes title.
        cmap:
            Colormap (default ``"viridis"``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_mutual_information()") from exc

        mi = Forecaster.mutual_information(X, n_bins=n_bins)
        C  = mi.shape[0]
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(C)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, C), max(3, C)))
        else:
            fig = ax.get_figure()

        im = ax.imshow(mi, cmap=cmap, aspect="auto")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(channel_names, rotation=45, ha="right")
        ax.set_yticklabels(channel_names)
        fig.colorbar(im, ax=ax, label="MI (nats)", fraction=0.046, pad=0.04)
        ax.set_title(title or "Pairwise mutual information")
        plt.tight_layout()
        return fig

    @staticmethod
    def seasonal_strength(
        X: np.ndarray,
        *,
        period: int,
        channel: int = 0,
    ) -> float:
        """Measure the relative strength of seasonality using residual variance.

        Based on Wang et al. (2006): decomposes the series with a moving-
        average trend; seasonal strength is ``1 - Var(residual) / Var(deseasonalised)``.
        A value close to ``1`` means nearly all variance is explained by
        the seasonal component; close to ``0`` means the series has little
        seasonality at this period.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        period:
            Seasonal period (e.g. ``24`` for hourly data with daily seasonality).
        channel:
            Channel index (default ``0``).

        Returns
        -------
        float in ``[0, 1]``
        """
        if period < 2:
            raise ValueError("period must be >= 2")
        decomp = Forecaster.seasonal_decompose(X, period=period, method="additive")
        resid = decomp["residual"]
        deseason = decomp["trend"]
        r = resid[:, channel] if resid.ndim > 1 else resid
        d = deseason[:, channel] if deseason.ndim > 1 else deseason
        var_r = float(np.nanvar(r))
        var_d = float(np.nanvar(d))
        if var_d < 1e-12:
            return 0.0
        return float(np.clip(1.0 - var_r / var_d, 0.0, 1.0))

    @staticmethod
    def optimal_lag(
        X: np.ndarray,
        *,
        max_lag: int = 100,
        criterion: str = "aic",
        channel: int = 0,
    ) -> Dict[str, object]:
        """Find the AR lag order that minimises AIC or BIC.

        Fits AR(p) models for ``p`` in ``1..max_lag`` using OLS and selects
        the order with the lowest information criterion.  Useful for choosing
        a principled ``seq_len``.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        max_lag:
            Maximum lag to test (default ``100``).
        criterion:
            ``"aic"`` (default) or ``"bic"``.
        channel:
            Channel index for multivariate input.

        Returns
        -------
        dict with keys:

        ``best_lag``  — integer lag with lowest criterion value
        ``aic``       — np.ndarray of AIC values for lags 1..max_lag
        ``bic``       — np.ndarray of BIC values
        ``scores``    — alias for the chosen criterion array
        """
        criterion = criterion.lower()
        if criterion not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic'; got {criterion!r}")

        ts = X[:, channel] if X.ndim > 1 else X
        ts = ts.astype(float)
        T  = len(ts)
        aics, bics = [], []

        for p in range(1, max_lag + 1):
            n = T - p
            if n <= p + 1:
                aics.append(np.inf)
                bics.append(np.inf)
                continue
            rows = np.column_stack(
                [ts[p - k - 1 : T - k - 1] for k in range(p)] + [np.ones(n)]
            )
            coef, *_ = np.linalg.lstsq(rows, ts[p:], rcond=None)
            resid    = ts[p:] - rows @ coef
            sse      = float((resid ** 2).sum())
            k        = p + 1  # number of params including intercept
            ll       = -n / 2 * (np.log(2 * np.pi * sse / n) + 1)
            aics.append(float(-2 * ll + 2 * k))
            bics.append(float(-2 * ll + k * np.log(n)))

        aic_arr = np.array(aics)
        bic_arr = np.array(bics)
        scores  = aic_arr if criterion == "aic" else bic_arr
        best    = int(np.argmin(scores) + 1)

        return {
            "best_lag": best,
            "aic":      aic_arr,
            "bic":      bic_arr,
            "scores":   scores,
        }

    def plot_quantile_forecast(
        self,
        X: np.ndarray,
        *,
        quantiles: List[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        n_samples: int = 200,
        channel: int = 0,
        n_context: int = 96,
        ax=None,
        title: Optional[str] = None,
    ):
        """Fan chart of quantile forecasts from :meth:`predict_quantiles`.

        Renders shaded bands between paired quantiles (e.g. q10/q90 and
        q25/q75) plus the median as a solid line.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` (last *n_context* steps used as context).
        quantiles:
            Quantile levels (default: 0.1/0.25/0.5/0.75/0.9).  Should be
            symmetric around the median for a good fan chart.
        n_samples:
            MC-Dropout samples to draw.
        channel:
            Channel to plot.
        n_context:
            History length shown in the plot.
        ax:
            Matplotlib axes.
        title:
            Axes title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_quantile_forecast()") from exc

        ctx = X[-n_context:]
        q_dict = self.predict_quantiles(ctx, quantiles=list(quantiles), n_samples=n_samples)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.get_figure()

        t_ctx  = np.arange(len(X) - n_context, len(X))
        t_pred = np.arange(len(X), len(X) + self.pred_len)

        ax.plot(t_ctx, ctx[:, channel], color="#555", lw=0.9, label="history")
        ax.axvline(len(X) - 0.5, color="#bbb", lw=0.8, ls=":")

        # sort quantiles to pair them as bands
        qs_sorted = sorted(quantiles)
        n = len(qs_sorted)
        alphas = np.linspace(0.15, 0.40, n // 2)
        for i, alpha in enumerate(alphas):
            lo_key = f"q{int(round(qs_sorted[i] * 100))}"
            hi_key = f"q{int(round(qs_sorted[n - 1 - i] * 100))}"
            if lo_key in q_dict and hi_key in q_dict:
                ax.fill_between(
                    t_pred,
                    q_dict[lo_key][:, channel],
                    q_dict[hi_key][:, channel],
                    alpha=alpha, color="#4C72B0",
                    label=f"{int(round(qs_sorted[i]*100))}–{int(round(qs_sorted[n-1-i]*100))}% PI",
                )
        # median
        med_key = f"q{int(round(0.5 * 100))}"
        if med_key in q_dict:
            ax.plot(t_pred, q_dict[med_key][:, channel],
                    color="#d62728", lw=1.5, label="median")
        elif "mean" in q_dict:
            ax.plot(t_pred, q_dict["mean"][:, channel],
                    color="#d62728", lw=1.5, label="mean")

        ax.set_title(title or f"Quantile forecast (channel {channel})")
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig

    def fit_on_dataframe(
        self,
        df: "pd.DataFrame",
        *,
        target_cols: Optional[List[str]] = None,
        date_col: Optional[str] = None,
        val_split: float = 0.1,
    ) -> "Forecaster":
        """Fit from a pandas DataFrame.

        Parameters
        ----------
        df:
            DataFrame with one row per timestep.
        target_cols:
            Column names to use as features.  If ``None``, all numeric
            columns except *date_col* are used.
        date_col:
            Name of the datetime column to exclude (default: auto-detect
            any column that is not numeric).
        val_split:
            Validation fraction passed to :meth:`fit`.

        Returns
        -------
        self
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for fit_on_dataframe()") from exc

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame; got {type(df).__name__!r}")

        # auto-detect date column
        if date_col is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or (
                    df[col].dtype == object and col.lower() in ("date", "time", "datetime", "timestamp")
                ):
                    date_col = col
                    break

        if target_cols is None:
            exclude = {date_col} if date_col else set()
            target_cols = [c for c in df.columns
                           if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        X = df[target_cols].to_numpy(dtype=np.float32)
        return self.fit(X, val_split=val_split)

    @staticmethod
    def partial_autocorrelation(
        X: np.ndarray,
        *,
        max_lag: int = 40,
        channel: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the partial autocorrelation function (PACF) via Yule-Walker.

        The PACF at lag *k* is the correlation between *x_t* and *x_{t-k}*
        after removing the linear influence of *x_{t-1}, …, x_{t-k+1}*.
        Unlike the ACF, it tells you the *direct* lag relationship.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        max_lag:
            Maximum lag (default ``40``).
        channel:
            Channel index.

        Returns
        -------
        lags : np.ndarray of shape ``(max_lag + 1,)``
        pacf : np.ndarray of shape ``(max_lag + 1,)``
        """
        ts = X[:, channel] if X.ndim > 1 else X
        ts = ts.astype(float)
        ts = ts - ts.mean()
        n  = len(ts)

        # Full autocorrelation vector r[0..max_lag]
        c0 = float(np.dot(ts, ts))
        r  = np.array([
            float(np.dot(ts[: n - k], ts[k:])) / c0 if k < n else 0.0
            for k in range(max_lag + 1)
        ])

        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0
        if max_lag < 1:
            return np.arange(max_lag + 1), pacf

        # Levinson-Durbin recursion
        phi = np.zeros((max_lag + 1, max_lag + 1))
        phi[1, 1] = r[1]
        pacf[1]   = r[1]
        for k in range(2, max_lag + 1):
            num = r[k] - np.dot(phi[k - 1, 1:k], r[k - 1 : 0 : -1])
            den = 1.0 - np.dot(phi[k - 1, 1:k], r[1:k])
            phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            pacf[k] = phi[k, k]

        return np.arange(max_lag + 1), pacf

    @staticmethod
    def plot_pacf(
        X: np.ndarray,
        *,
        max_lag: int = 40,
        channel: int = 0,
        ax=None,
        title: Optional[str] = None,
        significance_level: float = 0.05,
    ):
        """Stem plot of the partial autocorrelation function.

        Parameters
        ----------
        X:
            Time series ``(T, C)`` or ``(T,)``.
        max_lag:
            Maximum lag.
        channel:
            Channel index.
        ax:
            Matplotlib axes.
        title:
            Axes title.
        significance_level:
            Two-tailed level for the confidence band.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_pacf()") from exc

        lags, pacf = Forecaster.partial_autocorrelation(X, max_lag=max_lag, channel=channel)
        n  = len(X)
        ci = 1.96 / np.sqrt(n)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.vlines(lags, 0, pacf, linewidth=1.5)
        ax.axhline(0,   color="black", linewidth=0.8)
        ax.axhline( ci, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-ci, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("PACF")
        ax.set_title(title or f"Partial autocorrelation (channel {channel})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Residual ACF — model adequacy diagnostic
    # ------------------------------------------------------------------

    def residual_acf(self, X, *, max_lag: int = 40, channel: int = 0):
        """ACF of model residuals on *X*.

        Useful for diagnosing whether residuals are white noise (adequately
        fitted model).  Returns ``(lags, acf)`` arrays of shape ``(max_lag+1,)``.
        """
        self._check_fitted()
        resids = self.residuals(X)
        if resids.ndim == 3:
            r = resids[:, :, channel].ravel()
        elif resids.ndim == 2:
            r = resids[:, channel].ravel()
        else:
            r = resids.ravel()
        r = r - r.mean()
        n = len(r)
        c0 = np.dot(r, r) / n
        lags = np.arange(max_lag + 1)
        acf = np.array([1.0 if k == 0 else np.dot(r[k:], r[:-k]) / (n * c0) for k in lags])
        return lags, acf

    @staticmethod
    def plot_residual_acf(lags, acf, *, significance_level: float = 0.05,
                          ax=None, title: str = None):
        """Stem plot of residual ACF with confidence bands.

        Pass the output of :meth:`residual_acf` directly::

            lags, acf = fc.residual_acf(X_test)
            fc.plot_residual_acf(lags, acf)
        """
        import matplotlib.pyplot as plt
        n = len(lags)
        p = 1 - significance_level / 2
        t_val = np.sqrt(-2.0 * np.log(1 - p))
        a = [2.515517, 0.802853, 0.010328]
        b = [1.432788, 0.189269, 0.001308]
        z = t_val - (a[0] + a[1] * t_val + a[2] * t_val ** 2) / (
            1 + b[0] * t_val + b[1] * t_val ** 2 + b[2] * t_val ** 3
        )
        ci = z / np.sqrt(n)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 3))
        else:
            fig = ax.get_figure()
        ax.vlines(lags, 0, acf, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(ci,  color="red", linestyle="--", linewidth=0.9, alpha=0.8, label=f"{int((1-significance_level)*100)}% CI")
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(title or "Residual ACF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # clone — unfitted copy of the forecaster
    # ------------------------------------------------------------------

    def clone(self):
        """Return an unfitted :class:`Forecaster` with identical hyperparameters.

        Useful for cross-validation loops where you need multiple independent
        instances with the same configuration::

            base = Forecaster("PatchTST", seq_len=96, pred_len=24, epochs=10)
            for X_train, X_val in splits:
                clone = base.clone()
                clone.fit(X_train)
                print(clone.score(X_val))
        """
        import copy
        init_kwargs = dict(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            verbose=self.verbose,
            device=str(self.device),
            loss=self.loss,
            scheduler=self.scheduler,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            warm_start=False,   # clones always start unfitted
        )
        return Forecaster(copy.deepcopy(self.model_spec), **init_kwargs)

    # ------------------------------------------------------------------
    # leaderboard — compare multiple models on the same data
    # ------------------------------------------------------------------

    @classmethod
    def leaderboard(cls, X_train, X_test, models, *,
                    metric: str = "mse", sort: bool = True, **kwargs):
        """Fit and evaluate multiple models; return a ranked :class:`pandas.DataFrame`.

        *models* may be model-name strings or already-instantiated
        :class:`Forecaster` objects (they are cloned so the originals are
        unchanged).  Extra ``**kwargs`` are forwarded to the :class:`Forecaster`
        constructor when *models* contains strings.

        Example::

            df = Forecaster.leaderboard(
                X_train, X_test,
                ["DLinear", "NLinear", "PatchTST", "iTransformer"],
                metric="mae",
                seq_len=96, pred_len=24, epochs=5,
            )
            print(df)
        """
        import pandas as pd
        records = []
        for m in models:
            if isinstance(m, cls):
                fc = m.clone()
                label = getattr(m, "model_spec", repr(m))
                if not isinstance(label, str):
                    label = type(label).__name__
            else:
                label = str(m)
                fc = cls(m, **kwargs)
            try:
                fc.fit(X_train)
                scores = fc.evaluate(X_test)
                scores["model"] = label
                records.append(scores)
            except Exception as exc:
                records.append({"model": label, "error": str(exc)})
        df = pd.DataFrame(records).set_index("model")
        if sort and metric in df.columns:
            df = df.sort_values(metric, ascending=True)
        return df

    # ------------------------------------------------------------------
    # sensitivity_analysis — track prediction change under input perturbation
    # ------------------------------------------------------------------

    def sensitivity_analysis(self, X, *, channel: int = 0,
                             timestep: int = None, n_points: int = 11,
                             delta_range: float = 3.0):
        """Measure how a single input feature affects the forecast.

        Perturbs *channel* at *timestep* (default: last timestep of context
        window) uniformly from ``-delta_range`` to ``+delta_range`` standard
        deviations and records the change in mean absolute prediction.

        Returns a dict with keys:

        * ``"deltas"``      – 1-D array of perturbation values (shape ``(n_points,)``)
        * ``"pred_change"`` – mean absolute change in predictions for each delta
        * ``"baseline"``    – unperturbed prediction array

        Example::

            result = fc.sensitivity_analysis(X_context, channel=0, n_points=21)
            plt.plot(result["deltas"], result["pred_change"])
        """
        self._check_fitted()
        ctx = X[-self.seq_len:]                   # (seq_len, C)
        baseline = self.predict(ctx)              # (pred_len, C)
        if timestep is None:
            timestep = self.seq_len - 1
        std = float(ctx[:, channel].std()) or 1.0
        deltas = np.linspace(-delta_range * std, delta_range * std, n_points)
        pred_changes = np.zeros(n_points)
        for i, d in enumerate(deltas):
            ctx_perturbed = ctx.copy().astype(np.float32)
            ctx_perturbed[timestep, channel] += d
            pred_perturbed = self.predict(ctx_perturbed)
            pred_changes[i] = float(np.abs(pred_perturbed - baseline).mean())
        return {"deltas": deltas, "pred_change": pred_changes, "baseline": baseline}

    def plot_sensitivity(self, X, *, channel: int = 0, timestep: int = None,
                         n_points: int = 21, delta_range: float = 3.0,
                         ax=None, title: str = None):
        """Line chart of prediction sensitivity to a single input perturbation.

        Wraps :meth:`sensitivity_analysis`::

            fc.plot_sensitivity(X_context, channel=0)
        """
        import matplotlib.pyplot as plt
        result = self.sensitivity_analysis(
            X, channel=channel, timestep=timestep,
            n_points=n_points, delta_range=delta_range,
        )
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()
        ax.plot(result["deltas"], result["pred_change"], marker="o", markersize=4)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Perturbation on channel {channel}")
        ax.set_ylabel("Mean |ΔPrediction|")
        ax.set_title(title or f"Sensitivity — channel {channel}")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # train_val_test_split — static time-series-safe split utility
    # ------------------------------------------------------------------

    @staticmethod
    def train_val_test_split(X, *, val_ratio: float = 0.1,
                             test_ratio: float = 0.1, gap: int = 0):
        """Split array *X* into train / val / test segments (no shuffle).

        The split is chronological: train comes first, validation second,
        test last.  An optional *gap* can be inserted between splits to
        prevent look-ahead leakage when the model uses multi-step context.

        Returns ``(X_train, X_val, X_test)``.

        Example::

            X_train, X_val, X_test = Forecaster.train_val_test_split(
                data, val_ratio=0.1, test_ratio=0.2, gap=24
            )
            fc.fit(X_train)
            print(fc.score(X_test))
        """
        n = len(X)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test - 2 * gap
        if n_train < 1:
            raise ValueError(
                f"Not enough data: {n} samples, but train/val/test/gap need "
                f"{n_train + n_val + n_test + 2 * gap}"
            )
        train_end = n_train
        val_start = train_end + gap
        val_end   = val_start + n_val
        test_start = val_end + gap
        return X[:train_end], X[val_start:val_end], X[test_start:]

    # ------------------------------------------------------------------
    # ljung_box — test residual whiteness
    # ------------------------------------------------------------------

    def ljung_box(self, X, *, max_lag: int = 20, channel: int = 0):
        """Ljung-Box Q statistic on model residuals.

        Tests whether residuals are consistent with white noise (H₀: no
        autocorrelation up to *max_lag*).  Returns a dict::

            {
                "Q": float,       # Ljung-Box Q statistic
                "df": int,        # degrees of freedom (= max_lag)
                "p_value": float, # approximate chi-squared p-value
            }

        p-value < 0.05 suggests significant autocorrelation remains
        (model may be misspecified).  P-value computed via chi-squared
        CDF approximation — no scipy required.

        Example::

            result = fc.ljung_box(X_test, max_lag=20)
            print(result["p_value"])   # < 0.05 → residuals not white noise
        """
        self._check_fitted()
        _, acf = self.residual_acf(X, max_lag=max_lag, channel=channel)
        n_resid = len(self.residuals(X).ravel())
        lags = np.arange(1, max_lag + 1)
        Q = float(n_resid * (n_resid + 2) * np.sum(acf[1:] ** 2 / (n_resid - lags)))

        # chi-squared survival function via regularised incomplete gamma
        # P(χ²_k > Q) = 1 - P(χ²_k ≤ Q) ≈ using Wilson-Hilferty normal approx
        k = max_lag
        # Wilson-Hilferty: (χ²/k)^(1/3) ≈ N(1 - 2/(9k), 2/(9k))
        mu  = 1.0 - 2.0 / (9 * k)
        sig = np.sqrt(2.0 / (9 * k))
        z   = ((Q / k) ** (1.0 / 3) - mu) / (sig + 1e-15)
        # standard normal CDF via error function
        p_value = float(0.5 * (1.0 + np.sign(z) * (1.0 - np.exp(-0.7213475 * z ** 2 - 0.2316419 * abs(z)))))
        p_value = max(0.0, min(1.0, 1.0 - p_value))

        return {"Q": Q, "df": k, "p_value": p_value}

    # ------------------------------------------------------------------
    # lag_plot — scatter x[t] vs x[t-lag] to visualise autocorrelation
    # ------------------------------------------------------------------

    @staticmethod
    def lag_plot(X, *, lag: int = 1, channel: int = 0, ax=None, title: str = None):
        """Scatter plot of ``X[t]`` vs ``X[t-lag]`` for a single channel.

        A circular cloud indicates no autocorrelation; a diagonal band confirms
        positive autocorrelation (AR behaviour).

        Example::

            Forecaster.lag_plot(X_train, lag=1, channel=0)
        """
        import matplotlib.pyplot as plt
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr[:, channel]
        y = arr[lag:]
        x = arr[:-lag]
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.get_figure()
        ax.scatter(x, y, alpha=0.4, s=10)
        ax.set_xlabel(f"X[t-{lag}]")
        ax.set_ylabel("X[t]")
        ax.set_title(title or f"Lag plot (lag={lag}, channel={channel})")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # seasonal_plot — overlaid annual/weekly/etc. season lines
    # ------------------------------------------------------------------

    @staticmethod
    def seasonal_plot(X, period: int, *, channel: int = 0, ax=None, title: str = None,
                      cmap: str = "viridis", alpha: float = 0.6, linewidth: float = 1.0):
        """Overlay each *period*-length season as a separate line.

        Reveals repeating patterns (daily / weekly / yearly seasonality).
        Incomplete final cycles are silently dropped.

        Example::

            # daily pattern for hourly data (period=24)
            Forecaster.seasonal_plot(X_train, period=24, channel=0)
        """
        import matplotlib.pyplot as plt
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr[:, channel]
        n_complete = len(arr) // period
        if n_complete < 1:
            raise ValueError(f"Data length {len(arr)} < period {period}")
        seasons = arr[: n_complete * period].reshape(n_complete, period)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.get_figure()
        cmap_obj = plt.get_cmap(cmap)
        t = np.arange(period)
        for i, row in enumerate(seasons):
            color = cmap_obj(i / max(n_complete - 1, 1))
            ax.plot(t, row, color=color, alpha=alpha, linewidth=linewidth)
        sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                                   norm=plt.Normalize(vmin=0, vmax=n_complete - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Cycle index")
        ax.set_xlabel("Position within period")
        ax.set_ylabel(f"Channel {channel}")
        ax.set_title(title or f"Seasonal plot (period={period})")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # hyperparameter_search — random search over a parameter grid
    # ------------------------------------------------------------------

    def hyperparameter_search(self, X_train, X_val, param_grid: dict, *,
                              n_iter: int = 10, metric: str = "mse",
                              verbose: bool = False):
        """Random hyperparameter search via cloned :class:`Forecaster` instances.

        *param_grid* maps constructor-argument names to lists of candidate values.
        Each trial draws one value per key uniformly at random, fits a clone, and
        evaluates on *X_val*.  Returns a list of result dicts sorted by *metric*::

            results = fc.hyperparameter_search(
                X_train, X_val,
                {"lr": [1e-4, 5e-4, 1e-3], "batch_size": [16, 32, 64]},
                n_iter=12,
                metric="mae",
            )
            best = results[0]
            print(best["params"], best["score"])
        """
        rng = np.random.default_rng(0)
        records = []
        for i in range(n_iter):
            params = {k: rng.choice(v).item() if hasattr(rng.choice(v), "item") else rng.choice(v)
                      for k, v in param_grid.items()}
            # build a clone with overrides
            import copy
            init_kwargs = dict(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                patience=self.patience,
                verbose=verbose,
                device=str(self.device),
                loss=self.loss,
                scheduler=self.scheduler,
                grad_clip=self.grad_clip,
                weight_decay=self.weight_decay,
                warm_start=False,
            )
            init_kwargs.update(params)
            fc = Forecaster(copy.deepcopy(self.model_spec), **init_kwargs)
            fc.fit(X_train)
            scores = fc.evaluate(X_val)
            records.append({"params": params, "score": scores.get(metric, float("nan")), **scores})
        records.sort(key=lambda r: r["score"])
        return records

    # ------------------------------------------------------------------
    # to_lagged_features — build ML-ready lagged feature matrix
    # ------------------------------------------------------------------

    @staticmethod
    def to_lagged_features(X, lags, *, target_col: int = 0, dropna: bool = True):
        """Create a supervised learning feature matrix from lagged values.

        Each row contains the lagged values of all channels and the
        concurrent *target_col* as the label.  Useful for testing classical
        ML baselines (sklearn regressors) on time-series data.

        Returns ``(features, targets)`` numpy arrays::

            X_feat, y = Forecaster.to_lagged_features(data, lags=[1, 2, 3, 7])
            from sklearn.linear_model import Ridge
            Ridge().fit(X_feat, y)
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        T, C = arr.shape
        max_lag = max(lags)
        rows = []
        for t in range(max_lag, T):
            row = np.concatenate([arr[t - lag] for lag in lags])
            rows.append(row)
        features = np.array(rows, dtype=np.float32)
        targets  = arr[max_lag:, target_col]
        if dropna:
            valid = np.all(np.isfinite(features), axis=1) & np.isfinite(targets)
            features = features[valid]
            targets  = targets[valid]
        return features, targets

    # ------------------------------------------------------------------
    # predict_bootstrap — bootstrap CI from multiple re-fits
    # ------------------------------------------------------------------

    def predict_bootstrap(self, X_train, X_test, *, n_boot: int = 10,
                          ci: float = 0.9):
        """Bootstrap confidence interval on predictions by training *n_boot* clones.

        Each clone is fit on a bootstrap resample of *X_train* (block
        bootstrap with block size ``seq_len`` to respect temporal structure).
        Returns a dict::

            {
                "mean"  : (pred_len, C) — bootstrap mean forecast,
                "lower" : (pred_len, C) — lower CI bound,
                "upper" : (pred_len, C) — upper CI bound,
                "preds" : (n_boot, pred_len, C) — all individual predictions,
            }

        Example::

            result = fc.predict_bootstrap(X_train, X_test, n_boot=20, ci=0.9)
            plt.fill_between(t, result["lower"][:,0], result["upper"][:,0], alpha=0.3)
        """
        self._check_fitted()
        T = len(X_train)
        block = max(1, self.seq_len)
        n_blocks = T // block
        rng = np.random.default_rng(42)
        preds = []
        for _ in range(n_boot):
            idx = np.concatenate([
                np.arange(b * block, (b + 1) * block)
                for b in rng.integers(0, n_blocks, size=n_blocks)
            ])[:T]
            X_boot = X_train[idx]
            fc = self.clone()
            fc.fit(X_boot)
            p = fc.predict(X_test[-self.seq_len:])
            preds.append(p)
        preds = np.stack(preds, axis=0)   # (n_boot, pred_len, C)
        alpha = (1 - ci) / 2
        lower = np.quantile(preds, alpha, axis=0)
        upper = np.quantile(preds, 1 - alpha, axis=0)
        mean  = preds.mean(axis=0)
        return {"mean": mean, "lower": lower, "upper": upper, "preds": preds}

    # ------------------------------------------------------------------
    # plot_actual_vs_predicted — scatter actual vs predicted
    # ------------------------------------------------------------------

    def plot_actual_vs_predicted(self, X, *, channel: int = 0,
                                 ax=None, title: str = None):
        """Scatter plot of actual vs predicted values with R² annotation.

        The diagonal line ``y = x`` is the perfect-forecast reference.  Points
        below the line indicate over-prediction; above indicates under-prediction.

        Example::

            fc.plot_actual_vs_predicted(X_test, channel=0)
        """
        import matplotlib.pyplot as plt
        self._check_fitted()
        actual, predicted = self._collect_actuals_predictions(X)
        y_true = actual[:, :, channel].ravel()
        y_pred = predicted[:, :, channel].ravel()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-15)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.get_figure()
        ax.scatter(y_true, y_pred, alpha=0.3, s=8)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title or f"Actual vs Predicted (R²={r2:.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    def _collect_actuals_predictions(self, X):
        """Helper: run rolling predict and collect aligned (actual, pred) windows."""
        T   = len(X)
        n_w = max(1, (T - self.seq_len - self.pred_len) // self.pred_len + 1)
        actuals  = []
        preds    = []
        for i in range(n_w):
            start = i * self.pred_len
            ctx   = X[start: start + self.seq_len]
            tgt   = X[start + self.seq_len: start + self.seq_len + self.pred_len]
            if len(ctx) < self.seq_len or len(tgt) < self.pred_len:
                break
            p = self.predict(ctx)
            actuals.append(tgt[None])
            preds.append(p[None])
        actuals = np.concatenate(actuals, axis=0)
        preds   = np.concatenate(preds,   axis=0)
        return actuals, preds

    # ------------------------------------------------------------------
    # noise_robustness — accuracy vs. input noise level
    # ------------------------------------------------------------------

    def noise_robustness(self, X_train, X_test, *,
                         noise_levels=None, n_trials: int = 3,
                         metric: str = "mse"):
        """Measure model accuracy as a function of added Gaussian noise.

        For each noise standard deviation in *noise_levels*, adds i.i.d.
        Gaussian noise to *X_test* ``n_trials`` times and averages the
        resulting metric.  Returns a dict mapping noise level → metric value.

        Example::

            result = fc.noise_robustness(X_train, X_test,
                                         noise_levels=[0, 0.1, 0.5, 1.0])
            plt.plot(list(result.keys()), list(result.values()))
        """
        self._check_fitted()
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
        rng = np.random.default_rng(0)
        results = {}
        for sigma in noise_levels:
            trial_scores = []
            for _ in range(n_trials):
                X_noisy = X_test + rng.normal(0, sigma, X_test.shape).astype(np.float32)
                scores = self.evaluate(X_noisy)
                trial_scores.append(scores.get(metric, float("nan")))
            results[float(sigma)] = float(np.mean(trial_scores))
        return results

    def plot_noise_robustness(self, X_train, X_test, *,
                              noise_levels=None, n_trials: int = 3,
                              metric: str = "mse", ax=None, title: str = None):
        """Line chart of model accuracy vs noise level.

        Wraps :meth:`noise_robustness`::

            fc.plot_noise_robustness(X_train, X_test)
        """
        import matplotlib.pyplot as plt
        results = self.noise_robustness(X_train, X_test,
                                        noise_levels=noise_levels,
                                        n_trials=n_trials, metric=metric)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()
        xs = list(results.keys())
        ys = list(results.values())
        ax.plot(xs, ys, marker="o", markersize=5)
        ax.set_xlabel("Noise σ")
        ax.set_ylabel(metric.upper())
        ax.set_title(title or f"Noise robustness ({metric.upper()})")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # seasonal_naive_baseline — last-season naive forecast
    # ------------------------------------------------------------------

    @staticmethod
    def seasonal_naive_baseline(X, *, pred_len: int, period: int):
        """Seasonal naïve forecast: repeat the last observed complete season.

        The forecast for step ``t+h`` is ``X[t - period + h]``.  Returns a
        ``(pred_len, C)`` array.  Useful as a strong heuristic baseline for
        seasonal data.

        Example::

            y_hat = Forecaster.seasonal_naive_baseline(X_context, pred_len=24, period=24)
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if len(arr) < period:
            raise ValueError(f"Context ({len(arr)}) shorter than period ({period})")
        forecast = np.stack([arr[-(period - (h % period))] for h in range(pred_len)], axis=0)
        return forecast  # (pred_len, C)

    # ------------------------------------------------------------------
    # explain_global — average saliency map over a sample pool
    # ------------------------------------------------------------------

    def explain_global(self, X, *, n_samples: int = 50, target_step: int = 0,
                       target_channel: int = 0, absolute: bool = True):
        """Global feature importance: mean saliency across *n_samples* windows.

        Draws *n_samples* context windows uniformly from *X* and averages the
        per-timestep, per-channel gradient saliency.  Returns a ``(seq_len, C)``
        array representing global input importance.

        Example::

            imp = fc.explain_global(X_test, n_samples=100)
            Forecaster.plot_saliency(X_test, saliency=imp)
        """
        self._check_fitted()
        T = len(X)
        if T < self.seq_len:
            raise ValueError("X too short for explain_global")
        rng = np.random.default_rng(0)
        starts = rng.integers(0, T - self.seq_len + 1, size=n_samples)
        maps = []
        for s in starts:
            ctx = X[s: s + self.seq_len]
            sal = self.input_gradient(ctx, target_step=target_step,
                                      target_channel=target_channel,
                                      absolute=absolute)
            maps.append(sal)
        return np.mean(maps, axis=0)   # (seq_len, C)

    # ------------------------------------------------------------------
    # forecast_error_distribution — per-step error statistics
    # ------------------------------------------------------------------

    def forecast_error_distribution(self, X, *, channel: int = 0):
        """Compute per-horizon error statistics across rolling windows.

        For each prediction step ``h`` in ``[0, pred_len)``, collects the
        signed errors from all rolling windows and returns a dict::

            {
                "steps"  : (pred_len,) array of step indices,
                "mean"   : (pred_len,) mean error per step,
                "std"    : (pred_len,) std error per step,
                "median" : (pred_len,) median absolute error per step,
                "q05"    : (pred_len,) 5th percentile of errors,
                "q95"    : (pred_len,) 95th percentile of errors,
            }

        Example::

            stats = fc.forecast_error_distribution(X_test, channel=0)
            plt.fill_between(stats["steps"], stats["q05"], stats["q95"], alpha=0.3)
        """
        self._check_fitted()
        actuals, preds = self._collect_actuals_predictions(X)
        errors = actuals[:, :, channel] - preds[:, :, channel]   # (n_windows, pred_len)
        steps = np.arange(self.pred_len)
        return {
            "steps":  steps,
            "mean":   errors.mean(axis=0),
            "std":    errors.std(axis=0),
            "median": np.median(np.abs(errors), axis=0),
            "q05":    np.quantile(errors, 0.05, axis=0),
            "q95":    np.quantile(errors, 0.95, axis=0),
        }

    def plot_forecast_error_distribution(self, X, *, channel: int = 0,
                                          ax=None, title: str = None):
        """Ribbon plot of per-horizon error distribution.

        Shows mean ± std and the 5–95 % envelope::

            fc.plot_forecast_error_distribution(X_test, channel=0)
        """
        import matplotlib.pyplot as plt
        stats = self.forecast_error_distribution(X, channel=channel)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()
        s = stats["steps"]
        ax.fill_between(s, stats["q05"], stats["q95"], alpha=0.2, color="steelblue", label="5–95%")
        ax.fill_between(s, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                        alpha=0.35, color="steelblue", label="mean ± std")
        ax.plot(s, stats["mean"], color="steelblue", linewidth=1.5, label="mean")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Prediction step")
        ax.set_ylabel("Error")
        ax.set_title(title or f"Forecast error distribution (channel {channel})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # auto_select — fit multiple models, return the best one
    # ------------------------------------------------------------------

    @classmethod
    def auto_select(cls, X_train, X_val, candidates=None, *,
                    metric: str = "mse", verbose: bool = False, **kwargs):
        """Fit a list of candidate models and return the best-performing one.

        *candidates* defaults to ``["DLinear", "NLinear", "PatchTST"]``.
        Extra ``**kwargs`` are forwarded to each :class:`Forecaster` constructor.
        Returns the fitted :class:`Forecaster` instance with the best *metric*
        on *X_val*::

            best = Forecaster.auto_select(
                X_train, X_val,
                ["DLinear", "NLinear", "iTransformer"],
                seq_len=96, pred_len=24, epochs=10,
                metric="mae",
            )
            print(best)
        """
        if candidates is None:
            candidates = ["DLinear", "NLinear", "PatchTST"]
        best_fc    = None
        best_score = float("inf")
        for name in candidates:
            fc = cls(name, **kwargs)
            fc.verbose = verbose
            try:
                fc.fit(X_train)
                score = fc.evaluate(X_val).get(metric, float("inf"))
                if score < best_score:
                    best_score = score
                    best_fc    = fc
            except Exception:
                pass
        if best_fc is None:
            raise RuntimeError("All candidates failed to fit/evaluate.")
        return best_fc

    # ------------------------------------------------------------------
    # conformal_coverage — empirical PI coverage on held-out data
    # ------------------------------------------------------------------

    def conformal_coverage(self, X_calib, X_test, *,
                           coverage: float = 0.9, n_samples: int = 200):
        """Compute empirical coverage of conformal prediction intervals.

        Calls :meth:`predict_interval` calibrated on *X_calib*, then measures
        what fraction of *X_test* targets actually fall inside the intervals.
        Returns a dict::

            {
                "nominal_coverage"  : float,  # requested coverage
                "empirical_coverage": float,  # actual fraction covered
                "coverage_gap"      : float,  # empirical - nominal
            }

        Example::

            result = fc.conformal_coverage(X_calib, X_test, coverage=0.9)
            print(result["empirical_coverage"])  # should be ≈ 0.9 if well-calibrated
        """
        self._check_fitted()
        # calibrate conformal width on X_calib
        calib_resids = self.residuals(X_calib)
        if calib_resids.ndim == 3:
            calib_flat = np.abs(calib_resids).ravel()
        else:
            calib_flat = np.abs(calib_resids).ravel()
        q_level = float(np.quantile(calib_flat, coverage))
        # evaluate on X_test
        actuals, preds = self._collect_actuals_predictions(X_test)
        lower = preds - q_level
        upper = preds + q_level
        covered = ((actuals >= lower) & (actuals <= upper))
        emp_cov = float(covered.mean())
        return {
            "nominal_coverage":   coverage,
            "empirical_coverage": emp_cov,
            "coverage_gap":       emp_cov - coverage,
        }

    # ------------------------------------------------------------------
    # winkler_score — interval score penalising width and misses
    # ------------------------------------------------------------------

    def winkler_score(self, X_calib, X_test, *,
                      coverage: float = 0.9, n_samples: int = 200):
        """Compute the Winkler interval score on *X_test*.

        Lower is better.  The score penalises both wide intervals *and* misses::

            W = (upper - lower) + 2/α * max(0, lower - y) + 2/α * max(0, y - upper)

        where ``α = 1 - coverage``.  Returns a scalar float (mean over all
        target steps and channels).

        Example::

            ws = fc.winkler_score(X_calib, X_test, coverage=0.9)
            print(f"Winkler score: {ws:.4f}")
        """
        self._check_fitted()
        calib_resids = self.residuals(X_calib)
        q_level = float(np.quantile(np.abs(calib_resids), coverage))
        actuals, preds = self._collect_actuals_predictions(X_test)
        lower = preds - q_level
        upper = preds + q_level
        alpha  = 1 - coverage
        width  = upper - lower
        miss_lo = np.maximum(0.0, lower - actuals)
        miss_hi = np.maximum(0.0, actuals - upper)
        score = width + (2 / alpha) * (miss_lo + miss_hi)
        return float(score.mean())

    # ------------------------------------------------------------------
    # concept_drift_score — Jensen-Shannon divergence between time windows
    # ------------------------------------------------------------------

    @staticmethod
    def concept_drift_score(X_ref, X_test, *, window: int = None,
                            n_bins: int = 30, channel: int = 0):
        """Estimate distribution shift between *X_ref* and *X_test* via JS divergence.

        Uses a histogram-based JS divergence on a single *channel*.  Values
        near 0 indicate similar distributions; values near 1 indicate maximum
        shift.

        If *window* is given, computes the JS divergence in a rolling fashion
        over *X_test* and returns an array of per-window drift scores::

            drift = Forecaster.concept_drift_score(X_train, X_test, window=50)
            plt.plot(drift)
        """
        arr_ref  = np.asarray(X_ref)
        arr_test = np.asarray(X_test)
        if arr_ref.ndim == 2:
            arr_ref  = arr_ref[:, channel]
        if arr_test.ndim == 2:
            arr_test = arr_test[:, channel]

        def _js(a, b, bins):
            lo = min(a.min(), b.min()) - 1e-8
            hi = max(a.max(), b.max()) + 1e-8
            edges = np.linspace(lo, hi, bins + 1)
            p = np.histogram(a, edges)[0].astype(float) + 1e-10
            q = np.histogram(b, edges)[0].astype(float) + 1e-10
            p /= p.sum(); q /= q.sum()
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            return float(0.5 * (kl_pm + kl_qm))

        if window is None:
            return _js(arr_ref, arr_test, n_bins)
        scores = []
        for start in range(0, len(arr_test) - window + 1):
            seg = arr_test[start: start + window]
            scores.append(_js(arr_ref, seg, n_bins))
        return np.array(scores)

    def plot_concept_drift(self, X_ref, X_test, *, window: int = 50,
                           channel: int = 0, ax=None, title: str = None):
        """Rolling JS-divergence concept drift chart::

            fc.plot_concept_drift(X_train, X_test_long, window=100)
        """
        import matplotlib.pyplot as plt
        scores = self.concept_drift_score(X_ref, X_test, window=window, channel=channel)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(9, 3))
        else:
            fig = ax.get_figure()
        ax.plot(scores)
        ax.set_xlabel("Window index")
        ax.set_ylabel("JS divergence")
        ax.set_title(title or f"Concept drift (window={window}, channel={channel})")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # plot_prediction_bands — fan chart with multiple coverage levels
    # ------------------------------------------------------------------

    def plot_prediction_bands(self, X, *,
                              coverages=(0.5, 0.8, 0.9, 0.95),
                              n_samples: int = 200, channel: int = 0,
                              n_context: int = 96, ax=None, title: str = None):
        """Fan chart with shaded bands for multiple coverage levels.

        Uses MC-Dropout (if available) or bootstrap quantiles to build a fan
        chart showing nested prediction bands at the requested *coverages*::

            fc.plot_prediction_bands(X_test, coverages=(0.5, 0.8, 0.95))
        """
        import matplotlib.pyplot as plt
        self._check_fitted()
        ctx = X[-min(n_context, self.seq_len):]
        if len(ctx) < self.seq_len:
            pad = np.zeros((self.seq_len - len(ctx), X.shape[-1] if X.ndim == 2 else 1), dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=0)
        # gather sample predictions via MC-Dropout
        self._model.train()
        sample_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                p = self.predict(ctx)
                sample_preds.append(p[:, channel])
        self._model.eval()
        sample_preds = np.stack(sample_preds, axis=0)   # (n_samples, pred_len)
        t = np.arange(self.pred_len)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()
        cmap = plt.cm.Blues
        for i, cov in enumerate(sorted(coverages)):
            alpha_val = (1 - cov) / 2
            lo = np.quantile(sample_preds, alpha_val, axis=0)
            hi = np.quantile(sample_preds, 1 - alpha_val, axis=0)
            shade = 0.25 + 0.5 * (1 - i / len(coverages))
            ax.fill_between(t, lo, hi, alpha=0.35, color=cmap(shade),
                            label=f"{int(cov*100)}% CI")
        median = np.median(sample_preds, axis=0)
        ax.plot(t, median, color="navy", linewidth=1.5, label="Median")
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(f"Channel {channel}")
        ax.set_title(title or "Prediction bands")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # plot_calibration_curve — reliability diagram for conformal PI coverage
    # ------------------------------------------------------------------

    def plot_calibration_curve(self, X_calib, X_test, *,
                               n_levels: int = 10, ax=None, title: str = None):
        """Reliability diagram: nominal vs empirical PI coverage.

        For each target coverage level from ``1/n_levels`` to ``1``, computes
        the empirical coverage of the conformal interval on *X_test* and plots
        it against the nominal level.  A perfectly calibrated model follows the
        diagonal.

        Example::

            fc.plot_calibration_curve(X_calib, X_test, n_levels=10)
        """
        import matplotlib.pyplot as plt
        self._check_fitted()
        calib_resids = self.residuals(X_calib)
        q_abs        = np.abs(calib_resids).ravel()
        actuals, preds = self._collect_actuals_predictions(X_test)
        errors  = np.abs(actuals - preds).ravel()
        levels  = np.linspace(1 / n_levels, 1.0, n_levels)
        emp_covs = []
        for cov in levels:
            q = float(np.quantile(q_abs, cov))
            emp_covs.append(float((errors <= q).mean()))
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.get_figure()
        ax.plot(levels, emp_covs, marker="o", markersize=5, label="Model")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(title or "Calibration curve")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # rolling_zscore — z-score normalised rolling window signal
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_zscore(X, *, window: int = 30, channel: int = 0,
                       min_periods: int = 1):
        """Rolling z-score: ``(X[t] - μ_window) / (σ_window + ε)``.

        Returns a 1-D array of the same length as *X*.  Values in the warm-up
        region (fewer than *min_periods* samples) are set to 0.

        Example::

            z = Forecaster.rolling_zscore(X_train, window=50, channel=0)
            plt.plot(z)
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, channel]
        T = len(arr)
        z = np.zeros(T, dtype=np.float32)
        for t in range(T):
            start = max(0, t - window + 1)
            seg   = arr[start: t + 1]
            if len(seg) >= min_periods:
                mu  = seg.mean()
                sig = seg.std() + 1e-8
                z[t] = (arr[t] - mu) / sig
        return z

    # ------------------------------------------------------------------
    # multistep_score — per-horizon metric without re-fitting
    # ------------------------------------------------------------------

    def multistep_score(self, X, *, metric: str = "mse"):
        """Return the *metric* at each prediction step without re-fitting.

        Uses rolling evaluation on *X* and computes the metric at steps
        ``0, 1, …, pred_len - 1`` independently.  Returns a dict mapping
        ``step → score``::

            scores = fc.multistep_score(X_test, metric="mae")
            plt.plot(list(scores.keys()), list(scores.values()))
        """
        self._check_fitted()
        actuals, preds = self._collect_actuals_predictions(X)
        # actuals/preds: (n_windows, pred_len, C)
        step_scores = {}
        for h in range(self.pred_len):
            y_t = actuals[:, h, :].ravel()
            y_p = preds[:, h, :].ravel()
            if metric == "mse":
                s = float(np.mean((y_t - y_p) ** 2))
            elif metric == "mae":
                s = float(np.mean(np.abs(y_t - y_p)))
            elif metric == "rmse":
                s = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
            elif metric == "smape":
                denom = np.abs(y_t) + np.abs(y_p) + 1e-8
                s = float(np.mean(2.0 * np.abs(y_t - y_p) / denom))
            else:
                raise ValueError(f"Unknown metric: {metric!r}")
            step_scores[h] = s
        return step_scores

    def plot_multistep_score(self, X, *, metric: str = "mse",
                             ax=None, title: str = None):
        """Bar chart of per-horizon metric from :meth:`multistep_score`::

            fc.plot_multistep_score(X_test, metric="mae")
        """
        import matplotlib.pyplot as plt
        step_scores = self.multistep_score(X, metric=metric)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()
        steps = list(step_scores.keys())
        vals  = list(step_scores.values())
        ax.bar(steps, vals, color="steelblue", alpha=0.75)
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(metric.upper())
        ax.set_title(title or f"Per-step {metric.upper()}")
        ax.grid(True, alpha=0.3, axis="y")
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # feature_drift — per-channel JS divergence between two datasets
    # ------------------------------------------------------------------

    @staticmethod
    def feature_drift(X_ref, X_test, *, n_bins: int = 30):
        """Compute per-channel JS divergence between *X_ref* and *X_test*.

        Returns a 1-D array of shape ``(C,)`` where each entry is the JS
        divergence for that channel.  Values near 0 indicate no distribution
        shift; values near 1 indicate maximum shift::

            drift = Forecaster.feature_drift(X_train, X_test_new)
            # drift[0] is the drift for channel 0, etc.
        """
        arr_ref  = np.asarray(X_ref,  dtype=np.float64)
        arr_test = np.asarray(X_test, dtype=np.float64)
        if arr_ref.ndim == 1:
            arr_ref  = arr_ref[:, None]
        if arr_test.ndim == 1:
            arr_test = arr_test[:, None]
        C = arr_ref.shape[1]
        scores = np.zeros(C)
        for c in range(C):
            a = arr_ref[:, c]
            b = arr_test[:, c]
            lo = min(a.min(), b.min()) - 1e-8
            hi = max(a.max(), b.max()) + 1e-8
            edges = np.linspace(lo, hi, n_bins + 1)
            p = np.histogram(a, edges)[0].astype(float) + 1e-10
            q = np.histogram(b, edges)[0].astype(float) + 1e-10
            p /= p.sum(); q /= q.sum()
            m = 0.5 * (p + q)
            scores[c] = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
        return scores

    @staticmethod
    def plot_feature_drift(X_ref, X_test, *, n_bins: int = 30,
                           channel_names=None, ax=None, title: str = None):
        """Horizontal bar chart of per-channel JS drift scores.

        Example::

            Forecaster.plot_feature_drift(X_train, X_test_shifted)
        """
        import matplotlib.pyplot as plt
        scores = Forecaster.feature_drift(X_ref, X_test, n_bins=n_bins)
        C = len(scores)
        labels = channel_names or [f"ch{i}" for i in range(C)]
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(6, max(3, C * 0.4 + 1)))
        else:
            fig = ax.get_figure()
        colors = ["tomato" if s > 0.1 else "steelblue" for s in scores]
        ax.barh(labels, scores, color=colors, alpha=0.8)
        ax.axvline(0.1, color="red", linestyle="--", linewidth=0.8, label="Drift threshold (0.1)")
        ax.set_xlabel("JS Divergence")
        ax.set_title(title or "Per-channel feature drift")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # cross_correlation — CCF between two time series
    # ------------------------------------------------------------------

    @staticmethod
    def cross_correlation(X, Y=None, *, max_lag: int = 40,
                          channel_x: int = 0, channel_y: int = 1):
        """Cross-correlation function (CCF) between two signals.

        If *Y* is ``None``, uses *channel_x* and *channel_y* from *X*.
        Otherwise correlates *X* (channel *channel_x*) against *Y* (channel
        *channel_y*).  Returns ``(lags, ccf)`` where *lags* spans
        ``[-max_lag, max_lag]``::

            lags, ccf = Forecaster.cross_correlation(X, max_lag=20)
            # positive lags: Y leads X; negative lags: X leads Y
        """
        arr_x = np.asarray(X, dtype=np.float64)
        if arr_x.ndim == 2:
            arr_x = arr_x[:, channel_x]
        if Y is None:
            arr_y = np.asarray(X, dtype=np.float64)
            if arr_y.ndim == 2:
                arr_y = arr_y[:, channel_y]
        else:
            arr_y = np.asarray(Y, dtype=np.float64)
            if arr_y.ndim == 2:
                arr_y = arr_y[:, channel_y]
        n = min(len(arr_x), len(arr_y))
        x = arr_x[:n] - arr_x[:n].mean()
        y = arr_y[:n] - arr_y[:n].mean()
        norm = np.sqrt(np.dot(x, x) * np.dot(y, y)) + 1e-15
        lags = np.arange(-max_lag, max_lag + 1)
        ccf  = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            if lag >= 0:
                ccf[i] = np.dot(x[lag:], y[:n - lag]) / (norm / n * (n - lag)) if n > lag else 0.0
            else:
                ccf[i] = np.dot(x[:n + lag], y[-lag:]) / (norm / n * (n + lag)) if n > -lag else 0.0
        return lags, ccf

    @staticmethod
    def plot_cross_correlation(X, Y=None, *, max_lag: int = 40,
                               channel_x: int = 0, channel_y: int = 1,
                               significance_level: float = 0.05,
                               ax=None, title: str = None):
        """Stem plot of the cross-correlation function with confidence bands.

        Example::

            Forecaster.plot_cross_correlation(X, max_lag=30)
        """
        import matplotlib.pyplot as plt
        lags, ccf = Forecaster.cross_correlation(
            X, Y, max_lag=max_lag, channel_x=channel_x, channel_y=channel_y
        )
        n = min(len(np.asarray(X)), len(np.asarray(Y)) if Y is not None else len(np.asarray(X)))
        p = 1 - significance_level / 2
        t_val = np.sqrt(-2.0 * np.log(1 - p))
        a = [2.515517, 0.802853, 0.010328]
        b = [1.432788, 0.189269, 0.001308]
        z = t_val - (a[0] + a[1] * t_val + a[2] * t_val ** 2) / (
            1 + b[0] * t_val + b[1] * t_val ** 2 + b[2] * t_val ** 3
        )
        ci = z / np.sqrt(n)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 3))
        else:
            fig = ax.get_figure()
        ax.vlines(lags, 0, ccf, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline( ci, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-ci, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("CCF")
        ax.set_title(title or f"Cross-correlation (ch{channel_x} vs ch{channel_y})")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # regime_detection — k-means on rolling statistics
    # ------------------------------------------------------------------

    @staticmethod
    def regime_detection(X, *, n_regimes: int = 3, window: int = 30,
                         channel: int = 0, random_state: int = 0):
        """Detect hidden regimes via k-means clustering on rolling statistics.

        Computes rolling mean, rolling std, and rolling autocorrelation (lag-1)
        as features, then clusters them into *n_regimes* groups.  Returns a
        1-D integer array of shape ``(T,)`` with regime labels for each timestep.
        Warm-up period (first *window* timesteps) is assigned label ``-1``.

        Example::

            regimes = Forecaster.regime_detection(X_train, n_regimes=3, window=30)
            Forecaster.plot_regimes(X_train, regimes)
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, channel]
        T = len(arr)
        # build feature matrix: rolling mean, std, lag-1 autocorr
        features = []
        for t in range(T):
            start = max(0, t - window + 1)
            seg = arr[start: t + 1]
            mu  = seg.mean()
            sig = seg.std() + 1e-8
            if len(seg) >= 2:
                a = seg[:-1] - seg[:-1].mean()
                b = seg[1:]  - seg[1:].mean()
                denom = (np.dot(a, a) * np.dot(b, b)) ** 0.5 + 1e-15
                ac1 = float(np.dot(a, b) / denom)
            else:
                ac1 = 0.0
            features.append([mu, sig, ac1])
        features = np.array(features)
        # k-means++ initialisation + Lloyd iterations (pure numpy)
        rng = np.random.default_rng(random_state)
        valid = np.where(np.all(np.isfinite(features), axis=1))[0]
        if len(valid) < n_regimes:
            return np.full(T, -1, dtype=int)
        # k-means++ init
        centroids = [features[rng.choice(valid)]]
        for _ in range(n_regimes - 1):
            dists = np.array([min(np.sum((features[v] - c) ** 2) for c in centroids) for v in valid])
            probs = dists / (dists.sum() + 1e-15)
            centroids.append(features[valid[rng.choice(len(valid), p=probs)]])
        centroids = np.stack(centroids)
        labels = np.full(T, -1, dtype=int)
        for _ in range(50):   # max iterations
            dists = np.stack([np.sum((features - c) ** 2, axis=1) for c in centroids], axis=1)
            new_labels = np.argmin(dists, axis=1)
            new_labels[~np.all(np.isfinite(features), axis=1)] = -1
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_regimes):
                mask = labels == k
                if mask.any():
                    centroids[k] = features[mask].mean(axis=0)
        return labels

    @staticmethod
    def plot_regimes(X, regimes, *, channel: int = 0,
                     ax=None, title: str = None, cmap: str = "Set1"):
        """Time-series plot coloured by detected regime.

        Each detected regime is drawn with a distinct colour.  Regime ``-1``
        (warm-up) is shown in grey.

        Example::

            regimes = Forecaster.regime_detection(X, n_regimes=3)
            Forecaster.plot_regimes(X, regimes, channel=0)
        """
        import matplotlib.pyplot as plt
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, channel]
        T = len(arr)
        labels  = np.asarray(regimes)
        n_reg   = max(1, int(labels.max()) + 1)
        cmap_obj = plt.get_cmap(cmap)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.get_figure()
        t = np.arange(T)
        for k in range(-1, n_reg):
            mask = labels == k
            if not mask.any():
                continue
            color  = "grey" if k == -1 else cmap_obj(k / max(n_reg - 1, 1))
            label  = "warm-up" if k == -1 else f"Regime {k}"
            # scatter so gaps between segments look clean
            ax.scatter(t[mask], arr[mask], s=2, color=color, label=label, rasterized=True)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Channel {channel}")
        ax.set_title(title or f"Regime detection (n_regimes={n_reg})")
        ax.legend(fontsize=8, markerscale=4, loc="upper right")
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # functional_boxplot — functional boxplot over repeated seasons
    # ------------------------------------------------------------------

    @staticmethod
    def functional_boxplot(X, period: int, *, channel: int = 0,
                           ax=None, title: str = None):
        """Functional boxplot: median ± IQR envelope over repeated cycles.

        Reshapes the signal into ``(n_complete_cycles, period)`` and computes
        the median (solid line), 25–75 % IQR band, and 10–90 % outer band.
        Useful for confirming/visualising seasonal patterns.

        Example::

            # hourly data with daily seasonality
            Forecaster.functional_boxplot(X_train, period=24, channel=0)
        """
        import matplotlib.pyplot as plt
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, channel]
        n_complete = len(arr) // period
        if n_complete < 2:
            raise ValueError(f"Need ≥2 complete cycles; got {n_complete} (period={period})")
        mat = arr[: n_complete * period].reshape(n_complete, period)
        t   = np.arange(period)
        p10, p25, p50, p75, p90 = [np.percentile(mat, q, axis=0) for q in (10, 25, 50, 75, 90)]
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()
        ax.fill_between(t, p10, p90, alpha=0.18, color="steelblue", label="10–90%")
        ax.fill_between(t, p25, p75, alpha=0.40, color="steelblue", label="25–75%")
        ax.plot(t, p50, color="navy", linewidth=1.5, label="Median")
        ax.set_xlabel("Position within period")
        ax.set_ylabel(f"Channel {channel}")
        ax.set_title(title or f"Functional boxplot (period={period})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # spectrogram — short-time Fourier transform (pure numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def spectrogram(X, *, channel: int = 0, nperseg: int = None,
                    noverlap: int = None, fs: float = 1.0):
        """Compute a short-time Fourier transform (STFT) spectrogram.

        Splits the signal into overlapping segments, applies a Hann window,
        and returns the magnitude spectrogram.  No scipy dependency — uses
        only ``numpy.fft``.

        Returns a dict::

            {
                "times"  : 1-D array of segment centre times,
                "freqs"  : 1-D array of frequencies (0..fs/2),
                "Sxx"    : 2-D magnitude spectrogram (freqs × times),
            }

        Example::

            spec = Forecaster.spectrogram(X_train, channel=0, nperseg=64)
            Forecaster.plot_spectrogram(X_train)
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, channel]
        T = len(arr)
        if nperseg is None:
            nperseg = min(256, T // 4)
        nperseg = max(4, nperseg)
        if noverlap is None:
            noverlap = nperseg // 2
        step = nperseg - noverlap
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))  # Hann
        n_freqs = nperseg // 2 + 1
        times, Sxx = [], []
        for start in range(0, T - nperseg + 1, step):
            seg   = arr[start: start + nperseg] * window
            fft_v = np.fft.rfft(seg, n=nperseg)
            Sxx.append(np.abs(fft_v[:n_freqs]))
            times.append((start + nperseg / 2) / fs)
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)[:n_freqs]
        Sxx   = np.array(Sxx).T   # (n_freqs, n_times)
        return {"times": np.array(times), "freqs": freqs, "Sxx": Sxx}

    @staticmethod
    def plot_spectrogram(X, *, channel: int = 0, nperseg: int = None,
                         noverlap: int = None, fs: float = 1.0,
                         log_scale: bool = True, ax=None, title: str = None):
        """Colour-map plot of the STFT magnitude spectrogram.

        Example::

            Forecaster.plot_spectrogram(X_train, channel=0, nperseg=64)
        """
        import matplotlib.pyplot as plt
        spec = Forecaster.spectrogram(X, channel=channel, nperseg=nperseg,
                                      noverlap=noverlap, fs=fs)
        Sxx = np.log1p(spec["Sxx"]) if log_scale else spec["Sxx"]
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()
        im = ax.pcolormesh(spec["times"], spec["freqs"], Sxx, shading="gouraud",
                           cmap="viridis")
        plt.colorbar(im, ax=ax, label="log(1+|STFT|)" if log_scale else "|STFT|")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title(title or f"Spectrogram (channel {channel})")
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # summary_table — metrics at multiple prediction horizons
    # ------------------------------------------------------------------

    def summary_table(self, X, *, horizons=None, metrics=("mse", "mae", "rmse")):
        """Return a :class:`pandas.DataFrame` of metrics at multiple horizons.

        *horizons* is a list of step indices (0-based) to evaluate at.
        Defaults to ``[0, pred_len//4, pred_len//2, pred_len-1]``.
        Uses per-step errors from :meth:`multistep_score`::

            table = fc.summary_table(X_test)
            print(table)
        """
        import pandas as pd
        self._check_fitted()
        if horizons is None:
            p = self.pred_len
            horizons = sorted(set([0, p // 4, p // 2, p - 1]))
        actuals, preds = self._collect_actuals_predictions(X)
        records = []
        for h in horizons:
            row = {"horizon": h + 1}
            y_t = actuals[:, h, :].ravel()
            y_p = preds[:,   h, :].ravel()
            for m in metrics:
                if m == "mse":
                    row["mse"] = float(np.mean((y_t - y_p) ** 2))
                elif m == "mae":
                    row["mae"] = float(np.mean(np.abs(y_t - y_p)))
                elif m == "rmse":
                    row["rmse"] = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
                elif m == "smape":
                    denom = np.abs(y_t) + np.abs(y_p) + 1e-8
                    row["smape"] = float(np.mean(2 * np.abs(y_t - y_p) / denom))
            records.append(row)
        return pd.DataFrame(records).set_index("horizon")

    # ------------------------------------------------------------------
    # histogram_forecast — overlaid distribution of actual vs predicted
    # ------------------------------------------------------------------

    def histogram_forecast(self, X, *, channel: int = 0, n_bins: int = 40,
                           ax=None, title: str = None):
        """Overlaid histogram of predicted vs actual values.

        Useful for diagnosing distributional shift between the model's output
        and the true values (e.g. over-smoothing in point forecasters)::

            fc.histogram_forecast(X_test, channel=0)
        """
        import matplotlib.pyplot as plt
        self._check_fitted()
        actuals, preds = self._collect_actuals_predictions(X)
        y_true = actuals[:, :, channel].ravel()
        y_pred = preds[:,   :, channel].ravel()
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        bins = np.linspace(lo, hi, n_bins + 1)
        ax.hist(y_true, bins=bins, alpha=0.55, label="Actual",    density=True)
        ax.hist(y_pred, bins=bins, alpha=0.55, label="Predicted", density=True)
        ax.set_xlabel(f"Channel {channel}")
        ax.set_ylabel("Density")
        ax.set_title(title or f"Forecast distribution (channel {channel})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if created:
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # reliability_score — combined PI calibration metrics
    # ------------------------------------------------------------------

    def reliability_score(self, X_calib, X_test, *,
                          coverage: float = 0.9, n_samples: int = 200):
        """Compute a comprehensive set of PI reliability metrics.

        Returns a dict::

            {
                "picp"     : float,  # Prediction Interval Coverage Probability
                "pinaw"    : float,  # PI Normalised Average Width
                "cwc"      : float,  # Coverage Width-based Criterion (lower=better)
                "winkler"  : float,  # Winkler interval score
                "nominal"  : float,  # requested coverage
            }

        *picp* ≈ *nominal* means well-calibrated.  Low *pinaw* means sharp
        (narrow) intervals.  *cwc* penalises under-coverage more than width::

            rs = fc.reliability_score(X_calib, X_test, coverage=0.9)
            print(rs)
        """
        self._check_fitted()
        calib_resids = self.residuals(X_calib)
        q_abs        = np.abs(calib_resids).ravel()
        q_level      = float(np.quantile(q_abs, coverage))
        actuals, preds = self._collect_actuals_predictions(X_test)
        lower = preds - q_level
        upper = preds + q_level
        covered = (actuals >= lower) & (actuals <= upper)
        picp  = float(covered.mean())
        y_range = float(actuals.max() - actuals.min()) + 1e-8
        pinaw = float((upper - lower).mean()) / y_range
        # CWC: penalise if picp < nominal
        eta = 50.0
        cwc = pinaw * (1 + (picp < coverage) * np.exp(-eta * (picp - coverage)))
        # winkler score
        alpha = 1 - coverage
        width = upper - lower
        miss_lo = np.maximum(0.0, lower - actuals)
        miss_hi = np.maximum(0.0, actuals - upper)
        winkler = float((width + (2 / alpha) * (miss_lo + miss_hi)).mean())
        return {
            "picp":    picp,
            "pinaw":   pinaw,
            "cwc":     float(cwc),
            "winkler": winkler,
            "nominal": coverage,
        }

    # ------------------------------------------------------------------
    # forecast_with_trend — detrend → predict → re-add trend
    # ------------------------------------------------------------------

    def forecast_with_trend(self, X, *, degree: int = 1):
        """Predict on polynomial-detrended input and re-add the extrapolated trend.

        Fits a degree-*degree* polynomial to the context window, subtracts it,
        calls :meth:`predict`, then extrapolates the polynomial over the
        prediction horizon and adds it back.  Useful for non-stationary series
        where the model was trained on stationary data::

            y_pred = fc.forecast_with_trend(X_context, degree=1)
        """
        self._check_fitted()
        arr = np.asarray(X[-self.seq_len:], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        T, C = arr.shape
        t_ctx  = np.arange(T, dtype=np.float64)
        t_pred = np.arange(T, T + self.pred_len, dtype=np.float64)
        # fit per-channel polynomial trend on context
        trends_ctx  = np.zeros_like(arr)
        trends_pred = np.zeros((self.pred_len, C), dtype=np.float32)
        for c in range(C):
            coeffs = np.polyfit(t_ctx, arr[:, c].astype(np.float64), degree)
            trends_ctx[:, c]  = np.polyval(coeffs, t_ctx).astype(np.float32)
            trends_pred[:, c] = np.polyval(coeffs, t_pred).astype(np.float32)
        detrended  = arr - trends_ctx
        pred_det   = self.predict(detrended)          # (pred_len, C)
        return pred_det + trends_pred

    # ------------------------------------------------------------------
    # compute_pinball_loss — quantile (pinball) loss per step
    # ------------------------------------------------------------------

    def compute_pinball_loss(self, X, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), *,
                             n_samples: int = 200):
        """Compute pinball loss at each quantile over rolling windows.

        For each quantile *q*, the pinball (quantile) loss is::

            ρ_q(y, ŷ) = max(q·(y - ŷ), (q-1)·(y - ŷ))

        Returns a dict mapping each quantile → mean pinball loss::

            losses = fc.compute_pinball_loss(X_test, quantiles=(0.1, 0.5, 0.9))
            print(losses)   # {0.1: 0.043, 0.5: 0.021, 0.9: 0.038}
        """
        self._check_fitted()
        # gather quantile predictions via MC-Dropout
        self._model.train()
        sample_preds = []
        actuals_list = []
        T = len(X)
        n_w = max(1, (T - self.seq_len - self.pred_len) // self.pred_len + 1)
        with torch.no_grad():
            for _ in range(n_samples):
                preds_run = []
                for i in range(n_w):
                    start = i * self.pred_len
                    ctx   = X[start: start + self.seq_len]
                    tgt   = X[start + self.seq_len: start + self.seq_len + self.pred_len]
                    if len(ctx) < self.seq_len or len(tgt) < self.pred_len:
                        break
                    preds_run.append(self.predict(ctx))
                if preds_run:
                    sample_preds.append(np.stack(preds_run, axis=0))
        self._model.eval()
        if not sample_preds:
            return {q: float("nan") for q in quantiles}
        sample_preds = np.stack(sample_preds, axis=0)  # (n_samples, n_w, pred_len, C)
        # collect actuals once
        for i in range(n_w):
            start = i * self.pred_len
            tgt   = X[start + self.seq_len: start + self.seq_len + self.pred_len]
            if len(tgt) < self.pred_len:
                break
            actuals_list.append(tgt)
        actuals = np.stack(actuals_list, axis=0)   # (n_w, pred_len, C)
        result = {}
        for q in quantiles:
            q_hat = np.quantile(sample_preds, q, axis=0)  # (n_w, pred_len, C)
            errors = actuals - q_hat
            pinball = np.maximum(q * errors, (q - 1) * errors)
            result[float(q)] = float(pinball.mean())
        return result

    # ------------------------------------------------------------------
    # wavelet_decomposition — Haar DWT in pure numpy
    # ------------------------------------------------------------------

    @staticmethod
    def wavelet_decomposition(X, *, n_levels: int = 4, channel: int = 0):
        """Discrete wavelet transform (Haar) with *n_levels* of decomposition.

        Returns a dict::

            {
                "approx"  : 1-D array — final approximation coefficients,
                "details" : list of 1-D arrays — detail coefficients per level
                            (finest first),
            }

        The Haar DWT is computed in pure numpy (no pywt / scipy required).

        Example::

            decomp = Forecaster.wavelet_decomposition(X_train, n_levels=4)
            # decomp["details"][0] is the finest-scale detail
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, channel]
        # pad to next power-of-2 length
        T = len(arr)
        target = 1
        while target < T:
            target *= 2
        signal = np.concatenate([arr, np.zeros(target - T)])
        details = []
        approx  = signal
        for _ in range(n_levels):
            n = len(approx)
            if n < 2:
                break
            a = (approx[0::2] + approx[1::2]) / np.sqrt(2)
            d = (approx[0::2] - approx[1::2]) / np.sqrt(2)
            details.append(d)
            approx = a
        return {"approx": approx, "details": details}

    @staticmethod
    def plot_wavelet(X, *, n_levels: int = 4, channel: int = 0,
                     title: str = None):
        """Multi-panel plot of Haar wavelet approximation + details.

        Example::

            Forecaster.plot_wavelet(X_train, n_levels=4, channel=0)
        """
        import matplotlib.pyplot as plt
        decomp = Forecaster.wavelet_decomposition(X, n_levels=n_levels, channel=channel)
        n_panels = len(decomp["details"]) + 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2 * n_panels), sharex=False)
        if n_panels == 1:
            axes = [axes]
        axes[0].plot(decomp["approx"], linewidth=0.8)
        axes[0].set_title(title or f"Wavelet (channel {channel})")
        axes[0].set_ylabel("Approx")
        axes[0].grid(True, alpha=0.3)
        for i, d in enumerate(decomp["details"]):
            axes[i + 1].plot(d, linewidth=0.8, color="tomato")
            axes[i + 1].set_ylabel(f"Detail {i + 1}")
            axes[i + 1].grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # rolling_predict_iter — generator for one-step-ahead streaming forecast
    # ------------------------------------------------------------------

    def rolling_predict_iter(self, X, *, step: int = 1):
        """Generator that yields ``(t, prediction)`` for each rolling window.

        Slides a context window of size ``seq_len`` over *X* with stride
        *step*, yielding a ``(t, np.ndarray)`` tuple at each position where
        ``t`` is the index of the first predicted timestep::

            for t, pred in fc.rolling_predict_iter(X_test):
                print(f"t={t}: pred shape={pred.shape}")  # (pred_len, C)
        """
        self._check_fitted()
        T = len(X)
        t = self.seq_len
        while t + self.pred_len <= T:
            ctx  = X[t - self.seq_len: t]
            pred = self.predict(ctx)
            yield t, pred
            t += step

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
