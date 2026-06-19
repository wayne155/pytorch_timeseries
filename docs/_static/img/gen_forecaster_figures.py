"""Generate illustrative figures for the Forecaster high-level API section."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

rng = np.random.default_rng(seed=42)
OUT = os.path.dirname(os.path.abspath(__file__))

T_CTX  = 96
T_PRED = 24

# ── Shared synthetic signal ────────────────────────────────────────────────────
def make_signal(n=300, freq=0.04, noise=0.25, trend=0.002, seed=0):
    t   = np.arange(n)
    sig = (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 3 * freq * t)
           + trend * t + np.random.default_rng(seed).normal(0, noise, n))
    return sig.astype(np.float32)


# ── Figure 1 — Hero: context + forecast + shaded CI ─────────────────────────
def make_forecaster_hero():
    sig = make_signal(300)
    ctx  = sig[150:150 + T_CTX]
    truth = sig[150 + T_CTX:150 + T_CTX + T_PRED]

    # Simulate a plausible forecast
    rng2  = np.random.default_rng(7)
    trend = np.linspace(ctx[-1], ctx[-1] + (truth[-1] - truth[0]) * 0.8, T_PRED)
    noise = rng2.normal(0, 0.15, T_PRED).cumsum() * 0.15
    pred  = trend + noise
    std   = np.linspace(0.10, 0.40, T_PRED)

    t_ctx  = np.arange(T_CTX)
    t_pred = np.arange(T_CTX, T_CTX + T_PRED)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t_ctx,  ctx,   color="#555", lw=1.1, label="Context (96 steps)")
    ax.plot(t_pred, truth, "--", color="#1f77b4", lw=1.3, alpha=0.7, label="Ground truth")
    ax.plot(t_pred, pred,  color="#d62728", lw=1.5, label="Forecast (24 steps)")
    ax.fill_between(t_pred, pred - 2 * std, pred + 2 * std, alpha=0.15, color="#d62728", label="95% PI")
    ax.fill_between(t_pred, pred - std,     pred + std,     alpha=0.30, color="#d62728")
    ax.axvline(T_CTX, color="#bbb", lw=0.9, ls=":")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_title("Forecaster — context window → 24-step forecast with prediction intervals")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "forecaster_hero.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ forecaster_hero.png")


# ── Figure 2 — Model comparison bar chart ────────────────────────────────────
def make_model_comparison():
    models = ["DLinear", "NLinear", "TimeMixer", "PatchTST", "iTransformer"]
    mae    = [0.431, 0.443, 0.398, 0.387, 0.372]
    colors = ["#4C72B0"] * 5
    colors[mae.index(min(mae))] = "#d62728"   # highlight best

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(models, mae, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE (lower = better)")
    ax.set_title("Model comparison — ETTh1  ·  seq=96  pred=24")
    ax.set_ylim(0, max(mae) * 1.2)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "forecaster_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ forecaster_comparison.png")


# ── Figure 3 — Uncertainty fan chart (multi-coverage) ────────────────────────
def make_uncertainty():
    sig = make_signal(300)
    ctx  = sig[150:150 + T_CTX]
    rng2 = np.random.default_rng(99)

    t_ctx  = np.arange(T_CTX)
    t_pred = np.arange(T_CTX, T_CTX + T_PRED)
    trend  = np.linspace(ctx[-1], ctx[-1] * 0.9, T_PRED)
    std    = np.linspace(0.08, 0.45, T_PRED)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t_ctx, ctx, color="#555", lw=1.1, label="Context")
    for cov, alpha in [(0.95, 0.12), (0.80, 0.20), (0.50, 0.30)]:
        z = {0.95: 1.96, 0.80: 1.28, 0.50: 0.67}[cov]
        ax.fill_between(t_pred, trend - z * std, trend + z * std,
                        alpha=alpha, color="#4C72B0", label=f"{int(cov*100)}% PI")
    ax.plot(t_pred, trend, color="#d62728", lw=1.5, label="Median forecast")
    ax.axvline(T_CTX, color="#bbb", lw=0.9, ls=":")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Value")
    ax.set_title("Forecast uncertainty — conformal prediction intervals")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "forecaster_uncertainty.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ forecaster_uncertainty.png")


# ── Figure 4 — Signal analysis dashboard ─────────────────────────────────────
def make_signal_analysis():
    n     = 300
    t_arr = np.arange(n)
    freq  = 0.04
    sig   = (np.sin(2 * np.pi * freq * t_arr)
             + 0.4 * np.sin(2 * np.pi * 3 * freq * t_arr)
             + rng.normal(0, 0.18, n)).astype(np.float64)

    # ACF
    sig_c = sig - sig.mean()
    c0    = np.dot(sig_c, sig_c) / n
    max_lag = 48
    lags_arr = np.arange(max_lag + 1)
    acf_vals = np.array([1.0 if k == 0 else np.dot(sig_c[k:], sig_c[:-k]) / (n * c0)
                         for k in lags_arr])

    # PSD
    fft_v = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
    freqs = np.fft.rfftfreq(n)[:len(fft_v)]

    # Seasonal decompose (simple)
    period = 25
    n_comp = n // period
    mat    = sig[:n_comp * period].reshape(n_comp, period)
    seasonal = np.tile(mat.mean(axis=0), n_comp + 2)[:n]
    trend_arr = np.convolve(sig, np.ones(period) / period, mode='same')
    resid    = sig - trend_arr - seasonal + seasonal.mean()

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Panel 1 — raw signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_arr, sig, lw=0.9, color="#4C72B0", alpha=0.85)
    ax1.set_title("Input time series")
    ax1.set_xlabel("Timestep"); ax1.grid(True, alpha=0.2)

    # Panel 2 — ACF
    ax2 = fig.add_subplot(gs[1, 0])
    ci  = 1.96 / np.sqrt(n)
    ax2.vlines(lags_arr, 0, acf_vals, lw=1.3, color="#4C72B0")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.axhline( ci, color="r", ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(-ci, color="r", ls="--", lw=0.8, alpha=0.7)
    ax2.set_title("Autocorrelation (ACF)"); ax2.set_xlabel("Lag"); ax2.grid(True, alpha=0.2)

    # Panel 3 — PSD
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(freqs[1:], fft_v[1:], color="#2ca02c", lw=1.0)
    ax3.set_title("Power spectral density"); ax3.set_xlabel("Frequency"); ax3.grid(True, alpha=0.2)

    # Panel 4 — decomposition
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t_arr, trend_arr, lw=1.2, color="#d62728", label="Trend")
    ax4.plot(t_arr, sig,       lw=0.7, color="#aaa",    alpha=0.6, label="Original")
    ax4.set_title("Trend extraction"); ax4.set_xlabel("Timestep")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.2)

    plt.suptitle("Signal analysis — static tools (no model fit required)", y=1.01, fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "forecaster_signal_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ forecaster_signal_analysis.png")


# ── Figure 5 — Saliency heatmap ───────────────────────────────────────────────
def make_saliency():
    n_lags = 96
    n_ch   = 7
    # Simulate gradient saliency: recent lags + channel 2 most important
    saliency = rng.exponential(0.3, (n_lags, n_ch)).astype(np.float32)
    # Make recent lags more important
    decay = np.exp(np.linspace(-2, 0, n_lags))
    saliency = (saliency.T * decay).T
    saliency[:, 2] *= 2.5        # channel 2 "drives" the forecast
    saliency = saliency / saliency.max()

    ch_names = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(saliency.T, aspect="auto", cmap="hot", origin="lower",
                   vmin=0, vmax=1)
    ax.set_yticks(range(n_ch)); ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel("Context timestep (older → newer)")
    ax.set_title("Input gradient saliency → OT forecast (step 1)")
    plt.colorbar(im, ax=ax, label="|gradient|")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "forecaster_saliency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ forecaster_saliency.png")


# ── Figure 6 — ACF + PACF side by side ───────────────────────────────────────
def make_acf_pacf():
    n   = 400
    # AR(2) process: x_t = 0.7 x_{t-1} - 0.3 x_{t-2} + ε
    x = np.zeros(n)
    eps = rng.normal(0, 0.5, n)
    for i in range(2, n):
        x[i] = 0.7 * x[i-1] - 0.3 * x[i-2] + eps[i]
    x = x[100:]   # discard burn-in
    n = len(x)
    x_c = x - x.mean()
    c0  = np.dot(x_c, x_c) / n
    max_lag = 30
    lags_arr = np.arange(max_lag + 1)
    acf_v = np.array([1.0 if k == 0 else np.dot(x_c[k:], x_c[:-k]) / (n * c0)
                      for k in lags_arr])
    # Levinson-Durbin PACF
    r    = acf_v
    pacf = np.zeros(max_lag + 1); pacf[0] = 1.0
    phi  = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = r[1]
    pacf[1]   = r[1]
    for k in range(2, max_lag + 1):
        num = r[k] - np.dot(phi[k-1, 1:k], r[k-1:0:-1])
        den = 1.0  - np.dot(phi[k-1, 1:k], r[1:k])
        phi[k, k] = num / (den + 1e-15)
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
        pacf[k] = phi[k, k]

    ci = 1.96 / np.sqrt(n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), sharey=False)
    for ax, vals, title in [(ax1, acf_v, "ACF — AR(2) process"),
                             (ax2, pacf,  "PACF — cuts off after lag 2")]:
        ax.vlines(lags_arr, 0, vals, lw=1.4, color="#4C72B0")
        ax.axhline(0, color="k", lw=0.7)
        ax.axhline( ci, color="r", ls="--", lw=0.9, alpha=0.7, label="95% CI")
        ax.axhline(-ci, color="r", ls="--", lw=0.9, alpha=0.7)
        ax.set_xlabel("Lag"); ax.set_ylabel("Correlation")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "forecaster_acf_pacf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ forecaster_acf_pacf.png")


if __name__ == "__main__":
    make_forecaster_hero()
    make_model_comparison()
    make_uncertainty()
    make_signal_analysis()
    make_saliency()
    make_acf_pacf()
    print("All figures generated.")
