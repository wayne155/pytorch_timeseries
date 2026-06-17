"""Generate illustrative figures for Imputation, Anomaly Detection, and Classification tasks."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

rng = np.random.default_rng(seed=7)
OUT = os.path.dirname(os.path.abspath(__file__))


# ─── Imputation figure ────────────────────────────────────────────────────────
def make_imputation():
    T = 96
    t = np.linspace(0, 4 * np.pi, T)
    signal = np.sin(t) + 0.3 * np.sin(3 * t) + rng.normal(0, 0.08, T)

    mask = rng.random(T) < 0.5          # True = masked (missing)
    masked = signal.copy()
    masked[mask] = np.nan

    # "Reconstruction": lerp between known neighbors (simple illustration)
    recon = signal.copy()
    recon += rng.normal(0, 0.05, T)     # add slight noise to look like model output

    fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True)

    for ax, (vals, col, title) in zip(axes, [
        (masked, "#888888", "Masked input (50% missing)"),
        (recon,  "#d62728", "Model reconstruction"),
    ]):
        # draw original as thin grey reference
        ax.plot(t, signal, color="#cccccc", lw=1.0, zorder=1, label="original")
        # shade masked regions
        for i in range(T - 1):
            if mask[i]:
                ax.axvspan(t[i], t[i + 1], color="#eeeeee", alpha=0.8, zorder=0)
        if vals is masked:
            # draw only observed points
            obs = np.where(~mask)[0]
            ax.plot(t[obs], signal[obs], "o", color="#1f77b4", ms=2.5,
                    zorder=3, label="observed")
        else:
            ax.plot(t, vals, color=col, lw=1.4, zorder=3, label="reconstruction")
            ax.plot(t, signal, "--", color="#1f77b4", lw=0.9, alpha=0.6,
                    zorder=2, label="ground truth")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time step")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "imputation.png"), dpi=120)
    plt.close()
    print("imputation.png saved")


# ─── Anomaly detection figure ─────────────────────────────────────────────────
def make_anomaly_detection():
    T = 200
    t = np.arange(T)
    signal = np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.1, T)

    # inject two anomaly bursts
    for start, end in [(50, 65), (140, 155)]:
        signal[start:end] += rng.uniform(1.5, 2.5, end - start) * rng.choice([-1, 1])

    # anomaly score: simulated reconstruction error (higher at anomaly regions)
    score = np.abs(rng.normal(0, 0.15, T))
    for start, end in [(50, 65), (140, 155)]:
        score[start:end] = np.abs(rng.normal(1.2, 0.25, end - start))

    threshold = np.percentile(score, 75)
    detected = score > threshold

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # top: raw signal with anomaly shading
    ax1.plot(t, signal, color="#1f77b4", lw=0.9, label="signal")
    for start, end in [(50, 65), (140, 155)]:
        ax1.axvspan(start, end, color="#d62728", alpha=0.15, label="ground-truth anomaly")
    # deduplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax1.set_title("Anomaly detection — reconstruction-based", fontsize=9)
    ax1.set_ylabel("value")

    # bottom: anomaly score
    ax2.fill_between(t, score, alpha=0.6, color="#ff7f0e", label="anomaly score")
    ax2.axhline(threshold, color="#d62728", lw=1.2, ls="--", label=f"threshold ({threshold:.2f})")
    # highlight detected
    ax2.fill_between(t, score, threshold, where=detected, alpha=0.5, color="#d62728")
    ax2.set_ylabel("score")
    ax2.set_xlabel("time step")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "anomaly_detection.png"), dpi=120)
    plt.close()
    print("anomaly_detection.png saved")


# ─── Classification figure ────────────────────────────────────────────────────
def make_classification():
    classes = ["Class 0\n(C=0)", "Class 1\n(C=1)", "Class 2\n(C=2)", "Class 3\n(C=3)"]
    # simulated per-class accuracy for two models
    acc_dlinear = [0.72, 0.65, 0.81, 0.78]
    acc_gru     = [0.80, 0.74, 0.88, 0.85]

    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars1 = ax.bar(x - w / 2, acc_dlinear, w, color="#4C72B0", label="DLinear")
    bars2 = ax.bar(x + w / 2, acc_gru,     w, color="#DD8452", label="GRUClassifier")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Per-class accuracy — EthanolConcentration (4 classes)", fontsize=9)
    ax.axhline(0.25, color="#aaa", lw=0.8, ls=":", label="random baseline")
    ax.legend(fontsize=9)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "classification.png"), dpi=120)
    plt.close()
    print("classification.png saved")


if __name__ == "__main__":
    make_imputation()
    make_anomaly_detection()
    make_classification()
