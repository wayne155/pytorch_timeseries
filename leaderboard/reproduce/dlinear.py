#!/usr/bin/env python3
"""Reproduce DLinear long-term forecast results and populate leaderboard/results/.

Runs every (pred_len, seed) combination, sharded across the available GPUs
(one worker process per card, pinned via ``CUDA_VISIBLE_DEVICES``), then
builds the leaderboard JSON.

Usage:
    python leaderboard/reproduce/dlinear.py                       # all 4 GPUs
    python leaderboard/reproduce/dlinear.py --devices 0 1         # only GPU 0,1
    python leaderboard/reproduce/dlinear.py --pred-lens 96 192
    python leaderboard/reproduce/dlinear.py --epochs 5 --patience 3   # smoke test
    python leaderboard/reproduce/dlinear.py --cpu                 # single CPU process
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "leaderboard" / "results"
BUILD_SCRIPT = ROOT / "scripts" / "build_leaderboard.py"

PRED_LENS = [96, 192, 336, 720]
SEEDS = [0, 1, 2, 3, 4]


def _run_jobs(jobs, args, device: str) -> None:
    """Run a list of (pred_len, seed) jobs sequentially on one device."""
    from torch_timeseries.leaderboard.experiment import LeaderboardExperiment

    for pred_len, seed in jobs:
        print(f"[{device}] DLinear/{args.dataset} "
              f"windows={args.windows} pred_len={pred_len} seed={seed}")
        LeaderboardExperiment(
            model="DLinear",
            task="Forecast",
            dataset=args.dataset,
            results_dir=str(RESULTS_DIR),
            # ForecastConfig
            windows=args.windows,
            pred_len=pred_len,
            horizon=1,
            train_ratio=0.7,
            test_ratio=0.2,
            time_enc=1,
            # RuntimeConfig
            epochs=args.epochs,
            patience=args.patience,
            batch_size=32,
            lr=0.0001,
            device=device,
            save_dir=str(ROOT / "results"),
        ).run(seeds=[seed])


def _worker(args) -> None:
    """Worker entry point: runs the jobs passed via ``--jobs`` on cuda:0.

    Launched by the controller with ``CUDA_VISIBLE_DEVICES`` pinned to one GPU,
    so the single visible card is always ``cuda:0``.
    """
    jobs = [tuple(int(x) for x in spec.split(":")) for spec in args.jobs]
    device = "cpu" if args.cpu else "cuda:0"
    _run_jobs(jobs, args, device)


def _controller(args) -> None:
    """Build the full job list, shard it across devices, and dispatch workers."""
    jobs = [(p, s) for p in args.pred_lens for s in args.seeds]
    print(
        f"DLinear / {args.dataset}  windows={args.windows}  "
        f"pred_lens={args.pred_lens}  seeds={args.seeds}  "
        f"epochs={args.epochs}  patience={args.patience}  "
        f"({len(jobs)} runs)"
    )

    if args.cpu or not args.devices:
        # Single in-process run on CPU.
        _run_jobs(jobs, args, "cpu")
    else:
        # One subprocess per GPU, each pinned via CUDA_VISIBLE_DEVICES.
        shards = {gpu: [] for gpu in args.devices}
        for i, job in enumerate(jobs):
            shards[args.devices[i % len(args.devices)]].append(job)

        procs = []
        for gpu, gpu_jobs in shards.items():
            if not gpu_jobs:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [
                sys.executable, str(pathlib.Path(__file__).resolve()),
                "--worker",
                "--dataset", args.dataset,
                "--windows", str(args.windows),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--jobs", *[f"{p}:{s}" for p, s in gpu_jobs],
            ]
            print(f"GPU {gpu}: {len(gpu_jobs)} runs -> {gpu_jobs}")
            procs.append((gpu, subprocess.Popen(cmd, env=env, cwd=str(ROOT))))

        failed = []
        for gpu, proc in procs:
            if proc.wait() != 0:
                failed.append(gpu)
        if failed:
            print(f"Worker(s) on GPU {failed} failed", file=sys.stderr)
            sys.exit(1)

    print("\nAll runs complete. Building leaderboard JSON...")
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)], cwd=str(ROOT), text=True
    )
    if result.returncode != 0:
        print("build_leaderboard.py failed", file=sys.stderr)
        sys.exit(1)
    print("\nDone. View results at: leaderboard/webapp/  (npm run dev)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-lens", nargs="+", type=int, default=PRED_LENS)
    parser.add_argument("--windows", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--devices", nargs="*", type=int, default=[0, 1, 2, 3],
                        help="GPU ids to shard runs across (empty -> CPU)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force a single CPU process")
    # Internal: worker mode.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", nargs="*", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
