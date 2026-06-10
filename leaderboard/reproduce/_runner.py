"""Shared lane-parallel runner for the reproduce scripts.

Each model script defines its exact hyperparameters and calls ``run_grid``.
Jobs (dataset, pred_len, seed) are sharded across lanes — GPU x lanes-per-GPU,
one worker process per lane pinned via ``CUDA_VISIBLE_DEVICES`` — and the
leaderboard JSON is rebuilt at the end.

Common CLI flags (available on every script):
    --devices 0 1 2 3      GPUs to use (empty -> CPU)
    --lanes-per-gpu 2      worker processes per GPU
    --datasets / --pred-lens / --seeds   override the script's grid
    --cpu                  single CPU process (debug)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "leaderboard" / "results"
BUILD_SCRIPT = ROOT / "leaderboard" / "build_leaderboard.py"


def _axis_flag(axis: str) -> str:
    flag = axis if axis.endswith("s") else f"{axis}s"
    return "--" + flag.replace("_", "-")


def _axis_dest(axis: str) -> str:
    return f"grid_{axis}"


def _coerce_value(raw: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    return raw


def _grid_jobs(datasets: List[str], seeds: List[int], grid: Dict[str, List[Any]]):
    axes = list(grid)
    value_lists = [grid[axis] for axis in axes]
    combos = itertools.product(*value_lists) if axes else [()]
    jobs = []
    for dataset in datasets:
        for combo in combos:
            overrides = dict(zip(axes, combo))
            for seed in seeds:
                jobs.append((dataset, overrides, seed))
    return jobs


def _run_jobs(model: str, task: str, jobs, params: Dict, device: str) -> None:
    """Run jobs sequentially on one device."""
    sys.path.insert(0, str(ROOT))
    from torch_timeseries.leaderboard.experiment import LeaderboardExperiment

    for dataset, overrides, seed in jobs:
        run_params = dict(params)
        run_params.update(overrides)
        label = " ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
        label = f" {label}" if label else ""
        print(f"[{device}] {model}/{task}/{dataset}{label} seed={seed}")
        LeaderboardExperiment(
            model=model,
            task=task,
            dataset=dataset,
            results_dir=str(RESULTS_DIR),
            device=device,
            data_path=str(ROOT / "data"),
            save_dir=str(ROOT / "results"),
            **run_params,
        ).run(seeds=[seed])


def run_matrix(
    model: str,
    task: str,
    datasets: List[str],
    seeds: List[int],
    params: Dict,
    grid: Optional[Dict[str, List[Any]]] = None,
    argv: Optional[List[str]] = None,
    build: bool = True,
) -> None:
    grid = grid or {}
    parser = argparse.ArgumentParser(description=f"Reproduce {model} {task} results")
    parser.add_argument("--datasets", nargs="+", default=datasets)
    parser.add_argument("--seeds", nargs="+", type=int, default=seeds)
    parser.add_argument("--devices", nargs="*", type=int, default=[0, 1, 2, 3],
                        help="GPU ids to shard runs across (empty -> CPU)")
    parser.add_argument("--lanes-per-gpu", type=int, default=2,
                        help="parallel worker processes per GPU")
    parser.add_argument("--cpu", action="store_true", help="single CPU process")
    # Internal: worker mode.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", nargs="*", default=[], help=argparse.SUPPRESS)
    for axis, values in grid.items():
        reference = values[0] if values else ""
        parser.add_argument(
            _axis_flag(axis),
            dest=_axis_dest(axis),
            nargs="+",
            default=None,
            type=lambda raw, ref=reference: _coerce_value(raw, ref),
        )
    args = parser.parse_args(argv)

    if args.worker:
        jobs = [
            (dataset, overrides, int(seed))
            for dataset, overrides, seed in (json.loads(spec) for spec in args.jobs)
        ]
        _run_jobs(model, task, jobs, params, "cpu" if args.cpu else "cuda:0")
        return

    selected_grid = {}
    for axis, values in grid.items():
        selected = getattr(args, _axis_dest(axis))
        selected_grid[axis] = selected if selected is not None else values

    jobs = _grid_jobs(args.datasets, args.seeds, selected_grid)
    grid_label = ", ".join(f"{key}={value}" for key, value in selected_grid.items())
    print(f"{model} / {task}  datasets={args.datasets}  {grid_label}  "
          f"seeds={args.seeds}  ({len(jobs)} runs)")
    print(f"params: {params}")

    if args.cpu or not args.devices:
        _run_jobs(model, task, jobs, params, "cpu")
    else:
        lanes = [gpu for gpu in args.devices for _ in range(args.lanes_per_gpu)]
        shards: List[list] = [[] for _ in lanes]
        for i, job in enumerate(jobs):
            shards[i % len(lanes)].append(job)

        procs = []
        for lane_idx, (gpu, lane_jobs) in enumerate(zip(lanes, shards)):
            if not lane_jobs:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [
                sys.executable, sys.argv[0], "--worker",
                "--jobs",
                *[
                    json.dumps([dataset, overrides, seed], separators=(",", ":"))
                    for dataset, overrides, seed in lane_jobs
                ],
            ]
            print(f"lane {lane_idx} (GPU {gpu}): {len(lane_jobs)} runs")
            procs.append((lane_idx, gpu, subprocess.Popen(cmd, env=env, cwd=str(ROOT))))

        failed = [(i, g) for i, g, p in procs if p.wait() != 0]
        if failed:
            print(f"lane(s) failed: {failed}", file=sys.stderr)
            sys.exit(1)

    if build:
        print("\nAll runs complete. Building leaderboard JSON...")
        result = subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=str(ROOT), text=True)
        if result.returncode != 0:
            print("build_leaderboard.py failed", file=sys.stderr)
            sys.exit(1)
        print("\nDone. View results in webapp/leaderboard (npm run dev).")


def run_grid(
    model: str,
    task: str,
    datasets: List[str],
    pred_lens: List[int],
    seeds: List[int],
    params: Dict,
    argv: Optional[List[str]] = None,
) -> None:
    run_matrix(
        model=model,
        task=task,
        datasets=datasets,
        seeds=seeds,
        params=params,
        grid={"pred_len": pred_lens},
        argv=argv,
    )
