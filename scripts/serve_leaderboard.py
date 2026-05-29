#!/usr/bin/env python3
"""
Live leaderboard server. Serves webapp/dist/ and provides /api/refresh.

Usage:
    python scripts/serve_leaderboard.py           # port 8000
    python scripts/serve_leaderboard.py --port 9000
"""
import argparse
import pathlib
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST = ROOT / "webapp" / "dist"
BUILD = ROOT / "scripts" / "build_leaderboard.py"

app = FastAPI(title="Leaderboard Server")


@app.get("/api/refresh")
def refresh():
    result = subprocess.run(
        [sys.executable, str(BUILD)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr.strip())
    return {"status": "ok", "message": result.stdout.strip()}


if not DIST.exists():
    raise RuntimeError(
        "webapp/dist/ not found. Run: cd webapp && npm run build"
    )

app.mount("/", StaticFiles(directory=str(DIST), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
