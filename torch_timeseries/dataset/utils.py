"""Dataset download / integrity / extraction helpers.

Self-contained replacements for the handful of ``torchvision.datasets.utils``
functions this library used, so torchvision is no longer a dependency.
Signatures match torchvision's for drop-in compatibility.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

_USER_AGENT = "torch-timeseries"


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str) -> bool:
    return md5 == calculate_md5(fpath)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    """True if *fpath* exists (and matches *md5* when given)."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download *url* into ``root/filename``, skipping if already verified."""
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f"Using downloaded and verified file: {fpath}")
        return

    print(f"Downloading {url} to {fpath}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        progress = tqdm(total=total, unit="B", unit_scale=True) if tqdm else None
        with open(fpath, "wb") as fh:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                fh.write(chunk)
                if progress:
                    progress.update(len(chunk))
        if progress:
            progress.close()

    if md5 is not None and not check_md5(fpath, md5):
        raise RuntimeError(f"Downloaded file {fpath} failed md5 check ({md5}).")


def extract_archive(
    from_path: str,
    to_path: Optional[str] = None,
    remove_finished: bool = False,
) -> str:
    """Extract zip / tar(.gz|.bz2|.xz) / single-file .gz archives."""
    if to_path is None:
        to_path = os.path.dirname(from_path)

    lower = from_path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(from_path, "r") as zf:
            zf.extractall(to_path)
    elif lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(from_path, "r:*") as tf:
            tf.extractall(to_path)
    elif lower.endswith(".gz"):
        # single gzipped file, e.g. solar_AL.txt.gz -> solar_AL.txt
        target = os.path.join(to_path, os.path.basename(from_path)[: -len(".gz")])
        with gzip.open(from_path, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        raise ValueError(f"Unsupported archive format: {from_path}")

    if remove_finished:
        os.remove(from_path)
    return to_path


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    extract_archive(archive, extract_root, remove_finished)
