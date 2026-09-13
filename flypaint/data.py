"""Download and locate the MaleCNS v1.0 flat-connectome files.

Source: https://male-cns.janelia.org/download/ (bulk files live in the public
GCS bucket gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

GCS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

# file name -> expected size in bytes (from the bucket listing on 2026-09-13)
FILES = {
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": 1_051_241_946,
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": 14_483_314,
    "body-neurotransmitters-male-cns-v1.0.feather": 43_282_834,
}


def project_root() -> Path:
    return Path(os.environ.get("FLYPAINT_ROOT", Path(__file__).resolve().parent.parent))


def data_dir() -> Path:
    return Path(os.environ.get("FLYPAINT_DATA", project_root() / "data"))


def malecns_dir() -> Path:
    return data_dir() / "malecns"


def cache_dir() -> Path:
    return data_dir() / "cache"


def path_for(name: str) -> Path:
    return malecns_dir() / name


def _download(url: str, dest: Path, expected: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "flypaint/0.1"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if expected:
                sys.stderr.write(f"\r  {dest.name}: {done / 1e6:8.1f} / {expected / 1e6:.1f} MB")
        sys.stderr.write("\n")
    size = tmp.stat().st_size
    if expected and size != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: downloaded {size} bytes, expected {expected}")
    tmp.replace(dest)


def ensure_files(force: bool = False) -> dict[str, Path]:
    """Download any missing MaleCNS file. Returns name -> local path."""
    out = {}
    for name, size in FILES.items():
        dest = path_for(name)
        if force or not dest.exists() or dest.stat().st_size != size:
            print(f"downloading {name} ({size / 1e6:.0f} MB)", file=sys.stderr)
            _download(f"{GCS_BASE}/{name}", dest, size)
        out[name] = dest
    return out


def check_files() -> list[str]:
    """Return a list of problems (empty when everything is present)."""
    problems = []
    for name, size in FILES.items():
        p = path_for(name)
        if not p.exists():
            problems.append(f"missing {p}")
        elif p.stat().st_size != size:
            problems.append(f"{p} has {p.stat().st_size} bytes, expected {size}")
    return problems
