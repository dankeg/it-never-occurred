"""Unpack the per-run logs into `runs/` ($0, offline, standard library only).

The episodes, counterfactual branches and positive controls are shipped as one gzipped tar split into parts under
`data/` (each below GitHub's file-size limit). This checks every part against `data/SHA256SUMS`, then extracts the
archive at the repository root, which recreates `runs/jub_*`, `runs/fork_jub_*` and `runs/it25_*` with each run's
scenario, judged results, judge caches, decision log, trajectory, perceptions and engine events. Saved worlds and
server files are not included; they are not needed to recompute any result.

Usage: `python3 scripts/unpack_logs.py`
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def checked_parts() -> list[Path]:
    """The archive's parts in order, each verified against SHA256SUMS."""
    expected = dict(reversed(line.split()) for line in (DATA / "SHA256SUMS").read_text().splitlines() if line.strip())
    parts = sorted(DATA.glob("run_logs.tar.gz.part-*"))
    if sorted(part.name for part in parts) != sorted(expected):
        raise SystemExit(f"data/ should hold exactly {sorted(expected)}; found {[part.name for part in parts]}")
    for part in parts:
        if hashlib.sha256(part.read_bytes()).hexdigest() != expected[part.name]:
            raise SystemExit(f"{part.name} does not match its checksum; download it again")
    return parts


def main() -> int:
    """Verify and extract."""
    parts = checked_parts()
    stream = io.BytesIO(b"".join(part.read_bytes() for part in parts))
    with tarfile.open(fileobj=stream, mode="r:gz") as archive:
        members = archive.getmembers()
        unsafe = [member.name for member in members if not member.name.startswith("runs/") or ".." in member.name]
        if unsafe:
            raise SystemExit(f"refusing unexpected paths in the archive: {unsafe[:3]}")
        archive.extractall(ROOT, members=members)
    runs = {member.name.split("/")[1] for member in members}
    print(f"Unpacked {len(members)} files from {len(runs)} runs into runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
