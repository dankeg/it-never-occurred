"""Compare recomputed results with the committed ones ($0, offline).

Each analysis output in `runs/`, each generated table and each figure is compared with its version in git `HEAD`.
JSON is compared by value, and a list of rows is compared without regard to order (the order follows the order in
which runs were passed on the command line). Figures are compared pixel by pixel when Pillow is installed.

Usage: `python3 scripts/check_reproduction.py`
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ["runs/jubilee_analysis.json", "runs/jubilee_v2_analysis.json", "runs/jubilee_recruitment.json",
           "runs/jubilee_recruitment_fresh.json", "runs/jubilee_crossings.json", "runs/jubilee_crossings_summary.json",
           "runs/jubilee_v2_suspicion.json"] + [
    f"runs/jubilee_asked{'' if variant == 'asked' else '_' + variant}.json"
    for variant in ("asked", "jubilee", "candid", "self", "drive", "duty", "duty_jubilee", "counsellor")]
TABLES = sorted(f"docs/report/{path.name}" for path in (ROOT / "docs" / "report").glob("table_*.md"))
FIGURES = sorted(f"docs/report/{path.name}" for path in (ROOT / "docs" / "report").glob("*.png"))


def committed(path: str) -> bytes | None:
    """The file's bytes at HEAD, or None if it is not committed."""
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def canonical(value):
    """A JSON value with every list of rows sorted, so that row order does not matter."""
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [canonical(item) for item in value]
        if items and all(isinstance(item, dict) for item in items):
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
        return items
    return value


def same_json(path: str, old: bytes) -> bool:
    return canonical(json.loads(old)) == canonical(json.loads((ROOT / path).read_text(encoding="utf-8")))


def same_image(path: str, old: bytes) -> bool | None:
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return None
    before, after = Image.open(io.BytesIO(old)).convert("RGB"), Image.open(ROOT / path).convert("RGB")
    return before.size == after.size and ImageChops.difference(before, after).getbbox() is None


def main() -> int:
    """Print one line per output and return 1 if anything differs."""
    failures = 0
    for path in OUTPUTS + TABLES + FIGURES:
        old = committed(path)
        if old is None or not (ROOT / path).exists():
            print(f"  skipped    {path} (not committed or not produced)")
            continue
        if path.endswith(".json"):
            same = same_json(path, old)
        elif path.endswith(".png"):
            same = same_image(path, old)
        else:
            same = old == (ROOT / path).read_bytes()
        label = "identical" if same else ("unchecked" if same is None else "DIFFERS")
        failures += same is False
        print(f"  {label:<10} {path}")
    print("\nEvery recomputed result matches the committed one." if not failures
          else f"\n{failures} output(s) differ from the committed version; `git diff` shows how.")
    if any(path.endswith(".png") for path in FIGURES) and not failures:
        print("(Figures match pixel for pixel with the pinned matplotlib and Pillow on macOS, whose Helvetica Neue the "
              "figures use; elsewhere a substitute font changes pixels but not data.)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
