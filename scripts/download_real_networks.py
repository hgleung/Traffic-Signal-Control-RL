"""Download the Hangzhou (4x4) and Jinan (3x4) real-world benchmarks.

These are the recorded one-hour traffic datasets used by PressLight, CoLight,
MPLight and Advanced-XLight. They are fetched from the Advanced-XLight
repository (github.com/LiangZhang1996/Advanced_XLight, GPL-3.0) into
configs/real/<city>/, which is git-ignored, so they are not redistributed here.

Usage: python scripts/download_real_networks.py
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from traffic_env import CONFIGS_DIR

REPO = "https://github.com/LiangZhang1996/Advanced_XLight.git"
SOURCES = {"hangzhou": "data/Hangzhou/4_4", "jinan": "data/Jinan/3_4"}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", *SOURCES.values()], check=True)
        for city, source in SOURCES.items():
            target = CONFIGS_DIR / "real" / city
            target.mkdir(parents=True, exist_ok=True)
            for f in (Path(tmp) / source).glob("*.json"):
                shutil.copy(f, target / f.name)
            print(f"{city}: {sorted(p.name for p in target.glob('*.json'))}")


if __name__ == "__main__":
    main()
