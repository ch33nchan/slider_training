#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


OFFICIAL_REPO_URL = "https://github.com/Ahmednull/L2CS-Net"
OFFICIAL_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing"
OFFICIAL_FILENAME = "L2CSNet_gaze360.pkl"
DEFAULT_TARGET = "models/l2cs/l2cs_resnet50_gaze360.pth"
DEFAULT_CACHE_DIR = ".cache/l2cs_download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-url", default=OFFICIAL_DRIVE_FOLDER_URL)
    parser.add_argument("--filename", default=OFFICIAL_FILENAME)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown"],
            check=True,
        )


def download_folder(folder_url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            folder_url,
            "--output",
            str(output_dir),
            "--remaining-ok",
        ],
        check=True,
    )


def find_weight_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}. "
            f"Check the official source: {OFFICIAL_REPO_URL}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    target_path = Path(args.target).resolve()

    if target_path.exists() and not args.force:
        print(target_path)
        return

    ensure_gdown()
    download_folder(args.folder_url, cache_dir)

    source_path = find_weight_file(cache_dir, args.filename)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    print(target_path)


if __name__ == "__main__":
    main()
