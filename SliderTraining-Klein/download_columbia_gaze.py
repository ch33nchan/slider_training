"""
Download the Columbia Gaze Dataset and organize into neg/pos/neutral pairs.

Columbia Gaze Dataset: 56 subjects, genuine horizontal gaze at multiple angles.
We select frontal head pose (0P) and use:
  neg = -15H (looking left)
  pos = +15H (looking right)
  neutral = 0H (looking straight)

Saves to data/columbia_gaze/{neg,pos,neutral}/

Usage:
    python download_columbia_gaze.py
"""

import os
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve


DATASET_URL = "https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip"
SAVE_DIR = Path("data/columbia_gaze")
ZIP_PATH = Path("data/columbia_gaze.zip")


def download_with_progress(url, dest):
    def progress(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = downloaded / total * 100
            print(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)
    print(f"Downloading {url}")
    urlretrieve(url, dest, reporthook=progress)
    print()


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    neg_dir = SAVE_DIR / "neg"
    pos_dir = SAVE_DIR / "pos"
    neutral_dir = SAVE_DIR / "neutral"
    for d in [neg_dir, pos_dir, neutral_dir]:
        d.mkdir(exist_ok=True)

    # Download
    if not ZIP_PATH.exists():
        download_with_progress(DATASET_URL, ZIP_PATH)
    else:
        print(f"ZIP already exists at {ZIP_PATH}")

    # Extract
    extract_dir = Path("data/columbia_gaze_raw")
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(extract_dir)
    else:
        print(f"Already extracted to {extract_dir}")

    # Find all image files
    all_images = sorted(extract_dir.rglob("*.jpg")) + sorted(extract_dir.rglob("*.JPG"))
    if not all_images:
        all_images = sorted(extract_dir.rglob("*.png"))
    print(f"Found {len(all_images)} images")

    # Parse filename convention: XXXX_Ym_ZP_AV_BH.jpg
    # We want: 0P (frontal head), 0V (no vertical gaze), -15H/0H/+15H
    neg_count = pos_count = neutral_count = 0

    for img_path in all_images:
        name = img_path.stem  # e.g. 0001_2m_0P_0V_-15H
        parts = name.split("_")
        if len(parts) < 5:
            continue

        subject = parts[0]
        head_pose = [p for p in parts if p.endswith("P")]
        v_gaze = [p for p in parts if p.endswith("V")]
        h_gaze = [p for p in parts if p.endswith("H")]

        if not (head_pose and v_gaze and h_gaze):
            continue

        head_p = head_pose[0]
        v_g = v_gaze[0]
        h_g = h_gaze[0]

        # Only frontal head, no vertical gaze offset
        if head_p != "0P" or v_g != "0V":
            continue

        dest_name = f"subject_{subject}.jpg"

        if h_g in ("-15H", "m15H"):
            shutil.copy(img_path, neg_dir / dest_name)
            neg_count += 1
        elif h_g in ("+15H", "p15H", "15H"):
            shutil.copy(img_path, pos_dir / dest_name)
            pos_count += 1
        elif h_g == "0H":
            shutil.copy(img_path, neutral_dir / dest_name)
            neutral_count += 1

    print(f"\nOrganized:")
    print(f"  neg (left,  -15H): {neg_count}")
    print(f"  pos (right, +15H): {pos_count}")
    print(f"  neutral (0H):      {neutral_count}")
    print(f"\nDataset ready at {SAVE_DIR}")

    if neg_count == 0:
        # Print some filenames to help debug naming convention
        print("\nCould not parse filenames. Sample names found:")
        for p in all_images[:10]:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
