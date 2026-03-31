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


SAVE_DIR = Path("data/columbia_gaze")

# Try multiple known URLs for the Columbia Gaze Dataset
CANDIDATE_URLS = [
    "https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip",
    "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze.zip",
    "http://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip",
]


def try_download(dest: Path) -> bool:
    import urllib.request
    for url in CANDIDATE_URLS:
        print(f"Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                print(f"  Content-Type: {content_type}")
                if "html" in content_type.lower():
                    print(f"  Got HTML (likely a registration page) — skipping")
                    continue
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r  {downloaded/total*100:.1f}% ({downloaded//1024//1024}MB/{total//1024//1024}MB)", end="", flush=True)
                print(f"\n  Downloaded {downloaded//1024//1024}MB")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
    return False


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    neg_dir = SAVE_DIR / "neg"
    pos_dir = SAVE_DIR / "pos"
    neutral_dir = SAVE_DIR / "neutral"
    for d in [neg_dir, pos_dir, neutral_dir]:
        d.mkdir(exist_ok=True)

    ZIP_PATH = Path("data/columbia_gaze.zip")

    # Download
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1_000_000:
        print(f"ZIP already exists at {ZIP_PATH} ({ZIP_PATH.stat().st_size//1024//1024}MB)")
    else:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        ok = try_download(ZIP_PATH)
        if not ok:
            print("\nAll download attempts failed.")
            print("Please download manually from:")
            print("  https://cave.cs.columbia.edu/old/databases/columbia_gaze/")
            print("and save as data/columbia_gaze.zip")
            return

    # Verify it's actually a zip
    if not zipfile.is_zipfile(ZIP_PATH):
        print(f"ERROR: {ZIP_PATH} is not a valid zip file.")
        print("The server likely returned an HTML page requiring registration.")
        print("\nManual download instructions:")
        print("  1. Visit: https://cave.cs.columbia.edu/old/databases/columbia_gaze/")
        print("  2. Download the dataset zip")
        print(f"  3. Place it at: {ZIP_PATH.absolute()}")
        print("  4. Re-run this script")
        return

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
