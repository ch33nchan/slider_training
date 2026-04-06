#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional fallback
    cv2 = None


CANDIDATE_URLS = [
    "https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip",
    "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze.zip",
    "http://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip",
]

ANGLE_TO_LEVEL = {
    -15: -2,
    -10: -1,
    0: 0,
    10: 1,
    15: 2,
}
SUPPLEMENTAL_SUFFIX_TO_LEVEL = {
    "left": -2,
    "neutral": 0,
    "right": 2,
}


@dataclass(frozen=True)
class ColumbiaImage:
    subject_id: str
    distance_token: str
    head_pose_token: str
    vertical_token: str
    horizontal_deg: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data/columbia_gaze.zip")
    parser.add_argument("--extract-dir", default="data/columbia_gaze_raw")
    parser.add_argument("--output-dir", default="data/columbia_5level")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--preferred-distance", default="2m")
    parser.add_argument("--supplemental-triplet-dir", default="../gaze_pairs/gaze_pairs")
    parser.add_argument("--supplemental-extreme-yaw-deg", type=float, default=15.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def level_token(level: int) -> str:
    return "0" if level == 0 else f"{level:+d}"


def try_download(dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in CANDIDATE_URLS:
        print(f"Trying {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "")
                if "html" in content_type.lower():
                    continue
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with dest.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\r  {downloaded / total * 100:5.1f}% "
                                f"({downloaded // (1 << 20)}MB/{total // (1 << 20)}MB)",
                                end="",
                                flush=True,
                            )
                print()
                if zipfile.is_zipfile(dest):
                    return
        except Exception:
            continue
    raise RuntimeError(
        "Failed to download Columbia Gaze zip automatically. "
        "Download it manually and place it at data/columbia_gaze.zip."
    )


def ensure_zip(zip_path: Path, force_download: bool) -> None:
    if force_download or not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        try_download(zip_path)


def ensure_extract(zip_path: Path, extract_dir: Path, force_extract: bool) -> None:
    if force_extract and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if extract_dir.exists():
        return
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)


def parse_horizontal_token(token: str) -> Optional[int]:
    token = token.strip()
    if not token.endswith("H"):
        return None
    base = token[:-1]
    if base.startswith("m"):
        return -int(base[1:])
    if base.startswith("p"):
        return int(base[1:])
    return int(base)


def discover_columbia_images(extract_dir: Path) -> list[ColumbiaImage]:
    images = sorted(extract_dir.rglob("*.jpg")) + sorted(extract_dir.rglob("*.JPG")) + sorted(extract_dir.rglob("*.png"))
    discovered: list[ColumbiaImage] = []
    for image_path in images:
        parts = image_path.stem.split("_")
        if len(parts) < 5:
            continue
        subject_id = parts[0]
        distance_token = next((part for part in parts if part.endswith("m")), "")
        head_pose_token = next((part for part in parts if part.endswith("P")), "")
        vertical_token = next((part for part in parts if part.endswith("V")), "")
        horizontal_token = next((part for part in parts if part.endswith("H")), "")
        horizontal_deg = parse_horizontal_token(horizontal_token)
        if not distance_token or not head_pose_token or not vertical_token or horizontal_deg is None:
            continue
        discovered.append(
            ColumbiaImage(
                subject_id=subject_id,
                distance_token=distance_token,
                head_pose_token=head_pose_token,
                vertical_token=vertical_token,
                horizontal_deg=horizontal_deg,
                path=image_path,
            )
        )
    return discovered


def choose_subject_images(images: list[ColumbiaImage], preferred_distance: str) -> list[ColumbiaImage]:
    grouped: dict[tuple[str, int], list[ColumbiaImage]] = defaultdict(list)
    for item in images:
        if item.head_pose_token != "0P" or item.vertical_token != "0V":
            continue
        if item.horizontal_deg not in ANGLE_TO_LEVEL:
            continue
        grouped[(item.subject_id, ANGLE_TO_LEVEL[item.horizontal_deg])].append(item)

    chosen: list[ColumbiaImage] = []
    for _, candidates in sorted(grouped.items()):
        preferred = [candidate for candidate in candidates if candidate.distance_token == preferred_distance]
        selected = sorted(preferred or candidates, key=lambda item: (item.distance_token, item.path.name))[0]
        chosen.append(selected)
    return chosen


def build_face_detector() -> Any:
    if cv2 is None:
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    return detector if not detector.empty() else None


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def default_face_bbox(width: int, height: int) -> tuple[int, int, int, int]:
    crop = int(round(min(width, height) * 0.78))
    x = max(0, (width - crop) // 2)
    y = max(0, (height - crop) // 2)
    return x, y, crop, crop


def default_eye_bbox(face_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = face_bbox
    eye_w = max(16, int(round(w * 0.58)))
    eye_h = max(12, int(round(h * 0.18)))
    eye_x = x + int(round(w * 0.21))
    eye_y = y + int(round(h * 0.25))
    return eye_x, eye_y, eye_w, eye_h


def detect_face_bbox(image: Image.Image, detector: Any) -> tuple[int, int, int, int]:
    width, height = image.size
    if detector is None:
        return default_face_bbox(width, height)
    image_np = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return default_face_bbox(width, height)
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return clamp_bbox((int(x), int(y), int(w), int(h)), width, height)


def scale_bbox(
    bbox: tuple[int, int, int, int],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    sx = dst_width / float(src_width)
    sy = dst_height / float(src_height)
    return clamp_bbox(
        (
            int(round(x * sx)),
            int(round(y * sy)),
            max(1, int(round(w * sx))),
            max(1, int(round(h * sy))),
        ),
        dst_width,
        dst_height,
    )


def save_entry(
    *,
    image: Image.Image,
    output_dir: Path,
    level: int,
    stem: str,
    yaw_deg: float,
    detector: Any,
    dataset_name: str,
    metadata: list[dict[str, Any]],
    size: int,
) -> None:
    src_width, src_height = image.size
    face_bbox = detect_face_bbox(image, detector)
    eye_bbox = default_eye_bbox(face_bbox)

    resized = image.resize((size, size), Image.LANCZOS)
    face_bbox_scaled = scale_bbox(face_bbox, src_width, src_height, size, size)
    eye_bbox_scaled = scale_bbox(eye_bbox, src_width, src_height, size, size)

    level_name = level_token(level)
    rel_path = Path(f"level_{level_name}") / f"{stem}.png"
    (output_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_dir / rel_path)

    metadata.append(
        {
            "subject_id": stem,
            "level": level,
            "yaw_deg": yaw_deg,
            "yaw_rad": math.radians(yaw_deg),
            "pitch_rad": 0.0,
            "face_bbox": list(face_bbox_scaled),
            "eye_bbox": list(eye_bbox_scaled),
            "file_path": rel_path.as_posix(),
            "image_width": size,
            "image_height": size,
            "dataset": dataset_name,
        }
    )


def append_supplemental_triplets(
    metadata: list[dict[str, Any]],
    output_dir: Path,
    triplet_dir: Path,
    detector: Any,
    size: int,
    extreme_yaw_deg: float,
) -> int:
    if not triplet_dir.exists():
        return 0

    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(triplet_dir.iterdir()):
        if not path.is_file():
            continue
        lower = path.stem.lower()
        for suffix in SUPPLEMENTAL_SUFFIX_TO_LEVEL:
            token = f"_{suffix}"
            if lower.endswith(token):
                key = path.stem[: -len(token)]
                grouped[key][suffix] = path
                break

    saved = 0
    yaw_by_level = {
        -2: -abs(extreme_yaw_deg),
        0: 0.0,
        2: abs(extreme_yaw_deg),
    }
    for subject_key, paths in tqdm(sorted(grouped.items()), desc="Adding gaze_pairs"):
        required = {"left", "neutral", "right"}
        if not required.issubset(paths):
            continue
        for suffix, level in SUPPLEMENTAL_SUFFIX_TO_LEVEL.items():
            image = Image.open(paths[suffix]).convert("RGB")
            save_entry(
                image=image,
                output_dir=output_dir,
                level=level,
                stem=f"supp_{subject_key}_{suffix}",
                yaw_deg=yaw_by_level[level],
                detector=detector,
                dataset_name="gaze_pairs",
                metadata=metadata,
                size=size,
            )
        saved += 1
    return saved


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    supplemental_triplet_dir = Path(args.supplemental_triplet_dir).resolve()

    ensure_zip(zip_path, force_download=args.force_download)
    ensure_extract(zip_path, extract_dir, force_extract=args.force_extract)

    detector = build_face_detector()
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in (-2, -1, 0, 1, 2):
        (output_dir / f"level_{level_token(level)}").mkdir(parents=True, exist_ok=True)

    discovered = discover_columbia_images(extract_dir)
    chosen = choose_subject_images(discovered, preferred_distance=args.preferred_distance)

    metadata: list[dict[str, Any]] = []
    for item in tqdm(chosen, desc="Preparing Columbia Gaze"):
        level = ANGLE_TO_LEVEL[item.horizontal_deg]
        image = Image.open(item.path).convert("RGB")
        save_entry(
            image=image,
            output_dir=output_dir,
            level=level,
            stem=f"columbia_{item.subject_id}_{item.distance_token}_{level_token(level)}",
            yaw_deg=float(item.horizontal_deg),
            detector=detector,
            dataset_name="columbia_gaze",
            metadata=metadata,
            size=args.size,
        )

    supplemental_subjects = append_supplemental_triplets(
        metadata=metadata,
        output_dir=output_dir,
        triplet_dir=supplemental_triplet_dir,
        detector=detector,
        size=args.size,
        extreme_yaw_deg=args.supplemental_extreme_yaw_deg,
    )

    payload = {
        "format": "columbia_5level",
        "count": len(metadata),
        "levels": [-2, -1, 0, 1, 2],
        "items": metadata,
        "notes": {
            "frontal_subset": "Only 0P head pose and 0V vertical gaze images are kept from Columbia Gaze.",
            "level_mapping": {
                "level_-2": -15,
                "level_-1": -10,
                "level_0": 0,
                "level_+1": 10,
                "level_+2": 15,
            },
            "supplemental_subjects": supplemental_subjects,
            "supplemental_triplet_dir": str(supplemental_triplet_dir) if supplemental_triplet_dir.exists() else None,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"Saved metadata: {metadata_path}")
    for level in (-2, -1, 0, 1, 2):
        folder = output_dir / f"level_{level_token(level)}"
        count = len(list(folder.glob("*.png")))
        print(f"{folder.name}: {count}")


if __name__ == "__main__":
    main()
