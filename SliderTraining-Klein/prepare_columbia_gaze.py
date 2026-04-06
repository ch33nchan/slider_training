#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional fallback
    cv2 = None


DATASET_URLS = [
    "https://ceal.cs.columbia.edu/static/materials/columbiagaze/columbia_gaze_data_set.zip",
    "https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip",
    "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze.zip",
    "http://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze.zip",
]
EYE_CORNER_URLS = [
    "https://ceal.cs.columbia.edu/static/materials/columbiagaze/eye_corner_locations.zip",
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
    parser.add_argument("--zip-url", action="append", default=[])
    parser.add_argument("--extract-dir", default="data/columbia_gaze_raw")
    parser.add_argument("--corner-zip-path", default="data/eye_corner_locations.zip")
    parser.add_argument("--corner-zip-url", action="append", default=[])
    parser.add_argument("--corner-extract-dir", default="data/columbia_gaze_corners")
    parser.add_argument("--output-dir", default="data/columbia_5level")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--preferred-distance", default="2m")
    parser.add_argument("--supplemental-triplet-dir", default=None)
    parser.add_argument("--supplemental-extreme-yaw-deg", type=float, default=15.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-corner-download", action="store_true")
    parser.add_argument("--force-corner-extract", action="store_true")
    return parser.parse_args()


def level_token(level: int) -> str:
    return "0" if level == 0 else f"{level:+d}"


def try_download(dest: Path, urls: list[str]) -> None:
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
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
        f"Failed to download archive automatically for {dest.name}. "
        f"Download it manually and place it at {dest}."
    )


def ensure_zip(zip_path: Path, urls: list[str], force_download: bool) -> None:
    if force_download or not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        try_download(zip_path, urls=urls)


def ensure_extract(zip_path: Path, extract_dir: Path, force_extract: bool) -> None:
    if force_extract and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if extract_dir.exists():
        return
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)


def normalize_corner_key(value: str) -> str:
    normalized = sanitize_text(value).replace("\\", "/").strip().strip("\"'")
    name = PurePosixPath(normalized).name
    stem = PurePosixPath(name).stem.lower()
    return stem


def corner_key_variants(value: str) -> list[str]:
    normalized = sanitize_text(value).replace("\\", "/").strip().strip("\"'")
    normalized = re.sub(r"/+", "/", normalized).lstrip("./").lower()
    if not normalized:
        return []

    path = PurePosixPath(normalized)
    variants: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip().strip("\"'").lower()
        if candidate and candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)

    add(normalized)
    add(path.name)
    add(path.stem)
    stem_tokens = [token for token in re.split(r"[_\-/]+", path.stem) if token]
    if stem_tokens:
        add(stem_tokens[0])
        if stem_tokens[0].isdigit():
            token_int = int(stem_tokens[0])
            add(str(token_int))
            add(f"{token_int:03d}")
    numeric_tokens = re.findall(r"\d+", path.stem)
    if numeric_tokens:
        add(numeric_tokens[0])
        token_int = int(numeric_tokens[0])
        add(str(token_int))
        add(f"{token_int:03d}")
    if path.suffix:
        add(normalized[: -len(path.suffix)])
    for length in (2, 3, 4):
        if len(path.parts) >= length:
            add("/".join(path.parts[-length:]))
            tail = PurePosixPath("/".join(path.parts[-length:]))
            add(tail.stem)
            if tail.suffix:
                add(str(tail)[: -len(tail.suffix)])
    return variants


def subject_id_variants(subject_id: str) -> list[str]:
    cleaned = subject_id.strip()
    if not cleaned:
        return []
    variants = [cleaned]
    if cleaned.isdigit():
        as_int = int(cleaned)
        variants.append(str(as_int))
        variants.append(f"{as_int:03d}")
    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped


def read_text_robust(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def sanitize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    return cleaned


def parse_numeric_tokens(text: str) -> list[float]:
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return [float(match) for match in matches]


def parse_corner_line(line: str) -> tuple[Optional[str], list[tuple[float, float]]]:
    stripped = line.strip()
    if not stripped:
        return None, []
    image_match = re.search(r"([A-Za-z0-9_\-]+\.(?:jpg|jpeg|png))", stripped, flags=re.IGNORECASE)
    image_key = normalize_corner_key(image_match.group(1)) if image_match else None
    numeric_source = stripped
    if image_match is not None:
        numeric_source = f"{stripped[:image_match.start()]} {stripped[image_match.end():]}"
    values = parse_numeric_tokens(numeric_source)
    if len(values) < 4 or len(values) % 2 != 0:
        return image_key, []
    points = [(values[index], values[index + 1]) for index in range(0, len(values), 2)]
    return image_key, points


def merge_point_map(
    target: dict[str, list[tuple[float, float]]],
    keys: Optional[str | list[str]],
    points: list[tuple[float, float]],
) -> None:
    if keys is None or not points:
        return
    if isinstance(keys, str):
        key_iterable = corner_key_variants(keys)
    else:
        key_iterable = []
        for key in keys:
            key_iterable.extend(corner_key_variants(key))
    for key in key_iterable:
        target[key] = points


def parse_points_blob(text: str) -> list[tuple[float, float]]:
    candidate_blobs = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate_blobs.insert(0, text[start + 1 : end])

    for blob in candidate_blobs:
        per_line_points: list[tuple[float, float]] = []
        for line in blob.splitlines():
            values = parse_numeric_tokens(line)
            if len(values) == 2:
                per_line_points.append((values[0], values[1]))
        if len(per_line_points) >= 2:
            return per_line_points

        values = parse_numeric_tokens(blob)
        for offset in range(min(4, max(0, len(values) - 3))):
            trimmed = values[offset:]
            if len(trimmed) >= 4 and len(trimmed) % 2 == 0:
                return [
                    (trimmed[index], trimmed[index + 1])
                    for index in range(0, len(trimmed), 2)
                ]
    return []


def load_text_corner_annotation_file(path: Path, text: str) -> dict[str, list[tuple[float, float]]]:
    point_map: dict[str, list[tuple[float, float]]] = {}
    found_explicit_keys = False
    for line in text.splitlines():
        key, points = parse_corner_line(line)
        if key is None or not points:
            continue
        merge_point_map(point_map, key, points)
        found_explicit_keys = True

    if found_explicit_keys:
        return point_map

    points = parse_points_blob(text)
    if points:
        merge_point_map(point_map, str(path), points)
    return point_map


def load_table_corner_annotation_file(path: Path, text: str) -> dict[str, list[tuple[float, float]]]:
    point_map: dict[str, list[tuple[float, float]]] = {}
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return point_map

    sample = lines[0]
    delimiter = "\t" if "\t" in sample else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    fieldnames = {field.strip().upper(): field for field in (reader.fieldnames or [])}

    image_field = fieldnames.get("IMAGE")
    expected_fields = [
        fieldnames.get("RIGHT_EYE_IN_X"),
        fieldnames.get("RIGHT_EYE_IN_Y"),
        fieldnames.get("RIGHT_EYE_OUT_X"),
        fieldnames.get("RIGHT_EYE_OUT_Y"),
        fieldnames.get("LEFT_EYE_IN_X"),
        fieldnames.get("LEFT_EYE_IN_Y"),
        fieldnames.get("LEFT_EYE_OUT_X"),
        fieldnames.get("LEFT_EYE_OUT_Y"),
    ]
    if image_field is None or any(field is None for field in expected_fields):
        return point_map

    for row in reader:
        raw_key = row.get(image_field, "").strip()
        if not raw_key:
            continue
        try:
            numeric = [float(row[field].strip()) for field in expected_fields]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        points = [
            (numeric[index], numeric[index + 1])
            for index in range(0, len(numeric), 2)
        ]
        merge_point_map(point_map, raw_key, points)
    return point_map


def load_corner_annotations(corner_root: Path) -> dict[str, list[tuple[float, float]]]:
    point_map: dict[str, list[tuple[float, float]]] = {}
    if not corner_root.exists():
        return point_map

    text_suffixes = {".txt", ".csv", ".tsv", ".pts"}
    for path in sorted(corner_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in text_suffixes:
            text = sanitize_text(read_text_robust(path))
            table_points = {}
            if suffix in {".csv", ".tsv"}:
                table_points = load_table_corner_annotation_file(path, text)
            point_map.update(table_points or load_text_corner_annotation_file(path, text))
            continue
        if suffix == ".json":
            payload = json.loads(read_text_robust(path))
            if isinstance(payload, dict):
                items = payload.items()
            elif isinstance(payload, list):
                items = enumerate(payload)
            else:
                continue
            for raw_key, raw_value in items:
                key = str(raw_key) if isinstance(raw_key, str) else None
                values = raw_value if isinstance(raw_value, list) else raw_value.get("points", []) if isinstance(raw_value, dict) else []
                numeric = [float(v) for v in values if isinstance(v, (int, float))]
                if len(numeric) >= 4 and len(numeric) % 2 == 0:
                    points = [(numeric[index], numeric[index + 1]) for index in range(0, len(numeric), 2)]
                    merge_point_map(point_map, key, points)
    return point_map


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


def bbox_from_eye_corners(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    eye_w = max_x - min_x
    eye_h = max_y - min_y
    pad_x = max(8.0, eye_w * 0.35)
    pad_y = max(6.0, eye_h * 1.4)
    eye_bbox = clamp_bbox(
        (
            int(round(min_x - pad_x)),
            int(round(min_y - pad_y * 0.55)),
            int(round(eye_w + 2 * pad_x)),
            int(round(eye_h + pad_y)),
        ),
        width,
        height,
    )

    face_w = eye_bbox[2] * 1.9
    face_h = face_w * 1.15
    face_x = eye_bbox[0] + eye_bbox[2] * 0.5 - face_w * 0.5
    face_y = eye_bbox[1] + eye_bbox[3] * 0.35 - face_h * 0.35
    face_bbox = clamp_bbox(
        (
            int(round(face_x)),
            int(round(face_y)),
            int(round(face_w)),
            int(round(face_h)),
        ),
        width,
        height,
    )
    return face_bbox, eye_bbox


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
    face_bbox_override: Optional[tuple[int, int, int, int]] = None,
    eye_bbox_override: Optional[tuple[int, int, int, int]] = None,
) -> None:
    src_width, src_height = image.size
    face_bbox = face_bbox_override or detect_face_bbox(image, detector)
    eye_bbox = eye_bbox_override or default_eye_bbox(face_bbox)

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
    triplet_dir: Optional[Path],
    detector: Any,
    size: int,
    extreme_yaw_deg: float,
) -> int:
    if triplet_dir is None or not triplet_dir.exists():
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
    corner_zip_path = Path(args.corner_zip_path).resolve()
    corner_extract_dir = Path(args.corner_extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    supplemental_triplet_dir = (
        Path(args.supplemental_triplet_dir).resolve()
        if args.supplemental_triplet_dir
        else None
    )

    dataset_urls = args.zip_url or DATASET_URLS
    corner_urls = args.corner_zip_url or EYE_CORNER_URLS

    ensure_zip(zip_path, urls=dataset_urls, force_download=args.force_download)
    ensure_extract(zip_path, extract_dir, force_extract=args.force_extract)
    ensure_zip(corner_zip_path, urls=corner_urls, force_download=args.force_corner_download)
    ensure_extract(corner_zip_path, corner_extract_dir, force_extract=args.force_corner_extract)

    detector = build_face_detector()
    corner_map = load_corner_annotations(corner_extract_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in (-2, -1, 0, 1, 2):
        (output_dir / f"level_{level_token(level)}").mkdir(parents=True, exist_ok=True)

    discovered = discover_columbia_images(extract_dir)
    chosen = choose_subject_images(discovered, preferred_distance=args.preferred_distance)

    metadata: list[dict[str, Any]] = []
    corner_hits = 0
    for item in tqdm(chosen, desc="Preparing Columbia Gaze"):
        level = ANGLE_TO_LEVEL[item.horizontal_deg]
        image = Image.open(item.path).convert("RGB")
        face_bbox_override = None
        eye_bbox_override = None
        match_keys = corner_key_variants(str(item.path)) + subject_id_variants(item.subject_id)
        for corner_key in match_keys:
            if corner_key in corner_map:
                face_bbox_override, eye_bbox_override = bbox_from_eye_corners(corner_map[corner_key], *image.size)
                corner_hits += 1
                break
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
            face_bbox_override=face_bbox_override,
            eye_bbox_override=eye_bbox_override,
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
            "supplemental_triplet_dir": str(supplemental_triplet_dir) if supplemental_triplet_dir is not None and supplemental_triplet_dir.exists() else None,
            "eye_corner_annotation_entries": len(corner_map),
            "eye_corner_annotations_used": corner_hits,
            "eye_corner_annotation_root": str(corner_extract_dir),
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
