#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm


BUCKETS_DEG = (-25.0, -12.0, 0.0, 12.0, 25.0)
LEVEL_BY_BUCKET = {-25.0: -2, -12.0: -1, 0.0: 0, 12.0: 1, 25.0: 2}


@dataclass(frozen=True)
class SourceRow:
    image_path: str
    subject_id: str
    gaze_yaw_deg: float
    gaze_pitch_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    face_bbox: tuple[int, int, int, int] | None
    eye_bbox: tuple[int, int, int, int] | None
    session: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="ETH-XGaze image root.")
    parser.add_argument("--annotations", required=True, help="CSV, TSV, JSON, JSONL, or PKL annotations file.")
    parser.add_argument("--output-dir", default="data/eth_xgaze_5level")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=4.0)
    parser.add_argument("--max-per-subject-level", type=int, default=1)
    parser.add_argument("--session-column", default="")
    parser.add_argument("--allowed-session", action="append", default=[])
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        text = path.read_text()
        if suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "rows", "data", "annotations"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        raise ValueError("Unsupported JSON annotation structure.")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    if suffix == ".pkl":
        payload = pickle.loads(path.read_bytes())
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "rows", "data", "annotations"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise ValueError("Unsupported PKL annotation structure.")
    raise ValueError(f"Unsupported annotation file: {path}")


def pick_value(row: dict[str, Any], aliases: Iterable[str], default: Any = None) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in ("", None):
            return row[alias]
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    if isinstance(value, (list, tuple)):
        raise TypeError("Expected scalar, got sequence.")
    return float(value)


def parse_bbox(row: dict[str, Any], prefixes: Iterable[str]) -> tuple[int, int, int, int] | None:
    for prefix in prefixes:
        bbox_value = row.get(prefix)
        if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
            x, y, w, h = [int(round(float(v))) for v in bbox_value]
            return x, y, max(1, w), max(1, h)
        if isinstance(bbox_value, dict):
            x = bbox_value.get("x", bbox_value.get("left"))
            y = bbox_value.get("y", bbox_value.get("top"))
            w = bbox_value.get("w", bbox_value.get("width"))
            h = bbox_value.get("h", bbox_value.get("height"))
            if None not in (x, y, w, h):
                return int(round(float(x))), int(round(float(y))), max(1, int(round(float(w)))), max(1, int(round(float(h))))

    for prefix in prefixes:
        x = pick_value(row, (f"{prefix}_x", f"{prefix}x"))
        y = pick_value(row, (f"{prefix}_y", f"{prefix}y"))
        w = pick_value(row, (f"{prefix}_w", f"{prefix}_width", f"{prefix}w"))
        h = pick_value(row, (f"{prefix}_h", f"{prefix}_height", f"{prefix}h"))
        if None not in (x, y, w, h):
            return int(round(float(x))), int(round(float(y))), max(1, int(round(float(w)))), max(1, int(round(float(h))))
    return None


def normalize_row(row: dict[str, Any], session_column: str) -> SourceRow:
    image_path = str(
        pick_value(
            row,
            ("image_path", "path", "file_path", "img_path", "image", "face_path"),
        )
    )
    if not image_path:
        raise ValueError("Missing image path.")

    subject_id = str(pick_value(row, ("subject_id", "subject", "person_id", "identity", "subject")))
    if not subject_id:
        raise ValueError("Missing subject id.")

    gaze_value = pick_value(row, ("gaze", "gaze_vector"))
    if isinstance(gaze_value, (list, tuple)) and len(gaze_value) >= 2:
        gaze_pitch_deg = math.degrees(float(gaze_value[0])) if abs(float(gaze_value[0])) <= math.pi else float(gaze_value[0])
        gaze_yaw_deg = math.degrees(float(gaze_value[1])) if abs(float(gaze_value[1])) <= math.pi else float(gaze_value[1])
    else:
        gaze_yaw_deg = as_float(pick_value(row, ("gaze_yaw_deg", "yaw_deg", "gaze_yaw", "yaw")))
        gaze_pitch_deg = as_float(pick_value(row, ("gaze_pitch_deg", "pitch_deg", "gaze_pitch", "pitch")))

    head_value = pick_value(row, ("head_pose", "head"))
    if isinstance(head_value, (list, tuple)) and len(head_value) >= 2:
        head_pitch_deg = math.degrees(float(head_value[0])) if abs(float(head_value[0])) <= math.pi else float(head_value[0])
        head_roll_deg = 0.0
    else:
        head_pitch_deg = as_float(pick_value(row, ("head_pitch_deg", "head_pitch", "pitch_head")), default=0.0)
        head_roll_deg = as_float(pick_value(row, ("head_roll_deg", "head_roll", "roll")), default=0.0)

    face_bbox = parse_bbox(row, ("face_bbox", "face"))
    eye_bbox = parse_bbox(row, ("eye_bbox", "eyes_bbox", "eye"))
    session = str(row.get(session_column)) if session_column and row.get(session_column) not in ("", None) else None

    return SourceRow(
        image_path=image_path,
        subject_id=subject_id,
        gaze_yaw_deg=gaze_yaw_deg,
        gaze_pitch_deg=gaze_pitch_deg,
        head_pitch_deg=head_pitch_deg,
        head_roll_deg=head_roll_deg,
        face_bbox=face_bbox,
        eye_bbox=eye_bbox,
        session=session,
    )


def choose_bucket(yaw_deg: float, tolerance_deg: float) -> tuple[int, float] | None:
    best_bucket = None
    best_delta = float("inf")
    for bucket in BUCKETS_DEG:
        delta = abs(yaw_deg - bucket)
        if delta <= tolerance_deg and delta < best_delta:
            best_bucket = bucket
            best_delta = delta
    if best_bucket is None:
        return None
    return LEVEL_BY_BUCKET[best_bucket], best_delta


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


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
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )


def default_face_bbox(width: int, height: int) -> tuple[int, int, int, int]:
    return 0, 0, width, height


def default_eye_bbox(face_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = face_bbox
    eye_w = max(16, int(round(w * 0.58)))
    eye_h = max(12, int(round(h * 0.18)))
    eye_x = x + int(round(w * 0.21))
    eye_y = y + int(round(h * 0.25))
    return eye_x, eye_y, eye_w, eye_h


def level_token(level: int) -> str:
    return "0" if level == 0 else f"{level:+d}"


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in (-2, -1, 0, 1, 2):
        (output_dir / f"level_{level_token(level)}").mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(row, session_column=args.session_column) for row in load_rows(Path(args.annotations).resolve())]
    allowed_sessions = set(args.allowed_session)
    grouped: dict[tuple[str, int], list[tuple[SourceRow, float]]] = defaultdict(list)
    skipped = {"head_pose": 0, "session": 0, "yaw_bucket": 0, "missing_image": 0}

    for row in rows:
        if abs(row.head_pitch_deg) > 10.0 or abs(row.head_roll_deg) > 5.0:
            skipped["head_pose"] += 1
            continue
        if allowed_sessions and row.session not in allowed_sessions:
            skipped["session"] += 1
            continue
        bucket = choose_bucket(row.gaze_yaw_deg, tolerance_deg=args.yaw_tolerance_deg)
        if bucket is None:
            skipped["yaw_bucket"] += 1
            continue
        image_path = (image_root / row.image_path).resolve()
        if not image_path.exists():
            skipped["missing_image"] += 1
            continue
        level, delta = bucket
        grouped[(row.subject_id, level)].append((row, delta))

    metadata: list[dict[str, Any]] = []
    for (subject_id, level), candidates in tqdm(sorted(grouped.items()), desc="Preparing ETH-XGaze"):
        candidates.sort(key=lambda item: (item[1], abs(item[0].head_pitch_deg), abs(item[0].gaze_pitch_deg)))
        kept = candidates[: args.max_per_subject_level]
        for index, (row, _) in enumerate(kept):
            image_path = (image_root / row.image_path).resolve()
            image = Image.open(image_path).convert("RGB")
            src_width, src_height = image.size
            face_bbox = clamp_bbox(row.face_bbox or default_face_bbox(src_width, src_height), src_width, src_height)
            eye_bbox = clamp_bbox(row.eye_bbox or default_eye_bbox(face_bbox), src_width, src_height)

            resized = image.resize((args.size, args.size), Image.LANCZOS)
            face_bbox_scaled = clamp_bbox(scale_bbox(face_bbox, src_width, src_height, args.size, args.size), args.size, args.size)
            eye_bbox_scaled = clamp_bbox(scale_bbox(eye_bbox, src_width, src_height, args.size, args.size), args.size, args.size)

            level_name = level_token(level)
            filename = f"subject_{subject_id}_level_{level_name}_{index:02d}.png"
            rel_path = Path(f"level_{level_name}") / filename
            resized.save(output_dir / rel_path)

            metadata.append(
                {
                    "subject_id": subject_id,
                    "level": level,
                    "yaw_rad": math.radians(row.gaze_yaw_deg),
                    "pitch_rad": math.radians(row.gaze_pitch_deg),
                    "head_pitch_rad": math.radians(row.head_pitch_deg),
                    "head_roll_rad": math.radians(row.head_roll_deg),
                    "file_path": str(rel_path.as_posix()),
                    "image_width": args.size,
                    "image_height": args.size,
                    "face_bbox": list(face_bbox_scaled),
                    "eye_bbox": list(eye_bbox_scaled),
                    "session": row.session,
                }
            )

    metadata_path = output_dir / "metadata.json"
    metadata_payload = {
        "format": "eth_xgaze_5level",
        "image_root": str(output_dir),
        "levels": [-2, -1, 0, 1, 2],
        "count": len(metadata),
        "items": metadata,
        "skipped": skipped,
        "notes": {
            "yaw_buckets_deg": list(BUCKETS_DEG),
            "yaw_tolerance_deg": args.yaw_tolerance_deg,
            "head_pitch_limit_deg": 10.0,
            "head_roll_limit_deg": 5.0,
            "head_roll_behavior": "If roll is not present in annotations, the script assumes 0.0 because ETH-XGaze normalized face crops are already roll-corrected.",
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))

    print(f"Wrote {len(metadata)} images to {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Skipped: {json.dumps(skipped, sort_keys=True)}")


if __name__ == "__main__":
    main()
