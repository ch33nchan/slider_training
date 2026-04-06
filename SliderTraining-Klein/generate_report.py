#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from azure_uploader import upload_single_file_to_storage


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "experiment_report"
ASSET_DIR = REPORT_DIR / "assets"
CACHE_PATH = REPORT_DIR / "cdn_cache.json"
HTML_PATH = REPORT_DIR / "lora_experiment_report.html"
KLEIN_HTML_PATH = REPORT_DIR / "flux_klein_experiments.html"
RECENT_HTML_PATH = REPORT_DIR / "recent_runs_last_3_days.html"
SAMPLE_ORDER = ("1", "2", "4", "5", "7", "tewo")


@dataclass(frozen=True)
class MetricRow:
    checkpoint: str
    input_to_zero: float
    minus_to_zero: float
    plus_to_zero: float
    asymmetry_ratio: float
    mask_minus: float
    mask_plus: float
    bg_minus: float
    bg_plus: float


@dataclass(frozen=True)
class AssetEntry:
    title: str
    local_path: Path
    caption: str


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def read_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def write_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def relative_report_path(path: Path) -> str:
    return os.path.relpath(path, REPORT_DIR).replace(os.sep, "/")


def upload_asset(local_path: Path, prefix: str, cache: dict[str, str]) -> str:
    cache_key = str(local_path.relative_to(ROOT))
    if cache_key in cache:
        return cache[cache_key]
    url = upload_single_file_to_storage(str(local_path), prefix=prefix)
    cache[cache_key] = url
    write_cache(cache)
    return url


def build_preview_data_url(local_path: Path, max_edge: int = 1400, quality: int = 88) -> str:
    image = Image.open(local_path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    draw.text(xy, text, font=font, fill=fill)


def fit_image(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / float(image.width)
    new_height = max(1, int(round(image.height * scale)))
    return image.resize((max_width, new_height), Image.LANCZOS)


def build_result_contact_sheet(root: Path, output_path: Path, title: str) -> Path:
    label_font = get_font(22)
    title_font = get_font(30)
    margin = 28
    gap = 20
    header_height = 74
    row_label_width = 100
    row_images: list[tuple[str, Image.Image]] = []
    max_result_width = 1440

    for sample_id in SAMPLE_ORDER:
        result_path = root / sample_id / "result.png"
        if not result_path.exists():
            continue
        row_images.append((sample_id, fit_image(Image.open(result_path).convert("RGB"), max_result_width)))

    if not row_images:
        raise FileNotFoundError(f"No result images found in {root}")

    content_width = row_label_width + max(img.width for _, img in row_images)
    total_height = header_height + margin
    total_height += sum(img.height for _, img in row_images)
    total_height += gap * (len(row_images) - 1)
    total_height += margin

    canvas = Image.new("RGB", (content_width + margin * 2, total_height), "#f5f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((14, 14, canvas.width - 14, canvas.height - 14), radius=20, outline="#d7d0c4", width=2)
    draw_text(draw, (margin, 20), title, title_font, "#111111")

    y = header_height
    for sample_id, image in row_images:
        draw_text(draw, (margin, y + 16), sample_id, label_font, "#4a4a4a")
        canvas.paste(image, (margin + row_label_width, y))
        y += image.height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_triplet_contact_sheet(source_dir: Path, stems: list[str], output_path: Path, title: str) -> Path:
    font = get_font(20)
    title_font = get_font(28)
    small_font = get_font(18)
    margin = 28
    gap = 12
    title_height = 68
    label_height = 28
    row_title_width = 150
    cell_width = 220
    cell_height = 220
    headers = ("Left", "Neutral", "Right")

    total_width = margin * 2 + row_title_width + len(headers) * cell_width + (len(headers) - 1) * gap
    total_height = title_height + margin
    total_height += len(stems) * (cell_height + label_height + gap)
    total_height += margin

    canvas = Image.new("RGB", (total_width, total_height), "#f5f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((14, 14, canvas.width - 14, canvas.height - 14), radius=20, outline="#d7d0c4", width=2)
    draw_text(draw, (margin, 18), title, title_font, "#111111")

    header_y = title_height - 4
    for index, header in enumerate(headers):
        x = margin + row_title_width + index * (cell_width + gap) + 10
        draw_text(draw, (x, header_y), header, small_font, "#4a4a4a")

    y = title_height + margin
    for stem in stems:
        draw_text(draw, (margin, y + 90), stem, font, "#4a4a4a")
        for column_index, suffix in enumerate(("left", "neutral", "right")):
            image_path = source_dir / f"{stem}_{suffix}.png"
            image = Image.open(image_path).convert("RGB").resize((cell_width, cell_height), Image.LANCZOS)
            x = margin + row_title_width + column_index * (cell_width + gap)
            canvas.paste(image, (x, y))
        y += cell_height + label_height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_pipeline_comparison_sheet(
    pipeline_dir: Path,
    klein_dir: Path,
    output_path: Path,
    title: str,
    direction: str,
) -> Path:
    label_font = get_font(18)
    title_font = get_font(28)
    header_font = get_font(16)
    margin = 24
    gap = 12
    row_label_width = 90
    cell_width = 190
    cell_height = 190
    header_height = 72
    columns = [
        ("Source", "source_full.png"),
        ("LivePortrait only", f"lp_{direction}.png"),
        ("Klein LoRA only", f"scale_{'-8.0' if direction == 'left' else '+8.0'}.png"),
        ("LivePortrait + Klein LoRA", f"lora_{direction}_full.png"),
    ]

    total_width = margin * 2 + row_label_width + len(columns) * cell_width + (len(columns) - 1) * gap
    total_height = header_height + margin * 2 + len(SAMPLE_ORDER) * cell_height + (len(SAMPLE_ORDER) - 1) * gap

    canvas = Image.new("RGB", (total_width, total_height), "#f5f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((14, 14, canvas.width - 14, canvas.height - 14), radius=20, outline="#d7d0c4", width=2)
    draw_text(draw, (margin, 18), title, title_font, "#111111")

    header_y = 64
    for column_index, (label, _) in enumerate(columns):
        x = margin + row_label_width + column_index * (cell_width + gap)
        draw_text(draw, (x + 8, header_y), label, header_font, "#4a4a4a")

    y = header_height + margin
    for sample_id in SAMPLE_ORDER:
        draw_text(draw, (margin, y + 78), sample_id, label_font, "#4a4a4a")
        sample_pipeline_dir = pipeline_dir / sample_id
        sample_klein_dir = klein_dir / sample_id
        for column_index, (_, filename) in enumerate(columns):
            if filename == "source_full.png":
                image_path = sample_pipeline_dir / filename
            elif filename.startswith("scale_"):
                image_path = sample_klein_dir / filename
            else:
                image_path = sample_pipeline_dir / filename
            image = Image.open(image_path).convert("RGB").resize((cell_width, cell_height), Image.LANCZOS)
            x = margin + row_label_width + column_index * (cell_width + gap)
            canvas.paste(image, (x, y))
        y += cell_height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_pipeline_sample_sheet(
    pipeline_dir: Path,
    klein_dir: Path,
    sample_id: str,
    output_path: Path,
) -> Path:
    title_font = get_font(24)
    header_font = get_font(16)
    margin = 24
    gap = 10
    row_gap = 18
    cell_width = 190
    cell_height = 190
    label_width = 72
    header_height = 76
    columns = [
        ("Source", "source_full.png", "pipeline"),
        ("LivePortrait only", None, "pipeline"),
        ("Klein LoRA only", None, "klein"),
        ("Both", None, "pipeline"),
    ]

    total_width = margin * 2 + label_width + len(columns) * cell_width + (len(columns) - 1) * gap
    total_height = header_height + margin + (cell_height * 2) + row_gap + margin
    canvas = Image.new("RGB", (total_width, total_height), "#f5f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((14, 14, canvas.width - 14, canvas.height - 14), radius=20, outline="#d7d0c4", width=2)
    draw_text(draw, (margin, 18), f"Pipeline detail - {sample_id}", title_font, "#111111")

    header_y = 62
    for column_index, (label, _, _) in enumerate(columns):
        x = margin + label_width + column_index * (cell_width + gap)
        draw_text(draw, (x + 8, header_y), label, header_font, "#4a4a4a")

    sample_pipeline_dir = pipeline_dir / sample_id
    sample_klein_dir = klein_dir / sample_id
    row_specs = [
        ("Left", "left"),
        ("Right", "right"),
    ]

    y = header_height + margin
    for row_label, direction in row_specs:
        draw_text(draw, (margin, y + 80), row_label, header_font, "#4a4a4a")
        file_map = {
            "pipeline_source": sample_pipeline_dir / "source_full.png",
            "pipeline_liveportrait": sample_pipeline_dir / f"lp_{direction}.png",
            "pipeline_both": sample_pipeline_dir / f"lora_{direction}_full.png",
            "klein_only": sample_klein_dir / f"scale_{'-8.0' if direction == 'left' else '+8.0'}.png",
        }
        for column_index, (_, _, origin) in enumerate(columns):
            if column_index == 0:
                image_path = file_map["pipeline_source"]
            elif column_index == 1:
                image_path = file_map["pipeline_liveportrait"]
            elif column_index == 2:
                image_path = file_map["klein_only"]
            else:
                image_path = file_map["pipeline_both"]
            image = Image.open(image_path).convert("RGB").resize((cell_width, cell_height), Image.LANCZOS)
            x = margin + label_width + column_index * (cell_width + gap)
            canvas.paste(image, (x, y))
        y += cell_height + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def compute_metrics(root: Path, minus_name: str, zero_name: str, plus_name: str) -> MetricRow:
    values = []
    mask_values = []
    bg_values = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        zero_path = sample_dir / zero_name
        input_path = sample_dir / "input.png"
        if not input_path.exists():
            input_path = zero_path

        input_rgb = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.float32)
        zero_rgb = np.asarray(Image.open(zero_path).convert("RGB"), dtype=np.float32)
        minus_rgb = np.asarray(Image.open(sample_dir / minus_name).convert("RGB"), dtype=np.float32)
        plus_rgb = np.asarray(Image.open(sample_dir / plus_name).convert("RGB"), dtype=np.float32)
        mask_path = sample_dir / "eye_mask.png"
        if mask_path.exists():
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        else:
            mask = np.ones(zero_rgb.shape[:2], dtype=np.float32)
        mask_sel = mask > 0.05
        bg_sel = ~mask_sel

        zero_diff = np.abs(input_rgb - zero_rgb).mean(axis=2)
        minus_diff = np.abs(zero_rgb - minus_rgb).mean(axis=2)
        plus_diff = np.abs(zero_rgb - plus_rgb).mean(axis=2)

        values.append((float(zero_diff.mean()), float(minus_diff.mean()), float(plus_diff.mean())))
        mask_values.append(
            (
                float(minus_diff[mask_sel].mean()) if mask_sel.any() else 0.0,
                float(plus_diff[mask_sel].mean()) if mask_sel.any() else 0.0,
            )
        )
        bg_values.append(
            (
                float(minus_diff[bg_sel].mean()) if bg_sel.any() else 0.0,
                float(plus_diff[bg_sel].mean()) if bg_sel.any() else 0.0,
            )
        )

    values_arr = np.asarray(values)
    mask_arr = np.asarray(mask_values)
    bg_arr = np.asarray(bg_values)
    minus_mean = float(values_arr[:, 1].mean())
    plus_mean = float(values_arr[:, 2].mean())
    return MetricRow(
        checkpoint=root.name,
        input_to_zero=float(values_arr[:, 0].mean()),
        minus_to_zero=minus_mean,
        plus_to_zero=plus_mean,
        asymmetry_ratio=minus_mean / max(plus_mean, 1e-6),
        mask_minus=float(mask_arr[:, 0].mean()),
        mask_plus=float(mask_arr[:, 1].mean()),
        bg_minus=float(bg_arr[:, 0].mean()),
        bg_plus=float(bg_arr[:, 1].mean()),
    )


def render_kv_table(data: dict[str, str]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in data.items()
    )
    return f"<table class='kv-table'>{rows}</table>"


def render_metric_table(rows: Iterable[MetricRow]) -> str:
    body = "".join(
        (
            "<tr>"
            f"<td>{html.escape(row.checkpoint)}</td>"
            f"<td>{row.input_to_zero:.3f}</td>"
            f"<td>{row.minus_to_zero:.3f}</td>"
            f"<td>{row.plus_to_zero:.3f}</td>"
            f"<td>{row.asymmetry_ratio:.2f}x</td>"
            f"<td>{row.mask_minus:.3f}</td>"
            f"<td>{row.mask_plus:.3f}</td>"
            "</tr>"
        )
        for row in rows
    )
    return (
        "<table class='metric-table'>"
        "<thead><tr>"
        "<th>Checkpoint</th><th>Input -> 0</th><th>-dir -> 0</th><th>+dir -> 0</th><th>Asymmetry</th>"
        "<th>Mask -dir</th><th>Mask +dir</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_list(items: list[str]) -> str:
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def make_individual_result_entries(root: Path, title_prefix: str, caption_prefix: str = "") -> list[AssetEntry]:
    entries: list[AssetEntry] = []
    for sample_id in SAMPLE_ORDER:
        result_path = root / sample_id / "result.png"
        if not result_path.exists():
            continue
        caption = f"{caption_prefix}sample {sample_id}".strip()
        entries.append(AssetEntry(f"{title_prefix} - {sample_id}", result_path, caption))
    return entries


def parse_top_level_config(config_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in config_path.read_text().splitlines():
        if not raw_line or raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_ai_toolkit_config(config_path: Path) -> dict[str, str]:
    text = config_path.read_text()

    def find(pattern: str, default: str = "-") -> str:
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else default

    name = find(r'^\s{2}name:\s*"([^"]+)"')
    output_dir = f"outputs/{name}" if name and name != "-" else "-"
    return {
        "name": name if name != "-" else config_path.stem,
        "config_style": "ai-toolkit extension",
        "config_path": str(config_path.relative_to(ROOT)),
        "output_dir": output_dir,
        "prompt": find(r'^\s{12}target_class:\s*"([^"]+)"'),
        "dataset": find(r'^\s{12}neg_folder:\s*"([^"]+)"'),
        "resolution": f"{find(r'^\\s{8}width:\\s*([^\\s#]+)')} x {find(r'^\\s{8}height:\\s*([^\\s#]+)')}",
        "steps": find(r'^\s{8}steps:\s*([^\s#]+)'),
        "rank": find(r'^\s{8}linear:\s*([^\s#]+)'),
        "alpha": find(r'^\s{8}linear_alpha:\s*([^\s#]+)'),
        "train_method": find(r"^\s{4}- type:\s*'([^']+)'"),
        "lr": find(r'^\s{8}lr:\s*([^\s#]+)'),
        "eta": "-",
        "base_model": find(r'^\s{8}arch:\s*"([^"]+)"'),
    }


def resolve_output_path(output_dir: str) -> Path | None:
    if not output_dir or output_dir == "-":
        return None
    cleaned = output_dir.lstrip("./")
    candidate = ROOT / cleaned
    if candidate == OUTPUT_DIR:
        return None
    return candidate


def find_preview_artifact(output_path: Path | None) -> tuple[Path | None, str]:
    if output_path is None or not output_path.exists():
        return None, "-"
    preferred_names = [
        "result.png",
        "result.jpg",
        "result.jpeg",
        "result.webp",
        "infer_fix_v2.png",
        "infer_000400_fix.png",
        "infer_000400.png",
        "scale_+0.0.png",
        "scale_+8.0.png",
        "scale_+5.0.png",
        "scale_+3.0.png",
        "scale_+1.5.png",
        "strip_p0.png",
        "loss.png",
    ]
    for name in preferred_names:
        candidate = output_path / name
        if candidate.exists():
            return candidate, name
    images = sorted(
        path
        for path in output_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if images:
        return images[0], images[0].name
    return None, "-"


def find_test_input_artifact(output_path: Path | None) -> tuple[Path | None, str]:
    if output_path is None or not output_path.exists():
        return None, "-"
    preferred_names = [
        "human1.jpg",
        "human1.png",
        "source_full.png",
        "source.png",
        "input.png",
        "reference.png",
        "reference.jpg",
        "scale_+0.0.png",
    ]
    for name in preferred_names:
        candidate = output_path / name
        if candidate.exists():
            return candidate, name
    candidates = []
    for path in sorted(output_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        lower_name = path.name.lower()
        if lower_name.startswith("result") or lower_name.startswith("scale_") or lower_name.startswith("loss") or lower_name.startswith("infer_") or lower_name.startswith("strip_") or lower_name == "eye_mask.png":
            continue
        candidates.append(path)
    if candidates:
        return candidates[0], candidates[0].name
    return None, "-"


def select_pre_flux_klein_experiments(experiments: list[dict[str, object]]) -> list[dict[str, object]]:
    preferred_order = [
        "eye_gaze_adult",
        "eye_gaze_neutral",
        "eye_gaze_v2",
        "eye_gaze_v3",
        "eye_gaze_v3b",
        "eye_gaze_v4",
        "eye_gaze_v5",
        "eye_gaze_v6",
        "eye_gaze_v7",
        "eye_gaze_v7b",
        "eye_gaze_v7c",
        "eye_gaze_v8",
        "eye_gaze_text_v1",
        "eye_gaze_text_v2",
        "eye_gaze_text_v3",
        "eye_gaze_horizontal_v1",
        "eye_gaze_horizontal_texture_v1",
        "eye_gaze_vertical_v1",
        "eye_gaze_vertical_texture_v1",
        "eye_gaze_klein_v1",
        "eye_gaze_klein_v2",
    ]
    order_index = {name: index for index, name in enumerate(preferred_order)}
    filtered = [
        item for item in experiments
        if str(item.get("name")) in order_index
    ]
    return sorted(filtered, key=lambda item: order_index[str(item["name"])])


def discover_klein_experiments() -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []
    seen_output_dirs: set[str] = set()
    for config_path in sorted((ROOT / "config").glob("*.yaml")):
        raw_text = config_path.read_text()
        if "klein-9b" not in raw_text and "flux2_klein_9b" not in raw_text:
            continue
        if 'job: extension' in raw_text and 'image_reference_slider_trainer' in raw_text:
            cfg = parse_ai_toolkit_config(config_path)
        else:
            cfg = parse_top_level_config(config_path)
            cfg = {
                "name": config_path.stem,
                "config_style": "flat slider config",
                "config_path": str(config_path.relative_to(ROOT)),
                "output_dir": cfg.get("output_dir", "-"),
                "prompt": cfg.get("prompt", "-"),
                "dataset": cfg.get("neg_image_dir", "-"),
                "resolution": f"{cfg.get('width', '-')} x {cfg.get('height', '-')}",
                "steps": cfg.get("max_train_steps", "-"),
                "rank": cfg.get("rank", "-"),
                "alpha": cfg.get("alpha", "-"),
                "train_method": cfg.get("train_method", "-"),
                "lr": cfg.get("lr", "-"),
                "eta": cfg.get("eta", "-"),
                "base_model": cfg.get("transformer_path", "-"),
            }
        output_path = resolve_output_path(str(cfg.get("output_dir", "-")))
        if output_path is not None:
            seen_output_dirs.add(str(output_path.relative_to(ROOT)))
        input_path, input_file = find_test_input_artifact(output_path)
        preview_path, preview_file = find_preview_artifact(output_path)
        experiments.append(
            {
                "name": cfg.get("name", config_path.stem),
                "config_style": cfg.get("config_style", "-"),
                "config_path": cfg.get("config_path", str(config_path.relative_to(ROOT))),
                "output_dir": cfg.get("output_dir", "-"),
                "prompt": cfg.get("prompt", "-"),
                "dataset": cfg.get("dataset", "-"),
                "resolution": cfg.get("resolution", "-"),
                "steps": cfg.get("steps", "-"),
                "rank": cfg.get("rank", "-"),
                "alpha": cfg.get("alpha", "-"),
                "train_method": cfg.get("train_method", "-"),
                "lr": cfg.get("lr", "-"),
                "eta": cfg.get("eta", "-"),
                "base_model": cfg.get("base_model", "-"),
                "input_path": input_path,
                "input_file": input_file,
                "preview_path": preview_path,
                "preview_file": preview_file,
                "status": "preview artifact available" if preview_path is not None else "config present, no local preview artifact",
            }
        )
    artifact_patterns = ("eye_gaze*", "gaze_*", "lora_gaze_test*")
    for pattern in artifact_patterns:
        for output_path in sorted(OUTPUT_DIR.glob(pattern)):
            if not output_path.is_dir():
                continue
            rel_output = str(output_path.relative_to(ROOT))
            if rel_output in seen_output_dirs:
                continue
            input_path, input_file = find_test_input_artifact(output_path)
            preview_path, preview_file = find_preview_artifact(output_path)
            if preview_path is None:
                continue
            experiments.append(
                {
                    "name": output_path.name,
                    "config_style": "artifact only",
                    "config_path": "-",
                    "output_dir": rel_output,
                    "prompt": "-",
                    "dataset": "-",
                    "resolution": "-",
                    "steps": "-",
                    "rank": "-",
                    "alpha": "-",
                    "train_method": "-",
                    "lr": "-",
                    "eta": "-",
                    "base_model": "-",
                    "input_path": input_path,
                    "input_file": input_file,
                    "preview_path": preview_path,
                    "preview_file": preview_file,
                    "status": "artifact discovered without matching config",
                }
            )
    return experiments


def render_klein_inventory_table(experiments: list[dict[str, object]]) -> str:
    rows = []
    for item in experiments:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['name']))}</td>"
            f"<td>{html.escape(str(item['config_style']))}</td>"
            f"<td>{html.escape(str(item['config_path']))}</td>"
            f"<td>{html.escape(str(item['dataset']))}</td>"
            f"<td>{html.escape(str(item['output_dir']))}</td>"
            f"<td>{html.escape(str(item['resolution']))}</td>"
            f"<td>{html.escape(str(item['steps']))}</td>"
            f"<td>{html.escape(str(item['rank']))}</td>"
            f"<td>{html.escape(str(item['alpha']))}</td>"
            f"<td>{html.escape(str(item['lr']))}</td>"
            f"<td>{html.escape(str(item['train_method']))}</td>"
            f"<td>{html.escape(str(item['eta']))}</td>"
            f"<td>{html.escape(str(item['preview_file']))}</td>"
            f"<td>{html.escape(str(item['status']))}</td>"
            "</tr>"
        )
    return (
        "<table class='metric-table'>"
        "<thead><tr>"
        "<th>Name</th><th>Config Style</th><th>Config</th><th>Dataset</th><th>Output Dir</th><th>Resolution</th><th>Steps</th>"
        "<th>Rank</th><th>Alpha</th><th>LR</th><th>Train Method</th><th>Eta</th><th>Preview</th><th>Status</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def make_assets() -> dict[str, Path]:
    assets: dict[str, Path] = {}

    assets["klein_500"] = build_result_contact_sheet(
        ROOT / "outputs/character_lora_checkpoints/horizontal_stronger_gpu1/slider_000500",
        ASSET_DIR / "klein_500_sheet.png",
        "Klein texture LoRA - checkpoint 500",
    )
    assets["klein_5000"] = build_result_contact_sheet(
        ROOT / "outputs/character_lora_checkpoints/horizontal_stronger_gpu1/slider_005000",
        ASSET_DIR / "klein_5000_sheet.png",
        "Klein texture LoRA - checkpoint 5000",
    )
    assets["klein_8000"] = build_result_contact_sheet(
        ROOT / "outputs/character_lora_checkpoints/horizontal_stronger_gpu1/slider_008000",
        ASSET_DIR / "klein_8000_sheet.png",
        "Klein texture LoRA - checkpoint 8000",
    )
    assets["klein_final"] = build_result_contact_sheet(
        ROOT / "outputs/character_lora",
        ASSET_DIR / "klein_final_sheet.png",
        "Klein texture LoRA - final chosen output",
    )

    for step in ("000250", "000500"):
        assets[f"quick_{step}"] = build_result_contact_sheet(
            ROOT / f"flux1dev_pair_slider/evals/step_{step}_eye",
            ASSET_DIR / f"flux_quick_{step}_sheet.png",
            f"FLUX.1-dev quick prompt slider - step {step}",
        )

    for step in ("000250", "000500"):
        assets[f"pair15_{step}"] = build_result_contact_sheet(
            ROOT / f"flux1dev_pair_slider/evals/pair_step_{step}_eye",
            ASSET_DIR / f"flux_pair15_{step}_sheet.png",
            f"FLUX.1-dev paired slider - 15 pairs - step {step}",
        )

    for step in ("000250", "000500", "000750", "001000"):
        assets[f"pair30_{step}"] = build_result_contact_sheet(
            ROOT / f"flux1dev_pair_slider/evals/pair30_step_{step}_eye",
            ASSET_DIR / f"flux_pair30_{step}_sheet.png",
            f"FLUX.1-dev paired slider - 30 pairs - step {step}",
        )

    for step in ("000250", "000500", "000750", "001000"):
        assets[f"pair30_eye_{step}"] = build_result_contact_sheet(
            ROOT / f"flux1dev_pair_slider/evals/pair30_eye_crop_step_{step}_eye",
            ASSET_DIR / f"flux_pair30_eye_{step}_sheet.png",
            f"FLUX.1-dev paired slider - 30 pairs eye-crop - step {step}",
        )

    triplet_source = ROOT / "outputs/liveportrait_only"
    stems = [f"face_{index:04d}" for index in range(30)]
    for chunk_index in range(3):
        chunk = stems[chunk_index * 10:(chunk_index + 1) * 10]
        assets[f"dataset_{chunk_index}"] = build_triplet_contact_sheet(
            triplet_source,
            chunk,
            ASSET_DIR / f"dataset_triplets_{chunk_index + 1}.png",
            f"LivePortrait triplets used to build the 30-pair FLUX dataset ({chunk[0]} to {chunk[-1]})",
        )

    assets["liveportrait_overview"] = build_triplet_contact_sheet(
        triplet_source,
        stems[:6],
        ASSET_DIR / "liveportrait_overview.png",
        "LivePortrait output overview used for the paired FLUX dataset",
    )
    assets["pipeline_left"] = build_pipeline_comparison_sheet(
        ROOT / "outputs/pipeline_lora_gaze_characters",
        ROOT / "outputs/character_lora",
        ASSET_DIR / "pipeline_left_comparison.png",
        "Left direction: source vs LivePortrait only vs Klein LoRA only vs both",
        "left",
    )
    assets["pipeline_right"] = build_pipeline_comparison_sheet(
        ROOT / "outputs/pipeline_lora_gaze_characters",
        ROOT / "outputs/character_lora",
        ASSET_DIR / "pipeline_right_comparison.png",
        "Right direction: source vs LivePortrait only vs Klein LoRA only vs both",
        "right",
    )
    for sample_id in SAMPLE_ORDER:
        assets[f"pipeline_sample_{sample_id}"] = build_pipeline_sample_sheet(
            ROOT / "outputs/pipeline_lora_gaze_characters",
            ROOT / "outputs/character_lora",
            sample_id,
            ASSET_DIR / f"pipeline_sample_{sample_id}.png",
        )
    return assets


def build_experiments(assets: dict[str, Path]) -> list[dict[str, object]]:
    klein_metrics = [
        compute_metrics(ROOT / "outputs/character_lora_checkpoints/horizontal_stronger_gpu1/slider_000500", "scale_-8.0.png", "scale_+0.0.png", "scale_+8.0.png"),
        compute_metrics(ROOT / "outputs/character_lora_checkpoints/horizontal_stronger_gpu1/slider_005000", "scale_-8.0.png", "scale_+0.0.png", "scale_+8.0.png"),
        compute_metrics(ROOT / "outputs/character_lora_checkpoints/horizontal_stronger_gpu1/slider_008000", "scale_-8.0.png", "scale_+0.0.png", "scale_+8.0.png"),
        compute_metrics(ROOT / "outputs/character_lora", "scale_-8.0.png", "scale_+0.0.png", "scale_+8.0.png"),
    ]
    quick_metrics = [
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/step_000250_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/step_000500_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
    ]
    pair15_metrics = [
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair_step_000250_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair_step_000500_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
    ]
    pair30_metrics = [
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_step_000250_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_step_000500_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_step_000750_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_step_001000_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
    ]
    pair30_eye_metrics = [
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_eye_crop_step_000250_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_eye_crop_step_000500_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_eye_crop_step_000750_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
        compute_metrics(ROOT / "flux1dev_pair_slider/evals/pair30_eye_crop_step_001000_eye", "scale_-1.0.png", "scale_+0.0.png", "scale_+1.0.png"),
    ]
    klein_inventory = discover_klein_experiments()
    klein_inventory_assets = [
        AssetEntry(
            f"Klein 9B experiment - {item['name']}",
            item["preview_path"],
            f"{item['config_path']} -> {item['preview_file']}",
        )
        for item in klein_inventory
        if item["preview_path"] is not None and item["preview_path"].exists()
    ]

    return [
        {
            "tab_id": "overview",
            "title": "Overview",
            "intro": "Summary of the FLUX Klein 9B experiment branches, the FLUX.1-dev experiment branches, the LivePortrait-derived paired dataset, and the combined LivePortrait-to-Klein refinement path.",
            "analysis": render_list(
                [
                    "SliderTraining-Klein was used as the main orchestration repo for dataset generation, Klein 9B LoRA training and inference, LivePortrait-based synthesis, and the later FLUX.1-dev branches.",
                    "The FLUX Klein 9B branch produced the strongest eye-region response, but directional control remained inconsistent.",
                    "The FLUX quick LoRA learned a stronger but artifact-prone edit.",
                    "The paired FLUX runs improved locality and identity preservation, but gaze motion remained weak.",
                    "The 30-pair eye-crop trainer improved symmetry and locality, but still did not cross the threshold for usable gaze control.",
                    "LivePortrait was critical both as a data generator for FLUX paired training and as the first stage of the combined LivePortrait-plus-Klein pipeline.",
                ]
            ),
            "content": (
                "<div class='overview-grid'>"
                f"<div>{render_metric_table([quick_metrics[-1], pair15_metrics[-1], pair30_metrics[-1], pair30_eye_metrics[-1]])}</div>"
                "<div class='callout'><h3>Bottom line</h3>"
                "<p>The pipeline and reporting are instrumented end to end, but the current supervision signal is still not sufficient to produce reliable horizontal gaze control.</p>"
                "</div></div>"
            ),
            "assets": [],
        },
        {
            "tab_id": "pipeline_lp_klein",
            "title": "LivePortrait + Klein LoRA",
            "intro": "Combined inference pipeline using LivePortrait as the gaze-retargeting stage and the Klein LoRA path as the downstream refinement stage.",
            "dataset": render_kv_table(
                {
                    "Input identities": "6 character images: 1, 2, 4, 5, 7, tewo",
                    "LivePortrait stage output": "outputs/pipeline_lora_gaze_characters/<id>/lp_left.png and lp_right.png",
                    "Klein-only stage output": "outputs/character_lora/<id>/scale_-8.0.png and scale_+8.0.png",
                    "Combined stage output": "outputs/pipeline_lora_gaze_characters/<id>/lora_left_full.png and lora_right_full.png",
                }
            ),
            "training": render_kv_table(
                {
                    "Pipeline script": "pipeline_lora_gaze.py",
                    "LivePortrait gaze settings": "left_gaze=-18.0, right_gaze=+18.0",
                    "Klein img2img strength": "0.4",
                    "Klein denoise steps": "28",
                    "Prompt": "a person",
                    "Output resolution": "Keeps source resolution for final save, runs Klein at a 16-aligned model size",
                }
            ),
            "lora": render_kv_table(
                {
                    "Base model": "Klein 9B",
                    "Klein LoRA config": "config/eye_gaze_v7b.yaml",
                    "LoRA rank / alpha": "64 / 1",
                    "LoRA scales at inference": "left_scale=-8.0, right_scale=+8.0",
                    "Final blend weight": "lora_mix=0.18",
                    "Eye mask blend": "eye_mask_scale=1.9, eye_mask_blur=41",
                }
            ),
            "content": (
                "<h3>How it works</h3>"
                "<ol class='flow-list'>"
                "<li>The source image is loaded at full resolution and an eye mask is estimated from the source frame.</li>"
                "<li>LivePortrait runs first and generates <code>lp_left.png</code> and <code>lp_right.png</code> using <code>eyeball_direction_x</code> values of <code>-18.0</code> and <code>+18.0</code>.</li>"
                "<li>Each LivePortrait warp is resized to the Klein model size and passed into the Klein img2img path as both the reference image and the starting latent.</li>"
                "<li>The Klein LoRA is applied during denoising with inference scales <code>-8.0</code> for left and <code>+8.0</code> for right.</li>"
                "<li>The refined Klein output is blended back into the source frame only inside the eye mask with <code>lora_mix=0.18</code>, producing <code>lora_left_full.png</code> and <code>lora_right_full.png</code>.</li>"
                "</ol>"
                "<div class='path-box'>"
                "<code>source_full.png -> LivePortrait (lp_left / lp_right) -> Klein LoRA denoise -> eye-mask blend -> lora_left_full / lora_right_full</code>"
                "</div>"
            ),
            "analysis": render_list(
                [
                    "The combined path is a two-stage inference pipeline rather than a single LoRA-only operation.",
                    "LivePortrait-only output preserves identity well and provides the directional prior.",
                    "Klein-only output can change the eye region more aggressively, but without the LivePortrait prior the edit is less constrained.",
                    "The combined output preserves the LivePortrait direction signal while attempting to sharpen or regularize the eye region through the Klein LoRA path.",
                    "The implementation is technically sound, but the final visual gain over LivePortrait-only output remains limited.",
                ]
            ),
            "assets": [
                AssetEntry("Left comparison", assets["pipeline_left"], "For each identity: source, LivePortrait only, Klein LoRA only, and the combined pipeline for the left direction."),
                AssetEntry("Right comparison", assets["pipeline_right"], "For each identity: source, LivePortrait only, Klein LoRA only, and the combined pipeline for the right direction."),
            ],
            "individuals": [
                AssetEntry(f"Pipeline detail - {sample_id}", assets[f"pipeline_sample_{sample_id}"], f"Per-sample left and right stage breakdown for {sample_id}.")
                for sample_id in SAMPLE_ORDER
            ],
        },
        {
            "tab_id": "klein",
            "title": "FLUX Klein 9B Experiments",
            "intro": "Klein 9B experiments run inside SliderTraining-Klein, covering direct LoRA training, checkpoint sweeps, text and neutral variants, texture variants, and the later LivePortrait-to-Klein refinement path.",
            "dataset": render_kv_table(
                {
                    "Primary horizontal dataset": "data/gaze_horizontal_texture/neg, pos, neutral, masks",
                    "Other Klein datasets": "columbia_gaze, gaze_s15, adult, text-based prompt variants, vertical gaze variants",
                    "Eval identities": "6 character images: 1, 2, 4, 5, 7, tewo",
                    "Repo role": "SliderTraining-Klein owns dataset generation, Klein training scripts, evaluation scripts, and downstream LivePortrait-to-Klein pipeline scripts.",
                }
            ),
            "training": render_kv_table(
                {
                    "Base model": "Klein 9B / flux2_klein_9b",
                    "Primary trainer": "train_slider.py",
                    "Alternative trainer branch": "ai-toolkit extension configs such as eye_gaze_klein_v1.yaml and eye_gaze_klein_v2.yaml",
                    "Primary inference scripts": "infer_gaze_slider.py, inference_slider.py, pipeline_gaze.py, pipeline_lora_gaze.py",
                    "Common method": "xattn LoRA training with paired neg and pos supervision, optional neutral images, and eye-region weighting",
                    "Duration": "Only some runs preserved wall-clock timestamps in the repo snapshot; many Klein-only runs retained configs and artifacts but not logs.",
                }
            ),
            "lora": render_kv_table(
                {
                    "Main horizontal texture run": "config/eye_gaze_horizontal_texture_v1.yaml",
                    "Chosen production-style run": "config/eye_gaze_v7b.yaml",
                    "Representative ranks": "16, 32, 64",
                    "Representative alpha values": "1, 16, 32",
                    "Representative eta values": "4, 6, 8, 10",
                    "Eval scales": "Classic Klein sweeps used -8/0/+8 or smaller scale ladders depending on branch",
                }
            ),
            "content": (
                "<h3>How SliderTraining-Klein is used in the Klein branch</h3>"
                "<ol class='flow-list'>"
                "<li><code>generate_gaze_dataset_v2.py</code> and related scripts build paired neg/neutral/pos image folders and eye masks.</li>"
                "<li><code>train_slider.py</code> or the ai-toolkit extension configs train a Klein 9B LoRA against those paired targets.</li>"
                "<li><code>infer_gaze_slider.py</code> and related inference scripts generate scale sweeps and per-identity outputs for review.</li>"
                "<li><code>pipeline_lora_gaze.py</code> takes the Klein LoRA branch and uses it after a LivePortrait warp when combined inference is required.</li>"
                "</ol>"
                "<div class='path-box'>"
                "<code>paired dataset -> Klein 9B LoRA train -> scale sweep inference -> checkpoint review -> optional LivePortrait + Klein pipeline</code>"
                "</div>"
                "<h3>Experiment inventory</h3>"
                f"{render_klein_inventory_table(klein_inventory)}"
            ),
            "analysis": render_list(
                [
                    "This tab is the FLUX Klein 9B branch in aggregate, not only the final chosen run.",
                    "The repo contains multiple Klein LoRA configs with different rank, alpha, eta, dataset, and resolution settings; the inventory table surfaces those differences directly.",
                    "The main horizontal texture branch produced the largest eye-region response among the available LoRA experiments in this snapshot.",
                    "Even the stronger Klein branches remained visually subtle and did not turn into a clean horizontal gaze slider.",
                ]
            ),
            "metrics": render_metric_table(klein_metrics),
            "assets": [
                AssetEntry("Checkpoint 500", assets["klein_500"], "Early checkpoint. Stronger edit magnitude, but not yet stable."),
                AssetEntry("Checkpoint 5000", assets["klein_5000"], "Middle checkpoint. Similar behavior, still not a clean gaze direction change."),
                AssetEntry("Checkpoint 8000", assets["klein_8000"], "Late checkpoint. Slightly cleaner but still weak and subtle."),
                AssetEntry("Final chosen output", assets["klein_final"], "Final Klein output that was kept in the main output folder."),
            ],
            "individuals": klein_inventory_assets,
        },
        {
            "tab_id": "flux_quick",
            "title": "FLUX.1-dev Quick LoRA",
            "intro": "Fast prompt-only FLUX.1-dev baseline using text supervision without paired image targets.",
            "dataset": render_kv_table(
                {
                    "Training dataset": "No paired image dataset. Prompt-only slider training.",
                    "Eval set": "6 character images: 1, 2, 4, 5, 7, tewo",
                    "Eval path": "flux1dev_pair_slider/evals/step_000250_eye and step_000500_eye",
                }
            ),
            "training": render_kv_table(
                {
                    "Model": "FLUX.1-dev",
                    "Prompt groups": "4 text prompt triplets for left/right gaze",
                    "Train steps": "1000 planned; evaluated at 250 and 500",
                    "LR": "0.001",
                    "Scheduler": "constant",
                    "Train method": "full",
                    "Duration": "Step 250 at 30m 27s, step 500 at 1h 00m 29s",
                }
            ),
            "lora": render_kv_table(
                {
                    "LoRA rank": "16",
                    "LoRA alpha": "1",
                    "Num sliders": "1",
                    "Eta": "6",
                    "Eval scales": "-1, 0, +1",
                    "Eval crop": "Eye-preserving eval enabled",
                }
            ),
            "analysis": render_list(
                [
                    "This baseline produced the strongest FLUX-side edit magnitude.",
                    "Directionality remained highly asymmetric, with the negative direction dominating the positive direction.",
                    "After the evaluation fix the edit stayed localized, but the visual change still resembled repainting rather than convincing iris motion.",
                ]
            ),
            "metrics": render_metric_table(quick_metrics),
            "assets": [
                AssetEntry("Step 250", assets["quick_000250"], "Prompt-only FLUX run at 250 steps."),
                AssetEntry("Step 500", assets["quick_000500"], "Prompt-only FLUX run at 500 steps. Stronger, but still artifact-prone."),
            ],
            "individuals": make_individual_result_entries(ROOT / "flux1dev_pair_slider/evals/step_000500_eye", "FLUX quick step 500", "Prompt-only FLUX output, "),
        },
        {
            "tab_id": "flux_pair15",
            "title": "FLUX.1-dev Paired 15",
            "intro": "First paired FLUX.1-dev run using a 15-triplet left-right dataset and eye masks for locality.",
            "dataset": render_kv_table(
                {
                    "Dataset": "15 synthetic gaze pairs with neutral and eye-mask images",
                    "Source": "Generated from the initial LivePortrait source set",
                    "Eval set": "6 character images: 1, 2, 4, 5, 7, tewo",
                }
            ),
            "training": render_kv_table(
                {
                    "Model": "FLUX.1-dev",
                    "Trainer": "Paired latent-direction trainer",
                    "Train steps": "500",
                    "Train method": "xattn",
                    "LR": "0.0005",
                    "Duration": "27m 39s for 500 steps",
                }
            ),
            "lora": render_kv_table(
                {
                    "LoRA rank": "16",
                    "LoRA alpha": "16",
                    "Eval scales": "-1, 0, +1",
                    "Mask-based weighting": "yes",
                    "Result goal": "Cleaner than quick FLUX, but weaker",
                }
            ),
            "analysis": render_list(
                [
                    "This run largely eliminated the full-frame drift seen in the quick FLUX baseline.",
                    "Identity preservation and eye-region locality improved substantially.",
                    "Visible gaze motion remained weak, especially in the positive direction.",
                ]
            ),
            "metrics": render_metric_table(pair15_metrics),
            "assets": [
                AssetEntry("Step 250", assets["pair15_000250"], "Paired 15 run at 250 steps."),
                AssetEntry("Step 500", assets["pair15_000500"], "Paired 15 run at 500 steps. Clean but too weak."),
            ],
            "individuals": make_individual_result_entries(ROOT / "flux1dev_pair_slider/evals/pair_step_000500_eye", "FLUX paired 15 step 500", "Paired 15 output, "),
        },
        {
            "tab_id": "flux_pair30",
            "title": "FLUX.1-dev Paired 30",
            "intro": "Paired FLUX.1-dev run scaled to 30 triplets built from LivePortrait left, neutral, and right output.",
            "dataset": render_kv_table(
                {
                    "Dataset": "30 paired left-right triplets plus eye masks",
                    "Source path": "outputs/liveportrait_only -> data/gaze_horizontal_texture_maskfix_30",
                    "Why this mattered": "This proved the FLUX LoRA really was trained on LivePortrait-generated gaze targets.",
                }
            ),
            "training": render_kv_table(
                {
                    "Model": "FLUX.1-dev",
                    "Train steps": "1000",
                    "Train method": "xattn",
                    "LR": "0.0005",
                    "Eta": "2.25",
                    "Duration": "55m 17s for 1000 steps",
                }
            ),
            "lora": render_kv_table(
                {
                    "LoRA rank": "16",
                    "LoRA alpha": "16",
                    "Eye region weight": "12.0",
                    "Non-eye weight": "0.15",
                    "Background preserve": "0.25",
                    "Eval scales": "-1, 0, +1",
                }
            ),
            "analysis": render_list(
                [
                    "The data source was cleaner than the 15-pair run and was directly traceable to LivePortrait triplets.",
                    "The run plateaued early. Checkpoints 250, 500, 750, and 1000 showed only marginal separation.",
                    "The positive direction improved slightly, but the result still did not become useful horizontal gaze control.",
                ]
            ),
            "metrics": render_metric_table(pair30_metrics),
            "assets": [
                AssetEntry("Step 250", assets["pair30_000250"], "30-pair FLUX run at 250 steps."),
                AssetEntry("Step 500", assets["pair30_000500"], "30-pair FLUX run at 500 steps."),
                AssetEntry("Step 750", assets["pair30_000750"], "30-pair FLUX run at 750 steps."),
                AssetEntry("Step 1000", assets["pair30_001000"], "30-pair FLUX run at 1000 steps."),
            ],
            "individuals": make_individual_result_entries(ROOT / "flux1dev_pair_slider/evals/pair30_step_001000_eye", "FLUX paired 30 step 1000", "Paired 30 output, "),
        },
        {
            "tab_id": "flux_pair30_eye",
            "title": "FLUX.1-dev Eye-Crop 30",
            "intro": "Paired FLUX.1-dev run with eye-crop preprocessing before VAE encoding and eye-region-only supervision.",
            "dataset": render_kv_table(
                {
                    "Dataset": "Same 30-pair LivePortrait-derived dataset as the previous tab",
                    "Change vs previous run": "Training used cropped eye regions before latent encoding",
                    "Output folder": "flux1dev_pair_slider/outputs_pair_30_eye_crop",
                }
            ),
            "training": render_kv_table(
                {
                    "Model": "FLUX.1-dev",
                    "Train steps": "1000",
                    "Train method": "xattn with eye-crop preprocessing",
                    "LR": "0.0005",
                    "Eta": "3.0",
                    "Duration": "Not captured in the local repo snapshot",
                }
            ),
            "lora": render_kv_table(
                {
                    "LoRA rank": "16",
                    "LoRA alpha": "16",
                    "Eye crop padding": "4.0",
                    "Eye crop min size": "192",
                    "Eye region weight": "24.0",
                    "Eval scales": "-1, 0, +1",
                }
            ),
            "analysis": render_list(
                [
                    "This is the cleanest FLUX trainer in the current sequence.",
                    "Locality improved and the positive direction became somewhat stronger.",
                    "The final output still did not become a convincing gaze slider. The visual change remains closer to localized repainting than explicit iris motion.",
                ]
            ),
            "metrics": render_metric_table(pair30_eye_metrics),
            "assets": [
                AssetEntry("Step 250", assets["pair30_eye_000250"], "Eye-crop FLUX run at 250 steps."),
                AssetEntry("Step 500", assets["pair30_eye_000500"], "Eye-crop FLUX run at 500 steps."),
                AssetEntry("Step 750", assets["pair30_eye_000750"], "Eye-crop FLUX run at 750 steps."),
                AssetEntry("Step 1000", assets["pair30_eye_001000"], "Eye-crop FLUX run at 1000 steps."),
            ],
            "individuals": make_individual_result_entries(ROOT / "flux1dev_pair_slider/evals/pair30_eye_crop_step_001000_eye", "FLUX eye-crop step 1000", "Eye-crop output, "),
        },
        {
            "tab_id": "liveportrait",
            "title": "LivePortrait Source",
            "intro": "Dataset provenance for the paired FLUX runs. LivePortrait generated the left, neutral, and right triplets used to assemble the 30-pair training set.",
            "analysis": render_list(
                [
                    "LivePortrait produced left, neutral, and right gaze images for each identity.",
                    "The triplets were stored in outputs/liveportrait_only.",
                    "The 30-pair FLUX datasets were assembled directly from those triplets.",
                    "This establishes a direct provenance chain from LivePortrait output to the paired FLUX training set.",
                ]
            ),
            "content": (
                "<div class='path-box'>"
                "<code>outputs/liveportrait_only/face_0000_left.png</code><br>"
                "<code>outputs/liveportrait_only/face_0000_neutral.png</code><br>"
                "<code>outputs/liveportrait_only/face_0000_right.png</code><br><br>"
                "<code>generate_gaze_dataset_v2.py --input_dir outputs/liveportrait_only --output_dir data/gaze_horizontal_texture_maskfix_30</code>"
                "</div>"
            ),
            "assets": [
                AssetEntry("Overview", assets["liveportrait_overview"], "LivePortrait left / neutral / right triplets that later became the 30-pair FLUX training set."),
            ],
        },
        {
            "tab_id": "dataset",
            "title": "Dataset",
            "intro": "Triplet dataset used to build the 30-pair paired FLUX training folders. Each row corresponds to one identity and each column corresponds to left, neutral, or right gaze.",
            "analysis": render_list(
                [
                    "Rows are labeled by face id.",
                    "These triplets were converted into neg, neutral, and pos folders for training.",
                    "The paired FLUX runs depended on this exact structure and file naming scheme.",
                ]
            ),
            "assets": [
                AssetEntry("Triplets 1 to 10", assets["dataset_0"], "First 10 identities."),
                AssetEntry("Triplets 11 to 20", assets["dataset_1"], "Middle 10 identities."),
                AssetEntry("Triplets 21 to 30", assets["dataset_2"], "Last 10 identities."),
            ],
        },
    ]


def build_tab_content(cache: dict[str, str], experiments: list[dict[str, object]]) -> tuple[str, str]:
    nav = []
    panels = []
    preview_cache: dict[Path, str] = {}
    for index, item in enumerate(experiments):
        active = " active" if index == 0 else ""
        nav.append(f"<button class='tab-btn{active}' data-tab='{item['tab_id']}'>{html.escape(item['title'])}</button>")

        asset_html = []
        for asset in item.get("assets", []):
            cache_key = str(asset.local_path.relative_to(ROOT))
            asset_url = cache.get(cache_key, asset.local_path.as_uri())
            preview_url = preview_cache.setdefault(asset.local_path, build_preview_data_url(asset.local_path, max_edge=1400))
            asset_html.append(
                "<figure class='asset-card'>"
                f"<a class='asset-link' href='{html.escape(asset_url)}' target='_blank' rel='noopener noreferrer' data-title='{html.escape(asset.title)}' data-caption='{html.escape(asset.caption)}' data-preview='{html.escape(preview_url)}'>"
                f"<img src='{html.escape(preview_url)}' alt='{html.escape(asset.title)}' loading='lazy'>"
                "</a>"
                f"<figcaption><strong>{html.escape(asset.title)}</strong><span>{html.escape(asset.caption)}</span></figcaption>"
                "</figure>"
            )

        individual_html = []
        for asset in item.get("individuals", []):
            cache_key = str(asset.local_path.relative_to(ROOT))
            asset_url = cache.get(cache_key, asset.local_path.as_uri())
            preview_url = preview_cache.setdefault(asset.local_path, build_preview_data_url(asset.local_path, max_edge=1000))
            individual_html.append(
                "<figure class='asset-card asset-card-small'>"
                f"<a class='asset-link' href='{html.escape(asset_url)}' target='_blank' rel='noopener noreferrer' data-title='{html.escape(asset.title)}' data-caption='{html.escape(asset.caption)}' data-preview='{html.escape(preview_url)}'>"
                f"<img src='{html.escape(preview_url)}' alt='{html.escape(asset.title)}' loading='lazy'>"
                "</a>"
                f"<figcaption><strong>{html.escape(asset.title)}</strong><span>{html.escape(asset.caption)}</span></figcaption>"
                "</figure>"
            )

        details_blocks = []
        if "dataset" in item:
            details_blocks.append(f"<section><h3>Dataset Used</h3>{item['dataset']}</section>")
        if "training" in item:
            details_blocks.append(f"<section><h3>Training Config</h3>{item['training']}</section>")
        if "lora" in item:
            details_blocks.append(f"<section><h3>LoRA Config</h3>{item['lora']}</section>")
        if "metrics" in item:
            details_blocks.append(f"<section><h3>Numbers</h3>{item['metrics']}</section>")
        if "analysis" in item:
            details_blocks.append(f"<section><h3>TL;DR</h3><ul>{item['analysis']}</ul></section>")
        if "content" in item:
            details_blocks.append(f"<section>{item['content']}</section>")

        panels.append(
            f"<section class='tab-panel{active}' id='tab-{item['tab_id']}'>"
            f"<header class='panel-header'><h2>{html.escape(item['title'])}</h2><p>{html.escape(item['intro'])}</p></header>"
            f"{''.join(details_blocks)}"
            f"<section><h3>Visuals</h3><div class='asset-grid'>{''.join(asset_html)}</div></section>"
            + (
                f"<section><h3>Individual Results</h3><div class='individual-grid'>{''.join(individual_html)}</div></section>"
                if individual_html
                else ""
            )
            +
            "</section>"
        )

    return "\n".join(nav), "\n".join(panels)


def build_html(cache: dict[str, str], experiments: list[dict[str, object]]) -> str:
    nav_html, panel_html = build_tab_content(cache, experiments)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Eye Gaze LoRA Experiments</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f2efe8;
      color: #181614;
    }}
    header.page-header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid #d8d1c4;
      background: #fbf9f4;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .page-header h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }}
    .page-header p {{
      margin: 10px 0 0;
      max-width: 1100px;
      font-size: 16px;
      line-height: 1.5;
      color: #534c44;
    }}
    nav.tabs {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 16px 32px 8px;
      background: #fbf9f4;
      border-bottom: 1px solid #d8d1c4;
    }}
    .tab-btn {{
      border: 1px solid #cfc6b8;
      background: #efe7d7;
      color: #2c2823;
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 14px;
      white-space: nowrap;
    }}
    .tab-btn.active {{
      background: #2c2823;
      color: #fbf9f4;
      border-color: #2c2823;
    }}
    main {{
      padding: 24px 32px 48px;
    }}
    .tab-panel {{
      display: none;
      max-width: 1320px;
      margin: 0 auto;
    }}
    .tab-panel.active {{
      display: block;
    }}
    .panel-header {{
      margin-bottom: 20px;
    }}
    .panel-header h2 {{
      margin: 0 0 10px;
      font-size: 26px;
    }}
    .panel-header p {{
      margin: 0;
      font-size: 17px;
      line-height: 1.6;
      color: #534c44;
      max-width: 1100px;
    }}
    section {{
      margin-bottom: 28px;
      padding: 20px 22px;
      background: #fbf9f4;
      border: 1px solid #d8d1c4;
      border-radius: 18px;
    }}
    section h3 {{
      margin: 0 0 12px;
      font-size: 19px;
    }}
    .kv-table,
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: #fffdf8;
    }}
    .kv-table th,
    .kv-table td,
    .metric-table th,
    .metric-table td {{
      border: 1px solid #ddd5c8;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    .kv-table th,
    .metric-table th {{
      width: 28%;
      background: #f4ecde;
      font-weight: 600;
    }}
    .metric-table th {{
      width: auto;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.7;
    }}
    .flow-list {{
      margin: 0 0 14px;
      padding-left: 20px;
      line-height: 1.7;
    }}
    .asset-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
    }}
    .individual-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .asset-card {{
      margin: 0;
      border: 1px solid #ddd5c8;
      border-radius: 14px;
      overflow: hidden;
      background: #fffdf8;
    }}
    .asset-link {{
      display: block;
      cursor: zoom-in;
    }}
    .asset-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #ece7dc;
    }}
    .asset-card figcaption {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 12px 14px 14px;
      font-size: 14px;
      line-height: 1.5;
      color: #534c44;
    }}
    .asset-card figcaption strong {{
      color: #181614;
      font-size: 15px;
    }}
    .asset-card-small figcaption {{
      padding: 10px 12px 12px;
      font-size: 13px;
    }}
    .overview-grid {{
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .callout {{
      background: #2c2823;
      color: #fbf9f4;
      border-radius: 16px;
      padding: 18px;
    }}
    .callout h3 {{
      margin-top: 0;
    }}
    .callout p {{
      margin-bottom: 0;
      line-height: 1.6;
    }}
    .path-box {{
      padding: 16px;
      border-radius: 12px;
      background: #fffdf8;
      border: 1px solid #ddd5c8;
      font-family: Menlo, Monaco, monospace;
      font-size: 13px;
      line-height: 1.8;
      overflow-x: auto;
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(15, 12, 10, 0.88);
      z-index: 1000;
    }}
    .lightbox.open {{
      display: flex;
    }}
    .lightbox-inner {{
      max-width: min(92vw, 1600px);
      max-height: 92vh;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .lightbox img {{
      display: block;
      max-width: 100%;
      max-height: calc(92vh - 96px);
      width: auto;
      height: auto;
      border-radius: 12px;
      background: #111;
    }}
    .lightbox-meta {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      color: #fbf9f4;
    }}
    .lightbox-meta strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 16px;
    }}
    .lightbox-meta span {{
      font-size: 14px;
      line-height: 1.5;
      color: #d6d0c7;
    }}
    .lightbox-close {{
      border: 1px solid #8b8376;
      background: transparent;
      color: #fbf9f4;
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 13px;
      white-space: nowrap;
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 20px 16px 40px;
      }}
      nav.tabs {{
        padding: 12px 16px 8px;
      }}
      header.page-header {{
        padding: 22px 16px 18px;
      }}
      .overview-grid {{
        grid-template-columns: 1fr;
      }}
      .asset-grid {{
        grid-template-columns: 1fr;
      }}
      .individual-grid {{
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <header class="page-header">
    <h1>Eye Gaze LoRA Experiments</h1>
    <p>Report scope: training configuration, dataset provenance, runtime, pipeline structure, and output behavior across the main experiment branches.</p>
  </header>
  <nav class="tabs">{nav_html}</nav>
  <main>{panel_html}</main>
  <div class="lightbox" id="lightbox" aria-hidden="true">
    <div class="lightbox-inner">
      <img id="lightbox-image" src="" alt="">
      <div class="lightbox-meta">
        <div>
          <strong id="lightbox-title"></strong>
          <span id="lightbox-caption"></span>
        </div>
        <button class="lightbox-close" id="lightbox-close" type="button">Close</button>
      </div>
    </div>
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll('.tab-btn'));
    const panels = Array.from(document.querySelectorAll('.tab-panel'));
    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        const tabId = button.dataset.tab;
        for (const other of buttons) other.classList.remove('active');
        for (const panel of panels) panel.classList.remove('active');
        button.classList.add('active');
        document.getElementById('tab-' + tabId).classList.add('active');
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }});
    }}

    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxTitle = document.getElementById('lightbox-title');
    const lightboxCaption = document.getElementById('lightbox-caption');
    const lightboxClose = document.getElementById('lightbox-close');
    const assetLinks = Array.from(document.querySelectorAll('.asset-link'));

    function closeLightbox() {{
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      lightboxImage.src = '';
      lightboxImage.alt = '';
      lightboxTitle.textContent = '';
      lightboxCaption.textContent = '';
    }}

    for (const link of assetLinks) {{
      link.addEventListener('click', (event) => {{
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {{
          return;
        }}
        event.preventDefault();
        lightboxImage.src = link.dataset.preview || link.href;
        lightboxImage.alt = link.dataset.title || '';
        lightboxTitle.textContent = link.dataset.title || '';
        lightboxCaption.textContent = link.dataset.caption || '';
        lightbox.classList.add('open');
        lightbox.setAttribute('aria-hidden', 'false');
      }});
    }}

    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (event) => {{
      if (event.target === lightbox) {{
        closeLightbox();
      }}
    }});
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape' && lightbox.classList.contains('open')) {{
        closeLightbox();
      }}
    }});
  </script>
</body>
</html>
"""
    HTML_PATH.write_text(html_doc)
    return html_doc


def render_klein_simple_page(cache: dict[str, str], experiments: list[dict[str, object]]) -> str:
    cards: list[str] = []
    for item in experiments:
        input_path = item.get("input_path")
        output_path = item.get("preview_path")

        input_block = "<div class='missing'>No saved test input artifact</div>"
        if input_path is not None and Path(input_path).exists():
            input_cache_key = str(Path(input_path).relative_to(ROOT))
            input_href = cache.get(input_cache_key, Path(input_path).as_uri())
            input_block = (
                f"<a class='image-link' href='{html.escape(input_href)}' target='_blank' rel='noopener noreferrer' "
                f"title='{html.escape(str(item['name']) + ' - test input')}'>"
                f"<img src='{html.escape(input_href)}' alt='{html.escape(str(item['name']) + ' - test input')}' loading='lazy'>"
                "</a>"
            )

        output_block = "<div class='missing'>No saved test output artifact</div>"
        if output_path is not None and Path(output_path).exists():
            output_cache_key = str(Path(output_path).relative_to(ROOT))
            output_href = cache.get(output_cache_key, Path(output_path).as_uri())
            output_block = (
                f"<a class='image-link' href='{html.escape(output_href)}' target='_blank' rel='noopener noreferrer' "
                f"title='{html.escape(str(item['name']) + ' - test output')}'>"
                f"<img src='{html.escape(output_href)}' alt='{html.escape(str(item['name']) + ' - test output')}' loading='lazy'>"
                "</a>"
            )

        params_table = render_kv_table(
            {
                "Config": str(item["config_path"]),
                "Dataset": str(item["dataset"]),
                "Output Dir": str(item["output_dir"]),
                "Resolution": str(item["resolution"]),
                "Steps": str(item["steps"]),
                "Rank / Alpha": f"{item['rank']} / {item['alpha']}",
                "LR / Eta": f"{item['lr']} / {item['eta']}",
                "Train Method": str(item["train_method"]),
            }
        )

        cards.append(
            "<section class='experiment-card'>"
            f"<h2>{html.escape(str(item['name']))}</h2>"
            f"<p class='status'>{html.escape(str(item['status']))}</p>"
            f"{params_table}"
            "<div class='image-row'>"
            "<div><h3>Test Input</h3>"
            f"{input_block}"
            f"<p class='file-note'>{html.escape(str(item['input_file']))}</p>"
            "</div>"
            "<div><h3>Test Output</h3>"
            f"{output_block}"
            f"<p class='file-note'>{html.escape(str(item['preview_file']))}</p>"
            "</div>"
            "</div>"
            "</section>"
        )

    if not cards:
        cards.append(
            "<section class='experiment-card'>"
            "<h2>No Klein experiment artifacts found</h2>"
            "<p class='status'>No pre-FLUX.1-dev Klein 9B experiment artifacts were found in the current repo snapshot.</p>"
            "</section>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FLUX Klein Experiments</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f4f1ea;
      color: #171512;
    }}
    header {{
      padding: 28px 32px 20px;
      background: #fbf9f4;
      border-bottom: 1px solid #d8d1c4;
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    header p {{
      margin: 0;
      max-width: 1000px;
      line-height: 1.6;
      color: #534c44;
    }}
    main {{
      padding: 24px 32px 48px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .experiment-card {{
      margin-bottom: 24px;
      padding: 20px 22px;
      background: #fbf9f4;
      border: 1px solid #d8d1c4;
      border-radius: 16px;
    }}
    .experiment-card h2 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .status {{
      margin: 0 0 16px;
      color: #5f584f;
      font-size: 14px;
    }}
    .kv-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 18px;
      background: #fffdf8;
    }}
    .kv-table th,
    .kv-table td {{
      border: 1px solid #ddd5c8;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    .kv-table th {{
      width: 24%;
      background: #f4ecde;
      font-weight: 600;
    }}
    .image-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .image-row h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .image-link {{
      display: block;
      cursor: zoom-in;
      border: 1px solid #ddd5c8;
      border-radius: 12px;
      overflow: hidden;
      background: #fffdf8;
    }}
    .image-link img {{
      display: block;
      width: 100%;
      height: auto;
      background: #ece7dc;
    }}
    .missing {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 240px;
      border: 1px dashed #cdbfaa;
      border-radius: 12px;
      background: #fffdf8;
      color: #6d655b;
      font-size: 14px;
    }}
    .file-note {{
      margin: 8px 0 0;
      font-size: 13px;
      color: #5f584f;
    }}
    @media (max-width: 900px) {{
      header {{
        padding: 22px 16px 18px;
      }}
      main {{
        padding: 20px 16px 40px;
      }}
      .image-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>FLUX Klein 9B Experiments Before FLUX.1-dev</h1>
    <p>This page lists the Klein 9B experiments that were run before moving to FLUX.1-dev. Each block shows the dataset, the main training parameters, and the saved testing input and output artifacts when those files exist in the repo snapshot.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    KLEIN_HTML_PATH.write_text(html_doc)
    return html_doc


def build_recent_preview_list(directory: Path, limit: int = 4) -> list[Path]:
    preferred_files = [
        "result.png",
        "infer_fix_v2.png",
        "infer_000400_fix.png",
        "infer_000400.png",
        "strip.png",
        "scale_+0.0.png",
        "input.png",
    ]
    previews: list[Path] = []

    for name in preferred_files:
        candidate = directory / name
        if candidate.exists():
            previews.append(candidate)
    if previews:
        return previews[:limit]

    for sample_id in SAMPLE_ORDER:
        candidate = directory / sample_id / "result.png"
        if candidate.exists():
            previews.append(candidate)
        if len(previews) >= limit:
            return previews[:limit]

    images = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    return images[:limit]


def discover_recent_training_runs(days: int) -> list[dict[str, object]]:
    cutoff = dt.datetime.now() - dt.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".safetensors", ".yaml"}
    klein_lookup: dict[str, dict[str, object]] = {}
    for item in discover_klein_experiments():
        output_path = resolve_output_path(str(item.get("output_dir", "-")))
        if output_path is not None:
            klein_lookup[str(output_path.resolve())] = item

    runs: list[dict[str, object]] = []

    for output_dir in sorted(OUTPUT_DIR.iterdir()):
        if not output_dir.is_dir() or output_dir.name == "experiment_report":
            continue
        recent_files = [
            path for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_mtime >= cutoff_ts
        ]
        if not recent_files:
            continue
        latest_path = max(recent_files, key=lambda path: path.stat().st_mtime)
        metadata = klein_lookup.get(str(output_dir.resolve()), {})
        runs.append(
            {
                "name": str(metadata.get("name", output_dir.name)),
                "category": "Klein training/output",
                "path": output_dir,
                "last_modified": dt.datetime.fromtimestamp(latest_path.stat().st_mtime),
                "dataset": str(metadata.get("dataset", "-")),
                "config_path": str(metadata.get("config_path", "-")),
                "params": " | ".join(
                    [
                        f"steps={metadata.get('steps', '-')}",
                        f"rank={metadata.get('rank', '-')}",
                        f"alpha={metadata.get('alpha', '-')}",
                        f"lr={metadata.get('lr', '-')}",
                        f"eta={metadata.get('eta', '-')}",
                        f"train_method={metadata.get('train_method', '-')}",
                    ]
                ),
                "summary": f"{len(recent_files)} recent files",
                "preview_paths": build_recent_preview_list(output_dir),
            }
        )

    flux_output_root = ROOT / "flux1dev_pair_slider"
    for run_dir in sorted(flux_output_root.glob("outputs*/*")):
        if not run_dir.is_dir():
            continue
        recent_files = [
            path for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_mtime >= cutoff_ts
        ]
        if not recent_files:
            continue
        latest_path = max(recent_files, key=lambda path: path.stat().st_mtime)
        config_values = parse_top_level_config(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else {}
        weight_files = sorted((run_dir / "weights").glob("*.safetensors")) if (run_dir / "weights").exists() else []
        latest_weight = weight_files[-1].name if weight_files else "-"
        runs.append(
            {
                "name": config_values.get("slider_name", run_dir.name),
                "category": "FLUX training",
                "path": run_dir,
                "last_modified": dt.datetime.fromtimestamp(latest_path.stat().st_mtime),
                "dataset": config_values.get("data_root", config_values.get("metadata_path", "-")),
                "config_path": str((run_dir / "config.yaml").relative_to(ROOT)) if (run_dir / "config.yaml").exists() else "-",
                "params": " | ".join(
                    [
                        f"steps={config_values.get('max_train_steps', '-')}",
                        f"rank={config_values.get('rank', '-')}",
                        f"alpha={config_values.get('alpha', '-')}",
                        f"lr={config_values.get('lr', '-')}",
                        f"train_method={config_values.get('train_method', '-')}",
                    ]
                ),
                "summary": f"{len(weight_files)} checkpoints, latest={latest_weight}",
                "preview_paths": [],
            }
        )

    return sorted(runs, key=lambda item: item["last_modified"], reverse=True)


def discover_recent_eval_runs(days: int) -> list[dict[str, object]]:
    cutoff = dt.datetime.now() - dt.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    eval_root = ROOT / "flux1dev_pair_slider" / "evals"
    if not eval_root.exists():
        return []

    runs: list[dict[str, object]] = []
    for eval_dir in sorted(eval_root.iterdir()):
        if not eval_dir.is_dir():
            continue
        recent_images = [
            path for path in eval_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and path.stat().st_mtime >= cutoff_ts
        ]
        if not recent_images:
            continue
        latest_path = max(recent_images, key=lambda path: path.stat().st_mtime)
        sample_dirs = [
            path for path in sorted(eval_dir.iterdir())
            if path.is_dir() and path.name != "orthogonal_basis"
        ]
        result_count = sum(1 for _ in eval_dir.rglob("result.png"))
        scale_count = sum(1 for _ in eval_dir.rglob("scale_*.png"))
        runs.append(
            {
                "name": eval_dir.name,
                "category": "FLUX eval",
                "path": eval_dir,
                "last_modified": dt.datetime.fromtimestamp(latest_path.stat().st_mtime),
                "dataset": "eval inputs from characters/ or source image set",
                "config_path": "-",
                "params": "-",
                "summary": f"samples={len(sample_dirs)} | result.png={result_count} | scale_images={scale_count}",
                "preview_paths": build_recent_preview_list(eval_dir),
            }
        )

    return sorted(runs, key=lambda item: item["last_modified"], reverse=True)


def render_recent_runs_page(days: int) -> str:
    now = dt.datetime.now()
    cutoff = now - dt.timedelta(days=days)
    training_runs = discover_recent_training_runs(days)
    eval_runs = discover_recent_eval_runs(days)

    def render_cards(items: list[dict[str, object]]) -> str:
        cards: list[str] = []
        for item in items:
            previews = []
            for preview_path in item["preview_paths"]:
                href = html.escape(relative_report_path(preview_path))
                label = html.escape(preview_path.name)
                previews.append(
                    f"<a class='image-link' href='{href}' target='_blank' rel='noopener noreferrer'>"
                    f"<img src='{href}' alt='{label}' loading='lazy'></a>"
                )
            preview_html = "".join(previews) if previews else "<div class='missing'>No preview images for this run</div>"
            cards.append(
                "<section class='experiment-card'>"
                f"<h2>{html.escape(str(item['name']))}</h2>"
                f"<p class='status'>{html.escape(str(item['category']))} | last modified {html.escape(str(item['last_modified']))}</p>"
                "<table class='kv-table'>"
                f"<tr><th>Path</th><td>{html.escape(str(item['path'].relative_to(ROOT)))}</td></tr>"
                f"<tr><th>Dataset</th><td>{html.escape(str(item['dataset']))}</td></tr>"
                f"<tr><th>Config</th><td>{html.escape(str(item['config_path']))}</td></tr>"
                f"<tr><th>Params</th><td>{html.escape(str(item['params']))}</td></tr>"
                f"<tr><th>Summary</th><td>{html.escape(str(item['summary']))}</td></tr>"
                "</table>"
                f"<div class='preview-grid'>{preview_html}</div>"
                "</section>"
            )
        return "".join(cards) if cards else "<p>No runs matched the selected date range.</p>"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Recent Runs Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: #f4f1ea; color: #171512; }}
    header {{ padding: 28px 32px 20px; background: #fbf9f4; border-bottom: 1px solid #d8d1c4; }}
    header h1 {{ margin: 0 0 8px; font-size: 30px; }}
    header p {{ margin: 0; max-width: 1000px; line-height: 1.5; color: #534c44; }}
    main {{ padding: 24px 32px 48px; max-width: 1480px; margin: 0 auto; }}
    h2.section-title {{ margin: 28px 0 14px; font-size: 24px; }}
    .experiment-card {{ margin-bottom: 22px; padding: 18px 20px; background: #fbf9f4; border: 1px solid #d8d1c4; border-radius: 14px; }}
    .experiment-card h2 {{ margin: 0 0 8px; font-size: 22px; }}
    .status {{ margin: 0 0 14px; color: #5f584f; font-size: 14px; }}
    .kv-table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; background: #fffdf8; }}
    .kv-table th, .kv-table td {{ border: 1px solid #ddd5c8; padding: 10px 12px; text-align: left; vertical-align: top; font-size: 14px; }}
    .kv-table th {{ width: 20%; background: #f4ecde; font-weight: 600; }}
    .preview-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .image-link {{ display: block; border: 1px solid #ddd5c8; border-radius: 10px; overflow: hidden; background: #fffdf8; }}
    .image-link img {{ display: block; width: 100%; height: auto; background: #ece7dc; }}
    .missing {{ display: flex; align-items: center; justify-content: center; min-height: 180px; border: 1px dashed #cdbfaa; border-radius: 10px; background: #fffdf8; color: #6d655b; font-size: 14px; }}
    @media (max-width: 900px) {{
      header {{ padding: 22px 16px 18px; }}
      main {{ padding: 20px 16px 40px; }}
      .preview-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Runs From Last {days} Days</h1>
    <p>Generated from local file modification times. This report currently covers {html.escape(cutoff.strftime("%B %d, %Y %H:%M"))} to {html.escape(now.strftime("%B %d, %Y %H:%M"))}.</p>
  </header>
  <main>
    <h2 class="section-title">Training Runs</h2>
    {render_cards(training_runs)}
    <h2 class="section-title">Evaluation Runs</h2>
    {render_cards(eval_runs)}
  </main>
</body>
</html>
"""
    RECENT_HTML_PATH.write_text(html_doc)
    return html_doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-days", type=int, default=3)
    parser.add_argument("--recent-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    render_recent_runs_page(args.recent_days)
    print(f"Recent report written to {RECENT_HTML_PATH}")

    if args.recent_only:
        return

    assets = make_assets()
    experiments = build_experiments(assets)
    klein_experiments = select_pre_flux_klein_experiments(discover_klein_experiments())
    cache = read_cache()
    upload_paths = {asset_path for asset_path in assets.values()}
    for experiment in experiments:
        for group_name in ("assets", "individuals"):
            for asset in experiment.get(group_name, []):
                upload_paths.add(asset.local_path)
    for item in klein_experiments:
        if item.get("input_path") is not None:
            upload_paths.add(Path(item["input_path"]))
        if item.get("preview_path") is not None:
            upload_paths.add(Path(item["preview_path"]))
    for asset_path in sorted(upload_paths, key=str):
        upload_asset(asset_path, prefix="gaze-report", cache=cache)
    build_html(cache, experiments)
    render_klein_simple_page(cache, klein_experiments)
    print(f"Report written to {HTML_PATH}")
    print(f"Klein report written to {KLEIN_HTML_PATH}")


if __name__ == "__main__":
    main()
