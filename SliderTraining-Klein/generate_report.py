#!/usr/bin/env python3
"""
Upload all experiment result images to Azure CDN and generate a single HTML report.
Run from SliderTraining-Klein directory with env vars set.
"""
import io
import os
import sys
import json
import uuid
import mimetypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.storage_utils import Storage

CDN_BASE = os.environ.get("CDN_BASE_URL", "https://dev-content.dashtoon.ai")
BUCKET = "frameo-tools/general_uploads"
CACHE_FILE = "outputs/cdn_urls.json"


def upload_image(storage, local_path, name):
    with open(local_path, "rb") as f:
        data = io.BytesIO(f.read())
    uid = uuid.uuid4().hex[:8]
    blob_name = f"gaze-report/{uid}_{name}"
    urls = storage.bulk_file_upload(
        files=[data],
        bucket=BUCKET,
        format="png",
        content_type="image",
        fileNames=[blob_name],
    )
    if urls:
        return urls[0]
    return f"{CDN_BASE}/{BUCKET}/{blob_name}"


def upload_all():
    storage = Storage()
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    experiments = [
        # (tab_id, tab_title, description, result_status, image_paths)
        (
            "direct_lora",
            "Direct LoRA on Klein",
            "LoRA trained directly on Klein 9B transformer using paired gaze images (neg=left, pos=right). Reference conditioning in Klein's architecture dominates generation and prevents LoRA from controlling gaze direction.",
            "No gaze change across any slider scale. Identity preserved but slider has zero effect.",
            [
                ("v7b", "outputs/eye_gaze_v7b/result.png"),
                ("v7c", "outputs/eye_gaze_v7c/result.png"),
                ("v8", "outputs/eye_gaze_v8/result.png"),
            ],
        ),
        (
            "text_slider",
            "Text-based Slider",
            "LoRA trained using text prompts as directional signal: positive='a person looking to the right', negative='a person looking to the left'. Same Klein architecture with reference conditioning.",
            "No gaze change. At extreme scales image degrades to noise. Reference conditioning still dominates.",
            [
                ("text_v1", "outputs/eye_gaze_text_v1/result.png"),
                ("text_v2", "outputs/eye_gaze_text_v2/result.png"),
                ("text_v3", "outputs/eye_gaze_text_v3/result.png"),
            ],
        ),
        (
            "output_steer",
            "Output Delta Steering",
            "Precomputed gaze delta tensors applied at inference time by steering transformer output activations. No LoRA training required - delta is computed from left/right gaze image pairs.",
            "Partial gaze shift visible. Some identity drift at higher scales.",
            [
                ("output_steer", "outputs/output_steer/result.png"),
            ],
        ),
        (
            "pipeline",
            "LivePortrait + Klein Pipeline",
            "LivePortrait warps the eye region using eyeball_direction_x parameter (+/-15 degrees). Klein then runs img2img at strength=0.4 to refine image quality while preserving the warped gaze. Bypasses LoRA entirely.",
            "Clear gaze direction change. Identity well preserved. Works on any face without training.",
            [
                ("single_face", "outputs/pipeline_gaze/result.png"),
                ("face_0000", "outputs/pipeline_gaze_batch/face_0000/result.png"),
                ("face_0010", "outputs/pipeline_gaze_batch/face_0010/result.png"),
                ("face_0020", "outputs/pipeline_gaze_batch/face_0020/result.png"),
                ("face_0030", "outputs/pipeline_gaze_batch/face_0030/result.png"),
                ("face_0040", "outputs/pipeline_gaze_batch/face_0040/result.png"),
                ("face_0050", "outputs/pipeline_gaze_batch/face_0050/result.png"),
                ("face_0060", "outputs/pipeline_gaze_batch/face_0060/result.png"),
                ("face_0070", "outputs/pipeline_gaze_batch/face_0070/result.png"),
                ("face_0080", "outputs/pipeline_gaze_batch/face_0080/result.png"),
                ("face_0090", "outputs/pipeline_gaze_batch/face_0090/result.png"),
            ],
        ),
        (
            "aitoolkit_lora",
            "ai-toolkit ImageReferenceSliderTrainer",
            "Klein 9B LoRA trained using ai-toolkit's ImageReferenceSliderTrainer (no reference conditioning during training). 2000 steps, rank 32, 1001 paired gaze images. Two inference modes tested.",
            "With reference conditioning: no gaze change (ref cond locks output). Without reference conditioning: identity lost, model generates random people.",
            [
                ("with_ref_cond", "outputs/lora_gaze_test/result.png"),
                ("no_ref_cond", "outputs/lora_gaze_test_v2/result.png"),
            ],
        ),
    ]

    print("Uploading images...")
    for tab_id, _, _, _, images in experiments:
        for img_id, img_path in images:
            key = f"{tab_id}/{img_id}"
            if key in cache:
                print(f"  cached: {key}")
                continue
            if not os.path.exists(img_path):
                print(f"  missing: {img_path}")
                continue
            name = f"{tab_id}_{img_id}.png"
            url = upload_image(storage, img_path, name)
            cache[key] = url
            print(f"  uploaded: {key} -> {url}")

    os.makedirs("outputs", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    return experiments, cache


def build_html(experiments, cache):
    tabs_nav = ""
    tabs_content = ""

    for i, (tab_id, tab_title, description, status, images) in enumerate(experiments):
        active = "active" if i == 0 else ""
        tabs_nav += f'<button class="tab-btn {active}" onclick="showTab(\'{tab_id}\')" id="btn-{tab_id}">{tab_title}</button>\n'

        imgs_html = ""
        for img_id, _ in images:
            key = f"{tab_id}/{img_id}"
            url = cache.get(key, "")
            if not url:
                continue
            label = img_id.replace("_", " ")
            imgs_html += f'''
            <div class="img-card">
                <img src="{url}" alt="{label}" loading="lazy">
                <p class="img-label">{label}</p>
            </div>'''

        tabs_content += f'''
        <div class="tab-panel {'active' if i == 0 else ''}" id="tab-{tab_id}">
            <div class="experiment-meta">
                <p class="description">{description}</p>
                <p class="status"><strong>Result:</strong> {status}</p>
            </div>
            <div class="images-grid">
                {imgs_html}
            </div>
        </div>'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eye Gaze Slider - Experiment Results</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f0f0f; color: #e0e0e0; }}
  header {{ padding: 28px 40px 20px; border-bottom: 1px solid #222; }}
  header h1 {{ font-size: 20px; font-weight: 600; letter-spacing: 0.3px; color: #fff; }}
  header p {{ font-size: 13px; color: #888; margin-top: 4px; }}
  .tabs-nav {{ display: flex; gap: 0; border-bottom: 1px solid #222; padding: 0 40px; overflow-x: auto; }}
  .tab-btn {{
    background: none; border: none; border-bottom: 2px solid transparent;
    color: #888; cursor: pointer; font-size: 13px; padding: 14px 18px;
    white-space: nowrap; transition: all 0.15s; font-family: inherit;
  }}
  .tab-btn:hover {{ color: #ccc; }}
  .tab-btn.active {{ color: #fff; border-bottom-color: #4a9eff; }}
  .tab-panel {{ display: none; padding: 28px 40px 40px; }}
  .tab-panel.active {{ display: block; }}
  .experiment-meta {{ max-width: 800px; margin-bottom: 24px; }}
  .description {{ font-size: 13px; color: #aaa; line-height: 1.6; margin-bottom: 10px; }}
  .status {{ font-size: 13px; color: #aaa; line-height: 1.6; }}
  .status strong {{ color: #e0e0e0; }}
  .images-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .img-card {{ background: #1a1a1a; border-radius: 6px; overflow: hidden; border: 1px solid #2a2a2a; }}
  .img-card img {{ display: block; max-width: 100%; height: auto; max-height: 220px; object-fit: contain; background: #111; }}
  .img-label {{ font-size: 11px; color: #666; padding: 6px 10px; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>Eye Gaze Slider - Experiment Results</h1>
  <p>FLUX.2-Klein 9B gaze direction control experiments</p>
</header>
<nav class="tabs-nav">
{tabs_nav}
</nav>
<main>
{tabs_content}
</main>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.getElementById('btn-' + id).classList.add('active');
}}
</script>
</body>
</html>"""

    out = "outputs/experiment_report.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"\nReport saved to {out}")
    return out


if __name__ == "__main__":
    experiments, cache = upload_all()
    build_html(experiments, cache)
