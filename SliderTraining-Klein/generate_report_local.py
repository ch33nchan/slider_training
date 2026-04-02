#!/usr/bin/env python3
"""
Generate a standalone HTML report with base64-embedded images.
No CDN upload needed — runs fully locally.
"""
import base64
from pathlib import Path


def img_to_data_uri(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def config_table(rows):
    """Render a list of (key, value) pairs as an HTML table."""
    trs = "".join(f"<tr><td class='cfg-key'>{k}</td><td class='cfg-val'>{v}</td></tr>" for k, v in rows)
    return f"<table class='cfg-table'>{trs}</table>"


# Each experiment entry:
# (tab_id, tab_title, description, result_status, per_image_configs, images)
#
# per_image_configs: dict of img_id -> list of (key, value) config rows
# If an img_id is not in the dict, fall back to key "batch" if present.

experiments = [
    # -------------------------------------------------------------------------
    # 1. Direct LoRA on Klein
    # -------------------------------------------------------------------------
    (
        "direct_lora",
        "Direct LoRA on Klein",
        "LoRA trained directly on Klein 9B transformer using paired gaze images (neg=left, pos=right). "
        "Reference conditioning in Klein's architecture dominates generation and prevents LoRA from controlling gaze direction.",
        "No gaze change across any slider scale. Identity preserved but slider has zero effect.",
        {
            "v7b": [
                ("Dataset",        "Columbia Gaze Dataset — neg / pos / neutral image splits"),
                ("Prompts",        "Target: \"a person\""),
                ("Train method",   "xattn (cross-attention layers only)"),
                ("Rank / Alpha",   "64 / 1"),
                ("LR",             "0.0005"),
                ("Eta",            "8"),
                ("Steps",          "8 000"),
                ("Resolution",     "512 x 512"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps", "28"),
                ("Strength",       "0.6 (img2img)"),
                ("Seed",           "42"),
            ],
            "v7c": [
                ("Dataset",        "Columbia Gaze Dataset — neg / pos / neutral image splits"),
                ("Prompts",        "Target: \"a person\""),
                ("Train method",   "xattn"),
                ("Rank / Alpha",   "64 / 1"),
                ("LR",             "0.001"),
                ("Eta",            "10"),
                ("Steps",          "8 000"),
                ("Resolution",     "512 x 512"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps", "28"),
                ("Strength",       "0.6 (img2img)"),
                ("Seed",           "42"),
            ],
            "v8": [
                ("Dataset",        "gaze_s15 — 15-pair synthetic gaze set, neg / pos / neutral splits"),
                ("Prompts",        "Target: \"a person\""),
                ("Train method",   "xattn"),
                ("Rank / Alpha",   "64 / 1"),
                ("LR",             "0.001"),
                ("Eta",            "8"),
                ("Steps",          "4 000"),
                ("Resolution",     "512 x 512"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps", "28"),
                ("Strength",       "0.6 (img2img)"),
                ("Seed",           "42"),
            ],
        },
        [
            ("v7b", "outputs/eye_gaze_v7b/result.png"),
            ("v7c", "outputs/eye_gaze_v7c/result.png"),
            ("v8",  "outputs/eye_gaze_v8/result.png"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 2. Text-based Slider
    # -------------------------------------------------------------------------
    (
        "text_slider",
        "Text-based Slider",
        "LoRA trained using text prompts as the directional signal — no image pairs. "
        "The slider direction is defined by the difference between positive and negative prompt embeddings. "
        "Same Klein architecture with reference conditioning at inference.",
        "No gaze change. At extreme scales image degrades to noise. Reference conditioning still dominates.",
        {
            "text_v1": [
                ("Dataset",          "Columbia Gaze Dataset — neutral images only (no pos/neg image pairs)"),
                ("Target prompt",    "\"a person\""),
                ("Positive prompt",  "\"a person looking to the right\""),
                ("Negative prompt",  "\"a person looking to the left\""),
                ("Train method",     "xattn"),
                ("Rank / Alpha",     "32 / 1"),
                ("LR",               "0.001"),
                ("Eta",              "6"),
                ("Steps",            "2 000"),
                ("Resolution",       "512 x 512"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps",  "28"),
                ("Strength",         "0.6 (img2img)"),
                ("Seed",             "42"),
            ],
            "text_v2": [
                ("Dataset",          "Columbia Gaze Dataset — neutral images only"),
                ("Target prompt",    "\"a person\""),
                ("Positive prompt",  "\"a person looking to the right\""),
                ("Negative prompt",  "\"a person looking to the left\""),
                ("Train method",     "xattn"),
                ("Rank / Alpha",     "64 / 1"),
                ("LR",               "0.001"),
                ("Eta",              "10"),
                ("Steps",            "3 000"),
                ("Resolution",       "512 x 512"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps",  "28"),
                ("Strength",         "0.6 (img2img)"),
                ("Seed",             "42"),
            ],
            "text_v3": [
                ("Dataset",          "Columbia Gaze Dataset — neutral images only"),
                ("Target prompt",    "\"portrait of a person\""),
                ("Positive prompt",  "\"portrait of a person with eyes looking to the right\""),
                ("Negative prompt",  "\"portrait of a person with eyes looking to the left\""),
                ("Train method",     "xattn"),
                ("Rank / Alpha",     "32 / 1"),
                ("LR",               "0.0005"),
                ("Eta",              "8"),
                ("Steps",            "3 000"),
                ("Resolution",       "512 x 512"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps",  "28"),
                ("Strength",         "0.6 (img2img)"),
                ("Seed",             "42"),
            ],
        },
        [
            ("text_v1", "outputs/eye_gaze_text_v1/result.png"),
            ("text_v2", "outputs/eye_gaze_text_v2/result.png"),
            ("text_v3", "outputs/eye_gaze_text_v3/result.png"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 3. Output Delta Steering
    # -------------------------------------------------------------------------
    (
        "output_steer",
        "Output Delta Steering",
        "No LoRA training. For each image pair, two frozen forward passes (neg ref, pos ref) are run with the same "
        "noisy latent. The mean difference — mean(pos_pred - neg_pred) — is accumulated across all pairs and timesteps "
        "into a delta tensor (shape 1024 x 128). At inference, scale * delta is added to every denoising step's "
        "prediction to steer the trajectory toward the desired gaze direction without modifying any model weights.",
        "Partial gaze shift visible. Some identity drift at higher scales.",
        {
            "output_steer": [
                ("Dataset",          "Columbia Gaze Dataset — up to 200 neg/pos pairs for delta computation"),
                ("Prompts",          "\"a person\" (text conditioning during delta computation and inference)"),
                ("Delta computation","max_pairs=200, samples_per_pair=3 timestep samples per pair"),
                ("Inference scales", "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps",  "28"),
                ("Resolution",       "512 x 512"),
                ("Strength",         "N/A — full denoising from noise with delta added each step"),
                ("Seed",             "42"),
            ],
        },
        [
            ("output_steer", "outputs/output_steer/result.png"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 4. LivePortrait + Klein Pipeline
    # -------------------------------------------------------------------------
    (
        "pipeline",
        "LivePortrait + Klein Pipeline",
        "No LoRA training. LivePortrait warps the eye region using the eyeball_direction_x parameter to produce "
        "left and right gaze variants. Klein 9B then runs img2img at strength=0.4 with the warped face as the "
        "reference image, refining quality and lighting while preserving the warped gaze direction.",
        "Clear gaze direction change. Identity well preserved. Works on any face without training.",
        {
            "single_face": [
                ("Dataset",              "Single FFHQ source face (face_0100.png)"),
                ("Prompts",              "\"a person\" (Klein img2img conditioning)"),
                ("LivePortrait param",   "eyeball_direction_x = +/-15 degrees"),
                ("Klein strength",       "0.4 (img2img — 40% noise added, 60% structure from warp)"),
                ("Inference steps",      "28"),
                ("Seed",                 "42"),
                ("Resolution",           "512 x 512"),
                ("Config",               "eye_gaze_v8.yaml"),
            ],
            "batch": [
                ("Dataset",              "10 FFHQ source faces — face_0000 through face_0090 (step 10)"),
                ("Prompts",              "\"a person\""),
                ("LivePortrait param",   "eyeball_direction_x = +/-15 degrees"),
                ("Klein strength",       "0.4 (img2img)"),
                ("Inference steps",      "28"),
                ("Seed",                 "42"),
                ("Resolution",           "512 x 512"),
                ("Config",               "eye_gaze_v8.yaml"),
            ],
        },
        [
            ("single_face", "outputs/pipeline_gaze/result.png"),
            ("face_0000",   "outputs/pipeline_gaze_batch/face_0000/result.png"),
            ("face_0010",   "outputs/pipeline_gaze_batch/face_0010/result.png"),
            ("face_0020",   "outputs/pipeline_gaze_batch/face_0020/result.png"),
            ("face_0030",   "outputs/pipeline_gaze_batch/face_0030/result.png"),
            ("face_0040",   "outputs/pipeline_gaze_batch/face_0040/result.png"),
            ("face_0050",   "outputs/pipeline_gaze_batch/face_0050/result.png"),
            ("face_0060",   "outputs/pipeline_gaze_batch/face_0060/result.png"),
            ("face_0070",   "outputs/pipeline_gaze_batch/face_0070/result.png"),
            ("face_0080",   "outputs/pipeline_gaze_batch/face_0080/result.png"),
            ("face_0090",   "outputs/pipeline_gaze_batch/face_0090/result.png"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 5. ai-toolkit ImageReferenceSliderTrainer
    # -------------------------------------------------------------------------
    (
        "aitoolkit_lora",
        "ai-toolkit ImageReferenceSliderTrainer",
        "Klein 9B LoRA trained using ai-toolkit's ImageReferenceSliderTrainer. Training runs without reference "
        "conditioning so the LoRA learns the gaze direction from image pairs alone. Two inference modes tested: "
        "(1) standard Klein pipeline with reference conditioning, (2) reference conditioning removed.",
        "With reference conditioning: no gaze change (ref cond locks output). Without reference conditioning: identity lost, model generates random people.",
        {
            "with_ref_cond": [
                ("Dataset",              "gaze_s15 — 1001 paired gaze images, neg=left-gaze, pos=right-gaze, 512 x 512"),
                ("Target class prompt",  "\"a person\""),
                ("Inference prompt",     "\"a person\""),
                ("Trainer",              "ImageReferenceSliderTrainer (ai-toolkit)"),
                ("Rank / Alpha",         "32 / 16"),
                ("LR",                   "1e-4"),
                ("Optimizer",            "adamw8bit"),
                ("Steps",                "2 000"),
                ("Batch size",           "1 (gradient_accumulation_steps=1)"),
                ("EMA",                  "decay=0.99"),
                ("network_weight",       "1.5"),
                ("weight_jitter",        "0.05"),
                ("Inference mode",       "Standard Klein ref cond pipeline (reference image passed as conditioning)"),
                ("Inference scales",     "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps",      "28"),
                ("Strength",             "0.6 (img2img)"),
                ("Seed",                 "42"),
            ],
            "no_ref_cond": [
                ("Dataset",              "gaze_s15 — 1001 paired gaze images"),
                ("Target class prompt",  "\"a person\""),
                ("Inference prompt",     "\"a person\""),
                ("Trainer",              "ImageReferenceSliderTrainer (ai-toolkit)"),
                ("Rank / Alpha",         "32 / 16"),
                ("LR",                   "1e-4"),
                ("Steps",                "2 000"),
                ("Inference mode",       "Reference conditioning removed — LoRA alone drives generation from noise"),
                ("Inference scales",     "-5, -2.5, 0, 2.5, 5"),
                ("Inference steps",      "28"),
                ("Strength",             "1.0 (full generation from noise, no img2img blending)"),
                ("Seed",                 "42"),
            ],
        },
        [
            ("with_ref_cond", "outputs/lora_gaze_test/result.png"),
            ("no_ref_cond",   "outputs/lora_gaze_test_v2/result.png"),
        ],
    ),
]

base_dir = Path(__file__).parent

tabs_nav = ""
tabs_content = ""

for i, (tab_id, tab_title, description, status, cfg_map, images) in enumerate(experiments):
    active = "active" if i == 0 else ""
    tabs_nav += f'<button class="tab-btn {active}" onclick="showTab(\'{tab_id}\')" id="btn-{tab_id}">{tab_title}</button>\n'

    imgs_html = ""
    for img_id, img_path in images:
        full_path = base_dir / img_path
        if not full_path.exists():
            print(f"  missing: {img_path}")
            continue
        uri = img_to_data_uri(full_path)
        label = img_id.replace("_", " ")

        cfg_rows = cfg_map.get(img_id) or cfg_map.get("batch") or []
        cfg_html = config_table(cfg_rows) if cfg_rows else ""

        imgs_html += f'''
        <div class="img-card">
            <img src="{uri}" alt="{label}" loading="lazy">
            <p class="img-label">{label}</p>
            {cfg_html}
        </div>'''
        print(f"  embedded: {img_path}")

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
  .experiment-meta {{ max-width: 820px; margin-bottom: 28px; }}
  .description {{ font-size: 13px; color: #aaa; line-height: 1.6; margin-bottom: 10px; }}
  .status {{ font-size: 13px; color: #aaa; line-height: 1.6; }}
  .status strong {{ color: #e0e0e0; }}
  .images-grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
  .img-card {{ background: #1a1a1a; border-radius: 6px; overflow: hidden; border: 1px solid #2a2a2a; max-width: 500px; }}
  .img-card img {{ display: block; width: 100%; height: auto; max-height: 260px; object-fit: contain; background: #111; }}
  .img-label {{ font-size: 12px; color: #bbb; padding: 8px 12px 6px; font-weight: 500; border-bottom: 1px solid #222; }}
  .cfg-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  .cfg-table tr:nth-child(even) {{ background: #1f1f1f; }}
  .cfg-key {{ color: #666; padding: 4px 10px; width: 38%; vertical-align: top; white-space: nowrap; }}
  .cfg-val {{ color: #aaa; padding: 4px 10px; line-height: 1.5; }}
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

out = base_dir / "outputs" / "experiment_report.html"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    f.write(html)
print(f"\nReport saved to {out}")
