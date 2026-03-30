# Setup

Directional LoRA slider trainer for FLUX Klein 9B. Trains a single LoRA that can push image attributes in both directions along a concept axis (e.g., frowning <-> smiling) with continuous control via a scale parameter.

The concept direction is defined by **image pairs** — a negative image and a positive image per pair. The visual difference between the two captures the attribute change (e.g., frowning face → smiling face).

## Requirements

- GPU with >= 40 GB VRAM (H100 80GB recommended, A100 40GB works)
- Python >= 3.10
- CUDA >= 11.8

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Download model weights

You need `huggingface-cli` installed (`pip install huggingface-hub`) and must be logged in (`huggingface-cli login`) for the gated Klein 9B repo.

```bash
# Klein 9B Transformer (~17 GB)
mkdir -p models/klein-9b
huggingface-cli download black-forest-labs/FLUX.2-Klein-9B flux-2-klein-9b.safetensors \
    --local-dir models/klein-9b

# FLUX2 VAE (~321 MB)
huggingface-cli download ai-toolkit/flux2_vae ae.safetensors \
    --local-dir models/klein-9b
```

The Qwen3-8B text encoder (~16 GB) auto-downloads on first run and caches in `~/.cache/huggingface/`.

After this step you should have:
```
models/klein-9b/
├── flux-2-klein-9b.safetensors   (17 GB)
└── ae.safetensors                (321 MB)
```

## 3. Prepare training data

Unzip the included sample dataset:

```bash
unzip sample_dataset.zip -d data/smile_slider/
```

This gives you paired directories:
```
data/smile_slider/
├── neg/
│   ├── pair_0000.jpg
│   ├── pair_0001.jpg
│   ├── ...
│   └── pair_0049.jpg
└── pos/
    ├── pair_0000.jpg
    ├── pair_0001.jpg
    ├── ...
    └── pair_0049.jpg
```

The `config/smile_slider.yaml` already points to `data/smile_slider/neg/` and `data/smile_slider/pos/`. Pairs are matched by filename stem — `neg/pair_0000.jpg` is paired with `pos/pair_0000.jpg` (extensions can differ between neg/pos as long as the stem matches).

### About text captions / .txt files

Per-image text captions (`.txt` files) are **not used** by the slider trainer. The direction is defined entirely by the image pairs — text plays no role in defining what changes.

A single neutral prompt (set via `prompt` in the config, e.g. `"a person"`) is used for all forward passes. This prompt just tells the model the general category of the content. You do not need per-image captions.

If your dataset source includes `.txt` files alongside images (e.g., from a captioning pipeline), they will be ignored — only image files (`.png`, `.jpg`, `.jpeg`, `.webp`) are loaded.

### Using your own data

- Create `neg/` and `pos/` directories with matching filenames
- **neg/** should contain images with the "starting" attribute (e.g., neutral/frowning faces)
- **pos/** should contain images with the "target" attribute (e.g., smiling faces)
- Each pair should be the **same person/scene** with only the target attribute changed
- Images are resized automatically; 512x512 recommended
- At least 20-50 pairs for good diversity
- Set `prompt` in the config to a neutral description of your content (e.g., `"a person"`, `"a landscape"`, `"a car"`)

## 4. Training

```bash
python train_slider.py --config config/smile_slider.yaml
```

### What the config controls

| Field | Default | Description |
|-------|---------|-------------|
| `neg_image_dir` | `data/smile_slider/neg/` | Folder of negative (starting) images |
| `pos_image_dir` | `data/smile_slider/pos/` | Folder of positive (target) images |
| `prompt` | `"a person"` | Neutral text prompt used for all passes |
| `rank` | 16 | LoRA rank (higher = more capacity) |
| `alpha` | 1 | LoRA alpha scaling |
| `lr` | 0.002 | Learning rate |
| `eta` | 2 | Directional loss strength |
| `max_train_steps` | 1000 | Total training steps |
| `sample_every` | 100 | Save sample visualization every N steps |
| `height` / `width` | 512 | Training resolution |
| `train_method` | `xattn` | Which layers to target (xattn = all attention) |

### How the direction is defined

The direction comes from **image pairs**, not text prompts:
- Each training step samples a random pair `(neg_image, pos_image)`
- Two forward passes with different reference images capture the visual difference
- The LoRA learns to add this direction to its output, controllable via scale

A single neutral `prompt` (e.g., `"a person"`) is used for all forward passes. The text doesn't define the direction — the images do.

### Training outputs

Everything goes to `outputs/` (configurable via `output_dir`):

```
outputs/
├── weights/
│   ├── slider_000100.safetensors   # Checkpoint at step 100
│   ├── slider_000200.safetensors   # Checkpoint at step 200
│   ├── ...
│   └── slider_latest.safetensors   # Final checkpoint
├── samples/
│   ├── step_000100.png             # Visualization at step 100
│   ├── step_000200.png             # Visualization at step 200
│   ├── ...
│   └── final.png                   # Final visualization
├── loss.png                        # Loss curve plot
└── config.yaml                     # Copy of training config
```

Sample visualizations show a neg image from the dataset at scales [-5, -2.5, 0, 2.5, 5] so you can monitor the slider effect during training.

Training takes ~15 minutes for 1000 steps on an H100 (~1 it/s).

## 5. Inference

Generate edited images from a source portrait at different slider scales:

```bash
python inference_slider.py \
    --config config/smile_slider.yaml \
    --lora_path outputs/weights/slider_latest.safetensors \
    --source_image /path/to/portrait.jpg \
    --prompt "a person" \
    --scales -3 -1.5 0 1.5 3 \
    --output outputs/result.png
```

### Scale guide

| Scale | Effect (smile slider example) |
|-------|-------------------------------|
| -5 to -3 | Strong opposite (frowning) |
| -1.5 | Slight opposite |
| 0 | No change (baseline) |
| +1.5 | Slight target (gentle smile) |
| +3 to +5 | Strong target (big smile) |

### What it outputs

- `outputs/result.png` — Side-by-side strip: source + all scales
- `outputs/scale_+1.5.png`, `outputs/scale_-3.0.png`, etc. — Individual images per scale

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to training config yaml |
| `--lora_path` | (required) | Path to trained LoRA safetensors |
| `--source_image` | (required) | Input portrait image |
| `--prompt` | `"a person"` | Text prompt for generation |
| `--scales` | `-5 -2.5 0 2.5 5` | Space-separated list of scales |
| `--num_steps` | 28 | Denoising steps |
| `--seed` | 42 | Random seed |
| `--output` | `outputs/inference_result.png` | Output path for strip visualization |

## VRAM Budget (~47-50 GB bf16)

| Component | VRAM |
|-----------|------|
| Klein 9B transformer | ~18 GB |
| Qwen3-8B text encoder | ~16 GB (offloaded to CPU after prompt encoding) |
| VAE | ~0.3 GB |
| LoRA params + gradients | ~0.5 GB |
| Activations (3 forward passes per step) | ~12-15 GB |

## How it works

The directional slider LoRA is trained using 3 forward passes per step:

1. Two passes **without** LoRA (same noisy image + text, different reference images) to compute a direction vector: `gt = neg_pred + eta * (pos_pred - neg_pred)`
2. One pass **with** LoRA to match that direction: `loss = MSE(lora_pred, gt)`

The direction comes from swapping reference tokens (neg_image vs pos_image) while keeping text constant. This captures the visual attribute change between the image pairs.

At inference, the LoRA scale acts as a continuous slider: positive pushes toward the positive image direction, negative pushes toward the negative image direction, zero = no effect.
