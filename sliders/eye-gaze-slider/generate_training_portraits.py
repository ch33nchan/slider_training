#!/usr/bin/env python3
"""
generate_training_portraits.py
--------------------------------
Generate 100 diverse photorealistic portrait images using FLUX.2-klein-9B
split across 2 GPUs in parallel.

Output: sliders/eye-gaze-slider/data/portrait_base/portrait_0000.png … portrait_0099.png

Usage:
    python generate_training_portraits.py
    python generate_training_portraits.py --outdir /custom/path --gpu_a 2 --gpu_b 3
"""

import argparse
import os
import time
import torch
import torch.multiprocessing as mp
from pathlib import Path

# ---------------------------------------------------------------------------
# 100 diverse portrait prompts
# 20 subject descriptions × 5 lighting setups = 100
# ---------------------------------------------------------------------------

SUBJECTS = [
    "a young East Asian woman in her mid-twenties",
    "a middle-aged Black man in his forties",
    "a young white man in his late twenties",
    "an older Latina woman in her mid-fifties",
    "a young Black woman in her early twenties",
    "a South Asian man in his thirties",
    "a white woman in her late thirties",
    "an older East Asian man in his sixties",
    "a young Middle Eastern woman in her twenties",
    "a mixed-race man in his early thirties",
    "a young Scandinavian woman in her mid-twenties",
    "a Black woman in her late forties",
    "a young Hispanic man in his mid-twenties",
    "an older white man in his early sixties",
    "a young South Asian woman in her late twenties",
    "a young white woman in her early twenties",
    "a middle-aged East Asian woman in her forties",
    "a young Black man in his mid-twenties",
    "an older Middle Eastern man in his fifties",
    "a young Latina woman in her late twenties",
]

LIGHTINGS = [
    "soft box studio lighting, clean grey background",
    "dramatic rembrandt side lighting, dark moody background",
    "natural diffused window light, neutral background",
    "outdoor golden hour warm light, bokeh background",
    "high key bright studio lighting, white background",
]

def build_prompts():
    prompts = []
    for subject in SUBJECTS:
        for lighting in LIGHTINGS:
            p = (
                f"professional portrait photograph of {subject}, "
                f"neutral expression, looking directly at the camera, "
                f"face and upper chest in frame, sharp focus on eyes, "
                f"{lighting}, photorealistic, 8k, high detail"
            )
            prompts.append(p)
    assert len(prompts) == 100
    return prompts


# ---------------------------------------------------------------------------
# Worker: runs on one GPU, generates its slice of prompts
# ---------------------------------------------------------------------------

def worker(gpu_id: int, indices: list, prompts: list, outdir: str,
           model_id: str, steps: int, guidance: float, size: int):
    """Runs inside a child process — one per GPU."""

    print(f"[GPU {gpu_id}] Starting — {len(indices)} images to generate")

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers import FluxPipeline as Flux2KleinPipeline

    device = f"cuda:{gpu_id}"
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(outdir, exist_ok=True)

    for rank, (idx, prompt) in enumerate(zip(indices, prompts)):
        out_path = Path(outdir) / f"portrait_{idx:04d}.png"
        if out_path.exists():
            print(f"[GPU {gpu_id}] {idx:04d} already exists, skipping")
            continue

        t0 = time.time()
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=size,
                width=size,
            ).images[0]

        image.save(str(out_path))
        elapsed = time.time() - t0
        print(f"[GPU {gpu_id}] [{rank+1:3d}/{len(indices)}]  portrait_{idx:04d}.png  "
              f"({elapsed:.1f}s)")

    print(f"[GPU {gpu_id}] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--outdir",   default=str(
        Path(__file__).resolve().parent / "data" / "portrait_base"))
    p.add_argument("--gpu_a",    type=int, default=2, help="First GPU id")
    p.add_argument("--gpu_b",    type=int, default=3, help="Second GPU id")
    p.add_argument("--steps",    type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--size",     type=int, default=1024)
    args = p.parse_args()

    prompts = build_prompts()
    os.makedirs(args.outdir, exist_ok=True)

    # Split: first 50 → GPU A, last 50 → GPU B
    half = len(prompts) // 2
    split = [
        (args.gpu_a, list(range(half)),       prompts[:half]),
        (args.gpu_b, list(range(half, 100)),  prompts[half:]),
    ]

    print(f"Generating {len(prompts)} portraits → {args.outdir}")
    print(f"GPU {args.gpu_a}: portraits 0000–{half-1:04d}")
    print(f"GPU {args.gpu_b}: portraits {half:04d}–0099")
    print()

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id, indices, subset in split:
        proc = mp.Process(
            target=worker,
            args=(gpu_id, indices, subset, args.outdir,
                  args.model, args.steps, args.guidance, args.size),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    # Count results
    generated = len(list(Path(args.outdir).glob("portrait_*.png")))
    print(f"\n✓ {generated}/100 portraits saved → {args.outdir}")


if __name__ == "__main__":
    main()
