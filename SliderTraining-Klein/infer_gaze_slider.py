"""
Inference script for eye gaze slider LoRA on FLUX.1-dev.
Generates images at multiple LoRA scales to evaluate gaze control.

Usage:
  python infer_gaze_slider.py \
    --lora_path output/eye_gaze_flux_dev_v1/eye_gaze_flux_dev_v1.safetensors \
    --output_dir outputs/gaze_inference \
    --device cuda:0
"""

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--model_path", default="/mnt/data1/models/base-models/black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output_dir", default="outputs/gaze_inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading FLUX.1-dev...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(args.device)

    lora_path = Path(args.lora_path).resolve()
    print(f"Loading LoRA from {lora_path}...")
    from safetensors.torch import load_file
    state_dict = load_file(str(lora_path))
    pipe.load_lora_weights(state_dict)

    prompts = [
        "a person",
        "portrait of a person",
        "a person looking straight ahead",
    ]

    scales = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt: '{prompt}'")
        row = []
        for scale in scales:
            print(f"  scale={scale:+.1f}", end="", flush=True)
            img = pipe(
                prompt=prompt,
                guidance_scale=args.guidance,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                joint_attention_kwargs={"scale": scale},
                generator=torch.Generator(device=args.device).manual_seed(args.seed),
            ).images[0]
            fname = out / f"p{prompt_idx}_scale_{scale:+.1f}.png"
            img.save(str(fname))
            row.append(img)
            print(f" -> saved", flush=True)

        # Make a strip for this prompt
        w, h = row[0].size
        strip = Image.new("RGB", (w * len(scales), h + 40), (255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(strip)
        for i, (img, scale) in enumerate(zip(row, scales)):
            strip.paste(img, (i * w, 40))
            draw.text((i * w + w // 2 - 15, 5), f"{scale:+.1f}", fill=(0, 0, 0))
        strip_path = out / f"strip_p{prompt_idx}.png"
        strip.save(str(strip_path))
        print(f"  Strip saved: {strip_path}")

    print(f"\nDone! Results in {out}/")


if __name__ == "__main__":
    main()
