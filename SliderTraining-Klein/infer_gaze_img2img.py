"""
Img2img inference for eye gaze slider LoRA on FLUX.1-dev.
Takes an input face image and shifts gaze left/right at multiple scales.

Usage:
  python infer_gaze_img2img.py \
    --lora_path /path/to/eye_gaze_flux_dev_img.safetensors \
    --input_image /path/to/face.png \
    --output_dir outputs/gaze_img2img
"""

import argparse
from pathlib import Path

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--model_path", default="/mnt/data1/models/base-models/black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output_dir", default="outputs/gaze_img2img")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--strength", type=float, default=0.4, help="img2img strength (0=no change, 1=full redraw)")
    parser.add_argument("--guidance", type=float, default=3.5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_img = Image.open(args.input_image).convert("RGB")
    w, h = input_img.size
    # Snap to multiples of 16 for FLUX
    w = (w // 16) * 16
    h = (h // 16) * 16
    input_img = input_img.resize((w, h))

    print("Loading FLUX.1-dev img2img pipeline...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(args.device)

    print(f"Loading LoRA from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path, adapter_name="gaze")

    scales = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    prompt = "a person"

    row = []
    for scale in scales:
        print(f"  scale={scale:+.1f}", end="", flush=True)
        pipe.set_adapters(["gaze"], adapter_weights=[scale])
        img = pipe(
            prompt=prompt,
            image=input_img,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
        ).images[0]
        fname = out / f"scale_{scale:+.1f}.png"
        img.save(str(fname))
        row.append(img)
        print(f" -> saved", flush=True)

    # Save input alongside results
    input_img.save(str(out / "input.png"))

    # Make strip: input | -3 -2 -1 0 +1 +2 +3
    iw, ih = input_img.size
    label_h = 40
    strip = Image.new("RGB", (iw * (len(scales) + 1), ih + label_h), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(strip)

    strip.paste(input_img, (0, label_h))
    draw.text((iw // 2 - 20, 10), "INPUT", fill=(0, 0, 0))

    for i, (img, scale) in enumerate(zip(row, scales)):
        x = (i + 1) * iw
        strip.paste(img.resize((iw, ih)), (x, label_h))
        draw.text((x + iw // 2 - 20, 10), f"{scale:+.1f}", fill=(0, 0, 0))

    strip.save(str(out / "strip.png"))
    print(f"\nStrip saved: {out}/strip.png")
    print(f"Done! Results in {out}/")


if __name__ == "__main__":
    main()
