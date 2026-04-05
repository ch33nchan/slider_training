from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image, ImageDraw


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--input-dir", default=str(repo_root / "characters"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", default="/mnt/data1/models/base-models/black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="portrait of a person")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--scales", nargs="+", type=float, default=[-3.0, 0.0, 3.0])
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def load_input_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    return image.resize((width, height))


def save_strip(images: Iterable[Image.Image], labels: Iterable[str], output_path: Path) -> None:
    images = list(images)
    labels = list(labels)
    width, height = images[0].size
    label_height = 40
    canvas = Image.new("RGB", (width * len(images), height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 16, 10), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    lora_path = Path(args.lora_path).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing LoRA path: {lora_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    pipe = FluxImg2ImgPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(args.device)
    pipe.load_lora_weights(str(lora_path), adapter_name="gaze")

    for image_path in image_paths:
        source = load_input_image(image_path)
        sample_dir = output_root / image_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        source.save(sample_dir / "input.png")

        generated_images = [source]
        labels = ["input"]
        for scale in args.scales:
            pipe.set_adapters(["gaze"], adapter_weights=[scale])
            image = pipe(
                prompt=args.prompt,
                image=source,
                strength=args.strength,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=torch.Generator(device=args.device).manual_seed(args.seed),
            ).images[0]
            label = f"{scale:+.1f}"
            image.save(sample_dir / f"scale_{label}.png")
            generated_images.append(image)
            labels.append(label)

        save_strip(generated_images, labels, sample_dir / "result.png")


if __name__ == "__main__":
    main()
