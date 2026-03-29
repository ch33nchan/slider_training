import os
import sys
import argparse
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='black-forest-labs/FLUX.2-klein-base-4B')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='path to trained .pt checkpoint')
    parser.add_argument('--prompt', type=str, default='a photo of a person',
                        help='prompt to generate images with')
    parser.add_argument('--scales', type=str, default='-5,-2.5,0,2.5,5',
                        help='comma-separated LoRA scales to test')
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument('--num_images', type=int, default=3,
                        help='number of seeds to test per scale')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--train_method', type=str, default='xattn')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    weight_dtype = torch.bfloat16
    scales = [float(s) for s in args.scales.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading pipeline from {args.model_id} ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"Loading LoRA from {args.lora_path} ...")
    network = LoRANetwork(
        pipe.transformer,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.alpha,
        train_method=args.train_method,
    ).to(device, dtype=weight_dtype)
    network.load_state_dict(torch.load(args.lora_path, map_location=device))
    # LoRA is hooked into the transformer linear layers via apply_to().
    # Deactivate by default (multiplier=0 after __exit__); enable with 'with network:'.
    network.__exit__(None, None, None)
    print("LoRA loaded.")

    seeds = [args.seed + i for i in range(args.num_images)]

    for seed in seeds:
        row = []

        for scale in scales:
            network.set_lora_slider(scale=scale)
            generator = torch.Generator(device=device).manual_seed(seed)

            with torch.no_grad():
                if scale == 0.0:
                    # No LoRA — call pipe normally (multiplier already 0)
                    result = pipe(
                        args.prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        output_type='pil',
                    )
                else:
                    # Enable LoRA for the full denoising pass
                    with network:
                        result = pipe(
                            args.prompt,
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_inference_steps,
                            generator=generator,
                            output_type='pil',
                        )

            row.append(result.images[0])

        # Save individual images
        for img, scale in zip(row, scales):
            fname = f"seed{seed}_scale{scale:+.1f}.png"
            img.save(os.path.join(args.output_dir, fname))
            print(f"  saved {fname}")

        # Save side-by-side strip
        total_w = sum(img.width for img in row)
        strip = Image.new('RGB', (total_w, args.height))
        x = 0
        for img in row:
            strip.paste(img, (x, 0))
            x += img.width
        strip_name = f"seed{seed}_strip_{'_'.join(str(s) for s in scales)}.png"
        strip.save(os.path.join(args.output_dir, strip_name))
        print(f"  saved strip: {strip_name}")

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == '__main__':
    main()
