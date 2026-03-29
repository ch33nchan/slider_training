import os
import gc
import copy
import random
import argparse
from contextlib import ExitStack

import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.optimization import get_scheduler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model_util import load_klein_models
from utils.train_util import encode_prompt_klein
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE
from utils.custom_klein_pipeline import KleinPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='black-forest-labs/FLUX.2-klein-base-4B')
    parser.add_argument('--target_prompt', type=str, required=True,
                        help='neutral subject prompt, e.g. "a photo of a person"')
    parser.add_argument('--positive_prompt', type=str, required=True,
                        help='prompt for the concept to enhance, e.g. "a photo of a person looking right"')
    parser.add_argument('--negative_prompt', type=str, required=True,
                        help='opposite concept, e.g. "a photo of a person looking left"')
    parser.add_argument('--slider_name', type=str, default='klein-slider')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--max_train_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=2.0)
    parser.add_argument('--train_method', type=str, default='xattn', choices=['xattn', 'full'])
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def get_sigmas(timesteps, noise_scheduler_copy, n_dim=4, device='cuda:0', dtype=torch.bfloat16):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def main():
    args = parse_args()
    device = args.device
    weight_dtype = torch.bfloat16

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading models from {args.model_id} ...")
    tokenizer, text_encoder, transformer, scheduler, vae = load_klein_models(
        args.model_id, weight_dtype=weight_dtype
    )

    vae.to(device)
    transformer.to(device)
    text_encoder.to(device)

    noise_scheduler_copy = copy.deepcopy(scheduler)

    print("Encoding prompts ...")
    with torch.no_grad():
        all_embeds = encode_prompt_klein(
            tokenizer,
            text_encoder,
            [args.target_prompt, args.positive_prompt, args.negative_prompt],
            max_sequence_length=args.max_sequence_length,
            device=device,
        )
        target_embeds, positive_embeds, negative_embeds = all_embeds.chunk(3)

    print(f"Setting up LoRA (rank={args.rank}, method={args.train_method}) ...")
    network = LoRANetwork(
        transformer,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.alpha,
        train_method=args.train_method,
    ).to(device, dtype=weight_dtype)

    optimizer = AdamW(network.prepare_optimizer_params(), lr=args.lr)
    optimizer.zero_grad()
    criteria = torch.nn.MSELoss()

    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer,
        num_warmup_steps=min(200, args.max_train_steps // 5),
        num_training_steps=args.max_train_steps,
    )

    pipe = KleinPipeline(scheduler, vae, text_encoder, tokenizer, transformer)
    pipe.set_progress_bar_config(disable=True)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

    # compute unpack shape once
    with torch.no_grad():
        sample_packed = pipe(
            args.target_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            from_timestep=0,
            till_timestep=1,
            output_type='latent',
        )
        model_input_shape = KleinPipeline._unpack_latents(
            sample_packed, args.height, args.width, vae_scale_factor
        ).shape

    weighting_scheme = 'none'
    logit_mean, logit_std, mode_scale = 0.0, 1.0, 1.29
    bsz = 1

    losses = []
    progress_bar = tqdm(range(args.max_train_steps), desc="Training")

    for step in range(args.max_train_steps):

        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
        timestep_to_infer = (
            indices[0] * (args.num_inference_steps / noise_scheduler_copy.config.num_train_timesteps)
        ).long().item()

        # get x_t: run denoising up to the sampled timestep
        with torch.no_grad():
            packed_noisy = pipe(
                args.target_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                num_images_per_prompt=bsz,
                from_timestep=0,
                till_timestep=timestep_to_infer,
                output_type='latent',
            )

        latent_image_ids = KleinPipeline._prepare_latent_image_ids(
            model_input_shape[0],
            model_input_shape[2],
            model_input_shape[3],
            device,
            weight_dtype,
        )

        # build txt_ids for target embeddings
        txt_ids = torch.zeros(
            bsz, target_embeds.shape[1], 3,
            device=device, dtype=weight_dtype,
        )

        # guidance is None — guidance_embeds=False for this model
        guidance = None

        def transformer_forward(embeds):
            return transformer(
                hidden_states=packed_noisy,
                timestep=timesteps / 1000,
                encoder_hidden_states=embeds,
                img_ids=latent_image_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]

        # forward with LoRA active (target prompt)
        with ExitStack() as stack:
            stack.enter_context(network)
            model_pred = transformer_forward(target_embeds)

        model_pred = KleinPipeline._unpack_latents(
            model_pred,
            int(model_input_shape[2] * vae_scale_factor / 2),
            int(model_input_shape[3] * vae_scale_factor / 2),
            vae_scale_factor,
        )

        # frozen forward passes for gt construction
        with torch.no_grad():
            target_pred = KleinPipeline._unpack_latents(
                transformer_forward(target_embeds),
                int(model_input_shape[2] * vae_scale_factor / 2),
                int(model_input_shape[3] * vae_scale_factor / 2),
                vae_scale_factor,
            )
            positive_pred = KleinPipeline._unpack_latents(
                transformer_forward(positive_embeds),
                int(model_input_shape[2] * vae_scale_factor / 2),
                int(model_input_shape[3] * vae_scale_factor / 2),
                vae_scale_factor,
            )
            negative_pred = KleinPipeline._unpack_latents(
                transformer_forward(negative_embeds),
                int(model_input_shape[2] * vae_scale_factor / 2),
                int(model_input_shape[3] * vae_scale_factor / 2),
                vae_scale_factor,
            )

            # concept slider target: steer toward positive, away from negative
            gt_pred = target_pred + args.eta * (positive_pred - negative_pred)
            gt_pred = (gt_pred / gt_pred.norm()) * positive_pred.norm()

        loss = torch.mean(
            ((model_pred.float() - gt_pred.float()) ** 2).reshape(gt_pred.shape[0], -1), 1
        ).mean()

        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.6f}")

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_train_steps:
            save_path = os.path.join(args.output_dir, f"{args.slider_name}_step{step+1}.pt")
            network.save_weights(save_path, dtype=weight_dtype)
            print(f"\nSaved checkpoint: {save_path}")

    print("\nTraining complete.")
    print(f"Final avg loss (last 50 steps): {np.mean(losses[-50:]):.4f}")


if __name__ == '__main__':
    main()
