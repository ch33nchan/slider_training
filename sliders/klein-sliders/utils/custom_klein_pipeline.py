"""
Klein slider pipeline utilities.

Provides latent geometry helpers and a partial_denoise() function used by
train_klein_slider.py to obtain noisy latents at a specific timestep index.

Inference uses Flux2KleinPipeline directly (see inference_klein_slider.py).
"""

from contextlib import nullcontext

import torch
from diffusers import Flux2KleinPipeline


# ---------------------------------------------------------------------------
# Latent geometry — 4D position IDs required by Klein's pos_embed
# ---------------------------------------------------------------------------

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """
    4D position IDs for Klein's pos_embed.
    height / width are the *unpacked* latent spatial dims (H//vae_scale, W//vae_scale).
    Packed token grid is (H//2) x (W//2).
    """
    h2, w2 = height // 2, width // 2
    ids = torch.zeros(h2, w2, 4)
    ids[..., 1] = ids[..., 1] + torch.arange(h2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w2)[None, :]
    ids = ids.reshape(-1, 4)[None].repeat(batch_size, 1, 1)
    return ids.to(device=device, dtype=dtype)


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack (B, C, H, W) latents into (B, H/2*W/2, C*4) tokens."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


# ---------------------------------------------------------------------------
# Partial denoising for training
# ---------------------------------------------------------------------------

def _cache_ctx(transformer):
    """Return cache_context("cond") if the transformer supports it, else nullcontext."""
    if hasattr(transformer, "cache_context"):
        return transformer.cache_context("cond")
    return nullcontext()


@torch.no_grad()
def partial_denoise(
    transformer,
    scheduler,
    packed_latents,
    encoder_hidden_states,
    latent_image_ids,
    txt_ids,
    weight_dtype,
    start_step=0,
    end_step=None,
):
    """
    Run the denoising loop from start_step to end_step and return packed latents.

    Uses scheduler.timesteps[start_step:end_step], so the caller must have
    called scheduler.set_timesteps() before invoking this function.

    Returns packed latents at the noise level corresponding to end_step.
    """
    timestep_list = scheduler.timesteps[start_step:end_step]
    for t in timestep_list:
        timestep = t.expand(packed_latents.shape[0]).to(weight_dtype)
        with _cache_ctx(transformer):
            noise_pred = transformer(
                hidden_states=packed_latents,
                timestep=timestep / 1000,
                encoder_hidden_states=encoder_hidden_states,
                img_ids=latent_image_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]
        # Klein output may include context tokens; slice to latent patch count.
        noise_pred = noise_pred[:, : packed_latents.size(1)]
        packed_latents = scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
    return packed_latents


# ---------------------------------------------------------------------------
# Thin alias — kept so any remaining import of KleinPipeline doesn't break
# ---------------------------------------------------------------------------

class KleinPipeline(Flux2KleinPipeline):
    """
    Flux2KleinPipeline subclass kept for backward compatibility.
    Training now uses partial_denoise() directly; inference uses
    Flux2KleinPipeline.__call__ with 'with network:' as context manager.
    """
    pass
