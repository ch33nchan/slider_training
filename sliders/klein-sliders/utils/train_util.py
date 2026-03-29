from typing import Optional, List

import torch
from diffusers.utils.torch_utils import randn_tensor

# VAE and latent geometry constants — same as FLUX.1
VAE_SCALE_FACTOR = 16   # FLUX 16-channel VAE: 2 ** len(block_out_channels)
LATENT_CHANNELS = 16

# Qwen3 hidden state layers to extract for text conditioning.
# layers 9, 18, 27 are intermediate representations (not the final layer).
# Confirm num_hidden_layers >= 27 after loading: print(text_encoder.config.num_hidden_layers)
QWEN_LAYER_INDICES = (9, 18, 27)


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def encode_prompt_klein(
    tokenizer,
    text_encoder,
    prompts: List[str],
    max_sequence_length: int = 512,
    device: str = "cuda",
    layer_indices: tuple = QWEN_LAYER_INDICES,
) -> torch.FloatTensor:
    """
    Encode prompts with the Qwen3 text encoder.

    Extracts hidden states at the specified layer indices and concatenates
    them along the hidden dimension. Returns encoder_hidden_states of shape
    [batch, seq_len, hidden_dim * len(layer_indices)].

    There is no pooled embedding — Klein does not use pooled_projections.
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = text_encoder(input_ids, output_hidden_states=True)

    # hidden_states[0] is the embedding output, [1..N] are transformer block outputs.
    # layer index k corresponds to hidden_states[k].
    layer_tensors = [outputs.hidden_states[i] for i in layer_indices]
    encoder_hidden_states = torch.cat(layer_tensors, dim=-1)

    return encoder_hidden_states.to(dtype=text_encoder.dtype, device=device)


# ---------------------------------------------------------------------------
# Noise prediction — no batch doubling, guidance is a scalar passed directly
# ---------------------------------------------------------------------------

def predict_noise_klein(
    transformer,
    timestep: torch.Tensor,
    packed_latents: torch.Tensor,
    encoder_hidden_states: torch.FloatTensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float = 3.5,
) -> torch.Tensor:
    """
    Single forward pass through Flux2Transformer2DModel.

    Klein handles guidance via a scalar tensor inside the transformer — no
    batch doubling. The guidance tensor is only constructed when the model
    config has guidance_embeds=True (true for the undistilled base model).

    IMPORTANT: verify the exact forward signature on the server before use:
        import inspect
        print(inspect.signature(transformer.forward))
    The kwargs below (img_ids, encoder_hidden_states, guidance) must match
    what Flux2Transformer2DModel.forward actually accepts.
    """
    if transformer.config.guidance_embeds:
        guidance = torch.tensor(
            [guidance_scale],
            device=packed_latents.device,
            dtype=packed_latents.dtype,
        ).expand(packed_latents.shape[0])
    else:
        guidance = None

    # txt_ids: zero tensor of shape [batch, seq_len, 3] — confirmed present in forward signature
    txt_ids = torch.zeros(
        packed_latents.shape[0],
        encoder_hidden_states.shape[1],
        3,
        device=packed_latents.device,
        dtype=packed_latents.dtype,
    )

    # pooled_projections is a FLUX.1 argument — not in Klein's forward signature
    kwargs = dict(
        hidden_states=packed_latents,
        timestep=timestep / 1000,
        encoder_hidden_states=encoder_hidden_states,
        img_ids=latent_image_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )
    if guidance is not None:
        kwargs["guidance"] = guidance

    noise_pred = transformer(**kwargs)[0]
    return noise_pred


@torch.no_grad()
def diffusion_klein(
    transformer,
    scheduler,
    packed_latents: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    latent_image_ids: torch.Tensor,
    guidance_scale: float = 3.5,
    total_timesteps: int = None,
    start_timesteps: int = 0,
) -> torch.FloatTensor:
    timestep_list = scheduler.timesteps[start_timesteps:total_timesteps]

    for t in timestep_list:
        timestep = t.expand(packed_latents.shape[0]).to(packed_latents.dtype)
        noise_pred = predict_noise_klein(
            transformer,
            timestep,
            packed_latents,
            encoder_hidden_states,
            latent_image_ids,
            guidance_scale=guidance_scale,
        )
        packed_latents = scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]

    return packed_latents


# ---------------------------------------------------------------------------
# Latent helpers — identical geometry to FLUX.1
# ---------------------------------------------------------------------------

def get_random_noise(
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator = None,
) -> torch.Tensor:
    return torch.randn(
        (batch_size, LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR),
        generator=generator,
        device="cpu",
    )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2
    step = 64
    min_step = min_resolution // step
    max_step = max_resolution // step
    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step
    return height, width


# ---------------------------------------------------------------------------
# Optimizers and schedulers — unchanged from flux-sliders
# ---------------------------------------------------------------------------

def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation
        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):
        import bitsandbytes as bnb
        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion
            return Lion
        elif name == "prodigy":
            import prodigyopt
            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError("Scheduler must be cosine, cosine_with_restarts, step, linear or constant")
