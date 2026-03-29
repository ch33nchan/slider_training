from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

# Flux2Transformer2DModel requires diffusers >= 0.33.0
# If you get an ImportError here, run: pip install --upgrade diffusers
try:
    from diffusers.models.transformers import Flux2Transformer2DModel
except ImportError:
    raise ImportError(
        "Flux2Transformer2DModel not found. "
        "Please upgrade diffusers: pip install 'diffusers>=0.33.0'"
    )

DIFFUSERS_CACHE_DIR = None


def load_klein_models(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load all components for FLUX.2-klein-base-4B.

    Returns:
        tokenizer       - AutoTokenizer (Qwen3)
        text_encoder    - AutoModel (Qwen3), frozen, output_hidden_states=True
        transformer     - Flux2Transformer2DModel, frozen
        scheduler       - FlowMatchEulerDiscreteScheduler
        vae             - AutoencoderKL, frozen

    After loading, run these checks on the server:
        import inspect
        print(inspect.signature(transformer.forward))
        print(transformer.config.guidance_embeds)
        print(text_encoder.config.num_hidden_layers)  # should be >= 27
        print(vae.config.scaling_factor, vae.config.shift_factor)
        for name, mod in transformer.named_modules():
            if mod.__class__.__name__ in ['Flux2Attention', 'Flux2ParallelSelfAttention']:
                print(name, '->', mod.__class__.__name__)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        cache_dir=DIFFUSERS_CACHE_DIR,
        padding_side="left",
        truncation_side="right",
    )

    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    text_encoder.requires_grad_(False)

    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    transformer.requires_grad_(False)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    vae.requires_grad_(False)

    return tokenizer, text_encoder, transformer, scheduler, vae
