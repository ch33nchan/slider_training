import torch
from diffusers import Flux2KleinPipeline

DIFFUSERS_CACHE_DIR = None


def load_klein_pipeline(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> Flux2KleinPipeline:
    """
    Load the full Flux2KleinPipeline.
    All components (tokenizer, text_encoder, transformer, vae, scheduler)
    are accessible via pipe.xxx.

    Use pipe.encode_prompt() for text encoding — it handles Qwen3 internally
    with the correct layer extraction.
    Use pipe.prepare_latents() for correct latent packing and latent_ids.
    """
    pipe = Flux2KleinPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe


# Keep the old signature for backward compat — just wraps load_klein_pipeline
def load_klein_models(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
):
    pipe = load_klein_pipeline(pretrained_model_name_or_path, weight_dtype)
    return pipe.tokenizer, pipe.text_encoder, pipe.transformer, pipe.scheduler, pipe.vae
