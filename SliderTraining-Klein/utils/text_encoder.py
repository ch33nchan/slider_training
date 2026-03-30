"""
Text encoder utilities for Klein 9B slider training.
Uses Qwen3-8B to encode text prompts into embeddings.

Extracted from ostris ai-toolkit Flux2Pipeline._get_qwen_prompt_embeds.
"""

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange

OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512


def load_text_encoder(model_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load Qwen3-8B text encoder and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    return text_encoder, tokenizer


@torch.no_grad()
def encode_prompt(
    text_encoder,
    tokenizer,
    prompt,
    device: str = "cuda",
    max_length: int = MAX_LENGTH,
) -> Tensor:
    """
    Encode text prompt using Qwen3-8B.

    Returns: prompt_embeds (B, L, 12288) — stacked hidden states from layers 9, 18, 27.
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    all_input_ids = []
    all_attention_masks = []

    for p in prompt:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        all_input_ids.append(model_inputs["input_ids"])
        all_attention_masks.append(model_inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    out = torch.stack(
        [output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1
    )
    prompt_embeds = rearrange(out, "b c l d -> b l (c d)")

    return prompt_embeds
