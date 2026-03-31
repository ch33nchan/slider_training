"""
LoRA modules for Klein 9B transformer.

Adapted from SWHL flux-sliders LoRA implementation for Klein's architecture.
Targets double_blocks (img_attn, txt_attn) and single_blocks (linear1, linear2).
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from safetensors.torch import save_file


def init_ortho_proj(rank: int, weight: nn.Parameter, save_dir: Path):
    """
    Initialize orthogonal projection for lora_up weights.

    Uses QR decomposition on a random (out_dim, rank) matrix instead of full
    SVD on (out_dim, out_dim). This is critical for Klein 9B where out_dim can
    be 36864 (single_blocks.linear1) — full SVD would require a 36864x36864
    matrix (~5GB) and hits MKL bugs.
    """
    seed = torch.seed()
    torch.manual_seed(int(datetime.now().timestamp()))

    out_dim = weight.size(0)
    x = torch.randn(out_dim, rank, dtype=weight.dtype)
    q, _ = torch.linalg.qr(x)  # q is (out_dim, rank) with orthonormal columns

    torch.manual_seed(seed)
    return nn.Parameter(q)


class LoRAModule(nn.Module):
    """
    LoRA module that replaces the forward method of the target Linear layer.

    Forward: output = original(x) + lora_up(lora_down(x)) * multiplier * scale
    Training: only lora_down weights are trainable.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        save_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        self.lora_up.weight = init_ortho_proj(
            lora_dim, self.lora_up.weight, save_dir or Path(".")
        )
        self.lora_up.weight.requires_grad_(False)

        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        """Replace the original module's forward with LoRA forward."""
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    """
    LoRA network for Klein 9B transformer.

    Target modules for train_method="xattn":
    - double_blocks[0-7].img_attn.qkv   — Linear(4096, 12288)
    - double_blocks[0-7].img_attn.proj   — Linear(4096, 4096)
    - double_blocks[0-7].txt_attn.qkv   — Linear(4096, 12288)
    - double_blocks[0-7].txt_attn.proj   — Linear(4096, 4096)
    - single_blocks[0-23].linear1       — Linear(4096, 36864)
    - single_blocks[0-23].linear2       — Linear(16384, 4096)
    Total: 8×4 + 24×2 = 80 modules

    Usage as context manager:
        network.set_lora_slider(scale=2.0)
        with network:
            output = transformer(...)  # LoRA active
        # LoRA inactive (multiplier=0) outside context
    """

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        train_method: str = "xattn",
        save_dir: Optional[str] = None,
    ):
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        self.train_method = train_method
        self.save_dir = Path(save_dir) if save_dir else Path(".")

        self.loras = self._create_modules(transformer, rank, multiplier, train_method)
        print(f"Created LoRA for Klein 9B: {len(self.loras)} modules.")

        # Verify no duplicate names
        lora_names = set()
        for lora in self.loras:
            assert lora.lora_name not in lora_names, (
                f"Duplicate LoRA name: {lora.lora_name}"
            )
            lora_names.add(lora.lora_name)

        # Apply LoRA hooks and register modules
        for lora in self.loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        torch.cuda.empty_cache()

    def _create_modules(
        self,
        transformer: nn.Module,
        rank: int,
        multiplier: float,
        train_method: str,
    ) -> list:
        loras = []

        if train_method == "xattn":
            # Double blocks: attention layers
            for i in range(len(transformer.double_blocks)):
                block = transformer.double_blocks[i]
                targets = [
                    (f"double_blocks_{i}_img_attn_qkv", block.img_attn.qkv),
                    (f"double_blocks_{i}_img_attn_proj", block.img_attn.proj),
                    (f"double_blocks_{i}_txt_attn_qkv", block.txt_attn.qkv),
                    (f"double_blocks_{i}_txt_attn_proj", block.txt_attn.proj),
                ]
                for lora_name, module in targets:
                    loras.append(
                        LoRAModule(
                            lora_name, module, multiplier, rank,
                            self.alpha, save_dir=self.save_dir,
                        )
                    )

            # Single blocks: fused linear layers
            for i in range(len(transformer.single_blocks)):
                block = transformer.single_blocks[i]
                targets = [
                    (f"single_blocks_{i}_linear1", block.linear1),
                    (f"single_blocks_{i}_linear2", block.linear2),
                ]
                for lora_name, module in targets:
                    loras.append(
                        LoRAModule(
                            lora_name, module, multiplier, rank,
                            self.alpha, save_dir=self.save_dir,
                        )
                    )

        elif train_method == "full":
            for name, module in transformer.named_modules():
                if isinstance(module, nn.Linear):
                    lora_name = name.replace(".", "_")
                    loras.append(
                        LoRAModule(
                            lora_name, module, multiplier, rank,
                            self.alpha, save_dir=self.save_dir,
                        )
                    )
        else:
            raise NotImplementedError(f"train_method={train_method} not implemented")

        return loras

    def prepare_optimizer_params(self, train_lora_up: bool = False):
        """Return trainable LoRA parameters.

        By default only lora_down is trained (lora_up is a fixed orthogonal
        projection). Set train_lora_up=True to also optimise lora_up, giving
        the LoRA full low-rank expressivity at the cost of 2× parameters.
        """
        params = []
        for lora in self.loras:
            params.extend(lora.lora_down.parameters())
            if train_lora_up:
                lora.lora_up.weight.requires_grad_(True)
                params.extend(lora.lora_up.parameters())
        return [{"params": params}]

    def save_weights(self, file: str, dtype=None, metadata: Optional[dict] = None):
        """Save LoRA weights to safetensors."""
        state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            v = state_dict[key].detach().clone().to("cpu").contiguous()
            if dtype is not None:
                v = v.to(dtype)
            state_dict[key] = v
        save_file(state_dict, file, metadata or {})

    def set_lora_slider(self, scale: float):
        """Set the slider scale for inference."""
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.loras:
            lora.multiplier = 0
