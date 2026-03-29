import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from safetensors.torch import save_file
from datetime import datetime

# FLUX.2-klein uses Flux2Attention and Flux2ParallelSelfAttention.
# There are no UNet-style ResNet/Conv blocks in this architecture.
TRANSFORMER_TARGET_REPLACE_MODULE = [
    "Attention",
    "Flux2Attention",
    "Flux2ParallelSelfAttention",
]

LORA_PREFIX_TRANSFORMER = "lora_transformer"

DEFAULT_TARGET_REPLACE = TRANSFORMER_TARGET_REPLACE_MODULE

# Note: several FLUX.1-dev train_methods (xattn-up, xattn-down, xattn-mid,
# noxattn, selfattn) relied on UNet block path naming (up_block, down_block,
# mid_block, attn1, attn2). Klein's transformer uses single_transformer_blocks
# and transformer_blocks — those names do not appear in module paths.
# For Klein, only "xattn" (all attention) and "full" (all layers) are supported.
# Run the following on the server to confirm attention module path names:
#   for name, mod in transformer.named_modules():
#       if mod.__class__.__name__ in ['Flux2Attention', 'Flux2ParallelSelfAttention']:
#           print(name)
TRAINING_METHODS = Literal[
    "xattn",   # train all attention modules
    "full",    # train all layers
]


def load_ortho_dict(n):
    base = os.path.expanduser('~/orthogonal_basis')
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f'{n:09}.ckpt')
    if os.path.isfile(path):
        try:
            return torch.load(path, weights_only=True)
        except Exception:
            os.remove(path)
    x = torch.randn(n, n, dtype=torch.float32)
    eig, _ = torch.linalg.qr(x)
    torch.save(eig, path)
    return eig


def init_ortho_proj(rank, weight):
    seed = torch.seed()
    torch.manual_seed(datetime.now().timestamp())
    q_index = torch.randint(high=weight.size(0), size=(rank,))
    torch.manual_seed(seed)
    ortho_q_init = load_ortho_dict(weight.size(0)).to(dtype=weight.dtype)[:, q_index]
    return nn.Parameter(ortho_q_init)


class LoRAModule(nn.Module):
    """
    Wraps the forward method of a Linear layer with a low-rank adapter.
    The original module's forward is replaced; the adapter adds
    lora_up(lora_down(x)) * multiplier * scale on top.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        train_method='xattn'
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        elif "Conv" in org_module.__class__.__name__:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        if train_method == 'full':
            nn.init.zeros_(self.lora_up.weight)
        else:
            self.lora_up.weight = init_ortho_proj(lora_dim, self.lora_up.weight)
            self.lora_up.weight.requires_grad_(False)

        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        train_method: TRAINING_METHODS = "full",
        layers: List[str] = ['Linear'],
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        self.train_method = train_method
        self.module = LoRAModule

        self.unet_loras = self.create_modules(
            LORA_PREFIX_TRANSFORMER,
            transformer,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
            layers=layers,
        )
        print(f"created LoRA for transformer: {len(self.unet_loras)} modules.")

        lora_names = set()
        for lora in self.unet_loras:
            assert lora.lora_name not in lora_names, \
                f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        del transformer
        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
        layers: List[str],
    ) -> list:
        filt_layers = []
        if 'Linear' in layers:
            filt_layers.extend(["Linear", "LoRACompatibleLinear"])
        if 'Conv' in layers:
            filt_layers.extend(["Conv2d", "LoRACompatibleConv"])

        loras = []
        names = []

        for name, module in root_module.named_modules():
            if train_method == "xattn":
                # Accept any module whose path contains "attn" as a substring.
                # Verify this matches Klein's attention paths on the server
                # by running: for n, m in transformer.named_modules(): print(n)
                if "attn" not in name and module.__class__.__name__ not in target_replace_modules:
                    continue
            elif train_method == "full":
                pass
            else:
                raise NotImplementedError(f"train_method '{train_method}' not supported for Klein.")

            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in filt_layers:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        lora = self.module(lora_name, child_module, multiplier, rank, self.alpha, train_method)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)

        return loras

    def prepare_optimizer_params(self):
        all_params = []
        if self.unet_loras:
            params = []
            if self.train_method == 'full':
                [params.extend(lora.parameters()) for lora in self.unet_loras]
            else:
                [params.extend(lora.lora_down.parameters()) for lora in self.unet_loras]
            all_params.append({"params": params})
        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key].detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
