from typing import Literal, Optional, Union, List

import yaml
from pathlib import Path

from pydantic import BaseModel, root_validator
import torch
import copy


ACTION_TYPES = Literal[
    "erase",
    "enhance",
]


class PromptEmbedsKlein:
    # Klein uses a single Qwen3 encoder with no pooled output.
    # encoder_hidden_states is the concatenation of hidden states
    # at layers 9, 18, and 27.
    encoder_hidden_states: torch.FloatTensor

    def __init__(self, encoder_hidden_states: torch.FloatTensor) -> None:
        self.encoder_hidden_states = encoder_hidden_states


# SDv1.x / SDv2.x use FloatTensor, Klein uses PromptEmbedsKlein
PROMPT_EMBEDDING = Union[torch.FloatTensor, PromptEmbedsKlein]


class PromptEmbedsCache:
    prompts: dict[str, PROMPT_EMBEDDING] = {}

    def __setitem__(self, __name: str, __value: PROMPT_EMBEDDING) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[PROMPT_EMBEDDING]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class PromptSettings(BaseModel):
    target: str
    positive: str = None
    unconditional: str = ""
    neutral: str = None
    action: ACTION_TYPES = "erase"
    guidance_scale: float = 3.5
    resolution: int = 512
    dynamic_resolution: bool = False
    batch_size: int = 1
    dynamic_crops: bool = False

    @root_validator(pre=True)
    def fill_prompts(cls, values):
        keys = values.keys()
        if "target" not in keys:
            raise ValueError("target must be specified")
        if "positive" not in keys:
            values["positive"] = values["target"]
        if "unconditional" not in keys:
            values["unconditional"] = ""
        if "neutral" not in keys:
            values["neutral"] = values["unconditional"]
        return values


class PromptEmbedsPair:
    target: PROMPT_EMBEDDING
    positive: PROMPT_EMBEDDING
    unconditional: PROMPT_EMBEDDING
    neutral: PROMPT_EMBEDDING

    guidance_scale: float
    resolution: int
    dynamic_resolution: bool
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: PROMPT_EMBEDDING,
        positive: PROMPT_EMBEDDING,
        unconditional: PROMPT_EMBEDDING,
        neutral: PROMPT_EMBEDDING,
        settings: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral

        self.guidance_scale = settings.guidance_scale
        self.resolution = settings.resolution
        self.dynamic_resolution = settings.dynamic_resolution
        self.batch_size = settings.batch_size
        self.dynamic_crops = settings.dynamic_crops
        self.action = settings.action

    def _erase(
        self,
        target_latents: torch.FloatTensor,
        positive_latents: torch.FloatTensor,
        unconditional_latents: torch.FloatTensor,
        neutral_latents: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.loss_fn(
            target_latents,
            neutral_latents
            - self.guidance_scale * (positive_latents - unconditional_latents)
        )

    def _enhance(
        self,
        target_latents: torch.FloatTensor,
        positive_latents: torch.FloatTensor,
        unconditional_latents: torch.FloatTensor,
        neutral_latents: torch.FloatTensor,
    ):
        return self.loss_fn(
            target_latents,
            neutral_latents
            + self.guidance_scale * (positive_latents - unconditional_latents)
        )

    def loss(self, **kwargs):
        if self.action == "erase":
            return self._erase(**kwargs)
        elif self.action == "enhance":
            return self._enhance(**kwargs)
        else:
            raise ValueError("action must be erase or enhance")


def load_prompts_from_yaml(path, attributes=[]):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    print(prompts)
    if len(prompts) == 0:
        raise ValueError("prompts file is empty")
    if len(attributes) != 0:
        newprompts = []
        for i in range(len(prompts)):
            for att in attributes:
                copy_ = copy.deepcopy(prompts[i])
                copy_['target'] = att + ' ' + copy_['target']
                copy_['positive'] = att + ' ' + copy_['positive']
                copy_['neutral'] = att + ' ' + copy_['neutral']
                copy_['unconditional'] = att + ' ' + copy_['unconditional']
                newprompts.append(copy_)
    else:
        newprompts = copy.deepcopy(prompts)

    print(newprompts)
    print(len(prompts), len(newprompts))
    prompt_settings = [PromptSettings(**prompt) for prompt in newprompts]

    return prompt_settings
