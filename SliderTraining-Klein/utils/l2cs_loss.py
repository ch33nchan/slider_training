from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError("Unsupported L2CS checkpoint format.")


def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _remap_official_l2cs_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("fc_yaw_gaze."):
            remapped[key.replace("fc_yaw_gaze.", "fc_yaw.")] = value
            continue
        if key.startswith("fc_pitch_gaze."):
            remapped[key.replace("fc_pitch_gaze.", "fc_pitch.")] = value
            continue
        if key.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
            remapped[f"features.{key}"] = value
            continue
    return remapped


class L2CSBackbone(nn.Module):
    def __init__(self, num_bins: int = 90) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc_yaw = nn.Linear(backbone.fc.in_features, num_bins)
        self.fc_pitch = nn.Linear(backbone.fc.in_features, num_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x).flatten(1)
        return self.fc_yaw(features), self.fc_pitch(features)


def load_l2cs_backbone(
    model_path: str,
    device: torch.device,
    num_bins: int = 90,
) -> nn.Module:
    checkpoint_path = Path(model_path)
    if checkpoint_path.suffix.lower() in {".pt", ".jit", ".torchscript"}:
        module = torch.jit.load(str(checkpoint_path), map_location=device)
        module.eval()
        return module

    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = _strip_prefixes(_unwrap_state_dict(payload))
    if "fc_yaw_gaze.weight" in state_dict or "fc_pitch_gaze.weight" in state_dict:
        state_dict = _remap_official_l2cs_keys(state_dict)
    model = L2CSBackbone(num_bins=num_bins)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("fc_finetune.")]
    if unexpected:
        raise ValueError(f"Unexpected keys in L2CS checkpoint: {unexpected}")
    if missing and not all(key.endswith(("num_batches_tracked",)) for key in missing):
        raise ValueError(f"Missing keys in L2CS checkpoint: {missing}")
    model.eval()
    return model.to(device)


class DifferentiableGazeLoss(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str,
        num_bins: int = 90,
        angle_min_deg: float = -42.0,
        angle_max_deg: float = 42.0,
    ) -> None:
        super().__init__()
        self.backbone = load_l2cs_backbone(model_path=model_path, device=torch.device(device), num_bins=num_bins)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.num_bins = num_bins
        self.angle_min_rad = torch.deg2rad(torch.tensor(angle_min_deg, dtype=torch.float32))
        self.angle_max_rad = torch.deg2rad(torch.tensor(angle_max_deg, dtype=torch.float32))
        self.register_buffer(
            "angle_bins",
            torch.linspace(self.angle_min_rad.item(), self.angle_max_rad.item(), num_bins, dtype=torch.float32),
            persistent=False,
        )

    def _decode_yaw(self, output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            if "yaw" in output:
                return output["yaw"].view(-1)
            if "yaw_rad" in output:
                return output["yaw_rad"].view(-1)
            if "logits_yaw" in output:
                output = output["logits_yaw"]
        if isinstance(output, (tuple, list)):
            yaw_logits = output[0]
        else:
            yaw_logits = output

        if yaw_logits.ndim == 1:
            yaw_logits = yaw_logits.unsqueeze(0)

        if yaw_logits.shape[-1] == 1:
            return yaw_logits.view(-1)

        if yaw_logits.shape[-1] != self.num_bins:
            raise ValueError(
                f"Expected yaw output dimension {self.num_bins} or 1, got {yaw_logits.shape[-1]}."
            )

        probs = F.softmax(yaw_logits.float(), dim=-1)
        return probs @ self.angle_bins

    def predict_yaw(self, face_crop: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(face_crop, size=(448, 448), mode="bilinear", align_corners=False)
        x = (x + 1.0) / 2.0
        x = (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)
        return self._decode_yaw(self.backbone(x))

    def forward(self, face_crop: torch.Tensor, target_yaw_rad: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(face_crop, size=(448, 448), mode="bilinear", align_corners=False)
        x = (x + 1.0) / 2.0
        x = (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)
        yaw_pred = self._decode_yaw(self.backbone(x))
        target = target_yaw_rad.to(device=yaw_pred.device, dtype=yaw_pred.dtype).view(-1)
        return F.l1_loss(yaw_pred, target)


def save_l2cs_metadata(path: str | Path, **kwargs: Any) -> None:
    Path(path).write_text(json.dumps(kwargs, indent=2, sort_keys=True))
