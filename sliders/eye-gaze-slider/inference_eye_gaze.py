"""
inference_eye_gaze.py
---------------------
GazeSliderInference — LivePortrait-based eye gaze redirection.

Uses LivePortrait's 3D keypoint manipulation (eyeball_direction_x/y) for
realistic holistic gaze control: iris + eyelid + sclera move together.

No LoRA required.  Works deterministically.

Usage:
    from inference_eye_gaze import GazeSliderInference
    engine = GazeSliderInference()
    out = engine.apply_gaze(image, gaze_x=0.8, gaze_y=0.0)

Convention (matches LivePortrait UI):
    gaze_x = +1  → look left    (eyeball_direction_x = +max_scale)
    gaze_x = -1  → look right   (eyeball_direction_x = -max_scale)
    gaze_y = +1  → look up      (eyeball_direction_y = -max_scale, sign flip)
    gaze_y = -1  → look down    (eyeball_direction_y = +max_scale)
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ── Find LivePortrait root ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

def _find_liveportrait() -> Path:
    """Walk up from here and find the LivePortrait directory."""
    candidates = [
        _HERE.parents[1] / "LivePortrait",   # sliders/../LivePortrait
        _HERE.parents[2] / "LivePortrait",   # sliders/../../LivePortrait
        Path.home() / "Desktop" / "slider training" / "LivePortrait",
    ]
    for c in candidates:
        if (c / "src" / "gradio_pipeline.py").exists():
            return c
    raise RuntimeError(
        "Cannot find LivePortrait directory. "
        "Expected it at ../LivePortrait relative to sliders/."
    )

_LP_ROOT = _find_liveportrait()
if str(_LP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LP_ROOT))

# ── Change cwd so LivePortrait's relative model paths resolve correctly ───────
_ORIG_CWD = os.getcwd()

from src.gradio_pipeline import GradioPipeline  # noqa: E402
from src.config.argument_config import ArgumentConfig  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.config.crop_config import CropConfig  # noqa: E402


# ── Inference class ───────────────────────────────────────────────────────────

class GazeSliderInference:
    """
    Redirect eye gaze in a portrait using LivePortrait's 3D keypoint system.

    Constructor signature intentionally matches the old LoRA-based version so
    that app_eye_gaze.py works without changes.  model_id / lora_h / lora_v /
    rank / alpha / train_method are silently ignored.
    """

    # LivePortrait eyeball_direction values that feel "full-range"
    # ±12 keeps gaze visible while minimising whole-face translation drift
    DEFAULT_MAX_SCALE: float = 12.0

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-9B",
        lora_h=None,          # ignored — kept for backward compat
        lora_v=None,          # ignored
        rank: int = 8,        # ignored
        alpha: float = 4.0,   # ignored
        train_method: str = "noxattn",  # ignored
        device: str = "cuda",
        dtype=None,           # ignored
    ):
        use_half = (device != "cpu")
        inference_cfg = InferenceConfig(flag_use_half_precision=use_half)
        crop_cfg      = CropConfig()
        args          = ArgumentConfig()

        # Run LivePortrait from its own root so internal relative paths resolve
        os.chdir(_LP_ROOT)
        print(f"[GazeWarp] Loading LivePortrait from {_LP_ROOT} …")
        self.pipeline = GradioPipeline(inference_cfg, crop_cfg, args)
        os.chdir(_ORIG_CWD)
        print("[GazeWarp] LivePortrait ready.")

        # Cached per-image state (avoids re-running face detection on repeats)
        self._cached_img_path: Optional[str] = None
        self._source_eye_ratio: float = 0.0
        self._source_lip_ratio: float = 0.0
        self._tmp_path: Optional[str] = None  # reusable temp file

    # ── internal helpers ──────────────────────────────────────────────────────

    def _img_to_tmp(self, image: Image.Image) -> str:
        """Save PIL image to a stable temp file, return path."""
        if self._tmp_path is None:
            fd, self._tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
        image.save(self._tmp_path)
        return self._tmp_path

    def _init_source(self, img_path: str, scale: float = 2.3):
        """Run face detection + ratio extraction (once per new image)."""
        # Use a stable hash to avoid re-running for the same pixel content
        import hashlib
        with open(img_path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if getattr(self, "_cached_hash", None) == h:
            return self._source_eye_ratio, self._source_lip_ratio

        os.chdir(_LP_ROOT)
        eye_r, lip_r = self.pipeline.init_retargeting_image(
            retargeting_source_scale=scale,
            source_eye_ratio=0.0,
            source_lip_ratio=0.0,
            input_image=img_path,
        )
        os.chdir(_ORIG_CWD)
        self._source_eye_ratio = eye_r
        self._source_lip_ratio = lip_r
        self._cached_hash = h
        print(f"[GazeWarp] Source ratios — eye={eye_r:.3f}  lip={lip_r:.3f}")
        return eye_r, lip_r

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def apply_gaze(
        self,
        image: Image.Image,
        gaze_x: float = 0.0,     # +1 = look left,  −1 = look right
        gaze_y: float = 0.0,     # +1 = look up,    −1 = look down
        eye_open: float = 0.0,   # +1 = wider open, −1 = squint/close
        brow_raise: float = 0.0, # +1 = raise brows, −1 = lower brows
        strength: float = 0.0,   # reserved for future Flux refinement pass
        max_scale: float = None,
        retargeting_source_scale: float = 2.3,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Return a new PIL image with the eye gaze redirected.

        gaze_x = +1  →  look left      gaze_x = -1  →  look right
        gaze_y = +1  →  look up        gaze_y = -1  →  look down
        eye_open = +1 → eyes wider     eye_open = -1 → eyes closed
        brow_raise = +1 → raise brows  brow_raise = -1 → lower brows
        """
        gaze_x = max(-1.0, min(1.0, float(gaze_x)))
        gaze_y = max(-1.0, min(1.0, float(gaze_y)))

        if max_scale is None:
            max_scale = self.DEFAULT_MAX_SCALE

        # Map joystick coords → LivePortrait eyeball_direction
        # Empirically determined sign convention (LP internal axes differ from ours):
        # gaze_x=+1 (look left)  → eye_dir_x = -max_scale
        # gaze_y=+1 (look up)    → eye_dir_y = +max_scale
        eye_dir_x = float(-gaze_x) * max_scale
        eye_dir_y = float(gaze_y) * max_scale

        print(f"[GazeWarp] gaze=({gaze_x:+.2f},{gaze_y:+.2f})  "
              f"LP_dir=({eye_dir_x:+.1f},{eye_dir_y:+.1f})")

        # Write input to temp file (LivePortrait needs a file path)
        img_path = self._img_to_tmp(image.convert("RGB"))

        # Init source (face detection + ratio extraction)
        eye_r, lip_r = self._init_source(img_path, scale=retargeting_source_scale)

        # eye_open slider: ±1 maps to ±0.15 delta on top of natural eye ratio
        # clamp to valid range [0.0, 0.5]
        eye_open  = max(-1.0, min(1.0, float(eye_open)))
        target_eye_ratio = float(np.clip(eye_r + eye_open * 0.15, 0.0, 0.5))

        # skip the whole pipeline if nothing will change
        no_gaze   = abs(eye_dir_x) < 0.05 and abs(eye_dir_y) < 0.05
        no_eye    = abs(target_eye_ratio - eye_r) < 0.005
        no_brow   = abs(brow_raise) < 0.02
        if no_gaze and no_eye and no_brow:
            print("[GazeWarp] All sliders near-zero — returning original.")
            return image

        print(f"[GazeWarp] eye_ratio {eye_r:.3f} → {target_eye_ratio:.3f}  "
              f"brow={brow_raise:+.2f}")

        # Run LivePortrait retargeting
        os.chdir(_LP_ROOT)
        try:
            _out_crop, out_blended = self.pipeline.execute_image_retargeting(
                input_eye_ratio=target_eye_ratio,
                input_lip_ratio=lip_r,
                input_head_pitch_variation=0,
                input_head_yaw_variation=0,
                input_head_roll_variation=0,
                mov_x=0.0,
                mov_y=0.0,
                mov_z=1.0,
                lip_variation_zero=0,
                lip_variation_one=0,
                lip_variation_two=0,
                lip_variation_three=0,
                smile=0,
                wink=0,
                eyebrow=float(brow_raise),
                eyeball_direction_x=eye_dir_x,
                eyeball_direction_y=eye_dir_y,
                input_image=img_path,
                retargeting_source_scale=retargeting_source_scale,
                flag_stitching_retargeting_input=True,
                flag_do_crop_input_retargeting_image=True,
            )
        finally:
            os.chdir(_ORIG_CWD)

        # Convert output to PIL (paste_back returns numpy uint8 HWC RGB)
        if isinstance(out_blended, np.ndarray):
            return Image.fromarray(out_blended)
        return out_blended  # already PIL in some LP versions

    def apply_gaze_batch(self, image, gaze_coords, **kwargs):
        return [self.apply_gaze(image, gx, gy, **kwargs) for gx, gy in gaze_coords]

    def __del__(self):
        # Clean up temp file
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except Exception:
                pass
