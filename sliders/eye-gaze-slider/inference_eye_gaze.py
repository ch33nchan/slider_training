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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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
        lora_h=None,
        lora_v=None,
        rank: int = 8,
        alpha: float = 4.0,
        train_method: str = "noxattn",
        device: str = "cuda",
        dtype=None,
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

        # ── Optional FLUX refinement ─────────────────────────────────────────
        self._flux_ready = False
        if lora_h or lora_v:
            self._setup_flux(model_id, lora_h, lora_v, rank, alpha,
                             train_method, device, dtype)

    # ── FLUX setup ────────────────────────────────────────────────────────────

    def _setup_flux(self, model_id, lora_h, lora_v, rank, alpha,
                    train_method, device, dtype):
        """Load FLUX pipeline + LoRA networks for refinement pass."""
        print("[GazeWarp] Loading FLUX pipeline for refinement …")
        _flux_root = _HERE.parents[1]  # slider_training/
        sys.path.insert(0, str(_flux_root / "flux-sliders"))
        from utils.lora import LoRANetwork  # noqa

        try:
            from diffusers import Flux2KleinPipeline as _FP
        except ImportError:
            from diffusers import FluxPipeline as _FP

        self._flux_dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = _FP.from_pretrained(model_id,
                                   torch_dtype=self._flux_dtype)
        pipe.to(device)
        self._transformer = pipe.transformer
        self._transformer.requires_grad_(False)
        self._vae = pipe.vae
        self._vae.requires_grad_(False)
        self._vae.eval()
        self._flux_pipe   = pipe          # keep alive for encode_text
        self._flux_device = torch.device(device)

        # VAE scale / shift
        _cfg = dict(self._vae.config)
        self._vae_scale = _cfg.get("scaling_factor", 0.18215)
        self._vae_shift = _cfg.get("shift_factor",  0.0)

        # Static geometry for 512×512 images
        self._spatial  = 32   # 512 // 16
        self._img_ids  = self._prepare_img_ids(1, 32, 32, device,
                                               self._flux_dtype)

        # Neutral text embedding (cached)
        self._seq_emb, self._pooled_emb, self._txt_ids = \
            self._encode_text("portrait photograph", device, self._flux_dtype)

        # LoRA networks
        self._net_h: Optional[object] = None
        self._net_v: Optional[object] = None

        for path, axis in [(lora_h, "H"), (lora_v, "V")]:
            if not path:
                continue
            net = LoRANetwork(
                self._transformer,
                rank=rank,
                multiplier=0.0,
                alpha=alpha,
                train_method=train_method,
            ).to(device=device, dtype=self._flux_dtype)
            from safetensors.torch import load_file
            state = load_file(path)
            net.load_state_dict(state, strict=False)
            net.eval()
            if axis == "H":
                self._net_h = net
            else:
                self._net_v = net
            print(f"[GazeWarp] LoRA-{axis} loaded from {path}")

        self._flux_ready = True
        print("[GazeWarp] FLUX refinement ready.")

    @staticmethod
    def _prepare_img_ids(B, H, W, device, dtype):
        h_idx  = torch.arange(H, device=device, dtype=dtype)
        w_idx  = torch.arange(W, device=device, dtype=dtype)
        gh, gw = torch.meshgrid(h_idx, w_idx, indexing="ij")
        ids    = torch.zeros(B, H * W, 4, device=device, dtype=dtype)
        ids[:, :, 2] = gh.reshape(-1).unsqueeze(0).expand(B, -1)
        ids[:, :, 3] = gw.reshape(-1).unsqueeze(0).expand(B, -1)
        return ids

    @torch.no_grad()
    def _encode_text(self, prompt, device, dtype):
        import inspect
        sig = inspect.signature(self._flux_pipe.encode_prompt).parameters
        kw  = dict(prompt=prompt, device=device, num_images_per_prompt=1)
        if "prompt_2" in sig:
            kw["prompt_2"] = None
        result = self._flux_pipe.encode_prompt(**kw)
        seq, pooled = result[0], result[1]
        txt_ids = result[2] if len(result) > 2 else None
        seq     = seq.to(dtype)
        pooled  = pooled.to(dtype) if pooled is not None else None
        if txt_ids is None:
            txt_ids = torch.zeros(seq.shape[0], seq.shape[1], 4,
                                  device=device, dtype=dtype)
        else:
            txt_ids = txt_ids.to(dtype)
        return seq, pooled, txt_ids

    @staticmethod
    def _call_transformer(transformer, x_packed, seq_emb, pooled_emb,
                          t_norm, img_ids, txt_ids):
        import inspect
        sig = inspect.signature(transformer.forward).parameters
        kw: dict = {"hidden_states": x_packed, "return_dict": False}
        if "encoder_hidden_states" in sig:
            kw["encoder_hidden_states"] = seq_emb
        if "timestep" in sig:
            kw["timestep"] = t_norm
        if pooled_emb is not None and "pooled_projections" in sig:
            kw["pooled_projections"] = pooled_emb
        if img_ids is not None and "img_ids" in sig:
            kw["img_ids"] = img_ids
        if txt_ids is not None and "txt_ids" in sig:
            kw["txt_ids"] = txt_ids
        return transformer(**kw)[0]

    @torch.no_grad()
    def _refine_with_flux(self, img: Image.Image, gaze_x: float,
                          gaze_y: float, strength: float) -> Image.Image:
        """FLUX img2img refinement with gaze LoRAs active."""
        if strength < 0.01 or not self._flux_ready:
            return img

        dev, dtype = self._flux_device, self._flux_dtype

        # ── Encode image → latents ─────────────────────────────────────────
        pv = TF.to_tensor(img.convert("RGB").resize((512, 512),
                          Image.LANCZOS)).unsqueeze(0).to(dev, dtype)
        pv = pv * 2.0 - 1.0
        z0 = self._vae.encode(pv).latent_dist.sample()
        z0 = (z0 - self._vae_shift) * self._vae_scale
        z0 = F.pixel_unshuffle(z0, 2)   # [1,128,32,32]

        # ── Add noise at t=strength ────────────────────────────────────────
        eps = torch.randn_like(z0)
        t   = float(strength)
        z_t = (1.0 - t) * z0 + t * eps

        # ── Euler denoising with LoRAs ─────────────────────────────────────
        n_steps = max(1, round(t * 8))
        dt      = t / n_steps
        t_cur   = t

        # Activate LoRAs with gaze direction as scale
        if self._net_h is not None:
            self._net_h.set_lora_slider(scale=float(gaze_x))
        if self._net_v is not None:
            self._net_v.set_lora_slider(scale=float(gaze_y))

        def _step(z):
            nonlocal t_cur
            t_norm   = torch.tensor([t_cur], device=dev, dtype=dtype)
            x_packed = z.reshape(1, 128, -1).permute(0, 2, 1)  # [1,HW,C]
            vel_packed = self._call_transformer(
                self._transformer, x_packed,
                self._seq_emb, self._pooled_emb,
                t_norm, self._img_ids, self._txt_ids,
            )
            vel = vel_packed.permute(0, 2, 1).reshape_as(z)
            t_cur -= dt
            return z - dt * vel

        if self._net_h is not None and self._net_v is not None:
            with self._net_h:
                with self._net_v:
                    for _ in range(n_steps):
                        z_t = _step(z_t)
        elif self._net_h is not None:
            with self._net_h:
                for _ in range(n_steps):
                    z_t = _step(z_t)
        elif self._net_v is not None:
            with self._net_v:
                for _ in range(n_steps):
                    z_t = _step(z_t)

        # ── Decode ────────────────────────────────────────────────────────
        z_dec = F.pixel_shuffle(z_t, 2)        # [1,32,64,64]
        z_dec = z_dec / self._vae_scale + self._vae_shift
        out_t = self._vae.decode(z_dec).sample  # [1,3,512,512]
        out_t = (out_t.clamp(-1, 1) + 1.0) / 2.0
        out_np = (out_t[0].permute(1, 2, 0).cpu().float().numpy() * 255
                  ).astype(np.uint8)
        # Resize back to original input dimensions
        return Image.fromarray(out_np).resize(img.size, Image.LANCZOS)

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
            result = Image.fromarray(out_blended)
        else:
            result = out_blended  # already PIL in some LP versions

        # Optional FLUX refinement pass (sharpens skin, amplifies gaze drama)
        if strength and strength > 0.01 and self._flux_ready:
            result = self._refine_with_flux(result, gaze_x, gaze_y, strength)

        return result

    def apply_gaze_batch(self, image, gaze_coords, **kwargs):
        return [self.apply_gaze(image, gx, gy, **kwargs) for gx, gy in gaze_coords]

    def __del__(self):
        # Clean up temp file
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except Exception:
                pass
