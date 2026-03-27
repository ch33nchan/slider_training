"""
inference_eye_gaze.py
---------------------
GazeSliderInference — MediaPipe iris detection + geometric warp to redirect
eye gaze in a portrait, with an optional Flux2Klein refinement pass to
clean up warp seams.

No LoRA required.  Works deterministically.

Usage:
    from inference_eye_gaze import GazeSliderInference
    engine = GazeSliderInference("black-forest-labs/FLUX.2-klein-9B")
    out = engine.apply_gaze(image, gaze_x=0.8, gaze_y=0.0, max_scale=5.0)
"""

import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    from diffusers import FluxPipeline as Flux2KleinPipeline   # fallback

# ── MediaPipe FaceMesh landmark indices ──────────────────────────────────────
# These are only available when refine_landmarks=True
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye outline — used to clamp the warp to the eye socket
LEFT_EYE_OUTLINE  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_OUTLINE = [33,  7,  163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]


# ── Core warp primitive ───────────────────────────────────────────────────────

def _shift_iris(img_np: np.ndarray,
                cx: float, cy: float, radius: float,
                dx: float, dy: float) -> np.ndarray:
    """
    Shift the iris disc centred at (cx, cy) with the given pixel radius
    by (dx, dy) pixels, with Gaussian feathering at the border.

    dx > 0 → iris moves right in the output image
    dy > 0 → iris moves down  in the output image
    """
    h, w  = img_np.shape[:2]
    Y, X  = np.mgrid[0:h, 0:w].astype(np.float32)
    dist  = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Weight: 1.0 inside iris, smooth fall-off to 0 at 1.9 × radius
    weight = np.zeros((h, w), dtype=np.float32)
    weight[dist <= radius] = 1.0
    fade = (dist > radius) & (dist < radius * 1.9)
    weight[fade] = 1.0 - (dist[fade] - radius) / (radius * 0.9)
    weight = cv2.GaussianBlur(weight, (15, 15), 4)

    # Inverse map: output pixel (x,y) ← source pixel (x-dx, y-dy)
    src_x = np.clip(X - dx, 0, w - 1)
    src_y = np.clip(Y - dy, 0, h - 1)

    shifted = cv2.remap(img_np, src_x, src_y,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    w3 = weight[:, :, None]
    return (shifted * w3 + img_np * (1.0 - w3)).astype(np.uint8)


# ── Inference class ───────────────────────────────────────────────────────────

class GazeSliderInference:
    """
    Redirect eye gaze using MediaPipe iris warping.

    Parameters accepted by __init__ match the old LoRA-based signature so the
    Gradio app works without changes; lora_h / lora_v / rank / alpha are
    silently ignored.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-9B",
        lora_h=None,          # ignored — kept for backward compat
        lora_v=None,          # ignored
        rank: int = 8,        # ignored
        alpha: float = 4.0,   # ignored
        train_method: str = "noxattn",  # ignored
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not HAS_MEDIAPIPE:
            raise RuntimeError(
                "mediapipe is required for iris detection.\n"
                "Install it with:  pip install mediapipe"
            )

        self.device = torch.device(device)
        self.dtype  = dtype

        print(f"[GazeWarp] Loading Flux2KleinPipeline from {model_id} …")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        print("[GazeWarp] Pipeline ready.")

        # MediaPipe face mesh (refine_landmarks=True gives iris landmarks 468–477)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
        print("[GazeWarp] MediaPipe FaceMesh ready.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _detect(self, img_np: np.ndarray):
        """Return MediaPipe face landmarks or None."""
        res = self._face_mesh.process(img_np)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0].landmark

    def _iris_params(self, lm, indices, h, w):
        """Return (cx, cy, radius) of an iris in pixel coords."""
        pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
        cx  = sum(p[0] for p in pts) / len(pts)
        cy  = sum(p[1] for p in pts) / len(pts)
        r   = max(
            max(abs(p[0] - cx) for p in pts),
            max(abs(p[1] - cy) for p in pts)
        ) * 1.5        # expand slightly to cover full iris
        return cx, cy, max(r, 4.0)

    # ── public API ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def apply_gaze(
        self,
        image: Image.Image,
        gaze_x: float = 0.0,   # +1 = look left,  −1 = look right
        gaze_y: float = 0.0,   # +1 = look up,    −1 = look down
        eye_open: float = 0.0,    # placeholder (no LoRA yet)
        brow_raise: float = 0.0,  # placeholder
        prompt: str = "professional portrait photograph, studio lighting, "
                      "photorealistic, sharp focus, high quality",
        strength: float = 0.20,   # 0 = warp only, >0 = Flux refinement
        num_inference_steps: int = 8,
        guidance_scale: float = 0.0,
        max_scale: float = 5.0,   # pixel shift = |gaze| × max_scale × 3
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Redirect gaze by warping the iris regions.

        gaze_x=+1 shifts both irises LEFT  (person looks left).
        gaze_x=-1 shifts both irises RIGHT (person looks right).
        gaze_y=+1 shifts both irises UP.
        gaze_y=-1 shifts both irises DOWN.

        strength>0 runs a low-noise Flux pass to smooth warp seams.
        """
        gaze_x   = max(-1.0, min(1.0, float(gaze_x)))
        gaze_y   = max(-1.0, min(1.0, float(gaze_y)))
        strength = max(0.0, min(1.0, float(strength)))

        orig_size = image.size
        img1024   = image.convert("RGB").resize((1024, 1024), Image.LANCZOS)
        img_np    = np.array(img1024)  # H×W×3 uint8

        # pixel displacement per joystick unit
        pixel_shift = max_scale * 3.0   # e.g. scale=5 → 15 px at gaze=1
        # +gaze_x (look left)  → iris moves left  → dx negative
        # +gaze_y (look up)    → iris moves up     → dy negative
        dx = -gaze_x * pixel_shift
        dy = -gaze_y * pixel_shift

        print(f"[GazeWarp] gaze=({gaze_x:+.2f},{gaze_y:+.2f})  "
              f"shift=({dx:+.1f},{dy:+.1f})px  refine_strength={strength:.2f}")

        # ── 1. detect irises ─────────────────────────────────────────────────
        lm = self._detect(img_np)
        if lm is None:
            print("[GazeWarp] ⚠ No face detected — returning original image")
            return image

        h, w = img_np.shape[:2]
        l_cx, l_cy, l_r = self._iris_params(lm, LEFT_IRIS,  h, w)
        r_cx, r_cy, r_r = self._iris_params(lm, RIGHT_IRIS, h, w)
        print(f"[GazeWarp] L iris=({l_cx:.0f},{l_cy:.0f}) r={l_r:.0f}px  "
              f"R iris=({r_cx:.0f},{r_cy:.0f}) r={r_r:.0f}px")

        if abs(dx) < 0.5 and abs(dy) < 0.5:
            print("[GazeWarp] Gaze near-zero, skipping warp.")
            return image

        # ── 2. warp both irises ──────────────────────────────────────────────
        warped = _shift_iris(img_np,  l_cx, l_cy, l_r, dx, dy)
        warped = _shift_iris(warped,  r_cx, r_cy, r_r, dx, dy)
        warped_pil = Image.fromarray(warped)

        # ── 3. optional Flux refinement ──────────────────────────────────────
        # Pass the warped image through Flux at low strength to smooth seams
        # while preserving the corrected iris position.
        if strength > 0.05:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # Build a partial sigma schedule
            img_t = self.pipe.image_processor.preprocess(warped_pil)
            img_t = img_t.to(self.device, self.dtype)
            vae_l = self.pipe.vae.encode(img_t).latent_dist.sample()
            sf    = getattr(self.pipe.vae.config, "scaling_factor",
                    getattr(self.pipe.vae.config, "scale_factor", 0.13025))
            vae_l = vae_l * sf
            spatial = F.pixel_unshuffle(vae_l, downscale_factor=2)  # [1,128,64,64]

            t      = float(strength)
            noise  = torch.randn(spatial.shape, device=self.device, dtype=self.dtype)
            if generator is not None:
                noise = torch.randn(spatial.shape, generator=generator,
                                    device=self.device, dtype=self.dtype)
            noisy  = (1.0 - t) * spatial + t * noise

            n_steps = max(2, int(num_inference_steps * t))
            sigmas  = torch.linspace(t, 0.001, n_steps + 1).tolist()

            refined = self.pipe(
                image=warped_pil,
                prompt=prompt,
                num_inference_steps=n_steps,
                sigmas=sigmas,
                latents=noisy,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            ).images[0]
            warped_pil = refined

        return warped_pil.resize(orig_size, Image.LANCZOS)

    def apply_gaze_batch(self, image, gaze_coords, **kwargs):
        return [self.apply_gaze(image, gx, gy, **kwargs) for gx, gy in gaze_coords]
