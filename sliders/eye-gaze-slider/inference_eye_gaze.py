"""
inference_eye_gaze.py
---------------------
GazeSliderInference — loads two trained LoRA sliders (horizontal + vertical)
on top of FLUX.2-klein-9B and applies them via img2img to redirect eye gaze
in an input portrait, while preserving skin texture and identity.

Two LoRAs applied simultaneously (both monkey-patched onto the same
transformer; they chain additively via org_forward):
  - network_h  :  left (scale > 0) ↔ right (scale < 0)
  - network_v  :  up   (scale > 0) ↔ down  (scale < 0)

Usage (standalone):
    from inference_eye_gaze import GazeSliderInference
    from PIL import Image

    engine = GazeSliderInference(
        model_id="black-forest-labs/FLUX.2-klein-9B",
        lora_h="models/eye_gaze_horizontal_rank4_alpha1.0/last.safetensors",
        lora_v="models/eye_gaze_vertical_rank4_alpha1.0/last.safetensors",
    )

    img_out = engine.apply_gaze(
        image=Image.open("portrait.jpg"),
        gaze_x=0.6,    # -1 (right) … +1 (left)
        gaze_y=0.0,    # -1 (down)  … +1 (up)
        strength=0.45, # how much the image is re-generated (0 = no change)
        prompt="professional portrait photograph, studio lighting, photorealistic",
        max_scale=5.0, # maps joystick ±1 to LoRA scale ±max_scale
    )
    img_out.save("output.jpg")
"""

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Path setup — reuse existing lora.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FLUX_UTILS = REPO_ROOT / "flux-sliders" / "utils"
sys.path.insert(0, str(FLUX_UTILS.parent))

from utils.lora import LoRANetwork  # noqa

# ---------------------------------------------------------------------------
# Diffusers imports — FluxImg2ImgPipeline requires diffusers >= 0.30
# ---------------------------------------------------------------------------
from diffusers import Flux2KleinPipeline


class GazeSliderInference:
    """
    Loads FLUX.2-klein-9B + two LoRA sliders and exposes a single
    `apply_gaze(image, gaze_x, gaze_y, ...)` method.

    The two LoRA networks are loaded onto the SAME transformer; they
    compose additively because each LoRAModule chains through org_forward.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (e.g. "black-forest-labs/FLUX.2-klein-9B")
    lora_h : str | Path
        Path to the horizontal gaze slider weights (.safetensors or .pt)
    lora_v : str | Path
        Path to the vertical gaze slider weights (.safetensors or .pt)
    rank : int
        LoRA rank used during training (must match saved weights)
    alpha : float
        LoRA alpha used during training (must match saved weights)
    train_method : str
        Layer filter used during training (must match)
    device : str
        "cuda" (H100) or "cpu"
    dtype : torch.dtype
        Inference dtype (bfloat16 recommended on H100)
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-9B",
        lora_h: Optional[Union[str, Path]] = None,
        lora_v: Optional[Union[str, Path]] = None,
        rank: int = 4,
        alpha: float = 1.0,
        train_method: str = "noxattn",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype  = dtype
        self.rank   = rank
        self.alpha  = alpha
        self.train_method = train_method

        print(f"[GazeSlider] Loading pipeline from {model_id} …")
        self._load_pipeline(model_id)

        # --- Attach LoRA sliders to the already-loaded transformer ---
        print("[GazeSlider] Attaching LoRA networks …")
        self.network_h = LoRANetwork(
            self.pipe.transformer,
            rank=rank, multiplier=0.0, alpha=alpha, train_method=train_method
        ).to(self.device).to(self.dtype)
        self.network_v = LoRANetwork(
            self.pipe.transformer,
            rank=rank, multiplier=0.0, alpha=alpha, train_method=train_method
        ).to(self.device).to(self.dtype)

        if lora_h:
            self._load_lora(self.network_h, lora_h, label="horizontal")
        else:
            print("[GazeSlider] ⚠ No horizontal LoRA path provided — H network randomly initialized")

        if lora_v:
            self._load_lora(self.network_v, lora_v, label="vertical")
        else:
            print("[GazeSlider] ⚠ No vertical LoRA path provided — V network randomly initialized")

        print("[GazeSlider] Ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self, model_id: str) -> None:
        """Load Flux2KleinPipeline — supports image+prompt img2img natively."""
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._img2img_mode = True
        print("[GazeSlider] Using Flux2KleinPipeline")

        pass  # VAE tiling left at default

    def _load_lora(self, network: LoRANetwork, path: Union[str, Path], label: str) -> None:
        """Load safetensors / pt weights into a LoRANetwork."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {path}")

        if path.suffix == ".safetensors":
            state = load_file(str(path), device=str(self.device))
        else:
            state = torch.load(str(path), map_location=self.device)

        missing, unexpected = network.load_state_dict(state, strict=False)
        if missing:
            print(f"[GazeSlider] ⚠ {label} LoRA — {len(missing)} missing keys")
        if unexpected:
            print(f"[GazeSlider] ⚠ {label} LoRA — {len(unexpected)} unexpected keys")
        print(f"[GazeSlider] ✓ Loaded {label} LoRA from {path.name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def apply_gaze(
        self,
        image: Image.Image,
        gaze_x: float = 0.0,
        gaze_y: float = 0.0,
        eye_open: float = 0.0,    # -1 close → +1 open  (placeholder, no LoRA yet)
        brow_raise: float = 0.0,  # -1 lower  → +1 raise (placeholder, no LoRA yet)
        prompt: str = "professional portrait photograph, studio lighting, "
                      "photorealistic, sharp focus, high quality",
        strength: float = 0.45,
        num_inference_steps: int = 8,
        guidance_scale: float = 0.0,
        max_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Redirect the eye gaze of a portrait image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input portrait (any size; will be resized to 1024×1024 internally)
        gaze_x : float
            Horizontal gaze offset, in [-1, +1].
            -1 = hard right,  0 = centre,  +1 = hard left
        gaze_y : float
            Vertical gaze offset, in [-1, +1].
            -1 = hard down,   0 = centre,  +1 = hard up
        prompt : str
            Describes the portrait style (NOT the gaze — that's handled by LoRA)
        strength : float
            Img2img strength.  0 = no change, 1 = full generation.
            Keep around 0.35–0.55 for natural-looking gaze edits.
        num_inference_steps : int
            Total FLUX denoising steps (before strength reduction).
        guidance_scale : float
            CFG scale. Use 0.0 for distilled models (FLUX.2-klein / schnell).
        max_scale : float
            LoRA scale that corresponds to joystick ±1.
            Higher = stronger effect; start at 5.0, increase if too subtle.
        seed : int | None
            Optional seed for reproducibility.

        Returns
        -------
        PIL.Image.Image — edited portrait
        """
        # --- Input validation ---
        gaze_x   = max(-1.0, min(1.0, float(gaze_x)))
        gaze_y   = max(-1.0, min(1.0, float(gaze_y)))
        strength = max(0.05, min(1.0, float(strength)))

        scale_h = gaze_x * max_scale   # +x joystick → left gaze
        scale_v = gaze_y * max_scale   # +y joystick → up   gaze

        print(f"[GazeSlider] scale_h={scale_h:+.3f}  scale_v={scale_v:+.3f}  "
              f"strength={strength:.2f}  steps={num_inference_steps}")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        orig_size     = image.size
        image_resized = image.convert("RGB").resize((1024, 1024), Image.LANCZOS)

        # ------------------------------------------------------------------
        # Diagnostic mode: skip custom latents/sigmas entirely and let the
        # pipeline run its default schedule.  This confirms whether the LoRA
        # itself produces a visible gaze change before we re-introduce img2img.
        # The `image` param still provides identity conditioning.
        # ------------------------------------------------------------------
        print(f"[GazeSlider] scale_h={scale_h:+.3f}  scale_v={scale_v:+.3f}  "
              f"strength={strength:.2f}  steps={num_inference_steps}")

        self.network_h.set_lora_slider(scale=scale_h)
        self.network_v.set_lora_slider(scale=scale_v)

        with self.network_h, self.network_v:
            result = self.pipe(
                image=image_resized,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            ).images[0]

        return result.resize(orig_size, Image.LANCZOS)

    def apply_gaze_batch(
        self,
        image: Image.Image,
        gaze_coords: list,
        **kwargs,
    ) -> list:
        """
        Convenience wrapper to generate multiple gaze directions in one call.

        Parameters
        ----------
        gaze_coords : list of (gaze_x, gaze_y) tuples
            e.g. [(-0.8, 0), (0, 0), (0.8, 0)]

        Returns
        -------
        list of PIL.Image.Image
        """
        return [
            self.apply_gaze(image, gaze_x=x, gaze_y=y, **kwargs)
            for x, y in gaze_coords
        ]
