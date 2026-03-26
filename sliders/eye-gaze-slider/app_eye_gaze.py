"""
app_eye_gaze.py
---------------
Gradio app for interactive eye-gaze redirection using Flux.2-klein + LoRA sliders.

UI layout (mirrors your LivePortrait screenshot):
  - Head / Eye / Mouth tabs  (only Eye is wired for now)
  - 2D joystick canvas  (HTML/JS, dark theme, blue dot)
  - Denoising Strength slider
  - LoRA Intensity slider   (maps joystick ±1 → LoRA ±intensity)
  - Reset + Apply buttons
  - Input image (left) | Output image (right)

Run:
    python app_eye_gaze.py \
        --lora_h models/eye_gaze_horizontal_rank4_alpha1.0/last.safetensors \
        --lora_v models/eye_gaze_vertical_rank4_alpha1.0/last.safetensors \
        --model_id black-forest-labs/FLUX.2-klein-9B \
        --port 7860
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Lazy import of inference engine so Gradio can start without blocking GPU
# ---------------------------------------------------------------------------
_engine = None


def get_engine(model_id: str, lora_h: str, lora_v: str,
               rank: int, alpha: float) -> "GazeSliderInference":  # noqa: F821
    global _engine
    if _engine is None:
        # Import here so startup is fast if models aren't ready yet
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from inference_eye_gaze import GazeSliderInference  # noqa
        _engine = GazeSliderInference(
            model_id=model_id,
            lora_h=lora_h if lora_h else None,
            lora_v=lora_v if lora_v else None,
            rank=rank,
            alpha=alpha,
        )
    return _engine


# ===========================================================================
# 2-D joystick HTML + JavaScript
# ===========================================================================

JOYSTICK_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;gap:6px;user-select:none;">
  <canvas id="gaze-joystick"
    width="220" height="220"
    style="cursor:crosshair;border-radius:10px;display:block;touch-action:none;background:#1a1a1a;">
  </canvas>
  <div style="color:#888;font-size:11px;letter-spacing:0.04em;text-align:center;">
    DRAG  to set eye gaze direction
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Joystick init JS — run via demo.load(js=...) so Gradio actually executes it.
# (Scripts in gr.HTML are set via innerHTML and browsers skip them silently.)
# ---------------------------------------------------------------------------
JOYSTICK_INIT_JS = """
() => {
  function initJoystick() {
    const canvas = document.getElementById('gaze-joystick');
    if (!canvas) { setTimeout(initJoystick, 80); return; }
    if (canvas._gazeInit) return;   // already initialised
    canvas._gazeInit = true;

    const W = 220, H = 220, CX = W/2, CY = H/2;
    const RADIUS = CX - 16;
    let dotX = CX, dotY = CY, dragging = false;
    const ctx = canvas.getContext('2d');

    function rrect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r, y);
      ctx.lineTo(x+w-r, y);
      ctx.arc(x+w-r, y+r,   r, -Math.PI/2, 0);
      ctx.lineTo(x+w, y+h-r);
      ctx.arc(x+w-r, y+h-r, r,  0,          Math.PI/2);
      ctx.lineTo(x+r, y+h);
      ctx.arc(x+r,   y+h-r, r,  Math.PI/2,  Math.PI);
      ctx.lineTo(x,   y+r);
      ctx.arc(x+r,   y+r,   r,  Math.PI,   -Math.PI/2);
      ctx.closePath();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#1a1a1a'; rrect(0,0,W,H,10); ctx.fill();
      ctx.strokeStyle = '#2e2e2e'; ctx.lineWidth = 1.5; rrect(1,1,W-2,H-2,9); ctx.stroke();
      ctx.setLineDash([4,7]); ctx.strokeStyle='#3a3a3a'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(CX,10); ctx.lineTo(CX,H-10); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(10,CY); ctx.lineTo(W-10,CY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#505050'; ctx.font='bold 13px sans-serif';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('↑',CX,13); ctx.fillText('↓',CX,H-13);
      ctx.fillText('←',13,CY); ctx.fillText('→',W-13,CY);
      const grd = ctx.createRadialGradient(dotX,dotY,0,dotX,dotY,22);
      grd.addColorStop(0,'rgba(59,130,246,0.35)'); grd.addColorStop(1,'rgba(59,130,246,0)');
      ctx.beginPath(); ctx.arc(dotX,dotY,22,0,Math.PI*2); ctx.fillStyle=grd; ctx.fill();
      ctx.beginPath(); ctx.arc(dotX,dotY,11,0,Math.PI*2);
      ctx.fillStyle='#3b82f6'; ctx.fill();
      ctx.strokeStyle='#93c5fd'; ctx.lineWidth=1.5; ctx.stroke();
      if (Math.hypot(dotX-CX,dotY-CY)>6) {
        ctx.beginPath(); ctx.arc(CX,CY,3,0,Math.PI*2); ctx.fillStyle='#444'; ctx.fill();
      }
    }

    function clientXY(e) { const t=e.touches?e.touches[0]:e; return {cx:t.clientX,cy:t.clientY}; }
    function clamp(rx,ry) {
      const d=Math.hypot(rx,ry);
      if(d>RADIUS){const f=RADIUS/d; return{rx:rx*f,ry:ry*f};}
      return{rx,ry};
    }
    function updateDot(e) {
      const r=canvas.getBoundingClientRect();
      const {cx,cy}=clientXY(e);
      const {rx,ry}=clamp((cx-r.left)*(W/r.width)-CX,(cy-r.top)*(H/r.height)-CY);
      dotX=CX+rx; dotY=CY+ry; draw();
    }
    function pushToGradio() {
      const nx=(dotX-CX)/RADIUS, ny=-(dotY-CY)/RADIUS;
      function setNum(id,v) {
        const w=document.getElementById(id); if(!w)return;
        const inp=w.querySelector('input[type="number"]'); if(!inp)return;
        Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value')
          .set.call(inp,v.toFixed(4));
        inp.dispatchEvent(new Event('input',{bubbles:true}));
        inp.dispatchEvent(new Event('change',{bubbles:true}));
      }
      setNum('gaze-x-val',nx); setNum('gaze-y-val',ny);
    }
    canvas.addEventListener('mousedown', e=>{dragging=true; updateDot(e);});
    canvas.addEventListener('mousemove', e=>{if(dragging)updateDot(e);});
    window.addEventListener('mouseup',   ()=>{if(dragging){dragging=false;pushToGradio();}});
    canvas.addEventListener('touchstart',e=>{e.preventDefault();dragging=true;updateDot(e);},{passive:false});
    canvas.addEventListener('touchmove', e=>{e.preventDefault();if(dragging)updateDot(e);},{passive:false});
    window.addEventListener('touchend',  ()=>{if(dragging){dragging=false;pushToGradio();}});
    window.resetGazeJoystick=()=>{dotX=CX;dotY=CY;draw();pushToGradio();};
    draw();
  }
  initJoystick();
}
"""

RESET_JS = """
() => {
  if (typeof window.resetGazeJoystick === 'function') {
    window.resetGazeJoystick();
  }
  return [0, 0];
}
"""

# ===========================================================================
# Gradio inference callback
# ===========================================================================

def run_inference(
    input_image,
    gaze_x: float,
    gaze_y: float,
    eye_open: float,
    brow_raise: float,
    strength: float,
    lora_intensity: float,
    num_steps: int,
    prompt: str,
    # these are injected by the app.launch closure:
    _model_id: str,
    _lora_h: str,
    _lora_v: str,
    _rank: int,
    _alpha: float,
):
    if input_image is None:
        return None, "⚠ Please upload an input image first."

    try:
        engine = get_engine(_model_id, _lora_h, _lora_v, _rank, _alpha)
        from PIL import Image as PILImage
        if not isinstance(input_image, PILImage.Image):
            input_image = PILImage.fromarray(input_image)

        out = engine.apply_gaze(
            image=input_image,
            gaze_x=float(gaze_x),
            gaze_y=float(gaze_y),
            eye_open=float(eye_open),
            brow_raise=float(brow_raise),
            prompt=prompt,
            strength=float(strength),
            num_inference_steps=int(num_steps),
            max_scale=float(lora_intensity),
            guidance_scale=0.0,
        )
        status = (
            f"✓ Done  |  gaze ({gaze_x:+.2f}, {gaze_y:+.2f})  "
            f"eye_open={eye_open:+.2f}  brow={brow_raise:+.2f}  "
            f"strength={strength:.2f}  intensity={lora_intensity:.1f}"
        )
        return out, status

    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"


# ===========================================================================
# Build Gradio UI
# ===========================================================================

def build_app(model_id, lora_h, lora_v, rank, alpha):

    # Wrap inference so CLI args are baked in without globals
    def _infer(img, gx, gy, eye_open, brow_raise, strength, intensity, steps, prompt):
        return run_inference(img, gx, gy, eye_open, brow_raise, strength, intensity,
                             steps, prompt, model_id, lora_h, lora_v, rank, alpha)

    with gr.Blocks(title="Eye Gaze Slider — FLUX.2-klein") as demo:

        gr.Markdown(
            "## 👁  Eye Gaze & Expressions  —  FLUX.2-klein",
            elem_id="title",
        )

        with gr.Row():
            # ---------------------------------------------------------------
            # LEFT — Input image
            # ---------------------------------------------------------------
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Input Portrait",
                    type="pil",
                    height=480,
                )

            # ---------------------------------------------------------------
            # CENTRE — Controls
            # ---------------------------------------------------------------
            with gr.Column(scale=1, min_width=280):

                # Tab bar (Head / Eye / Mouth  — only Eye is active)
                with gr.Tabs():
                    with gr.Tab("🗣 Head", interactive=False):
                        gr.Markdown("*Head pose controls — coming soon*")

                    with gr.Tab("👁 Eye"):

                        # 2-D joystick
                        gr.HTML(JOYSTICK_HTML)

                        # Hidden Number components that JS writes to
                        gaze_x_val = gr.Number(
                            value=0.0, visible=False, elem_id="gaze-x-val", label="gaze_x"
                        )
                        gaze_y_val = gr.Number(
                            value=0.0, visible=False, elem_id="gaze-y-val", label="gaze_y"
                        )

                        eye_open_sl = gr.Slider(
                            label="Open / Close Eyes",
                            minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                            info="-1 = close  ·  0 = neutral  ·  +1 = open",
                        )
                        brow_raise_sl = gr.Slider(
                            label="Raise / Lower Eyebrows",
                            minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                            info="-1 = lower  ·  0 = neutral  ·  +1 = raise",
                        )

                        gr.Markdown("---")

                        strength_sl = gr.Slider(
                            label="Denoising Strength",
                            minimum=0.15, maximum=0.85, step=0.05, value=0.45,
                            info="Lower = preserves more of original image",
                        )
                        intensity_sl = gr.Slider(
                            label="LoRA Intensity",
                            minimum=1.0, maximum=10.0, step=0.5, value=5.0,
                            info="Maps joystick ±1 → LoRA scale ±intensity",
                        )
                        steps_sl = gr.Slider(
                            label="Inference Steps",
                            minimum=4, maximum=50, step=1, value=8,
                            info="4–12 recommended for FLUX.2-klein (distilled)",
                        )

                        prompt_box = gr.Textbox(
                            label="Style Prompt",
                            value="professional portrait photograph, studio lighting, "
                                  "photorealistic, sharp focus, high quality",
                            lines=2,
                        )

                        with gr.Row():
                            reset_btn = gr.Button("Reset", variant="secondary")
                            apply_btn = gr.Button("Apply", variant="primary")

                    with gr.Tab("👄 Mouth", interactive=False):
                        gr.Markdown("*Mouth controls — coming soon*")

                # Status bar
                status_box = gr.Textbox(
                    label="Status", value="Idle", interactive=False,
                    lines=1, elem_id="status-box",
                )

            # ---------------------------------------------------------------
            # RIGHT — Output image
            # ---------------------------------------------------------------
            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="Output",
                    type="pil",
                    height=480,
                    interactive=False,
                )

        # -------------------------------------------------------------------
        # Event wiring
        # -------------------------------------------------------------------

        # Apply button → run inference
        apply_btn.click(
            fn=_infer,
            inputs=[
                input_img, gaze_x_val, gaze_y_val,
                eye_open_sl, brow_raise_sl,
                strength_sl, intensity_sl, steps_sl, prompt_box,
            ],
            outputs=[output_img, status_box],
        )

        # Reset button → JS recentres dot + zero hidden numbers
        reset_btn.click(
            fn=lambda: (0.0, 0.0),
            inputs=[],
            outputs=[gaze_x_val, gaze_y_val],
            js=RESET_JS,
        )

        # Optional: live preview on joystick release
        # (fires whenever gaze_x_val or gaze_y_val changes via JS)
        gaze_x_val.change(fn=None, inputs=[], outputs=[])  # placeholder
        gaze_y_val.change(fn=None, inputs=[], outputs=[])  # placeholder

        # ---- Initialise joystick canvas via Gradio's JS hook ----
        # gr.HTML scripts are inserted via innerHTML and browsers skip them;
        # demo.load(js=...) is the correct way to run JS in Gradio 4.x.
        demo.load(fn=None, js=JOYSTICK_INIT_JS)

    return demo


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Eye Gaze Slider Gradio App")
    p.add_argument("--model_id",  default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--lora_h",    default="",  help="Path to horizontal gaze LoRA .safetensors")
    p.add_argument("--lora_v",    default="",  help="Path to vertical gaze LoRA .safetensors")
    p.add_argument("--rank",      type=int,   default=4)
    p.add_argument("--alpha",     type=float, default=1.0)
    p.add_argument("--port",      type=int,   default=7860)
    p.add_argument("--share",     action="store_true", help="Create public Gradio share link")
    p.add_argument("--server",    default="0.0.0.0", help="Server hostname (0.0.0.0 for GPU server)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_app(
        model_id=args.model_id,
        lora_h=args.lora_h or None,
        lora_v=args.lora_v or None,
        rank=args.rank,
        alpha=args.alpha,
    )
    demo.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="gray").set(
            body_background_fill="#111111",
            block_background_fill="#1c1c1c",
            block_border_color="#2a2a2a",
            input_background_fill="#222222",
            button_primary_background_fill="#3b82f6",
            button_primary_background_fill_hover="#2563eb",
        ),
        css="""
          .tab-nav button { font-size: 13px; padding: 6px 18px; }
          #status-box textarea { font-size: 12px; color: #aaa; }
          #gaze-x-val, #gaze-y-val { display: none !important; }
        """,
    )
