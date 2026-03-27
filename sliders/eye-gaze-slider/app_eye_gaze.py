"""
app_eye_gaze.py
---------------
Gradio app for interactive eye-gaze redirection using LivePortrait's
3D keypoint manipulation.  No LoRA required.

Controls:
  - 2D joystick  →  gaze direction (drag to aim)
  - Open / Close Eyes slider
  - Raise / Lower Eyebrows slider
  - Gaze Intensity  (LivePortrait eyeball_direction scale, default 12)

Run:
    python app_eye_gaze.py --port 7860
    python app_eye_gaze.py --port 7860 --server 0.0.0.0
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Lazy engine load
# ---------------------------------------------------------------------------
_engine = None


def get_engine(device: str = "cuda"):
    global _engine
    if _engine is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from inference_eye_gaze import GazeSliderInference
        _engine = GazeSliderInference(device=device)
    return _engine


# ===========================================================================
# 2-D joystick HTML + JS
# ===========================================================================

JOYSTICK_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;gap:6px;user-select:none;">
  <canvas id="gaze-joystick"
    width="220" height="220"
    style="cursor:crosshair;border-radius:10px;display:block;touch-action:none;background:#1a1a1a;">
  </canvas>
  <div style="color:#888;font-size:11px;letter-spacing:0.04em;text-align:center;">
    DRAG to set gaze direction
  </div>
</div>
"""

JOYSTICK_INIT_JS = """
() => {
  function initJoystick() {
    const canvas = document.getElementById('gaze-joystick');
    if (!canvas) { setTimeout(initJoystick, 80); return; }
    if (canvas._gazeInit) return;
    canvas._gazeInit = true;

    const W = 220, H = 220, CX = W/2, CY = H/2;
    const RADIUS = CX - 16;
    let dotX = CX, dotY = CY, dragging = false;
    const ctx = canvas.getContext('2d');

    function rrect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r, y); ctx.lineTo(x+w-r, y);
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
  if (typeof window.resetGazeJoystick === 'function') window.resetGazeJoystick();
  return [0, 0];
}
"""

# ===========================================================================
# Inference callback
# ===========================================================================

def run_inference(input_image, gaze_x, gaze_y, eye_open, brow_raise, intensity, device):
    if input_image is None:
        return None, "⚠ Upload a portrait first."
    try:
        engine = get_engine(device)
        from PIL import Image as PILImage
        if not isinstance(input_image, PILImage.Image):
            input_image = PILImage.fromarray(input_image)

        out = engine.apply_gaze(
            image=input_image,
            gaze_x=float(gaze_x),
            gaze_y=float(gaze_y),
            eye_open=float(eye_open),
            brow_raise=float(brow_raise),
            max_scale=float(intensity),
        )
        status = (f"✓  gaze ({float(gaze_x):+.2f}, {float(gaze_y):+.2f})  "
                  f"eye_open={float(eye_open):+.2f}  brow={float(brow_raise):+.2f}  "
                  f"intensity={float(intensity):.0f}")
        return out, status
    except Exception as e:
        import traceback
        return None, f"❌ {e}\n{traceback.format_exc()}"


# ===========================================================================
# Build UI
# ===========================================================================

def build_app(device: str = "cuda"):

    def _infer(img, gx, gy, eye_open, brow_raise, intensity):
        return run_inference(img, gx, gy, eye_open, brow_raise, intensity, device)

    with gr.Blocks(title="Eye Gaze — LivePortrait") as demo:

        gr.Markdown("## 👁  Eye Gaze & Expressions  —  LivePortrait")

        with gr.Row():

            # ── INPUT ────────────────────────────────────────────────────────
            with gr.Column(scale=1):
                input_img = gr.Image(label="Input Portrait", type="pil", height=480)

            # ── CONTROLS ─────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):

                with gr.Tabs():
                    with gr.Tab("👁 Eye"):

                        gr.HTML(JOYSTICK_HTML)

                        # Hidden number inputs written by JS
                        gaze_x_val = gr.Number(
                            value=0.0, elem_id="gaze-x-val", label="gaze_x",
                            elem_classes=["gaze-hidden"],
                        )
                        gaze_y_val = gr.Number(
                            value=0.0, elem_id="gaze-y-val", label="gaze_y",
                            elem_classes=["gaze-hidden"],
                        )

                        eye_open_sl = gr.Slider(
                            label="Open / Close Eyes",
                            minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                            info="-1 = close  ·  0 = neutral  ·  +1 = wide open",
                        )
                        brow_raise_sl = gr.Slider(
                            label="Raise / Lower Eyebrows",
                            minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                            info="-1 = lower  ·  0 = neutral  ·  +1 = raise",
                        )
                        intensity_sl = gr.Slider(
                            label="Gaze Intensity",
                            minimum=2.0, maximum=20.0, step=1.0, value=12.0,
                            info="Higher = more extreme gaze shift (12 recommended)",
                        )

                        with gr.Row():
                            reset_btn = gr.Button("↺  Reset", variant="secondary")
                            apply_btn = gr.Button("▶  Apply", variant="primary")

                    with gr.Tab("🗣 Head", interactive=False):
                        gr.Markdown("*Head pose — coming soon*")
                    with gr.Tab("👄 Mouth", interactive=False):
                        gr.Markdown("*Mouth controls — coming soon*")

                status_box = gr.Textbox(
                    label="Status", value="Idle", interactive=False,
                    lines=1, elem_id="status-box",
                )

            # ── OUTPUT ───────────────────────────────────────────────────────
            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="Output", type="pil", height=480, interactive=False,
                )

        # ── Wiring ───────────────────────────────────────────────────────────
        apply_btn.click(
            fn=_infer,
            inputs=[input_img, gaze_x_val, gaze_y_val,
                    eye_open_sl, brow_raise_sl, intensity_sl],
            outputs=[output_img, status_box],
        )
        reset_btn.click(
            fn=lambda: (0.0, 0.0),
            inputs=[],
            outputs=[gaze_x_val, gaze_y_val],
            js=RESET_JS,
        )
        gaze_x_val.change(fn=None, inputs=[], outputs=[])
        gaze_y_val.change(fn=None, inputs=[], outputs=[])

        demo.load(fn=None, js=JOYSTICK_INIT_JS)

    return demo


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--server", default="0.0.0.0")
    p.add_argument("--share",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_app(device=args.device)
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
          .gaze-hidden {
            position: absolute !important;
            width: 1px !important; height: 1px !important;
            opacity: 0 !important; pointer-events: none !important;
            overflow: hidden !important;
          }
        """,
    )
