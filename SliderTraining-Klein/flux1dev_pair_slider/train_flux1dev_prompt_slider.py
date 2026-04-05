from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_template = script_dir / "config" / "horizontal_flux1dev_quick.yaml"
    default_output = script_dir / "outputs"
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-config", default=str(default_template))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--model-path", default="/mnt/data1/models/base-models/black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--slider-name", default="gaze_horizontal_flux1dev_quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    flux_repo = repo_root / "flux-sliders"
    if not flux_repo.exists():
        raise FileNotFoundError(f"Missing flux-sliders repo: {flux_repo}")

    template_path = Path(args.template_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(template_path)
    cfg.pretrained_model_name_or_path = args.model_path
    cfg.device = args.device
    cfg.max_train_steps = args.steps
    cfg.sample_every = args.sample_every
    cfg.output_dir = str(output_dir)
    cfg.slider_name = args.slider_name

    effective_cfg_path = output_dir / f"{args.slider_name}.yaml"
    OmegaConf.save(cfg, effective_cfg_path)

    command = [
        sys.executable,
        "-c",
        (
            "from flux_sliders.text_sliders import FLUXTextSliders; "
            f"FLUXTextSliders(r'{effective_cfg_path}').train()"
        ),
    ]
    subprocess.run(command, cwd=flux_repo, check=True)


if __name__ == "__main__":
    main()
