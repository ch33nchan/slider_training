from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(script_dir / "config" / "horizontal_flux1dev_pair.yaml"),
    )
    parser.add_argument("--flux-repo", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from flux1dev_pair_slider.pair_training import train_pair_slider

    train_pair_slider(config_path=args.config, flux_repo=args.flux_repo)


if __name__ == "__main__":
    main()
