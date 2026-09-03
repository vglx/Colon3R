from __future__ import annotations

import argparse

from colon3r.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colon3r.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--source-manifest")
    parser.add_argument("--target-manifest")
    parser.add_argument("--c3vd-fov-mask")
    parser.add_argument("--total-updates", type=int, default=6000)
    arguments = parser.parse_args()
    run_training(
        config_path=arguments.config,
        output_directory=arguments.output,
        resume=arguments.resume,
        source_manifest=arguments.source_manifest,
        target_manifest=arguments.target_manifest,
        c3vd_fov_mask=arguments.c3vd_fov_mask,
        total_updates=arguments.total_updates,
    )


if __name__ == "__main__":
    main()
