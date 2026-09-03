from __future__ import annotations

import argparse
import json

from colon3r.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/colon3r_10k.pt")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/colon3r.yaml")
    parser.add_argument("--vggt-path")
    parser.add_argument("--export-ply", action="store_true")
    arguments = parser.parse_args()
    result = run_inference(
        input_values=arguments.input,
        checkpoint=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        output_directory=arguments.output,
        config_path=arguments.config,
        vggt_path=arguments.vggt_path,
        export_ply=arguments.export_ply,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
