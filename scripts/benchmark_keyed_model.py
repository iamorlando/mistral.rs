#!/usr/bin/env python3
"""Compare complete greedy model decoding with CPU, default Metal, and keyed Metal."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    command = [
        str(args.binary.resolve()),
        "bench",
        "-m", str(args.model.resolve()),
        "--dtype", "f32",
        "--format", "plain",
        "--token-source", "none",
        "--paged-attn", "off",
        "--device-layers", str(config["num_hidden_layers"]),
        "--prompt-len", "0",
        "--depth", "128,512",
        "--gen-len", "128",
        "--iterations", "5",
        "--warmup", "2",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"features": args.features, "runs": []}
    for mode in ("cpu_legacy", "metal_legacy", "metal_keyed"):
        env = os.environ.copy()
        env.pop("MISTRALRS_SAMPLING_RNG", None)
        env["NO_COLOR"] = "1"
        if mode == "metal_keyed":
            env["MISTRALRS_SAMPLING_RNG"] = "keyed-threefry2x32-v1"
        run = command + (["--cpu"] if mode == "cpu_legacy" else [])
        print(f"START {mode}", flush=True)
        result = subprocess.run(
            run, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        (args.output / f"model-{mode}.log").write_text(result.stdout.rstrip() + "\n")
        if result.returncode:
            raise SystemExit(f"{mode} failed; inspect its log")
        rows = re.findall(
            r"Decode \(128 tokens @ d(\d+)\).*?(\d+\.\d+) \u00b1 "
            r"(\d+\.\d+).*?(\d+\.\d+) ms TPOT",
            result.stdout,
        )
        if len(rows) != 2:
            raise SystemExit(f"Missing decode results for {mode}")
        summary["runs"].append(
            {
                "mode": mode,
                "command": run,
                "results": [
                    {
                        "depth": int(d),
                        "tokens_per_second": float(t),
                        "spread": float(s),
                        "tpot_ms": float(ms),
                    }
                    for d, t, s, ms in rows
                ],
            }
        )
        print(f"FINISH {mode}: {rows}", flush=True)
    (args.output / "model-results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
