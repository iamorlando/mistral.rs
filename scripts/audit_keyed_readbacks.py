#!/usr/bin/env python3
"""Count tensor copies and shared-record reads under LLDB, outside timing runs."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="mistralrs-core test executable")
    parser.add_argument("candle_source", type=Path, help="pinned candle-core source directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=[1, 8], default=1)
    parser.add_argument("--model", type=Path, help="audit CLI decode using this local model instead")
    parser.add_argument("--legacy", action="store_true", help="audit the default CLI sampler")
    args = parser.parse_args()
    source = args.candle_source / "src/metal_backend/mod.rs"
    lines = source.read_text().splitlines()
    line = next(i for i, value in enumerate(lines, 1) if "fn to_cpu<T: Clone>" in value) + 1
    keyed_source = Path(__file__).resolve().parents[1] / "mistralrs-keyed-rng/src/metal.rs"
    keyed_line = next(i for i, value in enumerate(keyed_source.read_text().splitlines(), 1)
                      if "fn readback_boundary(" in value) + 1
    environment = f"KEYED_BENCH_AUDIT=1 KEYED_BENCH_QUICK=1 KEYED_BENCH_REPEATS=1 KEYED_BENCH_STEPS=4 KEYED_BENCH_BATCH={args.batch}"
    run = "benchmark_keyed_sampling --ignored --nocapture --test-threads=1"
    if args.model:
        layers = json.loads((args.model / "config.json").read_text())["num_hidden_layers"]
        environment = "MISTRALRS_SAMPLING_RNG=" + ("legacy" if args.legacy else "keyed-threefry2x32-v1")
        run = (f"bench -m {json.dumps(str(args.model.resolve()))} --dtype f32 --format plain "
               f"--token-source none --paged-attn off --device-layers {layers} --prompt-len 0 "
               "--depth 128 --gen-len 8 --iterations 1 --warmup 1")
    commands = f"""target create {json.dumps(str(args.binary.resolve()))}
settings set target.env-vars {environment}
breakpoint set --file {json.dumps(str(source.resolve()))} --line {line}
breakpoint command add 1
script print("CANDLE_METAL_READBACK", flush=True)
continue
DONE
breakpoint set --file {json.dumps(str(keyed_source))} --line {keyed_line}
breakpoint command add 2
script print("KEYED_SHARED_READBACK", flush=True)
continue
DONE
run {run}
breakpoint list
quit
"""
    with tempfile.TemporaryDirectory(prefix="keyed-readback-audit-") as temporary:
        command_file = Path(temporary) / "audit.lldb"
        command_file.write_text(commands)
        result = subprocess.run(
            ["lldb", "--batch", "--source", str(command_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(line.rstrip() for line in result.stdout.splitlines()) + "\n")
    success = "Benchmark Results" if args.model else "test result: ok"
    if result.returncode or success not in result.stdout:
        raise SystemExit(f"LLDB audit failed; inspect {args.output}")
    counts = {}
    kinds = {}
    active = None
    for line in result.stdout.splitlines():
        if args.model and "Iteration 1/1..." in line:
            active = "model_legacy" if args.legacy else "model_keyed"
            counts[active] = 0
            kinds[active] = {"candle_copy": 0, "shared_records": 0}
        elif args.model and "Benchmark Results" in line:
            active = None
        begin = re.search(r"AUDIT_BEGIN mode=(\w+)", line)
        if begin:
            active = begin.group(1)
            counts[active] = 0
            kinds[active] = {"candle_copy": 0, "shared_records": 0}
        elif "AUDIT_END" in line:
            active = None
        elif active and line.strip() in ("CANDLE_METAL_READBACK", "KEYED_SHARED_READBACK"):
            counts[active] += 1
            kind = "candle_copy" if line.strip() == "CANDLE_METAL_READBACK" else "shared_records"
            kinds[active][kind] += 1
    expected = {
        "cpu_legacy": 0,
        "cpu_keyed": 0,
        "readback_cpu_legacy": 4,
        "readback_cpu_keyed": 4,
        "metal_compact": 4,
        "metal_queued": 0,
    }
    if args.model:
        expected = {"model_legacy" if args.legacy else "model_keyed": 8}
    print(json.dumps(counts, indent=2))
    if counts != expected:
        raise SystemExit(f"Unexpected readbacks: expected {expected}; inspect {args.output}")
    for mode, count in expected.items():
        shared = mode in ("metal_compact", "model_keyed")
        expected_kinds = {"candle_copy": 0 if shared else count,
                          "shared_records": count if shared else 0}
        if kinds[mode] != expected_kinds:
            raise SystemExit(f"Unexpected readback type for {mode}: {kinds[mode]}")
    args.output.with_suffix(".json").write_text(
        json.dumps({"batch": args.batch, "steps": 8 if args.model else 4, "readbacks": counts, "kinds": kinds}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
