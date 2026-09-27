#!/usr/bin/env python3
"""Count real Candle Metal readbacks under LLDB, outside performance measurements."""

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
    args = parser.parse_args()
    source = args.candle_source / "src/metal_backend/mod.rs"
    lines = source.read_text().splitlines()
    line = next(i for i, value in enumerate(lines, 1) if "fn to_cpu<T: Clone>" in value) + 1
    commands = f"""target create {json.dumps(str(args.binary.resolve()))}
settings set target.env-vars KEYED_BENCH_AUDIT=1 KEYED_BENCH_QUICK=1 KEYED_BENCH_REPEATS=1 KEYED_BENCH_STEPS=4 KEYED_BENCH_BATCH={args.batch}
breakpoint set --file {json.dumps(str(source.resolve()))} --line {line}
breakpoint command add 1
script print("CANDLE_METAL_READBACK", flush=True)
continue
DONE
run benchmark_keyed_sampling --ignored --nocapture --test-threads=1
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
    if result.returncode or "test result: ok" not in result.stdout:
        raise SystemExit(f"LLDB audit failed; inspect {args.output}")
    counts = {}
    active = None
    for line in result.stdout.splitlines():
        begin = re.search(r"AUDIT_BEGIN mode=(\w+)", line)
        if begin:
            active = begin.group(1)
            counts[active] = 0
        elif "AUDIT_END" in line:
            active = None
        elif active and line.strip() == "CANDLE_METAL_READBACK":
            counts[active] += 1
    expected = {
        "cpu_legacy": 0,
        "cpu_keyed": 0,
        "readback_cpu_legacy": 4,
        "readback_cpu_keyed": 4,
        "metal_compact": 4,
        "metal_queued": 0,
    }
    print(json.dumps(counts, indent=2))
    if counts != expected:
        raise SystemExit(f"Unexpected readbacks: expected {expected}; inspect {args.output}")
    args.output.with_suffix(".json").write_text(
        json.dumps({"batch": args.batch, "steps": 4, "readbacks": counts}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
