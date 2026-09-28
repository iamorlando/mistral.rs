#!/usr/bin/env python3
"""Check tensor copy sizes and shared-record reads under LLDB, outside timing runs."""

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
    parser.add_argument("--filter", choices=["top1", "topk40_p90"], default="topk40_p90")
    parser.add_argument("--dispatches", action="store_true", help="also count keyed GPU kernel dispatches")
    parser.add_argument("--model", type=Path, help="audit CLI decode using this local model instead")
    parser.add_argument("--legacy", action="store_true", help="audit the default CLI sampler")
    args = parser.parse_args()
    source = args.candle_source / "src/metal_backend/mod.rs"
    lines = source.read_text().splitlines()
    line = next(i for i, value in enumerate(lines, 1) if "fn to_cpu<T: Clone>" in value) + 1
    keyed_source = Path(__file__).resolve().parents[1] / "mistralrs-keyed-rng/src/metal.rs"
    keyed_line = next(i for i, value in enumerate(keyed_source.read_text().splitlines(), 1)
                      if "fn readback_boundary(" in value) + 1
    environment = (
        "KEYED_BENCH_AUDIT=1 KEYED_BENCH_VOCAB=32768 KEYED_BENCH_REPEATS=1 KEYED_BENCH_STEPS=4 "
        f"KEYED_BENCH_BATCH={args.batch} KEYED_BENCH_FILTER={args.filter}"
    )
    run = "benchmark_keyed_sampling --ignored --nocapture --test-threads=1"
    if args.model:
        model_config = json.loads((args.model / "config.json").read_text())
        layers = model_config["num_hidden_layers"]
        environment = "NO_COLOR=1"
        run = (f"bench -m {json.dumps(str(args.model.resolve()))} --dtype f32 --format plain "
               f"--token-source none --paged-attn off --device-layers {layers} --prompt-len 0 "
               "--depth 128 --gen-len 8 --iterations 1 --warmup 1")
        run += " --sampling-rng " + ("isaac64" if args.legacy else "keyed-threefry2x32-v1")
        if args.batch != 1:
            run += f" --batch-size {args.batch}"
    dispatch_commands = ""
    if args.dispatches:
        pipeline_line = next(i for i, value in enumerate(keyed_source.read_text().splitlines(), 1)
                             if value.startswith("fn pipeline(")) + 1
        dispatch_commands = f"""breakpoint set --file {json.dumps(str(keyed_source))} --line {pipeline_line}
breakpoint command add -s python 3
name = frame.FindVariable("name")
if not name.GetChildMemberWithName("length").GetValueAsUnsigned():
    name = frame.GetThread().GetFrameAtIndex(frame.GetFrameID() + 1).FindVariable("name")
pointer = name.GetChildMemberWithName("data_ptr").GetValueAsUnsigned()
length = name.GetChildMemberWithName("length").GetValueAsUnsigned()
error = lldb.SBError()
data = frame.GetThread().GetProcess().ReadMemory(pointer, length, error)
print("KEYED_DISPATCH " + (data.decode() if error.Success() else "UNAVAILABLE"), flush=True)
return False
DONE
"""
    commands = f"""target create {json.dumps(str(args.binary.resolve()))}
settings set target.env-vars {environment}
breakpoint set --file {json.dumps(str(source.resolve()))} --line {line}
breakpoint command add -s python 1
print("CANDLE_METAL_READBACK elements=" + str(frame.FindVariable("self").Dereference().GetChildMemberWithName("count").GetValueAsUnsigned()), flush=True)
return False
DONE
breakpoint set --file {json.dumps(str(keyed_source))} --line {keyed_line}
breakpoint command add 2
script print("KEYED_SHARED_READBACK", flush=True)
continue
DONE
{dispatch_commands}run {run}
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
    tensor_elements = {}
    dispatches = {}
    active = None
    for line in result.stdout.splitlines():
        if args.model and "Iteration 1/1..." in line:
            active = "model_legacy" if args.legacy else "model_keyed"
            counts[active] = 0
            kinds[active] = {"candle_copy": 0, "shared_records": 0}
            tensor_elements[active] = []
            dispatches[active] = {}
        elif args.model and "Benchmark Results" in line:
            active = None
        begin = re.search(r"AUDIT_BEGIN mode=(\w+)", line)
        if begin:
            active = begin.group(1)
            counts[active] = 0
            kinds[active] = {"candle_copy": 0, "shared_records": 0}
            tensor_elements[active] = []
            dispatches[active] = {}
        elif "AUDIT_END" in line:
            active = None
        elif active and line.startswith("KEYED_DISPATCH "):
            name = line.split(" ", 1)[1]
            dispatches[active][name] = dispatches[active].get(name, 0) + 1
        elif active and (line.startswith("CANDLE_METAL_READBACK elements=") or line.strip() == "KEYED_SHARED_READBACK"):
            counts[active] += 1
            kind = "candle_copy" if line.startswith("CANDLE_METAL_READBACK") else "shared_records"
            kinds[active][kind] += 1
            if kind == "candle_copy":
                tensor_elements[active].append(int(line.split("elements=")[1]))
    expected = {
        "cpu_legacy": 0,
        "cpu_keyed": 0,
        "readback_cpu_legacy": 4,
        "readback_cpu_keyed": 4,
        "metal_legacy": 4,
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
        if kinds[mode]["candle_copy"]:
            if args.model:
                expected_elements = 4 if args.batch == 1 else model_config["vocab_size"] * args.batch
            elif mode == "metal_legacy" and args.batch == 1:
                expected_elements = 4 if args.filter == "top1" else 82
            else:
                expected_elements = 32768 * args.batch
            if tensor_elements[mode] != [expected_elements] * count:
                raise SystemExit(f"Unexpected tensor readback size for {mode}: {tensor_elements[mode]}")
    if args.dispatches and args.batch == 1 and (args.model or args.filter == "top1"):
        for mode in ("model_keyed",) if args.model else ("metal_compact", "metal_queued"):
            if mode not in dispatches:
                continue
            steps = 8 if args.model else 4
            expected_dispatches = {"argmax_tiles": steps, "argmax_finish_commit": steps}
            if args.model:
                expected_dispatches["history_init"] = 1
            if dispatches[mode] != expected_dispatches:
                raise SystemExit(f"Unexpected sampling dispatches for {mode}: {dispatches[mode]}")
    if args.dispatches and args.model and args.batch > 1 and not args.legacy:
        expected_dispatches = {"history_init": args.batch, "logits_tiles": 8, "logits_finish": 8, "history_commit": 8 * args.batch}
        if dispatches["model_keyed"] != expected_dispatches:
            raise SystemExit(f"Unexpected batched sampling dispatches: {dispatches['model_keyed']}")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "batch": args.batch,
                "steps": 8 if args.model else 4,
                "readbacks": counts,
                "kinds": kinds,
                "tensor_elements": tensor_elements,
                "filter": "top1" if args.model else args.filter,
                "dispatches": dispatches if args.dispatches else None,
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
