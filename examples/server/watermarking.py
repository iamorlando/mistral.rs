"""Generate, inspect sampling traces, compare watermarking, and detect output."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

SCHEMES = ("synthid", "kgw", "unigram", "exponential", "inverse_transform", "mpac")


def post(base_url, endpoint, body):
    request = Request(
        f"{base_url.rstrip('/')}/v1/{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        return json.load(response)


def generated_text(response, endpoint):
    if endpoint == "completions":
        return response["choices"][0]["text"]
    if endpoint == "chat/completions":
        return response["choices"][0]["message"]["content"]
    if endpoint == "messages":
        return "".join(
            block["text"] for block in response["content"] if block["type"] == "text"
        )
    return "".join(
        block["text"]
        for item in response["output"]
        if item["type"] == "message"
        for block in item["content"]
        if block["type"] == "output_text"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scheme", choices=SCHEMES)
    parser.add_argument(
        "--endpoint",
        choices=("chat/completions", "completions", "responses", "messages"),
        default="chat/completions",
    )
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--trace", action="store_true", help="Include bounded sampling diagnostics")
    parser.add_argument("--compare", action="store_true", help="Trace both an unmarked and marked run")
    parser.add_argument("--trace-steps", type=int, default=32)
    parser.add_argument("--trace-candidates", type=int, default=32)
    parser.add_argument("--trace-layers", type=int, default=8)
    parser.add_argument(
        "--vocab-size", type=int, help="Model output vocabulary, including padding"
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Detect the returned text using model retokenization",
    )
    args = parser.parse_args()
    if (args.trace or args.compare) and args.endpoint not in ("chat/completions", "completions"):
        parser.error("sampling traces require chat/completions or completions")
    config_path = (
        Path(__file__).resolve().parents[1] / "watermarking" / f"{args.scheme}.json"
    )
    config = json.loads(config_path.read_text())
    config["key"] = os.environ["MISTRALRS_WATERMARK_KEY"]
    if args.vocab_size is not None and "vocab_size" in config:
        config["vocab_size"] = args.vocab_size
    text = "Write a long story about a lunar garden."
    body = {
        "model": args.model,
        "temperature": 0.8,
        "top_k": 40,
        "enable_thinking": False,
        "watermark": config,
        "seed": args.seed,
    }
    if args.endpoint == "completions":
        body.update(prompt=text, max_tokens=args.max_tokens)
    elif args.endpoint == "responses":
        body.update(input=text, max_output_tokens=args.max_tokens)
    else:
        body.update(messages=[{"role": "user", "content": text}], max_tokens=args.max_tokens)
    if args.trace or args.compare:
        body["sampling_trace"] = {
            "max_steps": args.trace_steps,
            "max_candidates": args.trace_candidates,
            "max_layers": args.trace_layers,
        }
        body["logprobs"] = 10 if args.endpoint == "completions" else True
        if args.endpoint == "chat/completions":
            body["top_logprobs"] = 10
    if args.compare:
        baseline = {name: value for name, value in body.items() if name != "watermark"}
        print(json.dumps({"without_watermark": post(args.base_url, args.endpoint, baseline)}, indent=2))
    response = post(args.base_url, args.endpoint, body)
    print(json.dumps(response, indent=2))
    if args.detect:
        evidence = post(
            args.base_url,
            "watermark/detect",
            {
                "watermark": config,
                "input": {
                    "type": "text",
                    "model": args.model,
                    "text": generated_text(response, args.endpoint),
                },
            },
        )
        print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
