"""Send a watermark configuration to any supported HTTP generation endpoint."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

SCHEMES = ("synthid", "kgw", "unigram", "exponential", "inverse_transform", "mpac")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scheme", choices=SCHEMES)
    parser.add_argument(
        "--endpoint",
        choices=("chat/completions", "completions", "responses", "messages"),
        default="chat/completions",
    )
    parser.add_argument("--base-url", default="http://localhost:1234")
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "watermarking" / f"{args.scheme}.json"
    config = json.loads(config_path.read_text())
    config["key"] = os.environ["MISTRALRS_WATERMARK_KEY"]
    text = "Write a long story about a lunar garden."
    body = {
        "model": "Qwen/Qwen3-4B",
        "temperature": 0.8,
        "top_k": 40,
        "enable_thinking": False,
        "watermark": config,
    }
    if args.endpoint == "completions":
        body.update(prompt=text, max_tokens=512)
    elif args.endpoint == "responses":
        body.update(input=text, max_output_tokens=512)
    else:
        body.update(messages=[{"role": "user", "content": text}], max_tokens=512)
    request = Request(
        f"{args.base_url}/v1/{args.endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
