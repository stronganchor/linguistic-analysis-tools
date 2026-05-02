#!/usr/bin/env python3
import argparse
import json
import os
import urllib.request
from pathlib import Path


API_BASE = "https://datacollective.mozillafoundation.org/api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Mozilla Data Collective dataset via the authenticated API")
    parser.add_argument("--dataset-id", required=True, help="Mozilla Data Collective dataset id")
    parser.add_argument("--output-path", required=True, help="Where the downloaded archive should be saved")
    parser.add_argument(
        "--api-key-env",
        default="MDC_API_KEY",
        help="Environment variable containing the Mozilla Data Collective API key",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env}")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        f"{API_BASE}/datasets/{args.dataset_id}/download",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as response:
        payload = json.load(response)

    download_url = payload["downloadUrl"]
    urllib.request.urlretrieve(download_url, output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".download.json")
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "metadata_path": str(meta_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
