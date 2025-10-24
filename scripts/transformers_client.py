#!/usr/bin/env python3
"""Simple client for transformers pipeline server.

Usage:
    # After SSH tunnel: ssh -L 5000:localhost:5000 jakko@server
    python transformers_client.py image.jpg "Describe this image"
"""

import argparse
import base64
import sys
from pathlib import Path

import requests


def analyze_image(
    server_url: str,
    image_path: str,
    prompt: str = None,
    max_tokens: int = 200,
    use_base64: bool = True
):
    """Send image to server for analysis."""

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if use_base64:
        # JSON with base64 (works over network)
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{server_url}/generate",
            json={
                "image_base64": image_b64,
                "prompt": prompt,
                "max_tokens": max_tokens
            },
            headers={"Content-Type": "application/json"}
        )
    else:
        # Multipart form (file upload)
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"max_tokens": max_tokens}
            if prompt:
                data["prompt"] = prompt

            response = requests.post(
                f"{server_url}/generate",
                files=files,
                data=data
            )

    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Transformers Pipeline Client")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("prompt", nargs="?", help="Optional text prompt")
    parser.add_argument("--url", default="http://localhost:5000",
                        help="Server URL (default: http://localhost:5000)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens to generate")
    parser.add_argument("--multipart", action="store_true",
                        help="Use multipart upload instead of base64")

    args = parser.parse_args()

    try:
        # Health check
        print("Connecting to server...")
        health = requests.get(f"{args.url}/").json()
        print(f"✓ Server: {health['status']}")
        print(f"  Model: {health['model']}")
        print(f"  Loaded: {health['pipeline_loaded']}")
        print()

        # Analyze image
        print(f"Analyzing: {args.image}")
        if args.prompt:
            print(f"Prompt: {args.prompt}")
        print("\nWaiting for response (first request loads model ~30-60s)...")
        print()

        result = analyze_image(
            args.url,
            args.image,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            use_base64=not args.multipart
        )

        print("="*80)
        print("RESULT:")
        print("="*80)
        print(result["text"])
        print("="*80)

        return 0

    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to {args.url}", file=sys.stderr)
        print("\nMake sure:", file=sys.stderr)
        print("1. Server is running: python scripts/transformers_pipeline_server.py", file=sys.stderr)
        print("2. SSH tunnel is active: ssh -L 5000:localhost:5000 user@server", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
