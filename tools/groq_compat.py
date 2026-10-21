#!/usr/bin/env python3
"""Minimal Groq/OpenAI-compatible chat wrapper for experimental use.

This repo is built around Claude Code and the .claude command system. This script is
not a drop-in replacement for that workflow. It is a lightweight compatibility shim
for running a Groq-compatible model from the terminal when you want to test a
non-Claude backend.

Example:
    export GROQ_API_KEY="..."
    python3 tools/groq_compat.py --prompt "Hello" --model "llama-3.1-8b-instant"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def _http_json(method: str, url: str, body: dict[str, Any], api_key: str) -> Any:
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API request failed ({exc.code}): {body_text}") from exc


def chat(prompt: str, *, model: str | None = None, system: str | None = None, temperature: float = 0.2) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before using this script.")

    model_name = model or DEFAULT_MODEL
    body: dict[str, Any] = {
        "model": model_name,
        "messages": [],
        "temperature": temperature,
    }

    if system:
        body["messages"].append({"role": "system", "content": system})

    body["messages"].append({"role": "user", "content": prompt})

    data = _http_json("POST", f"{DEFAULT_BASE_URL}/chat/completions", body, api_key)

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Groq response format: {data}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Groq-compatible chat wrapper")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the model")
    parser.add_argument("--system", default=None, help="System prompt to use")
    parser.add_argument("--model", default=None, help="Groq model name, default: GROQ_MODEL or llama-3.1-8b-instant")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        reply = chat(args.prompt, model=args.model, system=args.system, temperature=args.temperature)
        print(reply)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
