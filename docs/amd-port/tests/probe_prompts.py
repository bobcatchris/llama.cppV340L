#!/usr/bin/env python3
"""probe_prompts.py - Probe prompt generators for llama-server TP3 guards.

Provides standardized prompts for:
1. 2k context prefill probe (derived from probe_2k.txt retrospective prose).
2. 8k-10k context decode & acceptance probe (7,857 token prompt matching PLOG-101).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PROBE_2K_PATH = Path("/media/chris/ssd128/llamacpp/probe_2k.txt")
FALLBACK_REQ2K_PATH = Path("/home/chris/probe/req2k.json")
REQ10K_PATH = Path("/home/chris/probe/req10k.json")

UNIT_SENTENCE = "The lighthouse keeper logged every ship that passed the harbor. "


def get_2k_prompt() -> str:
    """Return the fixed ~2k prompt for prefill testing."""
    if PROBE_2K_PATH.is_file():
        try:
            content = PROBE_2K_PATH.read_text(encoding="utf-8", errors="replace").strip()
            return f"Summarize the key architectural and benchmark findings in the following technical text:\n\n{content}\n\nSummary:"
        except Exception:
            pass

    if FALLBACK_REQ2K_PATH.is_file():
        try:
            with open(FALLBACK_REQ2K_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "prompt" in data:
                    return data["prompt"]
        except Exception:
            pass

    # Deterministic synthetic fallback: ~2,075 tokens
    n = 175
    return f"Summarize the following text in one sentence. Text: {UNIT_SENTENCE * n}\nQuestion: What was logged?"


def get_10k_prompt() -> str:
    """Return the fixed 8k-10k prompt (~7,857 tokens) for decode testing."""
    if REQ10K_PATH.is_file():
        try:
            with open(REQ10K_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "prompt" in data:
                    return data["prompt"]
        except Exception:
            pass

    # Deterministic synthetic fallback matching ~7,857 tokens
    n = 655
    return f"Summarize the following text in one sentence. Text: {UNIT_SENTENCE * n}\nQuestion: How many records were logged?"


def get_determinism_prompt() -> str:
    """Return fixed greedy prompt for determinism verification."""
    return (
        "Explain the thermodynamic and orbital mechanics principles behind tidal locking "
        "in close-in exoplanetary systems. Include the role of gravitational torques and energy dissipation."
    )


def get_needle_prompt(depth: float, code: str, target_tokens: int = 8000) -> tuple[str, str]:
    """Generate a needle-in-a-haystack prompt at specified depth (e.g. 0.25, 0.50, 0.75).

    Returns (prompt, code).
    """
    needle_sentence = f" The secret verification code is {code}. Remember this code. "
    total_chars = int(target_tokens * 4.0)
    pre_chars = max(0, int(total_chars * depth) - len(needle_sentence))
    post_chars = max(0, total_chars - pre_chars - len(needle_sentence))

    pre_units = max(1, pre_chars // len(UNIT_SENTENCE))
    post_units = max(1, post_chars // len(UNIT_SENTENCE))

    haystack = (UNIT_SENTENCE * pre_units) + needle_sentence + (UNIT_SENTENCE * post_units)
    prompt = (
        f"Below is a long document with a hidden verification code.\n\n"
        f"{haystack}\n\n"
        f"Question: What is the 5-digit secret verification code mentioned in the text? "
        f"Answer with the 5-digit code only."
    )
    return prompt, code


if __name__ == "__main__":
    p2 = get_2k_prompt()
    p10 = get_10k_prompt()
    p_det = get_determinism_prompt()
    p_needle, c = get_needle_prompt(0.5, "48291", 8000)
    print(f"2k prompt: {len(p2)} chars")
    print(f"10k prompt: {len(p10)} chars")
    print(f"det prompt: {len(p_det)} chars")
    print(f"needle prompt (depth 0.5): {len(p_needle)} chars, expected code: {c}")

