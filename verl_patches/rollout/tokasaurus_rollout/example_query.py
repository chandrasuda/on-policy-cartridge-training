"""
Example: query the Tokasaurus model endpoint on Modal.

Usage:
    python example_query.py
"""

import requests, json, time

URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"

print(f"Querying {URL}")
print("(First request may take ~2 min if the container is cold-starting...)\n")


# --- 1. Plain completion (no cartridge) ---
print("1. Plain completion (no cartridge)...")
t0 = time.time()
resp = requests.post(
    f"{URL}/custom/cartridge/completions",
    json={
        "model": "default",
        "prompt": "Can you tell me about the patients?",
        "max_tokens": 256,
        "temperature": 0.7,
    },
    timeout=300,
)
dt = time.time() - t0
data = resp.json()
print(f"   ✓ {dt:.1f}s — {data['choices'][0]['text'][:120]}")


# --- 2. Completion WITH a pre-made KV cache (cartridge) ---
print("\n2. Completion with HuggingFace cartridge (KV cache injection)...")
print("   (First call downloads the cartridge — may take a moment)")
t0 = time.time()
resp = requests.post(
    f"{URL}/custom/cartridge/completions",
    json={
        "model": "default",
        "prompt": "Can you tell me about the patients?",
        "max_tokens": 128,
        "temperature": 0.0,
        "cartridges": [{
            "id": "hazyresearch/cartridge-wauoq23f",
            "source": "huggingface",
            "force_redownload": False,
        }],
    },
    timeout=300,
)
dt = time.time() - t0
data = resp.json()
if resp.status_code == 200:
    print(f"   ✓ {dt:.1f}s — {data['choices'][0]['text'][:200]}")
else:
    print(f"   ✗ HTTP {resp.status_code}: {resp.text[:200]}")
