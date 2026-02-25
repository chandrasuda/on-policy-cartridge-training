#!/usr/bin/env python3
"""
End-to-end test for the Tokasaurus rollout bridge.

Tests the full flow: build payload → POST to live Tokasaurus → parse response.
Requires a running Tokasaurus server (e.g. on Modal).

Usage:
    # Basic test (no cartridge):
    python test_e2e.py --url https://YOUR-MODAL-URL

    # With a pre-trained cartridge from HuggingFace:
    python test_e2e.py --url https://YOUR-MODAL-URL \
        --cartridge-id hazyresearch/cartridge-wauoq23f \
        --cartridge-source huggingface

    # With a local cartridge file (must be accessible to Tokasaurus):
    python test_e2e.py --url https://YOUR-MODAL-URL \
        --cartridge-id /tmp/cartridge.pt \
        --cartridge-source local
"""

import argparse
import asyncio
import json
import sys
import time


async def test_ping(url: str) -> bool:
    """Test /ping endpoint."""
    import aiohttp

    print("1. Ping...", end=" ", flush=True)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.get(f"{url}/ping") as r:
                ok = r.status == 200
                print(f"{'✓' if ok else '✗'} (status={r.status})")
                return ok
    except Exception as e:
        print(f"✗ ({e})")
        return False


async def test_basic_completion(url: str) -> bool:
    """Test basic text completion (no cartridge)."""
    import aiohttp

    print("2. Basic completion (no cartridge)...", end=" ", flush=True)
    payload = {
        "model": "default",
        "prompt": "The capital of France is",
        "max_tokens": 32,
        "temperature": 0.0,
        "logprobs_in_fingerprint": True,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
            t0 = time.time()
            async with s.post(f"{url}/custom/cartridge/completions", json=payload) as r:
                dt = time.time() - t0
                data = await r.json()
                if r.status != 200:
                    print(f"✗ HTTP {r.status}: {await r.text()}")
                    return False
                fp = json.loads(data.get("system_fingerprint", "{}"))
                ids = fp.get("completion_ids", [[]])[0]
                text = data.get("choices", [{}])[0].get("text", "")
                print(f"✓ ({len(ids)} tokens, {dt:.1f}s)")
                print(f"   Response: {text[:100]!r}")
                return len(ids) > 0
    except Exception as e:
        print(f"✗ ({e})")
        return False


async def test_token_ids_prompt(url: str) -> bool:
    """Test with token IDs as prompt (how veRL sends requests)."""
    import aiohttp

    print("3. Token-ID prompt (veRL-style)...", end=" ", flush=True)
    payload = {
        "model": "default",
        "prompt": [1, 450, 7483, 310, 6181, 338],  # "The capital of France is"
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs_in_fingerprint": True,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
            async with s.post(f"{url}/custom/cartridge/completions", json=payload) as r:
                data = await r.json()
                if r.status != 200:
                    print(f"✗ HTTP {r.status}: {await r.text()}")
                    return False
                fp = json.loads(data.get("system_fingerprint", "{}"))
                ids = fp.get("completion_ids", [[]])[0]
                print(f"✓ ({len(ids)} tokens: {ids[:5]}...)")
                return len(ids) > 0
    except Exception as e:
        print(f"✗ ({e})")
        return False


async def test_logprobs(url: str) -> bool:
    """Test logprob extraction from system_fingerprint."""
    import aiohttp
    import base64
    import numpy as np

    print("4. Logprobs extraction...", end=" ", flush=True)
    payload = {
        "model": "default",
        "prompt": "Hello",
        "max_tokens": 8,
        "temperature": 0.0,
        "logprobs_in_fingerprint": True,
        "logprobs": 1,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
            async with s.post(f"{url}/custom/cartridge/completions", json=payload) as r:
                data = await r.json()
                if r.status != 200:
                    print(f"✗ HTTP {r.status}")
                    return False
                fp = json.loads(data.get("system_fingerprint", "{}"))
                packed = fp.get("packed_chosen_logprobs")
                if packed and packed[0]:
                    lp = np.frombuffer(base64.b64decode(packed[0]), dtype=np.float32)
                    print(f"✓ ({len(lp)} logprobs, range=[{lp.min():.2f}, {lp.max():.2f}])")
                    return len(lp) > 0
                else:
                    print("✗ (no logprobs in response)")
                    return False
    except Exception as e:
        print(f"✗ ({e})")
        return False


async def test_cartridge(url: str, cartridge_id: str, source: str) -> bool:
    """Test completion with cartridge injection."""
    import aiohttp

    print(f"5. Cartridge injection ({source}: {cartridge_id[:40]})...", end=" ", flush=True)
    payload = {
        "model": "default",
        "prompt": "What is the main contribution of this paper?",
        "max_tokens": 64,
        "temperature": 0.0,
        "logprobs_in_fingerprint": True,
        "cartridges": [{"id": cartridge_id, "source": source}],
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
            t0 = time.time()
            async with s.post(f"{url}/custom/cartridge/completions", json=payload) as r:
                dt = time.time() - t0
                data = await r.json()
                if r.status != 200:
                    print(f"✗ HTTP {r.status}: {(await r.text())[:200]}")
                    return False
                fp = json.loads(data.get("system_fingerprint", "{}"))
                ids = fp.get("completion_ids", [[]])[0]
                text = data.get("choices", [{}])[0].get("text", "")
                print(f"✓ ({len(ids)} tokens, {dt:.1f}s)")
                print(f"   Response: {text[:120]!r}")
                return len(ids) > 0
    except Exception as e:
        print(f"✗ ({e})")
        return False


async def test_bridge_class(url: str, cartridge_id: str = None, source: str = None) -> bool:
    """Test the actual TokasaurusHttpServer class (the Ray actor code)."""
    from unittest.mock import MagicMock, patch

    print("6. TokasaurusHttpServer.generate()...", end=" ", flush=True)
    try:
        from verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server import (
            TokasaurusHttpServer,
        )

        cfg = MagicMock()
        cfg.temperature = 0.0
        cfg.top_p = 1.0
        cfg.response_length = 32

        carts = []
        if cartridge_id:
            carts = [{"id": cartridge_id, "source": source or "local"}]

        with patch(
            "verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server.omega_conf_to_dataclass",
            return_value=cfg,
        ):
            server = TokasaurusHttpServer(
                config=cfg, tokasaurus_url=url, cartridges=carts, max_retries=2, base_timeout=60.0
            )

        result = await server.generate(
            prompt_ids=[1, 450, 7483],
            sampling_params={"temperature": 0.0, "logprobs": True},
            request_id="e2e-test-001",
        )

        print(f"✓ (TokenOutput: {len(result.token_ids)} tokens, logprobs={'yes' if result.log_probs else 'no'})")
        print(f"   token_ids[:5] = {result.token_ids[:5]}")
        return len(result.token_ids) > 0
    except Exception as e:
        print(f"✗ ({e})")
        return False


async def main():
    parser = argparse.ArgumentParser(description="E2E test for Tokasaurus rollout bridge")
    parser.add_argument("--url", required=True, help="Tokasaurus server URL")
    parser.add_argument("--cartridge-id", default=None, help="Cartridge ID (path or HF repo)")
    parser.add_argument("--cartridge-source", default="local", help="Cartridge source (local/huggingface/wandb/s3)")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    print(f"Tokasaurus E2E Test — {url}\n{'='*50}")

    results = []
    results.append(await test_ping(url))
    results.append(await test_basic_completion(url))
    results.append(await test_token_ids_prompt(url))
    results.append(await test_logprobs(url))

    if args.cartridge_id:
        results.append(await test_cartridge(url, args.cartridge_id, args.cartridge_source))

    results.append(await test_bridge_class(url, args.cartridge_id, args.cartridge_source))

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
