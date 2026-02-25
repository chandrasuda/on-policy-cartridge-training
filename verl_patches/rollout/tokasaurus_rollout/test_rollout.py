"""
Test the veRL Tokasaurus rollout end-to-end.

Creates a TokasaurusReplica (the same class veRL's trainer uses),
launches the Ray actor, and generates tokens through your Modal server.

Usage:
    python test_rollout.py
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import ray

from verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server import (
    TokasaurusHttpServer,
    TokasaurusReplica,
)
from verl.workers.rollout.replica import TokenOutput

URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"


async def main():
    ray.init(num_cpus=2)

    # Build config (same shape veRL's trainer passes)
    config = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        response_length=64,
        custom={
            "tokasaurus_url": URL,
            "cartridges": [],
        },
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    with patch(
        "verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server.omega_conf_to_dataclass",
        return_value=config,
    ):
        replica = TokasaurusReplica(
            replica_rank=0,
            config=config,
            model_config=SimpleNamespace(),
            gpus_per_node=1,
        )

    # This creates a Ray actor and pings the server
    print("Launching servers (pinging Modal — may cold-start)...")
    await replica.launch_servers()
    print(f"✓ Connected to {URL}")

    server = replica.servers[0]

    # --- 1. Generate without cartridge ---
    print("\n1. Generate without cartridge...")
    result = await server.generate.remote(
        prompt_ids=[1, 450, 7483, 310, 6181, 338],  # "The capital of France is"
        sampling_params={"temperature": 0.0, "max_new_tokens": 32},
        request_id="rollout-test-001",
    )
    print(f"   ✓ {len(result.token_ids)} tokens: {result.token_ids[:10]}...")
    print(f"   logprobs: {'yes' if result.log_probs else 'no'}")

    # --- 2. Sync cartridge, then generate ---
    print("\n2. Syncing cartridge to all servers...")
    await replica.sync_cartridge("hazyresearch/cartridge-wauoq23f")
    print("   ✓ Cartridge synced")

    print("\n3. Generate WITH cartridge...")
    result2 = await server.generate.remote(
        prompt_ids=[1],
        sampling_params={"temperature": 0.0, "logprobs": True, "max_new_tokens": 64},
        request_id="rollout-test-cartridge",
    )
    print(f"   ✓ {len(result2.token_ids)} tokens: {result2.token_ids[:10]}...")
    print(f"   logprobs: {'yes' if result2.log_probs else 'no'}")
    if result2.log_probs:
        print(f"   first 5 logprobs: {result2.log_probs[:5]}")

    ray.shutdown()
    print("\n✓ Rollout test passed!")


if __name__ == "__main__":
    asyncio.run(main())
