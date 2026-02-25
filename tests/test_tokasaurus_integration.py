# Copyright 2025 - Tokasaurus veRL Integration
# Licensed under the Apache License, Version 2.0
"""
Integration test: veRL rollout bridge → live Tokasaurus on Modal.

Tests the ACTUAL veRL class hierarchy end-to-end:
  RolloutConfig → TokasaurusReplica → TokasaurusHttpServer (Ray actor) → Modal Tokasaurus → TokenOutput

Run:
    pytest tests/workers/rollout/test_tokasaurus_integration.py -v -s

Requirements:
    - Modal Tokasaurus server running at TOKASAURUS_URL
    - Ray (installed in venv)
    - No GPU needed (HTTP proxy only)
"""

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import ray

from verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server import (
    TokasaurusHttpServer,
    TokasaurusReplica,
)
from verl.workers.rollout.replica import TokenOutput, RolloutReplicaRegistry

# Default URL — override with TOKASAURUS_URL env var
TOKASAURUS_URL = os.getenv(
    "TOKASAURUS_URL",
    "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run",
)

# Skip all tests if server is unreachable
_server_reachable = None


def _check_server():
    global _server_reachable
    if _server_reachable is not None:
        return _server_reachable
    try:
        import requests
        r = requests.get(f"{TOKASAURUS_URL}/ping", timeout=180)
        _server_reachable = r.status_code == 200
    except Exception:
        _server_reachable = False
    return _server_reachable


requires_server = pytest.mark.skipif(
    not _check_server(),
    reason=f"Tokasaurus server not reachable at {TOKASAURUS_URL}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_rollout_config(**custom_overrides):
    """Build a mock RolloutConfig with Tokasaurus custom fields."""
    custom = {
        "tokasaurus_url": TOKASAURUS_URL,
        "cartridges": [],
        **custom_overrides,
    }
    cfg = MagicMock()
    cfg.temperature = 0.0
    cfg.top_p = 1.0
    cfg.response_length = 64
    cfg.custom = custom
    cfg.tensor_model_parallel_size = 1
    cfg.data_parallel_size = 1
    cfg.pipeline_model_parallel_size = 1
    return cfg


def _make_server(**kw):
    """Create a TokasaurusHttpServer (non-Ray, for direct testing)."""
    cfg = _mock_rollout_config(**{k: v for k, v in kw.items() if k in ("cartridges",)})
    defaults = dict(
        config=cfg,
        tokasaurus_url=kw.get("tokasaurus_url", TOKASAURUS_URL),
        cartridges=kw.get("cartridges", []),
        max_retries=3,
        base_timeout=120.0,
    )
    with patch(
        "verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server.omega_conf_to_dataclass",
        return_value=cfg,
    ):
        return TokasaurusHttpServer(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Direct TokasaurusHttpServer.generate() against live server
# ---------------------------------------------------------------------------

class TestLiveGenerate:
    """Tests the TokasaurusHttpServer class directly (no Ray) against the live Modal server."""

    @requires_server
    @pytest.mark.asyncio
    async def test_ping(self):
        server = _make_server()
        assert await server.ping()

    @requires_server
    @pytest.mark.asyncio
    async def test_generate_text_prompt(self):
        """Generate with a text prompt — basic smoke test."""
        server = _make_server()
        # Token IDs for "The capital of France is" (Llama tokenizer)
        prompt_ids = [1, 450, 7483, 310, 6181, 338]

        result = await server.generate(
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 32},
            request_id="integration-test-001",
        )

        assert isinstance(result, TokenOutput)
        assert len(result.token_ids) > 0
        assert result.stop_reason is not None

    @requires_server
    @pytest.mark.asyncio
    async def test_generate_with_logprobs(self):
        """Generate and extract logprobs — critical for KL distillation."""
        server = _make_server()
        prompt_ids = [1, 450, 7483, 310, 6181, 338]

        result = await server.generate(
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "logprobs": True, "max_new_tokens": 16},
            request_id="integration-test-logprobs",
        )

        assert isinstance(result, TokenOutput)
        assert len(result.token_ids) > 0
        assert result.log_probs is not None
        assert len(result.log_probs) == len(result.token_ids)
        # logprobs should be negative
        assert all(lp <= 0 for lp in result.log_probs)

    @requires_server
    @pytest.mark.asyncio
    async def test_generate_with_cartridge(self):
        """Generate with the HuggingFace pre-trained cartridge — the whole point."""
        server = _make_server(
            cartridges=[{
                "id": "hazyresearch/cartridge-wauoq23f",
                "source": "huggingface",
                "force_redownload": False,
            }]
        )

        result = await server.generate(
            prompt_ids=[1],  # just BOS — cartridge provides the context
            sampling_params={"temperature": 0.0, "max_new_tokens": 64},
            request_id="integration-test-cartridge",
        )

        assert isinstance(result, TokenOutput)
        assert len(result.token_ids) > 0


# ---------------------------------------------------------------------------
# Test 2: TokasaurusHttpServer as a Ray actor (how veRL actually uses it)
# ---------------------------------------------------------------------------

class TestRayActor:
    """Tests the TokasaurusHttpServer wrapped as a Ray actor."""

    @requires_server
    @pytest.mark.asyncio
    async def test_ray_actor_generate(self):
        """Create a Ray actor and call generate.remote() — exactly what veRL does."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)

        try:
            cfg = _mock_rollout_config()
            ServerActor = ray.remote(TokasaurusHttpServer)

            with patch(
                "verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server.omega_conf_to_dataclass",
                return_value=cfg,
            ):
                actor = ServerActor.remote(
                    config=cfg,
                    tokasaurus_url=TOKASAURUS_URL,
                    cartridges=[],
                    max_retries=3,
                    base_timeout=120.0,
                )

            # Ping
            alive = await actor.ping.remote()
            assert alive, "Ray actor ping failed"

            # Generate
            result = await actor.generate.remote(
                prompt_ids=[1, 450, 7483, 310, 6181, 338],
                sampling_params={"temperature": 0.0, "max_new_tokens": 16},
                request_id="ray-integration-001",
            )

            assert isinstance(result, TokenOutput)
            assert len(result.token_ids) > 0

            # Update cartridges (Phase 3 sync)
            await actor.update_cartridges.remote([
                {"id": "hazyresearch/cartridge-wauoq23f", "source": "huggingface"}
            ])

            result2 = await actor.generate.remote(
                prompt_ids=[1],
                sampling_params={"temperature": 0.0, "max_new_tokens": 32},
                request_id="ray-integration-cartridge",
            )
            assert len(result2.token_ids) > 0

        finally:
            ray.shutdown()


# ---------------------------------------------------------------------------
# Test 3: TokasaurusReplica.launch_servers() — the full veRL entry point
# ---------------------------------------------------------------------------

def _serializable_rollout_config(**custom_overrides):
    """Build a Ray-serializable config (SimpleNamespace, not MagicMock).

    Ray needs to serialize config when passing to remote actors.
    MagicMock causes infinite recursion during serialization.
    """
    custom = {
        "tokasaurus_url": TOKASAURUS_URL,
        "cartridges": [],
        **custom_overrides,
    }
    return SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        response_length=64,
        custom=custom,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


class TestReplica:
    """Tests TokasaurusReplica — the class veRL's trainer actually instantiates."""

    @requires_server
    @pytest.mark.asyncio
    async def test_replica_launch_and_generate(self):
        """Full flow: create replica → launch_servers → generate via server handle."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)

        try:
            config = _serializable_rollout_config()
            model_config = SimpleNamespace()

            with patch(
                "verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server.omega_conf_to_dataclass",
                return_value=config,
            ):
                replica = TokasaurusReplica(
                    replica_rank=0,
                    config=config,
                    model_config=model_config,
                    gpus_per_node=1,
                )

            # launch_servers creates the Ray actor and pings it
            await replica.launch_servers()

            assert len(replica.servers) == 1
            assert replica._server_address == TOKASAURUS_URL

            # Generate through the server handle (exactly what AsyncLLMServerManager does)
            server = replica.servers[0]
            result = await server.generate.remote(
                prompt_ids=[1, 450, 7483],
                sampling_params={"temperature": 0.0, "logprobs": True, "max_new_tokens": 16},
                request_id="replica-test-001",
            )

            assert isinstance(result, TokenOutput)
            assert len(result.token_ids) > 0
            assert result.log_probs is not None

            # Test cartridge sync
            await replica.sync_cartridge("hazyresearch/cartridge-wauoq23f")

        finally:
            ray.shutdown()
