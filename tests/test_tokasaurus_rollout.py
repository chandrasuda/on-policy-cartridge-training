# Copyright 2025 - Tokasaurus veRL Integration
# Licensed under the Apache License, Version 2.0
"""
Unit tests for the Tokasaurus rollout bridge.

Tests request formatting, response parsing, and the generate() flow
using mocks (no GPU or Tokasaurus server required).

Run: pytest tests/workers/rollout/test_tokasaurus_rollout.py -v
"""

import base64
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server import (
    TokasaurusHttpServer,
)
from verl.workers.rollout.replica import TokenOutput, RolloutReplicaRegistry


# ---- helpers ----


def _mock_config():
    cfg = MagicMock()
    cfg.temperature = 0.7
    cfg.top_p = 1.0
    cfg.response_length = 512
    cfg.custom = {"tokasaurus_url": "http://localhost:10210"}
    return cfg


def _fake_response(token_ids: list[int], logprobs: list[float] | None = None) -> dict:
    """Tokasaurus-style Completion response with system_fingerprint JSON."""
    fp: dict = {"completion_ids": [token_ids]}
    if logprobs is not None:
        fp["packed_chosen_logprobs"] = [
            base64.b64encode(np.array(logprobs, dtype=np.float32).tobytes()).decode()
        ]
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "choices": [{"text": "hello", "index": 0, "finish_reason": "stop"}],
        "system_fingerprint": json.dumps(fp),
    }


def _make_server(**kw):
    cfg = _mock_config()
    defaults = dict(
        config=cfg,
        tokasaurus_url="http://localhost:10210",
        cartridges=[{"id": "test-cart", "source": "local"}],
        max_retries=2,
        base_timeout=10.0,
    )
    defaults.update(kw)
    with patch(
        "verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server.omega_conf_to_dataclass",
        return_value=cfg,
    ):
        return TokasaurusHttpServer(**defaults)


# ---- response parsing ----


class TestParse:
    def test_token_ids(self):
        out = TokasaurusHttpServer._parse(_fake_response([10, 20, 30]), False)
        assert out.token_ids == [10, 20, 30]
        assert out.log_probs is None

    def test_with_logprobs(self):
        lp = [-0.5, -1.2, -0.1]
        out = TokasaurusHttpServer._parse(_fake_response([10, 20, 30], lp), True)
        assert len(out.log_probs) == 3
        np.testing.assert_allclose(out.log_probs, lp, atol=1e-6)

    def test_empty(self):
        out = TokasaurusHttpServer._parse(_fake_response([]), False)
        assert out.token_ids == []

    def test_missing_fingerprint(self):
        out = TokasaurusHttpServer._parse({"choices": [{"text": "hi"}]}, False)
        assert out.token_ids == []

    def test_corrupt_logprobs_returns_none(self):
        resp = _fake_response([1, 2])
        fp = json.loads(resp["system_fingerprint"])
        fp["packed_chosen_logprobs"] = ["not-valid-base64!!!"]
        resp["system_fingerprint"] = json.dumps(fp)
        out = TokasaurusHttpServer._parse(resp, True)
        assert out.log_probs is None


# ---- request payload ----


class TestGenerate:
    @pytest.mark.asyncio
    async def test_payload_format(self):
        server = _make_server()
        captured = {}

        async def mock_post(endpoint, payload):
            captured.update(payload)
            return _fake_response([42, 43])

        server._post = mock_post
        await server.generate([1, 2, 3], {"temperature": 0.5, "top_p": 0.9}, "req-1")

        assert captured["model"] == "default"
        assert captured["prompt"] == [1, 2, 3]
        assert captured["temperature"] == 0.5
        assert captured["top_p"] == 0.9
        assert captured["logprobs_in_fingerprint"] is True
        assert captured["cartridges"] == [{"id": "test-cart", "source": "local"}]

    @pytest.mark.asyncio
    async def test_max_new_tokens(self):
        server = _make_server()
        captured = {}
        server._post = lambda e, p: _set_and_return(captured, p, _fake_response([1]))
        await server.generate([1], {"max_new_tokens": 256}, "req")
        assert captured["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_no_cartridges(self):
        server = _make_server(cartridges=[])
        captured = {}
        server._post = lambda e, p: _set_and_return(captured, p, _fake_response([1]))
        await server.generate([1], {}, "req")
        assert "cartridges" not in captured

    @pytest.mark.asyncio
    async def test_does_not_mutate_params(self):
        server = _make_server()
        server._post = lambda e, p: asyncio.coroutine(lambda: _fake_response([1]))()
        params = {"max_new_tokens": 128, "logprobs": True, "temperature": 0.5}
        original = dict(params)

        import asyncio

        async def mock_post(e, p):
            return _fake_response([1])

        server._post = mock_post
        await server.generate([1], params, "req")
        assert params == original  # caller's dict unchanged


# ---- cartridge sync (Phase 3) ----


class TestCartridgeSync:
    def test_update_cartridges(self):
        server = _make_server()
        assert server.cartridges == [{"id": "test-cart", "source": "local"}]

        server.update_cartridges([
            {"id": "/tmp/new.pt", "source": "local", "force_redownload": True}
        ])
        assert server.cartridges == [
            {"id": "/tmp/new.pt", "source": "local", "force_redownload": True}
        ]

    @pytest.mark.asyncio
    async def test_force_redownload_in_payload(self):
        """After update_cartridges, subsequent generate() sends force_redownload."""
        server = _make_server()
        server.update_cartridges([
            {"id": "/tmp/step_1.pt", "source": "local", "force_redownload": True}
        ])

        captured = {}
        server._post = lambda e, p: _set_and_return(captured, p, _fake_response([1]))
        await server.generate([1, 2], {}, "req")

        assert captured["cartridges"] == [
            {"id": "/tmp/step_1.pt", "source": "local", "force_redownload": True}
        ]


# ---- registry ----


class TestRegistry:
    def test_tokasaurus_registered(self):
        from verl.workers.rollout.tokasaurus_rollout.async_tokasaurus_server import TokasaurusReplica

        assert RolloutReplicaRegistry.get("tokasaurus") is TokasaurusReplica


# ---- util ----

import asyncio


async def _set_and_return(store, payload, result):
    store.update(payload)
    return result
