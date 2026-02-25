# Copyright 2025 - Tokasaurus veRL Integration
# Licensed under the Apache License, Version 2.0
"""
Tokasaurus rollout server for veRL.

Connects veRL's rollout infrastructure to an external Tokasaurus inference server
that supports Cartridge KV tensor injection (bypassing prefill). This enables
on-policy Cartridge distillation by generating rollouts conditioned on a Cartridge.

Key differences from SGLang/vLLM rollout servers:
- Tokasaurus runs as an EXTERNAL server (not launched in-process)
- Requests include a `cartridges` array for KV tensor injection
- Uses /custom/cartridge/completions endpoint for token-in-token-out generation
"""

import asyncio
import base64
import json
import logging
import os
from typing import Any, Optional

import aiohttp
import numpy as np
import ray

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutReplica, TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 2.0


# ---------------------------------------------------------------------------
# TokasaurusHttpServer — Ray actor that proxies generate() to Tokasaurus
# ---------------------------------------------------------------------------

class TokasaurusHttpServer:
    """Proxies veRL generate() calls to an external Tokasaurus server.

    Accepts (prompt_ids, sampling_params, request_id) from AsyncLLMServerManager,
    POSTs to Tokasaurus /custom/cartridge/completions with the cartridges array,
    and returns TokenOutput with token_ids and optional logprobs.
    """

    def __init__(
        self,
        config: RolloutConfig,
        tokasaurus_url: str,
        cartridges: list[dict] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_timeout: float = DEFAULT_TIMEOUT,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.tokasaurus_url = tokasaurus_url.rstrip("/")
        self.cartridges = cartridges or []
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.retry_delay = retry_delay
        logger.info(f"TokasaurusHttpServer init: url={self.tokasaurus_url} cartridges={len(self.cartridges)}")

    # ---- public API (called by AsyncLLMServerManager via .remote()) ----

    async def ping(self) -> bool:
        """Health check. Allows up to 5 min for cold-start (Modal scales to zero)."""
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
                    async with s.get(f"{self.tokasaurus_url}/ping") as r:
                        if r.status == 200:
                            return True
            except Exception as e:
                logger.warning(f"Tokasaurus ping attempt {attempt+1}/5 failed: {e}")
                if attempt < 4:
                    await asyncio.sleep(15)
        return False

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate response tokens via Tokasaurus with Cartridge injection.

        Signature matches SGLangHttpServer.generate() so AsyncLLMServerManager
        can call server.generate.remote(...) identically.
        """
        # Don't mutate caller's dict
        params = dict(sampling_params)

        max_tokens = (
            params.pop("max_new_tokens", None)
            or params.pop("max_tokens", None)
            or self.config.response_length
        )
        return_logprob = params.pop("logprobs", False)

        payload: dict[str, Any] = {
            "model": "default",
            "prompt": list(prompt_ids) if not isinstance(prompt_ids, list) else prompt_ids,
            "max_tokens": max_tokens,
            "temperature": params.get("temperature", self.config.temperature),
            "top_p": params.get("top_p", self.config.top_p),
            "logprobs_in_fingerprint": True,
        }
        # Note: we do NOT send "logprobs": N here. Tokasaurus returns logprobs
        # via packed_chosen_logprobs in the system_fingerprint when
        # logprobs_in_fingerprint=True. Sending "logprobs" requires the server
        # to be configured with max_topk_logprobs, which may not be set.
        if self.cartridges:
            payload["cartridges"] = self.cartridges

        data = await self._post("/custom/cartridge/completions", payload)
        return self._parse(data, return_logprob)

    # ---- teacher logprobs (on-policy distillation) ----

    async def get_teacher_logprobs(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        document_ids: list[int],
        request_id: str,
    ) -> list[float]:
        """Get teacher logprobs for student-generated text.

        The teacher is the SAME model but with full document text as context
        instead of a cartridge. We send the document + question + student's
        response as one big prompt, ask for 1 token of generation, and extract
        the logprobs of the student's response tokens from the fingerprint.

        The trick: concatenate [document_ids + prompt_ids + response_ids] as
        the prompt, set max_tokens=1, and read packed_chosen_logprobs from
        the fingerprint. Tokasaurus evaluates logprobs on the entire prompt
        when logprobs_in_fingerprint=True.
        """
        # Full context: documents + question + student's response
        full_prompt = document_ids + prompt_ids + response_ids

        payload: dict[str, Any] = {
            "model": "default",
            "prompt": full_prompt,
            "max_tokens": 1,  # We only care about logprobs on the prompt tokens
            "temperature": 0.0,
            "logprobs_in_fingerprint": True,
            # NO cartridges — teacher sees full document context
        }

        data = await self._post("/custom/cartridge/completions", payload)
        fp = json.loads(data.get("system_fingerprint", "{}"))

        packed = fp.get("packed_chosen_logprobs")
        if packed and packed[0]:
            import base64
            all_logprobs = np.frombuffer(base64.b64decode(packed[0]), dtype=np.float32).tolist()
            # Extract only the logprobs for the response tokens
            # (skip document + prompt tokens)
            doc_prompt_len = len(document_ids) + len(prompt_ids)
            teacher_logprobs = all_logprobs[doc_prompt_len - 1:]  # shifted by 1 for next-token prediction
            return teacher_logprobs[:len(response_ids)]
        else:
            logger.warning(f"No teacher logprobs in response for {request_id}")
            return [0.0] * len(response_ids)

    # ---- cartridge sync (Phase 3) ----

    def update_cartridges(self, cartridges: list[dict]):
        """Replace the cartridge config for subsequent generate() calls.

        Typical usage after an RL optimizer step:
            1. ``cache.save("/shared/path/cartridge.pt")``
            2. ``server.update_cartridges.remote([{"id": "/shared/path/cartridge.pt",
                                                    "source": "local",
                                                    "force_redownload": True}])``
        Tokasaurus sees ``force_redownload: True`` and reloads the tensor from disk.
        """
        self.cartridges = cartridges
        logger.info(f"Updated cartridges config ({len(cartridges)} entries)")

    # ---- no-op lifecycle (external server) ----

    async def wake_up(self):
        pass

    async def sleep(self):
        pass

    async def clear_kv_cache(self):
        pass

    # ---- internals ----

    async def _post(self, endpoint: str, payload: dict) -> dict:
        """POST with exponential-backoff retries."""
        url = f"{self.tokasaurus_url}{endpoint}"
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.base_timeout * (1.5 ** attempt))
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}: {text}")
                        return await resp.json()
            except Exception as e:
                last_err = e
                logger.warning(f"Tokasaurus request failed ({attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError(f"Tokasaurus unreachable after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _parse(response: dict, return_logprob: bool) -> TokenOutput:
        """Parse Tokasaurus Completion → TokenOutput.

        Tokasaurus packs completion_ids and logprobs into the system_fingerprint JSON field.
        """
        fp = json.loads(response.get("system_fingerprint", "{}"))
        token_ids = fp.get("completion_ids", [[]])[0]

        if not token_ids and "choices" in response:
            logger.warning("No completion_ids in fingerprint; falling back to empty token list")
            token_ids = []

        log_probs = None
        if return_logprob:
            packed = fp.get("packed_chosen_logprobs")
            if packed:
                try:
                    log_probs = np.frombuffer(base64.b64decode(packed[0]), dtype=np.float32).tolist()
                except Exception as e:
                    logger.warning(f"Failed to decode logprobs: {e}")

        return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason="completed")


# Ray-remote wrapper
_TokasaurusServerActor = ray.remote(TokasaurusHttpServer)


# ---------------------------------------------------------------------------
# ServerAdapter — required by veRL's hybrid engine _build_rollout()
# ---------------------------------------------------------------------------

class ServerAdapter:
    """Minimal adapter for veRL's _ROLLOUT_REGISTRY.

    In hybrid engine mode, veRL calls ``get_rollout_class(name, mode)`` which
    expects a class with ``__init__(config, model_config, device_mesh)``.
    Tokasaurus runs externally so this adapter is mostly a no-op — the real
    work happens in TokasaurusReplica / TokasaurusHttpServer.
    """

    def __init__(self, config=None, model_config=None, device_mesh=None):
        self.config = config
        self.model_config = model_config

    def generate_sequences(self, *args, **kwargs):
        raise NotImplementedError("Tokasaurus uses external HTTP server, not in-process rollout")

    def init_weight_from_actor(self, *args, **kwargs):
        pass  # external server — no weight sync needed

    def update_weight_from_actor(self, *args, **kwargs):
        pass  # external server

    def resume_from_checkpoint(self, *args, **kwargs):
        pass

    async def resume(self, *args, **kwargs):
        pass  # external server — no weight sync

    async def offload(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        """Catch-all for any veRL lifecycle methods we haven't explicitly defined."""
        def noop(*args, **kwargs):
            pass
        async def async_noop(*args, **kwargs):
            pass
        # Return async noop for methods that are likely awaited
        if name in ("resume", "offload", "sleep", "wake_up", "clear_kv_cache"):
            return async_noop
        return noop


# ---------------------------------------------------------------------------
# TokasaurusReplica — veRL RolloutReplica for external Tokasaurus
# ---------------------------------------------------------------------------

class TokasaurusReplica(RolloutReplica):
    """Connects to a pre-running Tokasaurus server via HTTP.

    Configuration via ``RolloutConfig.custom``::

        actor_rollout_ref:
          rollout:
            name: tokasaurus
            custom:
              tokasaurus_url: "http://localhost:10210"
              cartridges:
                - id: "/tmp/cartridge.pt"
                  source: "local"
                  force_redownload: true
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        custom = self.config.custom or {}
        self.tokasaurus_url: str = custom.get("tokasaurus_url", "http://localhost:10210")
        self.cartridges: list[dict] = custom.get("cartridges", [])
        self.max_retries: int = custom.get("max_retries", DEFAULT_MAX_RETRIES)
        self.base_timeout: float = custom.get("base_timeout", DEFAULT_TIMEOUT)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        raise NotImplementedError("Tokasaurus runs externally; no in-process worker mode.")

    def rollout_worker_use_gpu(self) -> bool:
        return False  # HTTP proxy actor needs no GPU

    async def launch_servers(self):
        """Create TokasaurusHttpServer actor and verify connectivity."""
        server = _TokasaurusServerActor.options(
            name=f"tokasaurus_server_{self.replica_rank}",
        ).remote(
            config=self.config,
            tokasaurus_url=self.tokasaurus_url,
            cartridges=self.cartridges,
            max_retries=self.max_retries,
            base_timeout=self.base_timeout,
        )
        self.servers.append(server)

        if not await server.ping.remote():
            raise ConnectionError(
                f"Cannot reach Tokasaurus at {self.tokasaurus_url}. "
                f"Ensure it is running (geoff/cartridges branch)."
            )
        logger.info(f"TokasaurusReplica {self.replica_rank}: connected to {self.tokasaurus_url}")
        self._server_handle = server
        self._server_address = self.tokasaurus_url

    async def sync_cartridge(self, cartridge_path: str):
        """Tell all server actors to reload the cartridge from *cartridge_path*.

        Call this after the training worker saves an updated cartridge::

            worker.save_cartridge("/shared/cartridge.pt")
            await replica.sync_cartridge("/shared/cartridge.pt")
            # next generate() calls will use the updated cartridge
        """
        new_cartridges = [
            {"id": cartridge_path, "source": "local", "force_redownload": True}
        ]
        await asyncio.gather(
            *[server.update_cartridges.remote(new_cartridges) for server in self.servers]
        )
        logger.info(f"Synced cartridge to all servers: {cartridge_path}")

    async def wake_up(self):
        pass  # external server

    async def sleep(self):
        pass  # external server

    async def clear_kv_cache(self):
        pass  # external server
